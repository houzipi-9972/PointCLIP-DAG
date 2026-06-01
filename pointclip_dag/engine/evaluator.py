from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pointclip_dag.utils.metrics import IoUMeter
from pointclip_dag.utils.misc import move_to_device
from pointclip_dag.utils.prompt_metrics import PromptConsistencyAccumulator, make_prompt_variant_groups
from pointclip_dag.utils.semantic_metrics import (
    Agreement2D3DAccumulator,
    SemanticMetricAccumulator,
    compute_present_miou,
    semantic_confusions,
    text_similarity_baseline,
)


class Evaluator:
    def __init__(
        self,
        cfg,
        model,
        dataloader,
        vocabulary,
        device,
        logger=None,
        out_dir=None,
        semantic_probe_vocabulary=None,
        train_vocabulary=None,
    ):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.vocabulary = vocabulary
        self.semantic_probe_vocabulary = semantic_probe_vocabulary
        self.train_vocabulary = train_vocabulary
        self.device = device
        self.logger = logger
        self.out_dir = Path(out_dir) if out_dir else None

    @torch.no_grad()
    def evaluate(self, save_predictions: bool = False) -> dict:
        self.model.eval()
        self.model.set_vocabulary(self.vocabulary)
        meter3d = IoUMeter(self.vocabulary.num_classes, ignore_index=self.cfg.loss.ignore_index)
        meter2d = IoUMeter(self.vocabulary.num_classes, ignore_index=self.cfg.loss.ignore_index)
        ensemble_weights = _ensemble_weights(self.cfg)
        meter_ens = {
            float(alpha): IoUMeter(self.vocabulary.num_classes, ignore_index=self.cfg.loss.ignore_index)
            for alpha in ensemble_weights
        }
        pred_hist = np.zeros(self.vocabulary.num_classes, dtype=np.int64)
        gt_hist = np.zeros(self.vocabulary.num_classes, dtype=np.int64)
        pred2d_hist = np.zeros(self.vocabulary.num_classes, dtype=np.int64)
        gt2d_hist = np.zeros(self.vocabulary.num_classes, dtype=np.int64)
        logit_sum = np.zeros(self.vocabulary.num_classes, dtype=np.float64)
        logit_count = 0
        unseen_topk_hits = {1: np.zeros(self.vocabulary.num_classes, dtype=np.int64), 3: np.zeros(self.vocabulary.num_classes, dtype=np.int64), 5: np.zeros(self.vocabulary.num_classes, dtype=np.int64)}
        unseen_gt_count = np.zeros(self.vocabulary.num_classes, dtype=np.int64)
        probe_hist = None
        probe_hist_2d = None
        probe_names = []
        probe_text = None
        if self.semantic_probe_vocabulary is not None:
            probe_names = self.semantic_probe_vocabulary.names
            probe_hist = np.zeros(self.semantic_probe_vocabulary.num_classes, dtype=np.int64)
            probe_hist_2d = np.zeros(self.semantic_probe_vocabulary.num_classes, dtype=np.int64)
        ignored_points = 0
        total_points = 0
        projected_points = 0
        ignored_projected_points = 0

        pred_dir = None
        if save_predictions and self.out_dir is not None:
            pred_dir = self.out_dir / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)

        sem_cfg = self.cfg.get("semantic_metrics", {})
        sem_enabled = bool(sem_cfg.get("enabled", True))
        semantic_acc = SemanticMetricAccumulator(
            self.vocabulary.num_classes,
            self.vocabulary.seen_mask.cpu().numpy(),
            threshold=float(sem_cfg.get("semantic_threshold", 0.85)),
        )
        agreement_acc = Agreement2D3DAccumulator(temperature=float(self.cfg.loss.get("temperature", 2.0)))
        prompt_acc = PromptConsistencyAccumulator()
        prompt_variant_accs = {
            "aliases_only": PromptConsistencyAccumulator(),
            "point_cloud": PromptConsistencyAccumulator(),
            "driving_scene": PromptConsistencyAccumulator(),
        }
        prompt_batches = int(sem_cfg.get("prompt_consistency_num_batches", 10))
        prompt_groups = make_prompt_variant_groups(self.vocabulary)
        seen_biases = [float(value) for value in sem_cfg.get("seen_logit_biases", [])]
        meter_seen_bias = {
            bias: IoUMeter(self.vocabulary.num_classes, ignore_index=self.cfg.loss.ignore_index)
            for bias in seen_biases
        }

        for step, batch in enumerate(self.dataloader):
            batch = move_to_device(batch, self.device)
            outputs = self.model(batch)
            labels_t = outputs["point_labels"]
            logits3d_t = outputs["logits3d"]
            pred3d_t = logits3d_t.argmax(dim=-1)
            labels = labels_t.detach().cpu().numpy()
            pred3d = pred3d_t.detach().cpu().numpy()
            meter3d.update(pred3d, labels)
            total_points += labels.size
            valid_labels = (labels != self.cfg.loss.ignore_index) & (labels >= 0) & (labels < self.vocabulary.num_classes)
            ignored_points += int(labels.size - valid_labels.sum())
            if valid_labels.any():
                gt_hist += np.bincount(labels[valid_labels].astype(np.int64), minlength=self.vocabulary.num_classes)
                pred_hist += np.bincount(pred3d[valid_labels].astype(np.int64), minlength=self.vocabulary.num_classes)
                logit_sum += logits3d_t.detach().cpu().numpy()[valid_labels].sum(axis=0)
                logit_count += int(valid_labels.sum())
                _accumulate_unseen_topk(
                    unseen_topk_hits,
                    unseen_gt_count,
                    logits3d_t,
                    labels_t,
                    self.vocabulary.seen_mask.to(device=labels_t.device),
                    int(self.cfg.loss.ignore_index),
                )
                for bias, meter in meter_seen_bias.items():
                    calibrated = _apply_seen_bias(logits3d_t, self.vocabulary.seen_mask.to(logits3d_t.device), bias)
                    meter.update(calibrated.argmax(dim=-1).detach().cpu().numpy(), labels)
                if probe_hist is not None:
                    if probe_text is None:
                        probe_text = self.model.text_encoder.encode_groups(
                            self.semantic_probe_vocabulary.class_text_groups(),
                            device=self.device,
                        )
                        probe_text = torch.nn.functional.normalize(probe_text, dim=-1)
                    probe_logits = float(self.cfg.loss.get("logit_scale", 20.0)) * outputs["z3d"] @ probe_text.t()
                    probe_pred = probe_logits.argmax(dim=-1).detach().cpu().numpy()
                    probe_hist += np.bincount(probe_pred[valid_labels].astype(np.int64), minlength=len(probe_names))
            if sem_enabled and sem_cfg.get("compute_semantic_similarity", True):
                semantic_acc.update(
                    pred3d_t,
                    labels_t,
                    logits3d_t,
                    outputs["text_embeddings"],
                    int(self.cfg.loss.ignore_index),
                )
            if sem_enabled and sem_cfg.get("compute_prompt_consistency", True) and step < prompt_batches:
                alt_text = self.model.text_encoder.encode_groups(prompt_groups, device=self.device)
                alt_text = torch.nn.functional.normalize(alt_text, dim=-1)
                alt_logits = float(self.cfg.loss.get("logit_scale", 20.0)) * outputs["z3d"] @ alt_text.t()
                prompt_acc.update(logits3d_t, alt_logits, labels_t, int(self.cfg.loss.ignore_index))
                for variant_name, variant_acc in prompt_variant_accs.items():
                    variant_groups = make_prompt_variant_groups(self.vocabulary, mode=variant_name)
                    variant_text = self.model.text_encoder.encode_groups(variant_groups, device=self.device)
                    variant_text = torch.nn.functional.normalize(variant_text, dim=-1)
                    variant_logits = float(self.cfg.loss.get("logit_scale", 20.0)) * outputs["z3d"] @ variant_text.t()
                    variant_acc.update(logits3d_t, variant_logits, labels_t, int(self.cfg.loss.ignore_index))

            valid_mask = outputs["valid_point_mask"]
            if valid_mask.numel() == labels.shape[0] and valid_mask.any():
                labels_valid_t = outputs["labels_valid"]
                labels_valid = labels_valid_t.detach().cpu().numpy()
                pred2d_t = outputs["logits2d_points"].argmax(dim=-1)
                pred2d = pred2d_t.detach().cpu().numpy()
                meter2d.update(pred2d, labels_valid)
                for alpha, meter in meter_ens.items():
                    ens_logits = alpha * outputs["logits2d_points"] + (1.0 - alpha) * outputs["logits3d_valid"]
                    meter.update(ens_logits.argmax(dim=-1).detach().cpu().numpy(), labels_valid)
                if sem_enabled and sem_cfg.get("compute_2d3d_agreement", True):
                    agreement_acc.update(
                        outputs["logits2d_points"],
                        outputs["logits3d_valid"],
                        outputs["z2d_points"],
                        outputs["z3d_valid"],
                        labels_valid_t,
                        int(self.cfg.loss.ignore_index),
                    )
                projected_points += labels_valid.size
                valid_2d_labels = (
                    (labels_valid != self.cfg.loss.ignore_index)
                    & (labels_valid >= 0)
                    & (labels_valid < self.vocabulary.num_classes)
                )
                ignored_projected_points += int(labels_valid.size - valid_2d_labels.sum())
                if valid_2d_labels.any():
                    gt2d_hist += np.bincount(
                        labels_valid[valid_2d_labels].astype(np.int64),
                        minlength=self.vocabulary.num_classes,
                    )
                    pred2d_hist += np.bincount(
                        pred2d[valid_2d_labels].astype(np.int64),
                        minlength=self.vocabulary.num_classes,
                    )
                    if probe_hist_2d is not None:
                        if probe_text is None:
                            probe_text = self.model.text_encoder.encode_groups(
                                self.semantic_probe_vocabulary.class_text_groups(),
                                device=self.device,
                            )
                            probe_text = torch.nn.functional.normalize(probe_text, dim=-1)
                        probe_logits_2d = float(self.cfg.loss.get("logit_scale", 20.0)) * outputs["z2d_points"] @ probe_text.t()
                        probe_pred_2d = probe_logits_2d.argmax(dim=-1).detach().cpu().numpy()
                        probe_hist_2d += np.bincount(
                            probe_pred_2d[valid_2d_labels].astype(np.int64),
                            minlength=len(probe_names),
                        )

            if pred_dir is not None:
                np.save(pred_dir / f"batch_{step:06d}_pred3d.npy", pred3d)

        result3d = meter3d.compute(self.vocabulary.seen_mask.cpu().numpy())
        result2d = meter2d.compute(self.vocabulary.seen_mask.cpu().numpy())
        result_ens = {
            alpha: meter.compute(self.vocabulary.seen_mask.cpu().numpy()) for alpha, meter in meter_ens.items()
        }
        present_metrics = compute_present_miou(
            result3d["per_class_iou"],
            gt_hist,
            self.vocabulary.seen_mask.cpu().numpy(),
        )
        best_alpha, best_ens = _best_ensemble(result_ens)
        alpha05 = _closest_alpha(result_ens, 0.5)
        per_class_best_ens, per_class_best_alpha = _best_ensemble_per_class(result_ens, self.vocabulary.num_classes)
        metrics = {
            "all_mIoU": result3d["all_miou"],
            "mIoU_all_vocab": present_metrics["mIoU_all_vocab"],
            "present_mIoU": present_metrics["present_mIoU"],
            "seen_mIoU": result3d.get("seen_miou", 0.0),
            "unseen_mIoU": result3d.get("unseen_miou", 0.0),
            "seen_present_mIoU": present_metrics["seen_present_mIoU"],
            "unseen_present_mIoU": present_metrics["unseen_present_mIoU"],
            "mIoU_3d": result3d["all_miou"],
            "mIoU_2d_projected": result2d["all_miou"],
            "mIoU_ensemble": result_ens[alpha05]["all_miou"] if result_ens else 0.0,
            "best_ensemble_alpha": best_alpha,
            "best_ensemble_mIoU": best_ens,
            "per_class_iou": {
                name: _safe_float(result3d["per_class_iou"][idx]) for idx, name in enumerate(self.vocabulary.names)
            },
            "per_class_iou_2d_projected": {
                name: _safe_float(result2d["per_class_iou"][idx]) for idx, name in enumerate(self.vocabulary.names)
            },
            "per_class_iou_ensemble_best": {
                name: _safe_float(per_class_best_ens[idx]) for idx, name in enumerate(self.vocabulary.names)
            },
            "per_class_best_ensemble_alpha": {
                name: float(per_class_best_alpha[idx]) for idx, name in enumerate(self.vocabulary.names)
            },
            "per_class_count": {name: int(gt_hist[idx]) for idx, name in enumerate(self.vocabulary.names)},
            "per_class_seen": {name: bool(self.vocabulary.seen_mask[idx].item()) for idx, name in enumerate(self.vocabulary.names)},
            "pred_class_hist": {name: int(pred_hist[idx]) for idx, name in enumerate(self.vocabulary.names)},
            "gt_class_hist": {name: int(gt_hist[idx]) for idx, name in enumerate(self.vocabulary.names)},
            "pred_2d_projected_class_hist": {
                name: int(pred2d_hist[idx]) for idx, name in enumerate(self.vocabulary.names)
            },
            "gt_2d_projected_class_hist": {
                name: int(gt2d_hist[idx]) for idx, name in enumerate(self.vocabulary.names)
            },
            "ignored_point_ratio": float(ignored_points / max(total_points, 1)),
            "valid_projected_point_ratio": float(projected_points / max(total_points, 1)),
            "ignored_projected_label_ratio": float(ignored_projected_points / max(projected_points, 1)),
            "confusion_matrix": result3d["confusion_matrix"].tolist(),
        }
        for alpha, result in result_ens.items():
            metrics[f"mIoU_ensemble_alpha_{alpha}"] = result["all_miou"]
        if sem_enabled:
            metrics.update(semantic_acc.compute(self.vocabulary.names))
            metrics.update(agreement_acc.compute())
            metrics.update(prompt_acc.compute())
            for variant_name, variant_acc in prompt_variant_accs.items():
                for key, value in variant_acc.compute().items():
                    metrics[f"{key}_{variant_name}"] = value
            metrics["semantic_threshold"] = float(sem_cfg.get("semantic_threshold", 0.85))
            metrics.update(text_similarity_baseline(self.model.encode_text(self.device)))
            metrics["semantic_confusions"] = semantic_confusions(
                result3d["confusion_matrix"],
                self.model.encode_text(self.device),
                self.vocabulary.names,
                topk=int(sem_cfg.get("topk_confusions", 5)),
            )
            metrics.update(_unseen_topk_metrics(unseen_topk_hits, unseen_gt_count, self.vocabulary.names))
            metrics["class_logit_mean_3d"] = {
                name: _safe_float(logit_sum[idx] / max(logit_count, 1)) for idx, name in enumerate(self.vocabulary.names)
            }
            for bias, meter in meter_seen_bias.items():
                biased = meter.compute(self.vocabulary.seen_mask.cpu().numpy())
                metrics[f"mIoU_seen_bias_{bias}"] = biased["all_miou"]
        else:
            metrics.update(_empty_semantic_metrics())
        if self.semantic_probe_vocabulary is not None:
            metrics["semantic_probe_vocab_classes"] = self.semantic_probe_vocabulary.names
            if probe_hist is not None:
                metrics["semantic_probe_pred_hist_3d"] = {
                    name: int(probe_hist[idx]) for idx, name in enumerate(probe_names)
                }
                metrics["semantic_probe_top_classes_3d"] = _top_hist_items(probe_hist, probe_names, topk=10)
            if probe_hist_2d is not None:
                metrics["semantic_probe_pred_hist_2d_projected"] = {
                    name: int(probe_hist_2d[idx]) for idx, name in enumerate(probe_names)
                }
                metrics["semantic_probe_top_classes_2d_projected"] = _top_hist_items(probe_hist_2d, probe_names, topk=10)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.out_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            _write_metrics_summary_txt(
                self.out_dir / "metrics_summary.txt",
                metrics,
                self.train_vocabulary,
                self.vocabulary,
                result3d,
                result2d,
                result_ens,
                per_class_best_ens,
                per_class_best_alpha,
            )
        if self.logger:
            self.logger.info(
                "eval all_mIoU=%.4f seen=%.4f unseen=%.4f 2d=%.4f ensemble=%.4f ignored=%.4f",
                metrics["all_mIoU"],
                metrics["seen_mIoU"],
                metrics["unseen_mIoU"],
                metrics["mIoU_2d_projected"],
                metrics["mIoU_ensemble"],
                metrics["ignored_point_ratio"],
            )
            self.logger.info(
                "eval present_mIoU=%.4f seen_present=%.4f unseen_present=%.4f best_ensemble(alpha=%.2f)=%.4f",
                metrics["present_mIoU"],
                metrics["seen_present_mIoU"],
                metrics["unseen_present_mIoU"],
                metrics["best_ensemble_alpha"],
                metrics["best_ensemble_mIoU"],
            )
            self.logger.info("eval pred_hist=%s", metrics["pred_class_hist"])
            self.logger.info("eval gt_hist=%s", metrics["gt_class_hist"])
            self.logger.info("eval 2d_pred_hist=%s", metrics["pred_2d_projected_class_hist"])
            self.logger.info("eval 2d_gt_hist=%s", metrics["gt_2d_projected_class_hist"])
            self.logger.info(
                "eval projected valid_ratio=%.4f ignored_projected_label_ratio=%.4f",
                metrics["valid_projected_point_ratio"],
                metrics["ignored_projected_label_ratio"],
            )
        return metrics


def _safe_float(value) -> float:
    return 0.0 if np.isnan(value) else float(value)


def _ensemble_weights(cfg):
    ens_cfg = cfg.get("ensemble", {})
    if not bool(ens_cfg.get("enabled", True)):
        return [0.5]
    weights = ens_cfg.get("weights", [0.0, 0.25, 0.5, 0.75, 1.0])
    return [float(value) for value in weights]


def _best_ensemble(results):
    if not results:
        return 0.0, 0.0
    best_alpha = max(results, key=lambda alpha: results[alpha]["all_miou"])
    return float(best_alpha), float(results[best_alpha]["all_miou"])


def _best_ensemble_per_class(results, num_classes):
    best_iou = np.zeros(num_classes, dtype=np.float64)
    best_alpha = np.zeros(num_classes, dtype=np.float64)
    if not results:
        return best_iou, best_alpha
    for class_idx in range(num_classes):
        class_best_alpha = None
        class_best_iou = -np.inf
        for alpha, result in results.items():
            value = _safe_float(result["per_class_iou"][class_idx])
            if value > class_best_iou:
                class_best_iou = value
                class_best_alpha = alpha
        best_iou[class_idx] = max(class_best_iou, 0.0)
        best_alpha[class_idx] = float(class_best_alpha if class_best_alpha is not None else 0.0)
    return best_iou, best_alpha


def _closest_alpha(results, target):
    if not results:
        return 0.0
    return min(results, key=lambda alpha: abs(alpha - target))


def _empty_semantic_metrics():
    return {
        "semantic_similarity_score_all": 0.0,
        "semantic_similarity_score_seen": 0.0,
        "semantic_similarity_score_unseen": 0.0,
        "near_miss_semantic_accuracy_all": 0.0,
        "near_miss_semantic_accuracy_seen": 0.0,
        "near_miss_semantic_accuracy_unseen": 0.0,
        "exact_accuracy": 0.0,
        "near_miss_gain": 0.0,
        "agreement_2d3d_top1": 0.0,
        "agreement_2d3d_top5": 0.0,
        "feature_cosine_2d3d": 0.0,
        "kl_2d_to_3d": 0.0,
        "kl_3d_to_2d": 0.0,
        "prompt_top1_consistency": 0.0,
        "prompt_js_divergence": 0.0,
        "prompt_logit_cosine": 0.0,
        "text_alignment_margin_mean": 0.0,
        "text_alignment_margin_seen": 0.0,
        "text_alignment_margin_unseen": 0.0,
        "semantic_confusions": {},
    }


def _apply_seen_bias(logits, seen_mask, bias):
    calibrated = logits.clone()
    calibrated[:, seen_mask] = calibrated[:, seen_mask] - float(bias)
    return calibrated


def _accumulate_unseen_topk(hit_tables, gt_count, logits, labels, seen_mask, ignore_index):
    valid = (labels != ignore_index) & (labels >= 0) & (labels < seen_mask.numel())
    if valid.numel() == 0 or not bool(valid.any()):
        return
    safe_labels = labels.clamp(min=0, max=seen_mask.numel() - 1)
    unseen = valid & (~seen_mask[safe_labels])
    if not bool(unseen.any()):
        return
    labels_unseen = labels[unseen].detach().cpu().numpy().astype(np.int64)
    for class_idx, count in zip(*np.unique(labels_unseen, return_counts=True)):
        gt_count[int(class_idx)] += int(count)
    max_k = min(max(hit_tables), logits.shape[1])
    topk = logits[unseen].topk(max_k, dim=-1).indices
    labels_unseen_t = labels[unseen].view(-1, 1)
    for k, table in hit_tables.items():
        kk = min(k, topk.shape[1])
        hits = topk[:, :kk].eq(labels_unseen_t).any(dim=-1).detach().cpu().numpy()
        for class_idx in np.unique(labels_unseen):
            mask = labels_unseen == int(class_idx)
            table[int(class_idx)] += int(hits[mask].sum())


def _unseen_topk_metrics(hit_tables, gt_count, names):
    metrics = {}
    unseen_total = int(gt_count.sum())
    for k, table in hit_tables.items():
        metrics[f"unseen_top{k}_recall"] = float(table.sum() / max(unseen_total, 1))
        metrics[f"unseen_top{k}_recall_per_class"] = {
            name: (0.0 if gt_count[idx] == 0 else float(table[idx] / gt_count[idx])) for idx, name in enumerate(names)
        }
    return metrics


def _top_hist_items(hist, names, topk=10):
    order = np.argsort(-hist)[:topk]
    return [{"label": names[int(idx)], "count": int(hist[int(idx)])} for idx in order if int(hist[int(idx)]) > 0]


def _write_metrics_summary_txt(
    path,
    metrics,
    train_vocab,
    eval_vocab,
    result3d,
    result2d,
    result_ens,
    per_class_best_ens,
    per_class_best_alpha,
):
    lines = []
    lines.append("PointCLIP-DAG Evaluation Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Overall Metrics")
    lines.append("-" * 80)
    overall_rows = [
        ("all_mIoU", metrics.get("all_mIoU", 0.0)),
        ("present_mIoU", metrics.get("present_mIoU", 0.0)),
        ("seen_mIoU", metrics.get("seen_mIoU", 0.0)),
        ("unseen_mIoU", metrics.get("unseen_mIoU", 0.0)),
        ("mIoU_3d", metrics.get("mIoU_3d", 0.0)),
        ("mIoU_2d_projected", metrics.get("mIoU_2d_projected", 0.0)),
        ("mIoU_ensemble_alpha_0.5", metrics.get("mIoU_ensemble", 0.0)),
        ("best_ensemble_alpha", metrics.get("best_ensemble_alpha", 0.0)),
        ("best_ensemble_mIoU", metrics.get("best_ensemble_mIoU", 0.0)),
        ("valid_projected_point_ratio", metrics.get("valid_projected_point_ratio", 0.0)),
        ("ignored_projected_label_ratio", metrics.get("ignored_projected_label_ratio", 0.0)),
        ("unseen_top1_recall", metrics.get("unseen_top1_recall", 0.0)),
        ("unseen_top3_recall", metrics.get("unseen_top3_recall", 0.0)),
        ("unseen_top5_recall", metrics.get("unseen_top5_recall", 0.0)),
        ("prompt_top1_consistency", metrics.get("prompt_top1_consistency", 0.0)),
    ]
    lines.extend(_format_table(["metric", "value"], [(name, _fmt(value)) for name, value in overall_rows]))
    lines.append("")

    lines.append("Training Labels")
    lines.append("-" * 80)
    if train_vocab is None:
        lines.append("train_vocabulary was not provided to evaluator.")
    else:
        train_rows = []
        for idx, item in enumerate(train_vocab.classes):
            train_rows.append(
                (
                    idx,
                    item.name,
                    _raw_label_text(item),
                    item.mapping_label,
                    bool(item.seen),
                    ", ".join(item.aliases),
                )
            )
        lines.extend(_format_table(["id", "label", "raw_label", "mapping_label", "seen", "aliases"], train_rows))
    lines.append("")

    lines.append("Validation Labels")
    lines.append("-" * 80)
    val_rows = []
    iou3d = result3d["per_class_iou"]
    iou2d = result2d["per_class_iou"]
    for idx, item in enumerate(eval_vocab.classes):
        val_rows.append(
            (
                idx,
                item.name,
                _raw_label_text(item),
                item.mapping_label,
                bool(item.seen),
                int(metrics["per_class_count"].get(item.name, 0)),
                _fmt(_safe_float(iou3d[idx])),
                _fmt(_safe_float(iou2d[idx])),
                _fmt(_safe_float(per_class_best_ens[idx])),
                _fmt(float(per_class_best_alpha[idx]), digits=2),
            )
        )
    lines.extend(
        _format_table(
            ["id", "label", "raw_label", "mapping_label", "seen", "gt_count", "iou3d", "iou2d", "iouensemble", "alpha"],
            val_rows,
        )
    )
    lines.append("")

    lines.append("Ensemble Sweep")
    lines.append("-" * 80)
    sweep_rows = []
    for alpha in sorted(result_ens.keys()):
        sweep_rows.append((alpha, _fmt(result_ens[alpha]["all_miou"])))
    lines.extend(_format_table(["alpha", "mIoU"], sweep_rows))
    lines.append("")

    if metrics.get("semantic_probe_top_classes_3d"):
        lines.append("Semantic Probe Top Classes")
        lines.append("-" * 80)
        probe_rows = [(item["label"], item["count"]) for item in metrics["semantic_probe_top_classes_3d"]]
        lines.extend(_format_table(["label", "count"], probe_rows))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_table(headers, rows):
    rows = [tuple(str(cell) for cell in row) for row in rows]
    headers = [str(header) for header in headers]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    out = [sep]
    out.append("| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |")
    out.append(sep)
    for row in rows:
        out.append("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |")
    out.append(sep)
    return out


def _fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _raw_label_text(item):
    if not item.raw_labels:
        return ""
    if len(item.raw_labels) == 1:
        return str(item.raw_labels[0])
    return "[" + ",".join(str(value) for value in item.raw_labels) + "]"
