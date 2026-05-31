from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_present_miou(per_class_iou, gt_hist, seen_mask):
    per_class_iou = np.asarray(per_class_iou, dtype=np.float64)
    gt_hist = np.asarray(gt_hist, dtype=np.int64)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    present = gt_hist > 0
    return {
        "mIoU_all_vocab": _nanmean(per_class_iou),
        "present_mIoU": _nanmean(per_class_iou[present]),
        "seen_present_mIoU": _nanmean(per_class_iou[present & seen_mask]),
        "unseen_present_mIoU": _nanmean(per_class_iou[present & ~seen_mask]),
    }


class SemanticMetricAccumulator:
    def __init__(self, num_classes, seen_mask, threshold=0.55):
        self.num_classes = int(num_classes)
        self.seen_mask = torch.as_tensor(seen_mask, dtype=torch.bool)
        self.threshold = float(threshold)
        self.total = 0
        self.sss_sum = 0.0
        self.exact_sum = 0
        self.near_sum = 0
        self.seen_total = 0
        self.seen_sss_sum = 0.0
        self.seen_near_sum = 0
        self.unseen_total = 0
        self.unseen_sss_sum = 0.0
        self.unseen_near_sum = 0
        self.class_total = np.zeros(self.num_classes, dtype=np.int64)
        self.class_sss_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.margin_sum = 0.0
        self.margin_seen_sum = 0.0
        self.margin_unseen_sum = 0.0
        self.margin_class_sum = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, pred, target, logits, text_embeddings, ignore_index):
        valid = (target != ignore_index) & (target >= 0) & (target < self.num_classes)
        if valid.numel() == 0 or not bool(valid.any()):
            return
        pred = pred[valid].long()
        target = target[valid].long()
        logits = logits[valid]
        text = F.normalize(text_embeddings.float(), dim=-1)
        sim_table = text @ text.t()
        sims = sim_table[target, pred]
        exact = pred.eq(target)
        near = sims.ge(self.threshold)
        top2 = torch.topk(logits.float(), k=min(2, logits.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            margins = torch.zeros_like(top2[:, 0])
        else:
            margins = top2[:, 0] - top2[:, 1]

        seen = self.seen_mask.to(target.device)[target]
        self.total += int(target.numel())
        self.sss_sum += float(sims.sum().item())
        self.exact_sum += int(exact.sum().item())
        self.near_sum += int(near.sum().item())
        self.margin_sum += float(margins.sum().item())

        self.seen_total += int(seen.sum().item())
        self.seen_sss_sum += float(sims[seen].sum().item()) if bool(seen.any()) else 0.0
        self.seen_near_sum += int(near[seen].sum().item()) if bool(seen.any()) else 0
        unseen = ~seen
        self.unseen_total += int(unseen.sum().item())
        self.unseen_sss_sum += float(sims[unseen].sum().item()) if bool(unseen.any()) else 0.0
        self.unseen_near_sum += int(near[unseen].sum().item()) if bool(unseen.any()) else 0
        self.margin_seen_sum += float(margins[seen].sum().item()) if bool(seen.any()) else 0.0
        self.margin_unseen_sum += float(margins[unseen].sum().item()) if bool(unseen.any()) else 0.0

        target_np = target.detach().cpu().numpy()
        sims_np = sims.detach().cpu().numpy()
        margins_np = margins.detach().cpu().numpy()
        for cls in np.unique(target_np):
            mask = target_np == cls
            self.class_total[cls] += int(mask.sum())
            self.class_sss_sum[cls] += float(sims_np[mask].sum())
            self.margin_class_sum[cls] += float(margins_np[mask].sum())

    def compute(self, names):
        exact_acc = self.exact_sum / max(self.total, 1)
        nmsa_all = self.near_sum / max(self.total, 1)
        return {
            "semantic_similarity_score_all": self.sss_sum / max(self.total, 1),
            "semantic_similarity_score_seen": self.seen_sss_sum / max(self.seen_total, 1),
            "semantic_similarity_score_unseen": self.unseen_sss_sum / max(self.unseen_total, 1),
            "semantic_similarity_score_per_class": {
                name: float(self.class_sss_sum[idx] / max(self.class_total[idx], 1))
                for idx, name in enumerate(names)
            },
            "near_miss_semantic_accuracy_all": nmsa_all,
            "near_miss_semantic_accuracy_seen": self.seen_near_sum / max(self.seen_total, 1),
            "near_miss_semantic_accuracy_unseen": self.unseen_near_sum / max(self.unseen_total, 1),
            "exact_accuracy": exact_acc,
            "near_miss_gain": nmsa_all - exact_acc,
            "text_alignment_margin_mean": self.margin_sum / max(self.total, 1),
            "text_alignment_margin_seen": self.margin_seen_sum / max(self.seen_total, 1),
            "text_alignment_margin_unseen": self.margin_unseen_sum / max(self.unseen_total, 1),
            "text_alignment_margin_per_class": {
                name: float(self.margin_class_sum[idx] / max(self.class_total[idx], 1))
                for idx, name in enumerate(names)
            },
        }


class Agreement2D3DAccumulator:
    def __init__(self, temperature=2.0):
        self.temperature = float(temperature)
        self.total = 0
        self.top1 = 0
        self.top5 = 0
        self.feature_cosine_sum = 0.0
        self.kl_2d_to_3d_sum = 0.0
        self.kl_3d_to_2d_sum = 0.0

    def update(self, logits2d, logits3d, z2d, z3d, labels, ignore_index):
        valid = (labels != ignore_index) & (labels >= 0)
        if valid.numel() == 0 or not bool(valid.any()):
            return
        logits2d = logits2d[valid].float()
        logits3d = logits3d[valid].float()
        z2d = z2d[valid].float()
        z3d = z3d[valid].float()
        k = min(5, logits3d.shape[-1])
        pred2d = logits2d.argmax(dim=-1)
        pred3d = logits3d.argmax(dim=-1)
        top5_2d = torch.topk(logits2d, k=k, dim=-1).indices
        self.total += int(pred3d.numel())
        self.top1 += int(pred2d.eq(pred3d).sum().item())
        self.top5 += int((top5_2d == pred3d.unsqueeze(1)).any(dim=1).sum().item())
        self.feature_cosine_sum += float(F.cosine_similarity(z2d, z3d, dim=-1).sum().item())
        p2d = F.softmax(logits2d / self.temperature, dim=-1)
        p3d = F.softmax(logits3d / self.temperature, dim=-1)
        self.kl_2d_to_3d_sum += float((p2d * (p2d.clamp_min(1e-8).log() - p3d.clamp_min(1e-8).log())).sum(dim=-1).sum().item())
        self.kl_3d_to_2d_sum += float((p3d * (p3d.clamp_min(1e-8).log() - p2d.clamp_min(1e-8).log())).sum(dim=-1).sum().item())

    def compute(self):
        return {
            "agreement_2d3d_top1": self.top1 / max(self.total, 1),
            "agreement_2d3d_top5": self.top5 / max(self.total, 1),
            "feature_cosine_2d3d": self.feature_cosine_sum / max(self.total, 1),
            "kl_2d_to_3d": self.kl_2d_to_3d_sum / max(self.total, 1),
            "kl_3d_to_2d": self.kl_3d_to_2d_sum / max(self.total, 1),
        }


def semantic_confusions(confusion_matrix, text_embeddings, names, topk=5):
    confusion = np.asarray(confusion_matrix, dtype=np.int64)
    text = F.normalize(text_embeddings.detach().float().cpu(), dim=-1)
    sim = (text @ text.t()).numpy()
    out = {}
    for gt_idx, name in enumerate(names):
        row = confusion[gt_idx].copy()
        row[gt_idx] = 0
        pred_indices = np.argsort(row)[::-1]
        items = []
        for pred_idx in pred_indices:
            count = int(row[pred_idx])
            if count <= 0:
                continue
            items.append(
                {
                    "pred": names[pred_idx],
                    "count": count,
                    "text_similarity": float(sim[gt_idx, pred_idx]),
                }
            )
            if len(items) >= topk:
                break
        out[name] = items
    return out


def text_similarity_baseline(text_embeddings):
    text = F.normalize(text_embeddings.detach().float().cpu(), dim=-1)
    sim = text @ text.t()
    if sim.shape[0] <= 1:
        return {
            "text_similarity_offdiag_mean": 0.0,
            "text_similarity_offdiag_std": 0.0,
            "text_similarity_offdiag_p95": 0.0,
        }
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    values = sim[mask].numpy()
    return {
        "text_similarity_offdiag_mean": float(values.mean()),
        "text_similarity_offdiag_std": float(values.std()),
        "text_similarity_offdiag_p95": float(np.percentile(values, 95)),
    }


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    return float(values[finite].mean())
