from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenVocabularyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_3d_ce = float(cfg.get("w_3d_ce", 1.0))
        self.w_2d_ce = float(cfg.get("w_2d_ce", 0.5))
        self.w_2d_coarse_ce = float(cfg.get("w_2d_coarse_ce", 0.0))
        self.w_feat = float(cfg.get("w_feat", 1.0))
        self.w_kl = float(cfg.get("w_kl", 0.5))
        self.temperature = float(cfg.get("temperature", 2.0))
        self.ignore_index = int(cfg.get("ignore_index", -100))
        self.conf_thresh = cfg.get("conf_thresh", None)
        self.balance_2d_ce = bool(cfg.get("balance_2d_ce", True))

    def forward(self, outputs, batch=None):
        losses = {}
        labels = self._sanitize_labels(outputs["point_labels"], outputs["logits3d"].shape[1])
        if (labels != self.ignore_index).any():
            losses["loss_3d_ce"] = F.cross_entropy(outputs["logits3d"], labels, ignore_index=self.ignore_index)
        else:
            losses["loss_3d_ce"] = outputs["logits3d"].sum() * 0.0

        valid_labels = self._sanitize_labels(outputs["labels_valid"], outputs["logits3d"].shape[1])
        supervised_mask = self._label_mask(valid_labels)

        if supervised_mask.any():
            logits2d_sup = outputs["logits2d_points"][supervised_mask]
            labels_sup = valid_labels[supervised_mask]
            losses["loss_2d_ce"] = _ce_loss(
                logits2d_sup,
                labels_sup,
                ignore_index=self.ignore_index,
                balanced=self.balance_2d_ce,
            )
            coarse_logits = outputs.get("coarse_logits2d_points", None)
            if coarse_logits is not None and coarse_logits.numel() > 0 and self.w_2d_coarse_ce > 0:
                losses["loss_2d_coarse_ce"] = _ce_loss(
                    coarse_logits[supervised_mask],
                    labels_sup,
                    ignore_index=self.ignore_index,
                    balanced=self.balance_2d_ce,
                )
            else:
                losses["loss_2d_coarse_ce"] = outputs["logits3d"].sum() * 0.0
            pred2d_sup = logits2d_sup.argmax(dim=-1)
            losses["metric_2d_projected_acc"] = (pred2d_sup == labels_sup).float().mean()
            losses["metric_2d_projected_miou"] = _batch_miou(pred2d_sup, labels_sup, logits2d_sup.shape[1])
        else:
            zero = outputs["logits3d"].sum() * 0.0
            losses["loss_2d_ce"] = zero
            losses["loss_2d_coarse_ce"] = zero
            losses["metric_2d_projected_acc"] = zero
            losses["metric_2d_projected_miou"] = zero

        distill_mask = supervised_mask.clone()
        if self.conf_thresh is not None and outputs["logits2d_points"].numel() > 0:
            confidence = outputs["logits2d_points"].softmax(dim=-1).max(dim=-1).values
            distill_mask &= confidence >= float(self.conf_thresh)

        if distill_mask.any():
            logits2d = outputs["logits2d_points"][distill_mask]
            logits3d = outputs["logits3d_valid"][distill_mask]
            z2d = outputs["z2d_points"][distill_mask]
            z3d = outputs["z3d_valid"][distill_mask]
            losses["loss_feat"] = (1.0 - F.cosine_similarity(z3d, z2d.detach(), dim=-1)).mean()
            temp = self.temperature
            p2d = F.softmax(logits2d.detach() / temp, dim=-1)
            log_p3d = F.log_softmax(logits3d / temp, dim=-1)
            losses["loss_kl"] = F.kl_div(log_p3d, p2d, reduction="batchmean") * (temp * temp)
        else:
            zero = outputs["logits3d"].sum() * 0.0
            losses["loss_feat"] = zero
            losses["loss_kl"] = zero

        total_points = max(int(outputs["point_labels"].numel()), 1)
        projected_points = int(outputs["labels_valid"].numel())
        ignored_projected = int((valid_labels == self.ignore_index).sum().item()) if projected_points else 0
        supervised_projected = int(supervised_mask.sum().item()) if projected_points else 0
        distill_projected = int(distill_mask.sum().item()) if projected_points else 0
        device_zero = outputs["logits3d"].sum() * 0.0
        losses["metric_valid_projected_ratio"] = device_zero + projected_points / total_points
        losses["metric_ignored_projected_label_ratio"] = device_zero + ignored_projected / max(projected_points, 1)
        losses["metric_supervised_projected_ratio"] = device_zero + supervised_projected / max(projected_points, 1)
        losses["metric_distill_projected_ratio"] = device_zero + distill_projected / max(projected_points, 1)

        losses["loss"] = (
            self.w_3d_ce * losses["loss_3d_ce"]
            + self.w_2d_ce * losses["loss_2d_ce"]
            + self.w_2d_coarse_ce * losses["loss_2d_coarse_ce"]
            + self.w_feat * losses["loss_feat"]
            + self.w_kl * losses["loss_kl"]
        )
        return losses

    def _label_mask(self, labels: torch.Tensor) -> torch.Tensor:
        return labels != self.ignore_index

    def _sanitize_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        labels = labels.clone()
        labels[(labels < 0) | (labels >= num_classes)] = self.ignore_index
        return labels


def _batch_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    ious = []
    for class_idx in range(num_classes):
        pred_i = pred == class_idx
        target_i = target == class_idx
        union = pred_i | target_i
        if union.any():
            ious.append((pred_i & target_i).float().sum() / union.float().sum().clamp_min(1.0))
    if not ious:
        return pred.sum() * 0.0
    return torch.stack(ious).mean()


def _ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int, balanced: bool) -> torch.Tensor:
    if not balanced:
        return F.cross_entropy(logits, labels, ignore_index=ignore_index)
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    counts = torch.bincount(labels[valid], minlength=logits.shape[1]).float().to(logits.device)
    present = counts > 0
    weights = torch.ones(logits.shape[1], device=logits.device)
    weights[present] = (counts[present].sum() / counts[present].clamp_min(1.0)).sqrt()
    weights = weights / weights[present].mean().clamp_min(1e-6)
    return F.cross_entropy(logits, labels, weight=weights, ignore_index=ignore_index)
