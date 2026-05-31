from __future__ import annotations

import numpy as np


class IoUMeter:
    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target) -> None:
        pred = np.asarray(pred).reshape(-1)
        target = np.asarray(target).reshape(-1)
        mask = target != self.ignore_index
        mask &= target >= 0
        mask &= target < self.num_classes
        target = target[mask]
        pred = pred[mask]
        pred = np.clip(pred, 0, self.num_classes - 1)
        index = self.num_classes * target.astype(np.int64) + pred.astype(np.int64)
        hist = np.bincount(index, minlength=self.num_classes ** 2)
        self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def compute(self, seen_mask=None) -> dict:
        tp = np.diag(self.confusion).astype(np.float64)
        denom = self.confusion.sum(1) + self.confusion.sum(0) - tp
        iou = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
        result = {
            "per_class_iou": iou,
            "all_miou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "confusion_matrix": self.confusion.copy(),
        }
        if seen_mask is not None:
            seen_mask = np.asarray(seen_mask, dtype=bool)
            unseen_mask = ~seen_mask
            result["seen_miou"] = _masked_mean(iou, seen_mask)
            result["unseen_miou"] = _masked_mean(iou, unseen_mask)
        return result


def _masked_mean(values, mask) -> float:
    chosen = values[mask]
    return float(np.nanmean(chosen)) if np.any(~np.isnan(chosen)) else 0.0
