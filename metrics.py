"""
metrics.py
----------
Segmentation metrics for binary food masks.

All functions accept:
  preds   : [B, 1, H, W]  float in [0, 1]   (model sigmoid output)
  targets : [B, 1, H, W]  float in {0, 1}
  threshold: float         decision boundary (default 0.5)

Metrics implemented
───────────────────
  pixel_accuracy(preds, targets)   → float in [0, 1]
  iou_score     (preds, targets)   → float in [0, 1]   (Jaccard index)
  dice_score    (preds, targets)   → float in [0, 1]   (F1 of pixels)
  precision     (preds, targets)   → float
  recall        (preds, targets)   → float
  compute_all   (preds, targets)   → dict with all metrics
"""

import torch
import numpy as np

EPS = 1e-6


# ── core helpers ──────────────────────────────────────────────────────────────

def _binarise(preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert probability map to binary mask."""
    return (preds >= threshold).float()


def _flatten(preds: torch.Tensor,
             targets: torch.Tensor,
             threshold: float = 0.5):
    """Binarise preds and flatten both tensors to 1-D."""
    pred_bin = _binarise(preds, threshold).view(-1)
    tgt_bin  = targets.view(-1).float()
    return pred_bin, tgt_bin


# ── individual metrics ────────────────────────────────────────────────────────

def pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Pixel Accuracy = (TP + TN) / total_pixels

    Proportion of correctly classified pixels.
    """
    pred_bin, tgt_bin = _flatten(preds, targets, threshold)
    correct = (pred_bin == tgt_bin).sum().float()
    total   = tgt_bin.numel()
    return (correct / total).item()


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Intersection over Union (Jaccard Index).

    IoU = TP / (TP + FP + FN)
    """
    pred_bin, tgt_bin = _flatten(preds, targets, threshold)
    intersection = (pred_bin * tgt_bin).sum()
    union        = pred_bin.sum() + tgt_bin.sum() - intersection
    return ((intersection + EPS) / (union + EPS)).item()


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Dice Coefficient (F1 score of pixels).

    Dice = 2*TP / (2*TP + FP + FN)
    """
    pred_bin, tgt_bin = _flatten(preds, targets, threshold)
    intersection = (pred_bin * tgt_bin).sum()
    denom        = pred_bin.sum() + tgt_bin.sum()
    return ((2.0 * intersection + EPS) / (denom + EPS)).item()


def precision(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Precision = TP / (TP + FP)."""
    pred_bin, tgt_bin = _flatten(preds, targets, threshold)
    tp = (pred_bin * tgt_bin).sum()
    fp = (pred_bin * (1 - tgt_bin)).sum()
    return ((tp + EPS) / (tp + fp + EPS)).item()


def recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Recall = TP / (TP + FN)."""
    pred_bin, tgt_bin = _flatten(preds, targets, threshold)
    tp = (pred_bin * tgt_bin).sum()
    fn = ((1 - pred_bin) * tgt_bin).sum()
    return ((tp + EPS) / (tp + fn + EPS)).item()


# ── aggregate helper ──────────────────────────────────────────────────────────

def compute_all(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Returns a dict with all five metrics.

    Usage inside an eval loop:
        m = compute_all(preds, targets)
        print(m)   # {'pixel_acc': 0.97, 'iou': 0.84, ...}
    """
    return {
        "pixel_acc": pixel_accuracy(preds, targets, threshold),
        "iou":       iou_score(preds, targets, threshold),
        "dice":      dice_score(preds, targets, threshold),
        "precision": precision(preds, targets, threshold),
        "recall":    recall(preds, targets, threshold),
    }


# ── epoch-level averager ──────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Accumulates per-batch metrics and returns epoch-level averages.

    Usage
    -----
    acc = MetricAccumulator()
    for imgs, masks in val_loader:
        preds = model(imgs)
        acc.update(preds.cpu(), masks.cpu())
    results = acc.compute()   # dict of averages
    acc.reset()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._totals = {
            "pixel_acc": 0.0,
            "iou":       0.0,
            "dice":      0.0,
            "precision": 0.0,
            "recall":    0.0,
        }
        self._count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        m = compute_all(preds, targets, self.threshold)
        for k in self._totals:
            self._totals[k] += m[k]
        self._count += 1

    def compute(self) -> dict:
        if self._count == 0:
            return {k: 0.0 for k in self._totals}
        return {k: v / self._count for k, v in self._totals.items()}

    def pretty(self) -> str:
        m = self.compute()
        return (f"PixAcc={m['pixel_acc']:.4f}  "
                f"IoU={m['iou']:.4f}  "
                f"Dice={m['dice']:.4f}  "
                f"Prec={m['precision']:.4f}  "
                f"Rec={m['recall']:.4f}")


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preds   = torch.sigmoid(torch.randn(4, 1, 256, 256))
    targets = (torch.rand(4, 1, 256, 256) > 0.5).float()

    m = compute_all(preds, targets)
    print("Single batch metrics:", m)

    acc = MetricAccumulator()
    for _ in range(5):          # simulate 5 batches
        acc.update(preds, targets)
    print("Epoch average:", acc.pretty())
