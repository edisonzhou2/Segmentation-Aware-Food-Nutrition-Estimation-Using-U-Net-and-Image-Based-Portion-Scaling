"""
losses.py
---------
Loss functions for binary food segmentation.

Available losses
────────────────
  BCELoss          – standard binary cross-entropy
  DiceLoss         – soft Dice loss
  BCEDiceLoss      – weighted combination (default: 0.5 BCE + 0.5 Dice)
  FocalLoss        – focal loss for class-imbalanced masks
  TverskyLoss      – generalised Dice with α/β weighting

All losses expect:
  preds  : [B, 1, H, W]  float in [0, 1]   (sigmoid output)
  targets: [B, 1, H, W]  float in {0, 1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6   # numerical stability


# ── individual losses ─────────────────────────────────────────────────────────

class BCELoss(nn.Module):
    """Standard Binary Cross-Entropy."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(preds, targets)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss.

    DiceLoss = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)
    """

    def __init__(self, smooth: float = EPS):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / \
               (preds.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of BCE and Dice Loss.

    loss = bce_weight * BCE(p, t) + dice_weight * DiceLoss(p, t)

    Default equal weighting (0.5 / 0.5).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        assert abs(bce_weight + dice_weight - 1.0) < 1e-4, \
            "bce_weight + dice_weight should equal 1.0"
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

    def __repr__(self):
        return (f"BCEDiceLoss(bce={self.bce_weight}, "
                f"dice={self.dice_weight})")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.

    FL(p) = -α (1 - p)^γ log(p)

    Recommended: α=0.25, γ=2 for dense food masks.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        pt  = torch.where(targets == 1, preds, 1 - preds)
        fl  = self.alpha * (1 - pt) ** self.gamma * bce
        return fl.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss: generalises Dice by letting FP and FN have different weights.

    α controls penalty for false positives
    β controls penalty for false negatives  (β > α → recall-oriented)
    Dice is the special case α = β = 0.5.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = EPS):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (preds * targets).sum(dim=1)
        fp = (preds * (1 - targets)).sum(dim=1)
        fn = ((1 - preds) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / \
                  (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


# ── factory ───────────────────────────────────────────────────────────────────

def get_loss(name: str = "bce_dice", **kwargs) -> nn.Module:
    """
    name : 'bce'        → BCELoss
           'dice'       → DiceLoss
           'bce_dice'   → BCEDiceLoss  (default)
           'focal'      → FocalLoss
           'tversky'    → TverskyLoss
    kwargs are forwarded to the chosen loss constructor.
    """
    lut = {
        "bce":      BCELoss,
        "dice":     DiceLoss,
        "bce_dice": BCEDiceLoss,
        "focal":    FocalLoss,
        "tversky":  TverskyLoss,
    }
    if name not in lut:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(lut)}")
    loss_fn = lut[name](**kwargs)
    print(f"Using loss: {loss_fn}")
    return loss_fn


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 4
    preds   = torch.sigmoid(torch.randn(B, 1, 256, 256))
    targets = (torch.rand(B, 1, 256, 256) > 0.5).float()

    for name in ("bce", "dice", "bce_dice", "focal", "tversky"):
        loss = get_loss(name)
        val  = loss(preds, targets)
        print(f"  {name:10s}  →  {val.item():.4f}")
