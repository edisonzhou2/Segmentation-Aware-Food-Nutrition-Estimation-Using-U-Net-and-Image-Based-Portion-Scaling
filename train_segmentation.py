"""
train_segmentation.py
---------------------
End-to-end training script for the U-Net food segmentation model.

Colab quickstart
────────────────
    !python train_segmentation.py \
        --root  /content/Food_AI_Final/dataset \
        --backbone unet \
        --epochs 30 \
        --batch_size 8 \
        --lr 1e-4 \
        --save_dir /content/Food_AI_Final/checkpoints

Or import and call train() directly from a notebook cell.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Colab
import matplotlib.pyplot as plt

from dataset    import get_dataloaders
from model_unet import build_model
from losses     import get_loss
from metrics    import MetricAccumulator


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, best_iou, path):
    torch.save({
        "epoch":     epoch,
        "best_iou":  best_iou,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt      = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_iou"]


def plot_curves(history: dict, save_path: str):
    """Plot and save training / validation loss + IoU curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["train_iou"], label="Train IoU")
    axes[1].plot(epochs, history["val_iou"],   label="Val IoU")
    axes[1].set_title("IoU (Jaccard)"); axes[1].legend()
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# one-epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    acc = MetricAccumulator()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for imgs, masks in loader:
            imgs  = imgs.to(device,  non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            preds = model(imgs)
            loss  = criterion(preds, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            acc.update(preds.detach().cpu(), masks.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    metrics  = acc.compute()
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# main train() function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    root:        str   = "/content/Food_AI_Final/dataset",
    backbone:    str   = "unet",
    pretrained:  bool  = True,
    img_size:    int   = 256,
    epochs:      int   = 30,
    batch_size:  int   = 8,
    lr:          float = 1e-4,
    weight_decay:float = 1e-5,
    loss_name:   str   = "bce_dice",
    num_workers: int   = 2,
    save_dir:    str   = "/content/Food_AI_Final/checkpoints",
    resume:      str   = None,    # path to checkpoint to resume from
):
    # ── setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Food Segmentation Training")
    print(f" Device   : {device}")
    print(f" Backbone : {backbone}")
    print(f" Epochs   : {epochs}  |  Batch : {batch_size}  |  LR : {lr}")
    print(f"{'='*60}\n")

    os.makedirs(save_dir, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        root, img_size=(img_size, img_size),
        batch_size=batch_size, num_workers=num_workers,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(backbone=backbone, pretrained=pretrained).to(device)

    # ── loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = get_loss(loss_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── optionally resume ─────────────────────────────────────────────────────
    start_epoch = 0
    best_iou    = 0.0
    if resume and Path(resume).exists():
        start_epoch, best_iou = load_checkpoint(model, optimizer, resume, device)
        print(f"Resumed from {resume} (epoch {start_epoch}, best IoU {best_iou:.4f})")

    # ── history ───────────────────────────────────────────────────────────────
    history = {k: [] for k in
               ("train_loss", "val_loss", "train_iou", "val_iou",
                "train_dice", "val_dice")}

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()

        train_loss, train_m = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_m   = run_epoch(
            model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step()

        # record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_m["iou"])
        history["val_iou"].append(val_m["iou"])
        history["train_dice"].append(train_m["dice"])
        history["val_dice"].append(val_m["dice"])

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:03d}/{epochs}]  "
            f"Train loss={train_loss:.4f} IoU={train_m['iou']:.4f} "
            f"Dice={train_m['dice']:.4f}  |  "
            f"Val   loss={val_loss:.4f} IoU={val_m['iou']:.4f} "
            f"Dice={val_m['dice']:.4f} PixAcc={val_m['pixel_acc']:.4f}  "
            f"[{elapsed:.1f}s]"
        )

        # save best model
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            path = os.path.join(save_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_iou, path)
            print(f"  ✓ New best IoU {best_iou:.4f} → saved to {path}")

        # periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            path = os.path.join(save_dir, f"checkpoint_epoch{epoch:03d}.pth")
            save_checkpoint(model, optimizer, epoch, best_iou, path)

    # ── final artefacts ───────────────────────────────────────────────────────
    curve_path = os.path.join(save_dir, "training_curves.png")
    plot_curves(history, curve_path)

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net food segmentation")
    p.add_argument("--root",         default="/content/Food_AI_Final/dataset")
    p.add_argument("--backbone",     default="unet",
                   choices=["unet", "resnet50"])
    p.add_argument("--no_pretrained",action="store_true")
    p.add_argument("--img_size",     type=int,   default=256)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--loss",         default="bce_dice",
                   choices=["bce", "dice", "bce_dice", "focal", "tversky"])
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--save_dir",
                   default="/content/Food_AI_Final/checkpoints")
    p.add_argument("--resume",       default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        root         = args.root,
        backbone     = args.backbone,
        pretrained   = not args.no_pretrained,
        img_size     = args.img_size,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        loss_name    = args.loss,
        num_workers  = args.num_workers,
        save_dir     = args.save_dir,
        resume       = args.resume,
    )
