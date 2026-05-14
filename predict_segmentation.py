"""
predict_segmentation.py
-----------------------
Load a trained U-Net checkpoint and visualise predicted segmentation masks.

Usage – command line
────────────────────
    python predict_segmentation.py \
        --checkpoint /content/Food_AI_Final/checkpoints/best_model.pth \
        --image_dir  /content/Food_AI_Final/dataset/test/images \
        --mask_dir   /content/Food_AI_Final/dataset/test/masks \
        --out_dir    /content/Food_AI_Final/predictions \
        --backbone   unet \
        --n_samples  8

Usage – Colab notebook cell
────────────────────────────
    from predict_segmentation import predict_single, visualise_grid
    mask, overlay = predict_single("path/to/image.jpg", model)
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_unet import build_model
from metrics    import compute_all


# ── constants ─────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── image helpers ─────────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str,
                        img_size: tuple = (256, 256)) -> torch.Tensor:
    """
    Load an RGB image → normalised float tensor [1, 3, H, W].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size[::-1], Image.BILINEAR)   # PIL uses (W, H)
    tensor = TF.to_tensor(img)
    tensor = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0)   # [1, 3, H, W]


def load_mask(mask_path: str, img_size: tuple = (256, 256)) -> np.ndarray:
    """Load binary ground-truth mask as (H, W) uint8 array."""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(img_size[::-1], Image.NEAREST)
    mask_np = np.array(mask, dtype=np.uint8)
    return (mask_np > 0).astype(np.uint8)


def overlay_mask(
    image_path: str,
    pred_mask: np.ndarray,          # (H, W) binary float/int
    alpha: float = 0.45,
    color: tuple = (255, 80, 80),   # red tint for food region
) -> np.ndarray:
    """
    Blend a coloured mask over the original image.
    Returns RGB numpy array (H, W, 3) in [0, 255].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32)

    colour_layer = np.zeros_like(img_np)
    colour_layer[pred_mask == 1] = color

    blended = img_np.copy()
    food_px  = pred_mask == 1
    blended[food_px] = (
        (1 - alpha) * img_np[food_px] + alpha * colour_layer[food_px]
    )
    return blended.clip(0, 255).astype(np.uint8)


# ── single-image prediction ───────────────────────────────────────────────────

@torch.no_grad()
def predict_single(
    image_path:  str,
    model:       torch.nn.Module,
    img_size:    tuple  = (256, 256),
    threshold:   float  = 0.5,
    device:      str    = "cpu",
):
    """
    Run segmentation on a single image.

    Returns
    -------
    prob_map : np.ndarray (H, W) float in [0,1]
    pred_mask: np.ndarray (H, W) binary {0,1}
    overlay  : np.ndarray (H, W, 3) uint8
    """
    model.eval()
    tensor = load_and_preprocess(image_path, img_size).to(device)
    prob   = model(tensor)                          # [1, 1, H, W]
    prob   = prob.squeeze().cpu().numpy()           # (H, W)
    pred   = (prob >= threshold).astype(np.uint8)

    vis = overlay_mask(image_path, pred)
    return prob, pred, vis


# ── batch visualisation ───────────────────────────────────────────────────────

def visualise_grid(
    image_paths:  list,
    mask_paths:   list,
    model:        torch.nn.Module,
    out_path:     str,
    img_size:     tuple = (256, 256),
    threshold:    float = 0.5,
    device:       str   = "cpu",
    n_cols:       int   = 4,
):
    """
    Create a grid figure:  original | GT mask | predicted mask | overlay
    for up to len(image_paths) samples and save to *out_path*.
    """
    n = len(image_paths)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Image", "GT Mask", "Pred Mask", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    all_iou, all_dice = [], []

    for i, (img_path, msk_path) in enumerate(zip(image_paths, mask_paths)):
        prob, pred, vis = predict_single(
            img_path, model, img_size, threshold, device)
        gt = load_mask(msk_path, img_size)

        # Metrics
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
        gt_t   = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()
        m      = compute_all(pred_t, gt_t)
        all_iou.append(m["iou"])
        all_dice.append(m["dice"])

        # Original image
        orig = np.array(
            Image.open(img_path).convert("RGB").resize(
                img_size[::-1], Image.BILINEAR))

        axes[i][0].imshow(orig);      axes[i][0].axis("off")
        axes[i][1].imshow(gt,   cmap="gray"); axes[i][1].axis("off")
        axes[i][2].imshow(pred, cmap="gray"); axes[i][2].axis("off")
        axes[i][3].imshow(vis);       axes[i][3].axis("off")

        iou_str = f"IoU={m['iou']:.3f} Dice={m['dice']:.3f}"
        axes[i][0].set_ylabel(
            Path(img_path).name[:20] + "\n" + iou_str,
            fontsize=7, rotation=0, labelpad=80, va="center")

    avg_iou  = np.mean(all_iou)
    avg_dice = np.mean(all_dice)
    plt.suptitle(
        f"Segmentation Predictions  —  "
        f"Avg IoU: {avg_iou:.4f}  |  Avg Dice: {avg_dice:.4f}",
        fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Grid saved → {out_path}")
    print(f"Average IoU: {avg_iou:.4f}  |  Average Dice: {avg_dice:.4f}")
    return avg_iou, avg_dice


# ── run on test set ───────────────────────────────────────────────────────────

def evaluate_test_set(
    model:       torch.nn.Module,
    image_dir:   str,
    mask_dir:    str,
    out_dir:     str,
    img_size:    tuple = (256, 256),
    threshold:   float = 0.5,
    device:      str   = "cpu",
    n_samples:   int   = 8,
    save_individual: bool = True,
):
    """
    Run prediction on the test set, print aggregate metrics,
    and save overlay images.
    """
    exts = {".jpg", ".jpeg", ".png"}
    img_paths  = sorted([p for p in Path(image_dir).iterdir()
                          if p.suffix.lower() in exts])
    mask_paths = sorted([p for p in Path(mask_dir).iterdir()
                          if p.suffix.lower() in exts])

    assert len(img_paths) > 0, f"No images found in {image_dir}"
    print(f"Test set: {len(img_paths)} images")

    os.makedirs(out_dir, exist_ok=True)

    # Sample for the grid visualisation
    indices = random.sample(range(len(img_paths)), min(n_samples, len(img_paths)))
    sample_imgs  = [str(img_paths[i])  for i in indices]
    sample_masks = [str(mask_paths[i]) for i in indices]

    grid_path = os.path.join(out_dir, "prediction_grid.png")
    visualise_grid(sample_imgs, sample_masks, model,
                   grid_path, img_size, threshold, device)

    # Save individual overlays
    if save_individual:
        overlay_dir = os.path.join(out_dir, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        for img_path in img_paths[:50]:    # cap at 50 to avoid huge output
            _, _, vis = predict_single(str(img_path), model, img_size,
                                       threshold, device)
            out_name = Path(img_path).stem + "_overlay.png"
            Image.fromarray(vis).save(os.path.join(overlay_dir, out_name))
        print(f"Overlays saved → {overlay_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint: str, backbone: str, device):
    model = build_model(backbone=backbone, pretrained=False).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)      # handle both formats
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Predict & visualise food masks")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--mask_dir",   required=True)
    p.add_argument("--out_dir",    default="/content/Food_AI_Final/predictions")
    p.add_argument("--backbone",   default="unet",
                   choices=["unet", "resnet50"])
    p.add_argument("--img_size",   type=int, default=256)
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--n_samples",  type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_from_checkpoint(args.checkpoint, args.backbone, device)

    evaluate_test_set(
        model      = model,
        image_dir  = args.image_dir,
        mask_dir   = args.mask_dir,
        out_dir    = args.out_dir,
        img_size   = (args.img_size, args.img_size),
        threshold  = args.threshold,
        device     = str(device),
        n_samples  = args.n_samples,
    )
