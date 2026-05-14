"""
nutrition_with_segmentation.py
-------------------------------
Upgraded nutrition estimator that integrates U-Net segmentation masks
with the ResNet-50 food classifier and dish_ingredients.csv lookup.

Pipeline
────────
 1. Input: RGB food image
 2. U-Net → binary food mask  (0=background, 1=food)
 3. ResNet-50 classifier → top-k predicted dish labels
 4. Mask area → portion-size weight relative to a reference area
 5. dish_ingredients.csv → base nutritional values per label
 6. Output: adjusted calories, fat, carbs, protein

Comparison with old pipeline
─────────────────────────────
  OLD (classification-only):
    image → ResNet-50 → dish label → CSV lookup → fixed portion nutrition

  NEW (segmentation-based):
    image → U-Net mask → ResNet-50 → dish label → CSV lookup
            ↑
     mask area adjusts portion weight dynamically

Usage
─────
    python nutrition_with_segmentation.py \
        --image   /path/to/food.jpg \
        --seg_ckpt /content/Food_AI_Final/checkpoints/best_model.pth \
        --csv     /content/Food_AI_Final/dish_ingredients.csv

Or import NutritionEstimator and call .estimate(image_path).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_unet import build_model


# ─────────────────────────────────────────────────────────────────────────────
# constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Reference area used to calibrate "full plate" portion
# Assumes a 256×256 image where ~40% is a standard single serving
REFERENCE_FOOD_PIXEL_FRACTION = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str, size: int = 256) -> tuple:
    """
    Returns
    -------
    img_tensor : [1, 3, size, size] normalised float tensor
    pil_img    : original PIL image (for display)
    """
    pil = Image.open(image_path).convert("RGB")
    resized = pil.resize((size, size), Image.BILINEAR)
    tensor  = TF.to_tensor(resized)
    tensor  = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0), pil


# ─────────────────────────────────────────────────────────────────────────────
# segmentation
# ─────────────────────────────────────────────────────────────────────────────

class Segmentor:
    """Wraps the trained U-Net for mask inference."""

    def __init__(self, checkpoint: str, backbone: str = "unet",
                 img_size: int = 256, threshold: float = 0.5,
                 device: str = "cpu"):
        self.size      = img_size
        self.threshold = threshold
        self.device    = device

        self.model = build_model(backbone=backbone, pretrained=False).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[Segmentor] Loaded U-Net from {checkpoint}")

    @torch.no_grad()
    def predict(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        img_tensor : [1, 3, H, W]
        Returns binary mask (H, W) as uint8 {0, 1}
        """
        prob = self.model(img_tensor.to(self.device))
        prob = prob.squeeze().cpu().numpy()
        return (prob >= self.threshold).astype(np.uint8)

    def food_pixel_fraction(self, mask: np.ndarray) -> float:
        """Fraction of pixels classified as food."""
        return float(mask.sum()) / mask.size


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-50 classifier
# ─────────────────────────────────────────────────────────────────────────────

class FoodClassifier:
    """
    ResNet-50-based food/dish classifier.

    If a custom checkpoint is provided it must be a dict with keys
    'model' (state_dict) and optionally 'classes' (list of label strings).

    Falls back to ImageNet top-1 label as a zero-shot proxy when no
    custom checkpoint is available (useful for prototyping).
    """

    # Mapping: ImageNet class index → simplified food name (subset)
    _IMAGENET_FOOD_MAP = {
        924: "guacamole",   963: "pizza",   927: "hotdog",
        922: "sushi",       925: "burrito", 928: "taco",
        930: "ice cream",   932: "waffle",  934: "pretzel",
        936: "bagel",       937: "cheeseburger", 938: "hamburger",
        923: "carbonara",   959: "chocolate cake",
    }

    def __init__(self, num_classes: int = 1000,
                 class_names: list = None,
                 checkpoint: str  = None,
                 device: str = "cpu"):
        self.device      = device
        self.class_names = class_names   # None → use ImageNet labels

        weights = tv_models.ResNet50_Weights.DEFAULT
        self.model = tv_models.resnet50(weights=weights)

        if checkpoint and Path(checkpoint).exists():
            ckpt  = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model", ckpt)
            # Replace head if needed
            if num_classes != 1000:
                self.model.fc = nn.Linear(2048, num_classes)
            self.model.load_state_dict(state)
            if class_names is None and "classes" in ckpt:
                self.class_names = ckpt["classes"]
            print(f"[Classifier] Loaded custom weights from {checkpoint}")
        else:
            print("[Classifier] Using pretrained ImageNet ResNet-50 (zero-shot proxy)")

        self.model = self.model.to(device)
        self.model.eval()

        # Standard ImageNet preprocessing (already normalised input accepted)
        self.softmax = nn.Softmax(dim=1)

    @torch.no_grad()
    def predict_topk(self, img_tensor: torch.Tensor, k: int = 3) -> list:
        """
        Returns list of (label_string, confidence_float) tuples.
        img_tensor : [1, 3, H, W] normalised
        """
        logits = self.model(img_tensor.to(self.device))
        probs  = self.softmax(logits)[0]
        topk   = torch.topk(probs, k)

        results = []
        for score, idx in zip(topk.values, topk.indices):
            idx = idx.item()
            if self.class_names:
                label = self.class_names[idx]
            else:
                # ImageNet label via torchvision
                label = tv_models.ResNet50_Weights.DEFAULT \
                            .meta["categories"][idx].lower()
                # Map to simplified food name if in our subset
                label = self._IMAGENET_FOOD_MAP.get(idx, label)
            results.append((label, float(score)))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Nutrition CSV lookup
# ─────────────────────────────────────────────────────────────────────────────

class NutritionDB:
    """
    Loads dish_ingredients.csv and provides per-dish nutritional lookup.

    Expected CSV columns (case-insensitive, flexible naming):
        dish / dish_name / food / label   ← dish identifier
        calories / kcal
        fat / fat_g
        carbs / carbohydrates / carbs_g
        protein / protein_g

    Portion size columns (optional, used as reference):
        portion_g / serving_g / serving_size_g
    """

    _COL_ALIASES = {
        "dish":      ["dish", "dish_name", "food", "label", "name"],
        "calories":  ["calories", "kcal", "energy"],
        "fat":       ["fat", "fat_g", "total_fat"],
        "carbs":     ["carbs", "carbohydrates", "carbs_g", "carb"],
        "protein":   ["protein", "protein_g"],
        "portion_g": ["portion_g", "serving_g", "serving_size_g", "portion"],
    }

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        self.df = df
        self._col_map = self._resolve_columns()
        print(f"[NutritionDB] Loaded {len(df)} entries from {csv_path}")
        print(f"  Column mapping: {self._col_map}")

    def _resolve_columns(self) -> dict:
        resolved = {}
        for key, aliases in self._COL_ALIASES.items():
            for alias in aliases:
                if alias in self.df.columns:
                    resolved[key] = alias
                    break
        return resolved

    def _get(self, row, key, default=0.0) -> float:
        col = self._col_map.get(key)
        if col is None or pd.isna(row.get(col, np.nan)):
            return default
        return float(row[col])

    def lookup(self, dish_name: str) -> dict | None:
        """
        Fuzzy-ish lookup: exact → prefix → substring match (case-insensitive).
        Returns dict with keys: dish, calories, fat, carbs, protein, portion_g
        or None if not found.
        """
        dish_col = self._col_map.get("dish")
        if dish_col is None:
            return None

        name_lower = dish_name.lower()
        series = self.df[dish_col].str.lower()

        # 1. exact
        mask = series == name_lower
        if not mask.any():
            # 2. starts-with
            mask = series.str.startswith(name_lower)
        if not mask.any():
            # 3. contains
            mask = series.str.contains(name_lower, regex=False, na=False)
        if not mask.any():
            return None

        row = self.df[mask].iloc[0].to_dict()
        return {
            "dish":      row.get(dish_col, dish_name),
            "calories":  self._get(row, "calories"),
            "fat":       self._get(row, "fat"),
            "carbs":     self._get(row, "carbs"),
            "protein":   self._get(row, "protein"),
            "portion_g": self._get(row, "portion_g", 100.0),
        }

    def closest_match(self, candidates: list[str]) -> tuple[str, dict | None]:
        """Try each candidate label in order; return first match."""
        for label in candidates:
            result = self.lookup(label)
            if result:
                return label, result
        return candidates[0] if candidates else "unknown", None


# ─────────────────────────────────────────────────────────────────────────────
# Portion-size adjuster
# ─────────────────────────────────────────────────────────────────────────────

def compute_portion_weight(
    food_pixel_fraction: float,
    reference_fraction:  float = REFERENCE_FOOD_PIXEL_FRACTION,
) -> float:
    """
    Estimate a portion multiplier relative to a reference plate.

    portion_weight = food_pixel_fraction / reference_fraction

    Examples:
      food fills 40% of image (=reference) → weight = 1.0  (standard portion)
      food fills 80% of image              → weight = 2.0  (double portion)
      food fills 20% of image              → weight = 0.5  (half portion)
    """
    if reference_fraction <= 0:
        return 1.0
    weight = food_pixel_fraction / reference_fraction
    # Clamp to [0.1, 3.0] — sanity bounds
    return float(np.clip(weight, 0.1, 3.0))


def adjust_nutrition(base: dict, weight: float) -> dict:
    """Scale nutritional values by *weight*."""
    return {
        "dish":      base["dish"],
        "portion_g": round(base["portion_g"] * weight, 1),
        "calories":  round(base["calories"]  * weight, 1),
        "fat":       round(base["fat"]       * weight, 2),
        "carbs":     round(base["carbs"]     * weight, 2),
        "protein":   round(base["protein"]   * weight, 2),
        "portion_weight": round(weight, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# main estimator
# ─────────────────────────────────────────────────────────────────────────────

class NutritionEstimator:
    """
    Full pipeline: image → segmentation mask → classifier → nutrition.

    Parameters
    ----------
    seg_checkpoint : str   path to U-Net checkpoint (.pth)
    csv_path       : str   path to dish_ingredients.csv
    cls_checkpoint : str   optional path to custom ResNet-50 weights
    class_names    : list  optional list of dish label strings
    backbone       : str   'unet' or 'resnet50'
    img_size       : int   spatial resolution (default 256)
    topk           : int   number of classifier predictions to try
    device         : str   'cuda' or 'cpu'
    """

    def __init__(
        self,
        seg_checkpoint: str,
        csv_path:       str,
        cls_checkpoint: str  = None,
        class_names:    list = None,
        backbone:       str  = "unet",
        img_size:       int  = 256,
        topk:           int  = 3,
        device:         str  = "cpu",
    ):
        self.img_size = img_size
        self.topk     = topk
        self.device   = device

        self.segmentor  = Segmentor(seg_checkpoint, backbone, img_size,
                                    device=device)
        self.classifier = FoodClassifier(
            class_names=class_names,
            checkpoint=cls_checkpoint,
            device=device,
        )
        self.nutrition_db = NutritionDB(csv_path)

    def estimate(self, image_path: str, verbose: bool = True) -> dict:
        """
        Run the full pipeline on a single image.

        Returns
        -------
        dict with keys:
            image_path, predicted_dish, top_predictions,
            food_pixel_fraction, portion_weight,
            calories, fat, carbs, protein, portion_g,
            mask (np.ndarray H×W), overlay (np.ndarray H×W×3)
        """
        # 1. Preprocess
        img_tensor, pil_img = preprocess_image(image_path, self.img_size)

        # 2. Segmentation → mask + fraction
        mask     = self.segmentor.predict(img_tensor)
        fraction = self.segmentor.food_pixel_fraction(mask)

        # 3. Classification → top-k labels
        top_preds = self.classifier.predict_topk(img_tensor, k=self.topk)
        labels    = [lbl for lbl, _ in top_preds]

        # 4. Nutrition DB lookup
        matched_label, base_nutrition = self.nutrition_db.closest_match(labels)

        # 5. Portion weight from mask
        portion_weight = compute_portion_weight(fraction)

        # 6. Adjust nutrition
        if base_nutrition:
            result = adjust_nutrition(base_nutrition, portion_weight)
        else:
            # No match found — return raw prediction info with zero nutrition
            result = {
                "dish": matched_label, "portion_g": 0.0,
                "calories": 0.0, "fat": 0.0, "carbs": 0.0, "protein": 0.0,
                "portion_weight": portion_weight,
            }

        # 7. Build overlay
        from predict_segmentation import overlay_mask
        overlay = overlay_mask(image_path, mask)

        # Collect full output
        output = {
            "image_path":          image_path,
            "predicted_dish":      matched_label,
            "top_predictions":     top_preds,
            "food_pixel_fraction": round(fraction, 4),
            "portion_weight":      portion_weight,
            **{k: result[k] for k in
               ("calories", "fat", "carbs", "protein", "portion_g")},
            "mask":    mask,
            "overlay": overlay,
        }

        if verbose:
            self._print_report(output)

        return output

    @staticmethod
    def _print_report(r: dict):
        print("\n" + "="*55)
        print(f"  Food Nutrition Estimation Report")
        print("="*55)
        print(f"  Image   : {Path(r['image_path']).name}")
        print(f"  Dish    : {r['predicted_dish']}")
        print(f"  Top-k   : {r['top_predictions']}")
        print(f"\n  [Segmentation]")
        print(f"  Food pixel fraction : {r['food_pixel_fraction']*100:.1f}%")
        print(f"  Portion weight      : {r['portion_weight']:.2f}× reference")
        print(f"\n  [Estimated Nutrition  (adjusted for portion)]")
        print(f"  Calories : {r['calories']:.0f} kcal")
        print(f"  Fat      : {r['fat']:.1f} g")
        print(f"  Carbs    : {r['carbs']:.1f} g")
        print(f"  Protein  : {r['protein']:.1f} g")
        print(f"  Serving  : {r['portion_g']:.0f} g")
        print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# comparison: old vs new pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compare_pipelines(
    image_path: str,
    new_result: dict,
    nutrition_db: NutritionDB,
    topk_labels: list,
):
    """
    Print side-by-side comparison of old (classification-only) vs new pipeline.
    """
    matched_label, base = nutrition_db.closest_match(topk_labels)
    if base is None:
        print("No CSV match found for old pipeline comparison.")
        return

    print("\n" + "="*65)
    print("  Pipeline Comparison")
    print("="*65)
    print(f"  {'Metric':<25} {'OLD (no mask)':>15} {'NEW (with mask)':>15}")
    print("-"*65)
    rows = [
        ("Dish",     base['dish'],            new_result['predicted_dish']),
        ("Calories", f"{base['calories']:.0f} kcal",
                     f"{new_result['calories']:.0f} kcal"),
        ("Fat",      f"{base['fat']:.1f} g",    f"{new_result['fat']:.1f} g"),
        ("Carbs",    f"{base['carbs']:.1f} g",  f"{new_result['carbs']:.1f} g"),
        ("Protein",  f"{base['protein']:.1f} g",f"{new_result['protein']:.1f} g"),
        ("Portion",  f"{base['portion_g']:.0f} g",
                     f"{new_result['portion_g']:.0f} g"),
        ("Mask area","N/A",
                     f"{new_result['food_pixel_fraction']*100:.1f}%"),
        ("Portion ×","1.00× (fixed)",
                     f"{new_result['portion_weight']:.2f}×"),
    ]
    for label, old_val, new_val in rows:
        print(f"  {label:<25} {old_val:>15} {new_val:>15}")
    print("="*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def save_nutrition_figure(result: dict, out_path: str):
    """Save a 3-panel figure: original | mask | overlay + nutrition text."""
    pil = Image.open(result["image_path"]).convert("RGB") \
              .resize((256, 256), Image.BILINEAR)
    orig    = np.array(pil)
    mask    = result["mask"]
    overlay = result["overlay"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(orig);     axes[0].set_title("Original");      axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Food Mask"); axes[1].axis("off")
    axes[2].imshow(overlay);  axes[2].set_title("Overlay");       axes[2].axis("off")

    info = (
        f"Dish: {result['predicted_dish']}\n"
        f"Food area: {result['food_pixel_fraction']*100:.1f}%  "
        f"(×{result['portion_weight']:.2f} reference)\n"
        f"Calories: {result['calories']:.0f} kcal  |  "
        f"Fat: {result['fat']:.1f}g  |  "
        f"Carbs: {result['carbs']:.1f}g  |  "
        f"Protein: {result['protein']:.1f}g"
    )
    plt.suptitle(info, fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Segmentation-based nutrition estimation")
    p.add_argument("--image",     required=True,
                   help="Path to input food image")
    p.add_argument("--seg_ckpt",  required=True,
                   help="Path to U-Net segmentation checkpoint")
    p.add_argument("--csv",       required=True,
                   help="Path to dish_ingredients.csv")
    p.add_argument("--cls_ckpt",  default=None,
                   help="(Optional) Path to custom ResNet-50 classifier checkpoint")
    p.add_argument("--backbone",  default="unet",
                   choices=["unet", "resnet50"])
    p.add_argument("--img_size",  type=int, default=256)
    p.add_argument("--topk",      type=int, default=3)
    p.add_argument("--out_dir",   default="/content/Food_AI_Final/nutrition_output")
    p.add_argument("--compare",   action="store_true",
                   help="Print old vs new pipeline comparison table")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    estimator = NutritionEstimator(
        seg_checkpoint = args.seg_ckpt,
        csv_path       = args.csv,
        cls_checkpoint = args.cls_ckpt,
        backbone       = args.backbone,
        img_size       = args.img_size,
        topk           = args.topk,
        device         = device,
    )

    result = estimator.estimate(args.image, verbose=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(args.image).stem
    save_nutrition_figure(result, os.path.join(args.out_dir,
                                               f"{stem}_nutrition.png"))

    if args.compare:
        labels = [lbl for lbl, _ in result["top_predictions"]]
        compare_pipelines(args.image, result,
                          estimator.nutrition_db, labels)
