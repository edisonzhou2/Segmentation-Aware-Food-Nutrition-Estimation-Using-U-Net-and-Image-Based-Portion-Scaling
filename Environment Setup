----Upload Project Python Files----
from google.colab import files
uploaded = files.upload()  

import shutil
for fname in uploaded:
    shutil.move(fname, os.path.join(PROJECT_ROOT, fname))
    print(f'Moved {fname} → {PROJECT_ROOT}/')

----Prepare Dataset Splits----
import os, shutil, random
from pathlib import Path

DATASET_ROOT = Path("/content/Food_AI_Final/dataset")

train_img_dir = DATASET_ROOT / "train/images"
train_mask_dir = DATASET_ROOT / "train/masks"

val_img_dir = DATASET_ROOT / "val/images"
val_mask_dir = DATASET_ROOT / "val/masks"
test_img_dir = DATASET_ROOT / "test/images"
test_mask_dir = DATASET_ROOT / "test/masks"

for d in [val_img_dir, val_mask_dir, test_img_dir, test_mask_dir]:
    d.mkdir(parents=True, exist_ok=True)

image_files = sorted([
    f for f in train_img_dir.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

random.seed(42)
random.shuffle(image_files)

n = len(image_files)
val_count = int(0.15 * n)
test_count = int(0.15 * n)

val_files = image_files[:val_count]
test_files = image_files[val_count:val_count + test_count]

def find_mask(image_path):
    """
    Finds matching mask with same name but usually .png.
    Example:
    image: apple_001.jpg
    mask:  apple_001.png
    """
    possible_masks = [
        train_mask_dir / (image_path.stem + ".png"),
        train_mask_dir / (image_path.stem + ".jpg"),
        train_mask_dir / (image_path.stem + ".jpeg"),
    ]
    for m in possible_masks:
        if m.exists():
            return m
    return None

def copy_pair(files, out_img_dir, out_mask_dir):
    copied = 0
    missing = 0

    for img in files:
        mask = find_mask(img)

        if mask is None:
            print("Missing mask for:", img.name)
            missing += 1
            continue

        shutil.copy2(img, out_img_dir / img.name)
        shutil.copy2(mask, out_mask_dir / mask.name)
        copied += 1

    print(f"Copied {copied} pairs. Missing masks: {missing}")

copy_pair(val_files, val_img_dir, val_mask_dir)
copy_pair(test_files, test_img_dir, test_mask_dir)

----Load Dataset and Visualize Samples----
sys.path.insert(0, PROJECT_ROOT)
from dataset import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np

train_loader, val_loader, test_loader = get_dataloaders(
    DATASET_ROOT, img_size=(256, 256), batch_size=4, num_workers=2)

imgs, masks = next(iter(train_loader))
print('Image batch:', imgs.shape)   # [4, 3, 256, 256]
print('Mask  batch:', masks.shape)  # [4, 1, 256, 256]
print('Mask unique values:', masks.unique().tolist())

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])
for i in range(4):
    img_np = imgs[i].permute(1,2,0).numpy()
    img_np = (img_np * STD + MEAN).clip(0,1)
    axes[0][i].imshow(img_np);                    axes[0][i].axis('off')
    axes[1][i].imshow(masks[i,0], cmap='gray');   axes[1][i].axis('off')
axes[0][0].set_title('Images', fontweight='bold')
axes[1][0].set_title('GT Masks', fontweight='bold')
plt.suptitle('Training Samples — FoodSeg103', fontsize=13)
plt.tight_layout(); plt.show()

----Build and Verify U-Net Model----
from model_unet import build_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_unet = build_model('unet').to(device)
x = torch.randn(2, 3, 256, 256).to(device)
y = model_unet(x)
print(f'UNet      input {tuple(x.shape)} → output {tuple(y.shape)}')
print(f'Params: {sum(p.numel() for p in model_unet.parameters())/1e6:.1f}M')

model_res = build_model('resnet50', pretrained=True).to(device)
y2 = model_res(x)
print(f'ResNetUNet input {tuple(x.shape)} → output {tuple(y2.shape)}')
print(f'Params: {sum(p.numel() for p in model_res.parameters())/1e6:.1f}M')

----Test Loss Functions and Metrics----
from losses  import get_loss
from metrics import compute_all, MetricAccumulator

preds   = torch.sigmoid(torch.randn(4, 1, 256, 256))
targets = (torch.rand(4, 1, 256, 256) > 0.5).float()

print('── Loss Functions ──────────────────')
for name in ('bce', 'dice', 'bce_dice', 'focal', 'tversky'):
    loss = get_loss(name)
    print(f'  {name:10s} → {loss(preds, targets).item():.4f}')

print('\n── Metrics ─────────────────────────')
m = compute_all(preds, targets)
for k, v in m.items():
    print(f'  {k:<12} → {v:.4f}')

----Train Food Segmentation Model----
from train_segmentation import train

model, history = train(
    root        = DATASET_ROOT,
    backbone    = 'unet',       
    pretrained  = True,
    img_size    = 256,
    epochs      = 30,
    batch_size  = 8,
    lr          = 1e-4,
    loss_name   = 'bce_dice',
    num_workers = 2,
    save_dir    = CKPT_DIR,
)

----Evaluate Segmentation Model----
from predict_segmentation import load_model_from_checkpoint, evaluate_test_set

BEST_CKPT = f'{CKPT_DIR}/best_model.pth'
model_inf = load_model_from_checkpoint(BEST_CKPT, 'unet', str(device))

evaluate_test_set(
    model     = model_inf,
    image_dir = f'{DATASET_ROOT}/test/images',
    mask_dir  = f'{DATASET_ROOT}/test/masks',
    out_dir   = PRED_DIR,
    img_size  = (256, 256),
    threshold = 0.5,
    device    = str(device),
    n_samples = 8,
)

IPImage(filename=f'{PRED_DIR}/prediction_grid.png', width=900)

----Upload Fixed Nutrition CSV----
from google.colab import files
import shutil, os

uploaded = files.upload()  

for fname in uploaded:
    shutil.move(fname, os.path.join(PROJECT_ROOT, fname))
    print("Moved", fname, "to", PROJECT_ROOT)

----Initialize Nutrition Estimator----
from nutrition_with_segmentation import NutritionEstimator, compare_pipelines

estimator = NutritionEstimator(
    seg_checkpoint = BEST_CKPT,
    csv_path       = CSV_PATH,
    backbone       = 'unet',
    img_size       = 256,
    topk           = 3,
    device         = str(device),
)

import glob
test_images = sorted(glob.glob(f'{DATASET_ROOT}/test/images/*'))[:1]
if test_images:
    result = estimator.estimate(test_images[0], verbose=True)
else:
    print('No test images found — check DATASET_ROOT')

----Top Prediction----
import os
from pathlib import Path
from nutrition_with_segmentation import save_nutrition_figure

fig_path = os.path.join(NUTRI_DIR, Path(test_images[0]).stem + '_nutrition.png')
save_nutrition_figure(result, fig_path)
IPImage(filename=fig_path, width=900)

----Compare Pipeline and Run Batch Nutrition----
labels = [lbl for lbl, _ in result['top_predictions']]
compare_pipelines(test_images[0], result, estimator.nutrition_db, labels)

----Compare Pipeline and Run Batch Nutrition----
import pandas as pd, glob

test_imgs = sorted(glob.glob(f'{DATASET_ROOT}/test/images/*'))[:20]
rows = []
for img_path in test_imgs:
    r = estimator.estimate(img_path, verbose=False)
    rows.append({
        'image':             Path(img_path).name,
        'predicted_dish':    r['predicted_dish'],
        'food_area_%':       round(r['food_pixel_fraction']*100, 1),
        'portion_weight':    r['portion_weight'],
        'calories_kcal':     r['calories'],
        'fat_g':             r['fat'],
        'carbs_g':           r['carbs'],
        'protein_g':         r['protein'],
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(f'{NUTRI_DIR}/batch_nutrition_results.csv', index=False)
print(f'\nSaved → {NUTRI_DIR}/batch_nutrition_results.csv')

----Custom Food Image Demo----
import os, sys
from pathlib import Path
from google.colab import files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

print("  Select a food image from your computer...")
uploaded = files.upload()

if not uploaded:
    print("No file uploaded. Run the cell again and choose an image.")
else:
    img_filename = list(uploaded.keys())[0]
    img_path     = f"/content/{img_filename}"

    with open(img_path, "wb") as f:
        f.write(uploaded[img_filename])
    print(f"Saved: {img_path}")

    print("\n  Running segmentation + nutrition estimation...")

    result = estimator.estimate(img_path, verbose=False)

    print("\n" + "=" * 52)
    print("     NUTRITION ESTIMATION RESULTS")
    print("=" * 52)
    print(f"  {'predicted_dish':<20} {result['predicted_dish']}")
    print(f"  {'food_area_%':<20} {result['food_pixel_fraction']*100:.1f}%")
    print(f"  {'portion_weight':<20} {result['portion_weight']:.2f}×")
    print(f"  {'calories_kcal':<20} {result['calories']:.0f} kcal")
    print(f"  {'fat_g':<20} {result['fat']:.1f} g")
    print(f"  {'carbs_g':<20} {result['carbs']:.1f} g")
    print(f"  {'protein_g':<20} {result['protein']:.1f} g")
    print("=" * 52)

    orig_img = np.array(
        Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
    )
    mask    = result["mask"]     # (H, W) binary
    overlay = result["overlay"]  # (H, W, 3) uint8

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("white")

    axes[0].imshow(orig_img)
    axes[0].set_title("Your image", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Food mask  (U-Net)", fontsize=12, fontweight="bold", pad=8)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold", pad=8)
    axes[2].axis("off")

    summary = (
        f"Dish: {result['predicted_dish']}   |   "
        f"Food area: {result['food_pixel_fraction']*100:.1f}%   |   "
        f"Portion: {result['portion_weight']:.2f}×   |   "
        f"Calories: {result['calories']:.0f} kcal   |   "
        f"Fat: {result['fat']:.1f}g   |   "
        f"Carbs: {result['carbs']:.1f}g   |   "
        f"Protein: {result['protein']:.1f}g"
    )
    fig.text(0.5, -0.02, summary, ha="center", fontsize=9.5,
             color="#444", wrap=True)

    plt.tight_layout()
    plt.savefig("/content/my_food_result.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("\n  Figure saved → /content/my_food_result.png")

    labels = ["Fat", "Carbs", "Protein"]
    values = [result["fat"], result["carbs"], result["protein"]]
    colors = ["#D85A30", "#1D9E75", "#378ADD"]

    fig2, ax = plt.subplots(figsize=(5, 2.8))
    bars = ax.barh(labels, values, color=colors, height=0.45)
    ax.bar_label(bars, fmt="%.1f g", padding=4, fontsize=10)
    ax.set_xlabel("grams", fontsize=10)
    ax.set_title("Macronutrient breakdown", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.3 + 1)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    plt.savefig("/content/my_food_macros.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("  Macro chart saved → /content/my_food_macros.png")
