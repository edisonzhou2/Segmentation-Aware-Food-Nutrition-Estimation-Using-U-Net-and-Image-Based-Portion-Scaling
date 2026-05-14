"""
dataset.py
----------
PyTorch Dataset for loading image-mask pairs from FoodSeg103 / UECFOODPIXCOMPLETE.

Directory structure expected:
  root/
    train/images/*.jpg  (or .png)
    train/masks/*.png
    val/images/*.jpg
    val/masks/*.png
    test/images/*.jpg
    test/masks/*.png

Masks are binary:  0 = background, 1 = food
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


# ── helper ────────────────────────────────────────────────────────────────────

def get_image_paths(folder: str):
    """Return sorted list of image paths (jpg/jpeg/png) inside *folder*."""
    exts = {".jpg", ".jpeg", ".png"}
    folder = Path(folder)
    paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    return paths


# ── Dataset ───────────────────────────────────────────────────────────────────

class FoodSegDataset(Dataset):
    """
    Parameters
    ----------
    root : str
        Root dataset directory (e.g. '/content/Food_AI_Final/dataset').
    split : str
        One of 'train', 'val', 'test'.
    img_size : tuple
        (H, W) to resize images and masks to.
    augment : bool
        Apply random augmentations (only for training).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: tuple = (256, 256),
        augment: bool = False,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.img_size = img_size
        self.augment = augment

        img_dir  = Path(root) / split / "images"
        mask_dir = Path(root) / split / "masks"

        self.image_paths = get_image_paths(img_dir)
        self.mask_paths  = get_image_paths(mask_dir)

        assert len(self.image_paths) > 0, \
            f"No images found in {img_dir}. Check the path."
        assert len(self.image_paths) == len(self.mask_paths), (
            f"Mismatch: {len(self.image_paths)} images vs "
            f"{len(self.mask_paths)} masks in split='{split}'"
        )

        # ImageNet normalisation (used after converting to tensor)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )

    # ── internal transforms ───────────────────────────────────────────────────

    def _resize(self, img: Image.Image, mask: Image.Image):
        img  = TF.resize(img,  self.img_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)
        return img, mask

    def _augment(self, img: Image.Image, mask: Image.Image):
        """Synchronised augmentations applied to both image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.3:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # Color jitter on image only
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3,
                                saturation=0.2, hue=0.05)
        img = jitter(img)

        return img, mask

    # ── __len__ / __getitem__ ─────────────────────────────────────────────────

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load
        img  = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")   # grayscale

        # Resize
        img, mask = self._resize(img, mask)

        # Optional augmentation
        if self.augment:
            img, mask = self._augment(img, mask)

        # Image → tensor [3, H, W] float in [0,1], then normalise
        img_tensor = TF.to_tensor(img)          # float32, [0,1]
        img_tensor = self.normalize(img_tensor)

        # Mask → binary tensor [1, H, W]  (0 or 1)
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 0).astype(np.float32)  # binarise
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor


# ── convenience factory ───────────────────────────────────────────────────────

def get_dataloaders(
    root: str,
    img_size: tuple = (256, 256),
    batch_size: int = 8,
    num_workers: int = 2,
):
    """
    Returns (train_loader, val_loader, test_loader).
    Call this from train_segmentation.py.
    """
    train_ds = FoodSegDataset(root, split="train", img_size=img_size, augment=True)
    val_ds   = FoodSegDataset(root, split="val",   img_size=img_size, augment=False)
    test_ds  = FoodSegDataset(root, split="test",  img_size=img_size, augment=False)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    print(f"Dataset splits — train: {len(train_ds)} | "
          f"val: {len(val_ds)} | test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ── quick sanity-check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    ROOT = "/content/Food_AI_Final/dataset"
    train_loader, val_loader, test_loader = get_dataloaders(ROOT, batch_size=4)

    imgs, masks = next(iter(train_loader))
    print("Image batch shape :", imgs.shape)   # [4, 3, 256, 256]
    print("Mask  batch shape :", masks.shape)  # [4, 1, 256, 256]
    print("Mask unique values:", masks.unique())
