"""
model_unet.py
-------------
U-Net baseline for binary food segmentation.

Architecture
────────────
Encoder  : 4 downsampling blocks  (double-conv + MaxPool)
Bottleneck: double-conv
Decoder  : 4 upsampling blocks  (bilinear upsample + skip + double-conv)
Head     : 1×1 conv → 1 channel sigmoid output

Input  : [B, 3,  H, W]  — normalised RGB image
Output : [B, 1,  H, W]  — probability map in [0, 1]

Optional ResNet-50 encoder
──────────────────────────
Pass `backbone='resnet50'` to use a pretrained ResNet-50 as the encoder.
The decoder is kept the same custom U-Net decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── building blocks ───────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive (Conv → BN → ReLU) blocks."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv (decoder step)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad x to match skip spatial size (handles odd input dims)
        diffH = skip.size(2) - x.size(2)
        diffW = skip.size(3) - x.size(3)
        x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                       diffH // 2, diffH - diffH // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── baseline U-Net (scratch encoder) ─────────────────────────────────────────

class UNet(nn.Module):
    """
    Vanilla U-Net with 4 levels.

    Parameters
    ----------
    in_channels  : int   — 3 for RGB
    out_channels : int   — 1 for binary mask
    features     : list  — channel counts per encoder level
    """

    def __init__(
        self,
        in_channels:  int  = 3,
        out_channels: int  = 1,
        features: list     = [64, 128, 256, 512],
    ):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck
        self.bottleneck = Down(features[3], features[3] * 2)

        # Decoder
        self.up1 = Up(features[3] * 2, features[3], features[3])
        self.up2 = Up(features[3],     features[2], features[2])
        self.up3 = Up(features[2],     features[1], features[1])
        self.up4 = Up(features[1],     features[0], features[0])

        # Output head
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path (save skip connections)
        x1 = self.inc(x)        # [B, 64,  H,    W   ]
        x2 = self.down1(x1)     # [B, 128, H/2,  W/2 ]
        x3 = self.down2(x2)     # [B, 256, H/4,  W/4 ]
        x4 = self.down3(x3)     # [B, 512, H/8,  W/8 ]
        x5 = self.bottleneck(x4)# [B,1024, H/16, W/16]

        # Decoder path (skip connections from encoder)
        x = self.up1(x5, x4)   # [B, 512, H/8,  W/8 ]
        x = self.up2(x,  x3)   # [B, 256, H/4,  W/4 ]
        x = self.up3(x,  x2)   # [B, 128, H/2,  W/2 ]
        x = self.up4(x,  x1)   # [B, 64,  H,    W   ]

        logits = self.out_conv(x)   # [B, 1, H, W]
        return torch.sigmoid(logits)


# ── ResNet-50 encoder + U-Net decoder ────────────────────────────────────────

class ResNetUNet(nn.Module):
    """
    U-Net with a pretrained ResNet-50 encoder.

    Skip connections are taken from ResNet intermediate layers:
      layer0 (stem)  → 64 ch
      layer1         → 256 ch
      layer2         → 512 ch
      layer3         →1024 ch
      layer4         →2048 ch  (bottleneck)

    The decoder mirrors the vanilla U-Net decoder but uses 1×1 convs to
    project skip channels before concatenation.
    """

    def __init__(self, out_channels: int = 1, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base    = models.resnet50(weights=weights)

        # Encoder layers
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # /2
        self.pool   = base.maxpool                                      # /4
        self.layer1 = base.layer1    # /4   256 ch
        self.layer2 = base.layer2    # /8   512 ch
        self.layer3 = base.layer3    # /16  1024 ch
        self.layer4 = base.layer4    # /32  2048 ch

        # Decoder
        self.up1 = Up(2048, 1024, 512)
        self.up2 = Up(512,   512, 256)
        self.up3 = Up(256,   256, 128)
        self.up4 = Up(128,    64,  64)
        # After up4, spatial = /2 of input — one more upsample to recover
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(64, 32),
        )

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)   # /2   64
        xp = self.pool(x0)    # /4
        x1 = self.layer1(xp)  # /4   256
        x2 = self.layer2(x1)  # /8   512
        x3 = self.layer3(x2)  # /16  1024
        x4 = self.layer4(x3)  # /32  2048

        # Decoder with skips
        d = self.up1(x4, x3)  # /16  512
        d = self.up2(d,  x2)  # /8   256
        d = self.up3(d,  x1)  # /4   128
        d = self.up4(d,  x0)  # /2   64
        d = self.up5(d)        # /1   32

        logits = self.out_conv(d)
        return torch.sigmoid(logits)


# ── factory ───────────────────────────────────────────────────────────────────

def build_model(backbone: str = "unet", pretrained: bool = True,
                out_channels: int = 1) -> nn.Module:
    """
    backbone : 'unet'    → vanilla U-Net (trained from scratch)
               'resnet50'→ ResNet-50 encoder + U-Net decoder
    """
    if backbone == "resnet50":
        model = ResNetUNet(out_channels=out_channels, pretrained=pretrained)
        print(f"Built ResNet-50 U-Net (pretrained={pretrained})")
    else:
        model = UNet(in_channels=3, out_channels=out_channels)
        print("Built vanilla U-Net")
    return model


# ── quick sanity-check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in ("unet", "resnet50"):
        m = build_model(name).to(device)
        x = torch.randn(2, 3, 256, 256).to(device)
        y = m(x)
        print(f"{name:10s}  input {tuple(x.shape)}  →  output {tuple(y.shape)}")
        total = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"           {total:.1f}M parameters\n")
