"""
bev_monolayout.py — Learning-Based BEV Projection (MonoLayout-style)
=====================================================================
A MonoLayout-inspired architecture adapted for YOUR dataset:

Key differences from original MonoLayout
-----------------------------------------
  Original MonoLayout: uses temporal sensor fusion from video to
    self-generate BEV occupancy GT (no external GT sensor).
  This version: uses drone world_x/world_y as direct, accurate BEV GT —
    rendered as Gaussian occupancy heatmaps on the BEV grid.
    This is your dataset's core contribution over prior work.

Architecture
------------
  Encoder  : ResNet-18 (pretrained ImageNet)
             Extracts context features from street-view image.

  Decoder  : Transposed-conv upsampling head
             Outputs BEV occupancy grid: (B, num_classes, H_bev, W_bev)
             Default BEV grid: 64x64 cells covering 40m x 40m forward range.

  Discriminator: PatchGAN discriminator (from MonoLayout)
             Trained adversarially to enforce realistic BEV layout priors.
             Critical for hallucinating occluded regions — the key MonoLayout
             contribution.

Training supervision (drone GT)
--------------------------------
  For each frame, matched drone tracks provide (world_x, world_y) positions.
  These are rendered as 2D Gaussian blobs on the BEV grid per class.
  Loss = BCE(pred_grid, gt_grid) + λ_adv * adversarial_loss
       + λ_focal * focal_loss (handles class imbalance: few objects vs empty space)

BEV Grid convention
--------------------
  Grid origin: camera position (bike)
  X axis: forward (away from camera)
  Y axis: lateral (left positive)
  Grid size: bev_range_m x bev_range_m (default 40m x 40m)
  Resolution: bev_range_m / bev_h (default 40/64 = 0.625 m/cell)

Usage — Training
-----------------
  python bev_monolayout.py train \
      --street_manifest all_street.csv \
      --drone_manifest all_drone.csv \
      --track_map_csv all_track_map.csv \
      --camera_cfg camera_params.json \
      --coord_align_csv coord_align.csv \
      --out_dir checkpoints/bev_run1 \
      --epochs 50 --batch_size 4

Usage — Evaluation
-------------------
  python bev_monolayout.py eval \
      --street_manifest val_street.csv \
      --drone_manifest val_drone.csv \
      --track_map_csv val_track_map.csv \
      --camera_cfg camera_params.json \
      --checkpoint checkpoints/bev_run1/best.pth \
      --out_report bev_learned_eval.json \
      --vis_dir vis_bev/

Dependencies
------------
  pip install torch torchvision opencv-python pandas numpy scipy
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Optional: visualization
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

CLASSES = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

BEV_H = 64          # BEV grid height (forward axis)
BEV_W = 64          # BEV grid width  (lateral axis)

# Grid tuned to ACTUAL data distribution from diagnostic:
#   x_fwd: 8.3–17.5m  → cover 5–22m (17m range, with margin)
#   y_lat: -3.8–0.8m  → cover -5–5m (10m range, symmetric)
#
# Previous 25×25m grid had objects in only 6.6% of cells.
# New asymmetric grid: objects in ~35% of cells → model can actually learn.
#
# Resolution: 17m/64 = 0.27 m/cell fwd,  10m/64 = 0.16 m/cell lat
BEV_FWD_MIN =  5.0    # near edge (m)
BEV_FWD_MAX = 22.0    # far edge (m)
BEV_LAT_MIN = -5.0    # left edge (m)
BEV_LAT_MAX =  5.0    # right edge (m)
BEV_FWD_RANGE = BEV_FWD_MAX - BEV_FWD_MIN   # 17.0m
BEV_LAT_RANGE = BEV_LAT_MAX - BEV_LAT_MIN   # 10.0m

# Legacy aliases used throughout (BEV_RANGE_M = forward range, BEV_Y_OFFSET = lat min)
BEV_RANGE_M  = BEV_FWD_RANGE
BEV_Y_OFFSET = BEV_LAT_MIN

# Gaussian blob sigma per class (meters) — larger = easier for model to learn
# With 25m range / 64 cells = 0.39m/cell, sigma=2.0m ≈ 5 cells radius
CLASS_SIGMA_M = {
    "car": 3.0, "truck": 4.0, "bus": 5.0,
    "person": 1.5, "bicycle": 1.5, "motorcycle": 2.0,
}
DEFAULT_SIGMA_M = 2.0

LAMBDA_ADV   = 0.0    # disabled — with 1 scene, adversarial signal is noise
LAMBDA_FOCAL = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_class(x: str) -> str:
    x = str(x).strip().lower()
    if x in {"pedestrian", "person", "people"}:
        return "person"
    if x in {"bike", "bicycle", "cyclist"}:
        return "bicycle"
    if x in {"motorbike", "motorcycle"}:
        return "motorcycle"
    return x


def world_to_bev_pixel(
    world_x: float, world_y: float,
    bev_range_m: float = BEV_RANGE_M,
    bev_h: int = BEV_H, bev_w: int = BEV_W,
    offset_x: float = 0.0, offset_y: float = 0.0,
    rot_deg: float = 0.0,
    bev_y_offset: float = BEV_Y_OFFSET,
    **kwargs,
) -> Optional[Tuple[int, int]]:
    """
    Convert drone x_fwd/y_left → BEV grid pixel (row, col).

    Grid covers:
      Forward: BEV_FWD_MIN (5m) to BEV_FWD_MAX (22m)
      Lateral: BEV_LAT_MIN (-5m) to BEV_LAT_MAX (5m)

    Objects outside this window return None.
    """
    dx = world_x - offset_x
    dy = world_y - offset_y
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    x = c * dx + s * dy   # forward (camera frame)
    y = -s * dx + c * dy  # lateral (camera frame)

    # Forward: must be within [BEV_FWD_MIN, BEV_FWD_MAX]
    x_rel = x - BEV_FWD_MIN
    if x_rel < 0 or x_rel > BEV_FWD_RANGE:
        return None

    # Lateral: must be within [BEV_LAT_MIN, BEV_LAT_MAX]
    y_rel = y - BEV_LAT_MIN
    if y_rel < 0 or y_rel > BEV_LAT_RANGE:
        return None

    row = int((BEV_FWD_RANGE - x_rel) / BEV_FWD_RANGE * bev_h)
    col = int(y_rel / BEV_LAT_RANGE * bev_w)
    row = max(0, min(bev_h - 1, row))
    col = max(0, min(bev_w - 1, col))
    return row, col


def render_gaussian_blob(
    grid: np.ndarray,
    row: int, col: int,
    sigma_cells: float,
):
    """Add a 2D Gaussian blob to grid at (row, col) with sigma in cells."""
    H, W = grid.shape
    for dr in range(-int(3 * sigma_cells) - 1, int(3 * sigma_cells) + 2):
        for dc in range(-int(3 * sigma_cells) - 1, int(3 * sigma_cells) + 2):
            r, c = row + dr, col + dc
            if 0 <= r < H and 0 <= c < W:
                val = math.exp(-(dr ** 2 + dc ** 2) / (2 * sigma_cells ** 2))
                grid[r, c] = max(grid[r, c], val)


def build_bev_gt(
    positions: List[Tuple[int, int, int]],  # (class_idx, row, col)
    bev_h: int = BEV_H,
    bev_w: int = BEV_W,
    sigmas_cells: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Render BEV GT grid from object positions.
    Returns float32 array of shape (NUM_CLASSES, bev_h, bev_w) in [0, 1].
    """
    grid = np.zeros((NUM_CLASSES, bev_h, bev_w), dtype=np.float32)
    for cidx, row, col in positions:
        sigma = sigmas_cells.get(cidx, 2.0) if sigmas_cells else 2.0
        render_gaussian_blob(grid[cidx], row, col, sigma)
    return grid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CrossViewBEVDataset(Dataset):
    """
    Each sample: (street_image, bev_gt_grid, frame_meta)

    street_image : (3, H, W) normalized tensor
    bev_gt_grid  : (NUM_CLASSES, BEV_H, BEV_W) float32 occupancy heatmap
    """

    def __init__(
        self,
        street_manifest: pd.DataFrame,
        drone_manifest: pd.DataFrame,
        track_map: pd.DataFrame,
        img_dir: str,           # root directory of street frame crops/images
        align_params: Dict,     # {scene_id: {offset_x, offset_y, scale, rot_deg}}
        bev_range_m: float = BEV_RANGE_M,
        bev_h: int = BEV_H,
        bev_w: int = BEV_W,
        img_h: int = 256,
        img_w: int = 512,
        min_conf: float = 0.0,
        augment: bool = True,
    ):
        self.street = street_manifest
        self.drone = drone_manifest
        self.align = align_params
        self.bev_range_m = bev_range_m
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.img_dir = img_dir

        # Filter track map
        tm = track_map[track_map["drone_track_id"] != -1].copy()
        if "stitch_confidence" in tm.columns:
            tm = tm[tm["stitch_confidence"] >= min_conf]
        self.valid_s2d = {
            int(r["street_track_id"]): int(r["drone_track_id"])
            for _, r in tm.iterrows()
        }

        # Build per-class sigma in cells
        cells_per_meter = bev_h / bev_range_m
        self.sigmas_cells: Dict[int, float] = {
            CLASS_TO_IDX[c]: CLASS_SIGMA_M.get(c, DEFAULT_SIGMA_M) * cells_per_meter
            for c in CLASSES
        }

        # Index: list of (scene_id, frame) pairs that have ≥1 drone GT object
        self.frames = self._build_frame_index()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _build_frame_index(self) -> List[Tuple[str, int]]:
        """Collect all (scene_id, frame) pairs that have drone GT in BEV grid."""
        valid_stids = set(self.valid_s2d.keys())
        df = self.street[self.street["track_id"].isin(valid_stids)].copy()
        if "scene_id" not in df.columns:
            df["scene_id"] = "scene_00"
        frames = []
        total_positions = 0
        for (scene_id, frame), _ in df.groupby(
            [df["scene_id"], df["frame"].astype(int)]
        ):
            positions = self._get_bev_positions(str(scene_id), int(frame))
            if len(positions) > 0:
                frames.append((str(scene_id), int(frame)))
                total_positions += len(positions)

        if frames:
            avg_occ = total_positions / (len(frames) * self.bev_h * self.bev_w) * 100
            print(f"[dataset] {len(frames)} frames with GT  |  "
                  f"avg {total_positions/len(frames):.1f} objects/frame  |  "
                  f"grid occupancy ~{avg_occ:.3f}%")
        else:
            print("[dataset] WARNING: 0 frames have GT in BEV grid — check coord_align and BEV range")
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        scene_id, frame_num = self.frames[idx]

        # --- Load street frame image ---
        img = self._load_frame_image(scene_id, frame_num)  # (H, W, 3) uint8

        # --- Build BEV GT ---
        positions = self._get_bev_positions(scene_id, frame_num)
        bev_gt = build_bev_gt(positions, self.bev_h, self.bev_w, self.sigmas_cells)

        # --- Augmentation (color jitter on street image only) ---
        if self.augment:
            img = self._augment_image(img)

        img_tensor = self.transform(img)  # (3, H, W)
        bev_tensor = torch.from_numpy(bev_gt)  # (C, BEV_H, BEV_W)

        return img_tensor, bev_tensor, {"scene_id": scene_id, "frame": frame_num}

    def _load_frame_image(self, scene_id: str, frame_num: int) -> np.ndarray:
        """Load and resize street frame. Returns HxWx3 uint8.

        Tries three sources in order:
          1. img_dir/scene_id/XXXXXX.jpg  (full frames)
          2. img_dir/XXXXXX.jpg           (flat directory)
          3. crop_path from street manifest (falls back to first crop of that frame)
        """
        if HAS_CV2:
            for ext in [".jpg", ".png"]:
                # Standard: img_dir/scene_id/000001.jpg
                path = os.path.join(self.img_dir, scene_id, f"{frame_num:06d}{ext}")
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return cv2.resize(img, (self.img_w, self.img_h))
                # Flat: img_dir/000001.jpg
                path = os.path.join(self.img_dir, f"{frame_num:06d}{ext}")
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return cv2.resize(img, (self.img_w, self.img_h))

            # Fallback: use crop_path from street manifest for this frame
            if "crop_path" in self.street.columns:
                rows = self.street[self.street["frame"].astype(int) == frame_num]
                if len(rows) > 0:
                    crop_path = str(rows.iloc[0]["crop_path"])
                    if os.path.exists(crop_path):
                        img = cv2.imread(crop_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            return cv2.resize(img, (self.img_w, self.img_h))

        # Final fallback: blank image (BEV GT still trains, image branch gets zeros)
        return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

    def _get_bev_positions(self, scene_id: str, frame_num: int) -> List[Tuple[int, int, int]]:
        """Get BEV pixel positions for all GT objects in this frame."""
        align = self.align.get(scene_id, {
            "offset_x": 0.0, "offset_y": 0.0, "rot_deg": 0.0
        })

        street_col = self.street.copy()
        if "scene_id" not in street_col.columns:
            street_col["scene_id"] = "scene_00"
        s_frame = street_col[
            (street_col["scene_id"] == scene_id) &
            (street_col["frame"].astype(int) == frame_num)
        ]

        positions = []
        drone_frame_col = "street_frame" if "street_frame" in self.drone.columns else "drone_frame"
        has_fwd = "x_fwd" in self.drone.columns and "y_left" in self.drone.columns

        for _, sr in s_frame.iterrows():
            s_tid = int(sr["track_id"])
            if s_tid not in self.valid_s2d:
                continue
            d_tid = self.valid_s2d[s_tid]
            cname = norm_class(sr["class_name"])
            if cname not in CLASS_TO_IDX:
                continue
            cidx = CLASS_TO_IDX[cname]

            d_rows = self.drone[
                (self.drone["track_id"] == d_tid) &
                (self.drone[drone_frame_col].astype(int) == frame_num)
            ]
            if len(d_rows) == 0:
                continue
            dr = d_rows.iloc[0]

            # Use drone-relative metric coords; fall back to world_x/world_y if absent
            if has_fwd:
                wx, wy = float(dr["x_fwd"]), float(dr["y_left"])
            else:
                wx, wy = float(dr["world_x"]), float(dr["world_y"])

            pix = world_to_bev_pixel(wx, wy,
                self.bev_range_m, self.bev_h, self.bev_w, **align)
            if pix is not None:
                positions.append((cidx, pix[0], pix[1]))

        return positions

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Simple color jitter augmentation."""
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            img = img[:, ::-1, :].copy()  # horizontal flip
        return img


# ---------------------------------------------------------------------------
# Model: Encoder
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    ResNet-18 encoder (pretrained) with multi-scale feature extraction.
    Returns feature map at 1/8 resolution.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # stride 4
        self.layer2 = resnet.layer2   # stride 8
        self.layer3 = resnet.layer3   # stride 16
        self.layer4 = resnet.layer4   # stride 32

        # Feature compression: combine layer3 + layer4 features
        self.compress = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)             # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)            # (B, 512, H/32, W/32)
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([f3, f4_up], dim=1)  # (B, 768, H/16, W/16)
        return self.compress(fused)            # (B, 256, H/16, W/16)


# ---------------------------------------------------------------------------
# Model: BEV Decoder
# ---------------------------------------------------------------------------

class BEVDecoder(nn.Module):
    """
    Decodes encoder features to BEV occupancy grid.

    Key fix: replaces AdaptiveAvgPool2d (which discards all spatial info)
    with a column-wise feature projection that preserves horizontal position.

    Intuition: in a perspective image, horizontal position → lateral BEV position
    is approximately preserved. Vertical position → forward BEV distance.
    We exploit this by treating each vertical column of features as a
    "ray" into the scene, and learning which BEV cells it activates.

    Input:  (B, 256, H/16, W/16)
    Output: (B, NUM_CLASSES, BEV_H, BEV_W)
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 bev_h: int = BEV_H, bev_w: int = BEV_W):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Column projection: compress H dimension, keep W
        # (B, 256, H/16, W/16) → (B, 128, 1, W/16) → (B, 128, bev_w)
        self.col_compress = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, bev_w)),   # compress rows only, keep cols
        )

        # Row projection: compress W dimension, keep H → forward axis
        # (B, 256, H/16, W/16) → (B, 128, bev_h, 1)
        self.row_compress = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((bev_h, 1)),   # compress cols only, keep rows
        )

        # Combine column and row projections into BEV feature volume
        # outer product: (B, 128, bev_h, 1) × (B, 128, 1, bev_w) → (B, 128, bev_h, bev_w)
        self.combine = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 256, Hf, Wf)
        col_feat = self.col_compress(features)  # (B, 128, 1, bev_w)
        row_feat = self.row_compress(features)  # (B, 128, bev_h, 1)

        # Outer product: lateral from cols, forward from rows
        bev_feat = col_feat * row_feat          # broadcast → (B, 128, bev_h, bev_w)

        x = self.combine(bev_feat)
        return self.out_conv(x)                 # raw logits


# ---------------------------------------------------------------------------
# Model: PatchGAN Discriminator (from MonoLayout)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator — operates on BEV occupancy grids.
    Enforces realistic spatial layout priors (e.g. cars don't appear on sidewalks,
    clusters of objects are spatially coherent).

    Takes (B, NUM_CLASSES, BEV_H, BEV_W) as input.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        def block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            nn.Conv2d(num_classes, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MonoLayoutBEV(nn.Module):
    def __init__(self, pretrained_encoder: bool = True,
                 num_classes: int = NUM_CLASSES,
                 bev_h: int = BEV_H, bev_w: int = BEV_W):
        super().__init__()
        self.encoder = ContextEncoder(pretrained=pretrained_encoder)
        self.decoder = BEVDecoder(num_classes=num_classes, bev_h=bev_h, bev_w=bev_w)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.encoder(img)
        return self.decoder(features)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def heatmap_mse_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bg_weight: float = 0.02,
) -> torch.Tensor:
    """
    Weighted MSE for Gaussian heatmap regression.

    The GT is already Gaussian blobs (values 0–1). Regress directly to them.
    This has NO degenerate solution:
      - all-positive → MSE on 99.96% empty cells is huge
      - all-zero     → MSE on blob cells (target=0.5–1) is large
      - optimal      → predict the exact GT heatmap

    bg_weight=0.02: background cells contribute 2% of the loss weight
    so the model focuses on getting blob positions right, not on perfectly
    predicting every background cell as exactly 0.
    """
    pred = torch.sigmoid(logits)

    # Weight map: blob cells get weight 1.0, background gets bg_weight
    weight = torch.where(target > 0.01,
                         torch.ones_like(target),
                         torch.full_like(target, bg_weight))

    mse = (pred - target) ** 2
    return (weight * mse).mean()


def adversarial_gen_loss(disc: PatchDiscriminator, logits: torch.Tensor) -> torch.Tensor:
    """Generator wants discriminator to predict 'real' for generated BEV."""
    pred = torch.sigmoid(logits)
    fake_scores = disc(pred)
    return F.mse_loss(fake_scores, torch.ones_like(fake_scores))


def adversarial_disc_loss(
    disc: PatchDiscriminator,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Discriminator: real GT → 1, generated pred → 0."""
    pred = torch.sigmoid(logits.detach())
    real_scores = disc(target)
    fake_scores = disc(pred)
    real_loss = F.mse_loss(real_scores, torch.ones_like(real_scores))
    fake_loss = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
    return (real_loss + fake_loss) * 0.5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bev_metrics(
    logits: torch.Tensor,  # raw logits (B, C, H, W) — sigmoid applied here
    target: torch.Tensor,
    threshold: float = 0.5,
    class_names: List[str] = CLASSES,
) -> Dict:
    """
    Compute per-class and mean mIoU, Precision, Recall at given threshold.
    Accepts raw logits; applies sigmoid internally.
    """
    pred = torch.sigmoid(logits)
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    results = {}
    ious, aps = [], []

    for i, cname in enumerate(class_names):
        p = pred_bin[:, i]
        t = target_bin[:, i]

        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        union = tp + fp + fn

        iou = tp / union if union > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[cname] = {
            "IoU": round(iou, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }
        ious.append(iou)

    results["mIoU"] = round(float(np.mean(ious)), 4)
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # Load data
    street = pd.read_csv(args.street_manifest)
    drone = pd.read_csv(args.drone_manifest)
    track_map = pd.read_csv(args.track_map_csv)
    street["class_name"] = street["class_name"].apply(norm_class)
    if "class_name" in drone.columns:
        drone["class_name"] = drone["class_name"].apply(norm_class)
    if "scene_id" not in street.columns:
        street["scene_id"] = "scene_00"
    if "scene_id" not in drone.columns:
        drone["scene_id"] = "scene_00"

    # Camera / alignment configs
    align_params = {}
    if args.coord_align_csv and os.path.exists(args.coord_align_csv):
        align_df = pd.read_csv(args.coord_align_csv)
        for _, row in align_df.iterrows():
            align_params[str(row["scene_id"])] = {
                "offset_x": float(row.get("offset_x", 0.0)),
                "offset_y": float(row.get("offset_y", 0.0)),
                "rot_deg":  float(row.get("rot_deg",  0.0)),
            }

    # Train / val split
    # With multiple scenes: split by scene (keeps scenes intact, avoids leakage).
    # With only one scene: split by frame number (last 20% of frames → val).
    all_scenes = street["scene_id"].unique().tolist()

    if len(all_scenes) >= 5:
        # Scene-level split
        val_scenes  = all_scenes[-max(1, len(all_scenes) // 5):]
        train_scenes = [s for s in all_scenes if s not in val_scenes]
        train_street = street[street["scene_id"].isin(train_scenes)]
        val_street   = street[street["scene_id"].isin(val_scenes)]
        print(f"[train] scene-level split — train: {train_scenes}  val: {val_scenes}")
    else:
        # Frame-level split — sort by frame, last 20% → val
        all_frames   = sorted(street["frame"].astype(int).unique())
        n_val        = max(1, len(all_frames) // 5)
        val_frames   = set(all_frames[-n_val:])
        train_frames = set(all_frames[:-n_val])
        train_street = street[street["frame"].astype(int).isin(train_frames)]
        val_street   = street[street["frame"].astype(int).isin(val_frames)]
        print(f"[train] frame-level split ({len(all_scenes)} scene(s)) — "
              f"train frames: {len(train_frames)}  val frames: {len(val_frames)}")
        print(f"[train] train range: {min(train_frames)}–{max(train_frames)}  "
              f"val range: {min(val_frames)}–{max(val_frames)}")

    train_ds = CrossViewBEVDataset(
        train_street, drone, track_map,
        img_dir=args.img_dir,
        align_params=align_params,
        min_conf=args.min_conf,
        augment=True,
    )
    val_ds = CrossViewBEVDataset(
        val_street, drone, track_map,
        img_dir=args.img_dir,
        align_params=align_params,
        min_conf=args.min_conf,
        augment=False,
    )
    print(f"[train] train frames: {len(train_ds)}  val frames: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # Models
    model = MonoLayoutBEV(pretrained_encoder=True).to(device)
    disc = PatchDiscriminator().to(device)

    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_miou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # ---- Training ----
        model.train()
        disc.train()
        train_losses = defaultdict(list)

        for imgs, bev_gt, _ in train_loader:
            imgs   = imgs.to(device)
            bev_gt = bev_gt.to(device)

            logits = model(imgs)
            loss_g = heatmap_mse_loss(logits, bev_gt)

            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()

            # Discriminator step (minimal role)
            loss_d = adversarial_disc_loss(disc, logits, bev_gt)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            train_losses["focal"].append(loss_g.item())
            train_losses["adv_g"].append(0.0)
            train_losses["disc"].append(loss_d.item())

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_logits_list, val_gts = [], []
        with torch.no_grad():
            for imgs, bev_gt, _ in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                val_logits_list.append(logits.cpu())
                val_gts.append(bev_gt)

        all_logits = torch.cat(val_logits_list, dim=0)
        all_gt     = torch.cat(val_gts, dim=0)
        metrics    = compute_bev_metrics(all_logits, all_gt)

        miou = metrics["mIoU"]
        car_prec = metrics.get("car", {}).get("Precision", 0)
        car_rec  = metrics.get("car", {}).get("Recall", 0)
        mean_dice  = np.mean(train_losses["focal"])   # stores MSE now
        mean_act   = float(torch.sigmoid(all_logits).mean())

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"mse={mean_dice:.5f}  "
              f"val_mIoU={miou:.4f}  "
              f"car P={car_prec:.3f} R={car_rec:.3f}  "
              f"act={mean_act:.4f}", end="")

        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(args.out_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "disc_state": disc.state_dict(),
                "miou": miou,
                "metrics": metrics,
            }, ckpt_path)
            print(f"  ✓ saved (best mIoU={best_miou:.4f})", end="")
        print()

        history.append({
            "epoch": epoch,
            "dice_loss": round(mean_dice, 5),
            "val_mIoU": miou,
            "mean_activation": round(mean_act, 5),
            "val_metrics": metrics,
        })

    # Save training history
    hist_path = os.path.join(args.out_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[done] best mIoU: {best_miou:.4f}")
    print(f"[done] checkpoint: {os.path.join(args.out_dir, 'best.pth')}")
    print(f"[done] history:    {hist_path}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    street = pd.read_csv(args.street_manifest)
    drone = pd.read_csv(args.drone_manifest)
    track_map = pd.read_csv(args.track_map_csv)
    street["class_name"] = street["class_name"].apply(norm_class)
    if "class_name" in drone.columns:
        drone["class_name"] = drone["class_name"].apply(norm_class)
    if "scene_id" not in street.columns:
        street["scene_id"] = "scene_00"

    align_params = {}
    if args.coord_align_csv and os.path.exists(args.coord_align_csv):
        align_df = pd.read_csv(args.coord_align_csv)
        for _, row in align_df.iterrows():
           
            align_params[str(row["scene_id"])] = {
                "offset_x": float(row.get("offset_x", 0.0)),
                "offset_y": float(row.get("offset_y", 0.0)),
                "rot_deg":  float(row.get("rot_deg",  0.0)),
            }
    ds = CrossViewBEVDataset(
        street, drone, track_map,
        img_dir=args.img_dir,
        align_params=align_params,
        min_conf=args.min_conf,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MonoLayoutBEV().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded checkpoint (epoch={ckpt['epoch']}, "
          f"mIoU={ckpt.get('miou', '?'):.4f})")

    all_logits_list, all_gt = [], []
    vis_samples = []

    with torch.no_grad():
        for imgs, bev_gt, meta in loader:
            logits = model(imgs.to(device)).cpu()
            all_logits_list.append(logits)
            all_gt.append(bev_gt)
            if args.vis_dir and len(vis_samples) < 16:
                pred_vis = torch.sigmoid(logits)
                for i in range(len(imgs)):
                    vis_samples.append((imgs[i], pred_vis[i], bev_gt[i], meta))

    all_logits = torch.cat(all_logits_list, dim=0)
    all_gt     = torch.cat(all_gt, dim=0)

    metrics_50 = compute_bev_metrics(all_logits, all_gt, threshold=0.5)
    metrics_25 = compute_bev_metrics(all_logits, all_gt, threshold=0.25)

    # Print table
    print("\n" + "=" * 60)
    print("  BEV LEARNED MODEL EVALUATION")
    print("=" * 60)
    print(f"\n  mIoU @0.5  : {metrics_50['mIoU']:.4f}")
    print(f"  mIoU @0.25 : {metrics_25['mIoU']:.4f}")
    print()
    print(f"  {'Class':<14} {'IoU@.5':>7} {'IoU@.25':>8} {'Prec':>6} {'Rec':>6}")
    print("  " + "-" * 46)
    for cname in CLASSES:
        m50 = metrics_50.get(cname, {})
        m25 = metrics_25.get(cname, {})
        print(f"  {cname:<14} {m50.get('IoU', 0):>7.4f} {m25.get('IoU', 0):>8.4f} "
              f"{m50.get('Precision', 0):>6.4f} {m50.get('Recall', 0):>6.4f}")
    print()

    # Save visualization
    if args.vis_dir and HAS_CV2:
        os.makedirs(args.vis_dir, exist_ok=True)
        _save_visualizations(vis_samples, args.vis_dir)
        print(f"[done] visualizations: {args.vis_dir}")

    # Save report
    report = {
        "metrics_threshold_0.5": metrics_50,
        "metrics_threshold_0.25": metrics_25,
        "num_samples": len(all_logits),
    }
    if args.out_report:
        with open(args.out_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[done] report: {args.out_report}")


def _save_visualizations(samples, vis_dir: str):
    """Save side-by-side: street image | pred BEV | GT BEV."""
    if not HAS_CV2:
        return
    for i, (img_t, pred_t, gt_t, _) in enumerate(samples):
        # Denormalize street image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # BEV grids: max over classes for visualization
        pred_vis = pred_t.max(dim=0)[0].numpy()
        gt_vis = gt_t.max(dim=0)[0].numpy()

        pred_bgr = cv2.applyColorMap(
            (pred_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        gt_bgr = cv2.applyColorMap(
            (gt_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Resize BEV panels to match image height
        h = img_np.shape[0]
        pred_bgr = cv2.resize(pred_bgr, (h, h))
        gt_bgr = cv2.resize(gt_bgr, (h, h))

        combined = np.concatenate([img_np, pred_bgr, gt_bgr], axis=1)
        cv2.imwrite(os.path.join(vis_dir, f"sample_{i:04d}.jpg"), combined)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MonoLayout-style BEV model for CrossView dataset.")
    sub = ap.add_subparsers(dest="cmd")

    # Shared args
    def add_shared(p):
        p.add_argument("--street_manifest", required=True)
        p.add_argument("--drone_manifest", required=True)
        p.add_argument("--track_map_csv", required=True)
        p.add_argument("--img_dir", default="frames",
                       help="Root directory of street frame images: img_dir/scene_id/XXXXXX.jpg")
        p.add_argument("--coord_align_csv", default=None)
        p.add_argument("--min_conf", type=float, default=0.5)

    # Train
    p_train = sub.add_parser("train")
    add_shared(p_train)
    p_train.add_argument("--out_dir", default="checkpoints/bev")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)

    # Eval
    p_eval = sub.add_parser("eval")
    add_shared(p_eval)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--out_report", default=None)
    p_eval.add_argument("--vis_dir", default=None)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        sys.exit(1)

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()