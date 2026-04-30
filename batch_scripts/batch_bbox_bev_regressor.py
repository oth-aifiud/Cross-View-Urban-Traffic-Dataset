"""
bbox_bev_regressor.py — Detection-Conditioned BEV Position Regressor
=====================================================================
A lightweight MLP that takes 2D bounding box features from the street
camera and regresses to BEV (x_fwd, y_lat) positions in meters.

Why this works where MonoLayout doesn't:
  - MonoLayout: image (3×H×W) → BEV grid (C×64×64) — 12M params, needs 50k+ frames
  - This model: bbox geometry + IPM prior (7D) → BEV position (2D)
  - The bbox bottom-center pixel already encodes most of the perspective geometry
  - The model learns a residual correction on top of explicit IPM cues

Input features per detection (7D):
  [1-px/W, py/H, w/W, h/H, log(h/H), x_ipm, y_ipm]
  where (px,py) = bbox bottom-center, (w,h) = bbox size, W/H = image dims,
  and (x_ipm,y_ipm) is the ground-plane IPM projection of the bbox bottom-center.

Output:
  [x_fwd, y_lat] in meters (camera-relative BEV position)
Usage — Train (single-scene mode)
---------------------------------
  python bbox_bev_regressor.py train \
      --street_manifest  .../street_wedge_manifest.csv \
      --drone_manifest   .../drone_wedge_manifest.csv \
      --track_map_csv    .../track_mapping.csv \
      --coord_align_csv  .../coord_align.csv \
      --camera_cfg       .../camera_params.json \
      --out_dir          .../checkpoints/bbox_regressor \
      --epochs           300

Usage — Train (batch scene-manifest mode)
-----------------------------------------
  python bbox_bev_regressor.py train \
      --scene_csv        .../processed_scene_manifest.csv \
      --camera_cfg       .../camera_params.json \
      --train_split      train \
      --val_split        val \
      --out_dir          .../checkpoints/bbox_regressor \
      --epochs           300

Usage — Eval (single-scene mode)
--------------------------------
  python bbox_bev_regressor.py eval \
      --street_manifest  .../street_wedge_manifest.csv \
      --drone_manifest   .../drone_wedge_manifest.csv \
      --track_map_csv    .../track_mapping.csv \
      --coord_align_csv  .../coord_align.csv \
      --camera_cfg       .../camera_params.json \
      --checkpoint       .../checkpoints/bbox_regressor/best.pth \
      --out_report       .../bbox_bev_eval.json

Usage — Eval (batch scene-manifest mode)
----------------------------------------
  python bbox_bev_regressor.py eval \
      --scene_csv        .../processed_scene_manifest.csv \
      --camera_cfg       .../camera_params.json \
      --eval_split       test \
      --checkpoint       .../checkpoints/bbox_regressor/best.pth \
      --out_report       .../bbox_bev_eval.json
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_class(x):
    x = str(x).strip().lower()
    if x in {"pedestrian", "person", "people"}: return "person"
    if x in {"bike", "bicycle", "cyclist"}:     return "bicycle"
    if x in {"motorbike", "motorcycle"}:        return "motorcycle"
    return x


def build_K_R(cfg: Dict, actual_w: Optional[float] = None, actual_h: Optional[float] = None):
    fx = float(cfg["fx"])
    fy = float(cfg["fy"])
    cx = float(cfg["cx"])
    cy = float(cfg["cy"])

    # Camera parameters are often stored for a calibration image whose nominal
    # size is approximately (2*cx, 2*cy). Rescale intrinsics to the actual
    # street-frame resolution used by the manifests / video frames.
    cfg_w = 2.0 * cx
    cfg_h = 2.0 * cy
    if actual_w is not None and actual_h is not None and cfg_w > 1e-6 and cfg_h > 1e-6:
        sx = float(actual_w) / cfg_w
        sy = float(actual_h) / cfg_h
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    pitch = float(cfg.get("pitch_deg", -8.0))
    a = math.radians(-pitch)
    R_base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    Ry = np.array([
        [math.cos(a), 0, math.sin(a)],
        [0, 1, 0],
        [-math.sin(a), 0, math.cos(a)],
    ], dtype=np.float64)
    R = Ry @ R_base
    cam_h = float(cfg.get("cam_height_m", 1.1))
    return K, R, cam_h


def ipm_project(px, py, K, R, cam_h) -> Optional[Tuple[float,float]]:
    """Project image point to ground via IPM."""
    K_inv = np.linalg.inv(K)
    ray = R @ (K_inv @ np.array([px, py, 1.0]))
    if ray[2] >= -1e-6:
        return None
    t = -cam_h / ray[2]
    return float(t * ray[0]), float(t * ray[1])


def drone_to_cam_bev(wx, wy, ox, oy, rot_deg) -> Tuple[float, float]:
    """Convert drone x_fwd/y_left to camera-relative BEV via inverse coord_align."""
    dx, dy = wx - ox, wy - oy
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    return c*dx + s*dy, -s*dx + c*dy   # (x_fwd, y_lat) in camera frame



# ---------------------------------------------------------------------------
# Batch scene-manifest loading
# ---------------------------------------------------------------------------

def load_from_scene_manifest(
    scene_csv: str,
    cam_cfg_all: Dict,
    split: Optional[str] = None,
    require_align_ok: bool = True,
):
    sm = pd.read_csv(scene_csv, skipinitialspace=True)

    required = ["scene_id", "street_manifest_csv", "drone_manifest_csv", "track_mapping_csv", "coord_align_csv"]
    missing = [c for c in required if c not in sm.columns]
    if missing:
        raise RuntimeError(f"scene manifest missing required columns: {missing}")

    if split is not None:
        if "split" not in sm.columns:
            raise RuntimeError("scene manifest has no 'split' column")
        sm = sm[sm["split"].astype(str).str.strip() == str(split)]

    if require_align_ok and "align_status" in sm.columns:
        sm = sm[sm["align_status"].astype(str).str.strip().str.lower() == "ok"]

    if len(sm) == 0:
        raise RuntimeError("No scenes remain after filtering scene manifest.")

    street_parts = []
    drone_parts = []
    track_parts = []
    align_by_scene: Dict[str, Dict] = {}
    cam_by_scene: Dict[str, Dict] = {}
    img_size_by_scene: Dict[str, Tuple[float, float]] = {}

    for _, row in sm.iterrows():
        sid = str(row["scene_id"]).strip()

        S = pd.read_csv(str(row["street_manifest_csv"]).strip()).copy()
        D = pd.read_csv(str(row["drone_manifest_csv"]).strip()).copy()
        Tm = pd.read_csv(str(row["track_mapping_csv"]).strip()).copy()
        A = pd.read_csv(str(row["coord_align_csv"]).strip()).copy()

        S["scene_id"] = sid
        D["scene_id"] = sid

        if "track_id" in S.columns:
            S["track_id_orig"] = S["track_id"]
            S["track_id"] = S["track_id"].apply(lambda x: f"{sid}::{x}")
        if "track_id" in D.columns:
            D["track_id_orig"] = D["track_id"]
            D["track_id"] = D["track_id"].apply(lambda x: f"{sid}::{x}")
        if "street_track_id" in Tm.columns:
            Tm["street_track_id_orig"] = Tm["street_track_id"]
            Tm["street_track_id"] = Tm["street_track_id"].apply(lambda x: f"{sid}::{x}")
        if "drone_track_id" in Tm.columns:
            Tm["drone_track_id_orig"] = Tm["drone_track_id"]
            Tm["drone_track_id"] = Tm["drone_track_id"].apply(
                lambda x: -1 if str(x) in {"-1", "-1.0"} or x == -1 else f"{sid}::{x}"
            )

        street_parts.append(S)
        drone_parts.append(D)
        track_parts.append(Tm)

        arow = A.iloc[0]
        align_by_scene[sid] = {
            "offset_x": float(arow.get("offset_x", 0.0)),
            "offset_y": float(arow.get("offset_y", 0.0)),
            "rot_deg": float(arow.get("rot_deg", 0.0)),
        }

        cfg = cam_cfg_all.get(sid, cam_cfg_all.get("default", {}))
        cam_by_scene[sid] = cfg

        x2c_img = "bbox_x2" if "bbox_x2" in S.columns else "x2"
        y2c_img = "bbox_y2" if "bbox_y2" in S.columns else "y2"
        img_w = float(int(math.ceil(float(S[x2c_img].max()))))
        img_h = float(int(math.ceil(float(S[y2c_img].max()))))
        img_size_by_scene[sid] = (img_w, img_h)

    street = pd.concat(street_parts, ignore_index=True) if street_parts else pd.DataFrame()
    drone = pd.concat(drone_parts, ignore_index=True) if drone_parts else pd.DataFrame()
    track_map = pd.concat(track_parts, ignore_index=True) if track_parts else pd.DataFrame()

    return street, drone, track_map, align_by_scene, cam_by_scene, img_size_by_scene

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Replacement: new BBoxBEVDataset class
class BBoxBEVDataset(Dataset):
    """
    Each sample: (features_7d, target_2d, meta)

    features_7d: [1-px/W, py/H, w/W, h/H, log(h/H+1e-4), x_ipm, y_ipm]
    target_2d:   [x_fwd_m, y_lat_m] in camera-relative BEV
    """

    def __init__(
        self,
        street: pd.DataFrame,
        drone: pd.DataFrame,
        track_map: pd.DataFrame,
        align_by_scene: Dict[str, Dict],
        cam_by_scene: Dict[str, Dict],
        img_size_by_scene: Dict[str, Tuple[float, float]],
        bev_fwd_min: float = 5.0,
        bev_fwd_max: float = 22.0,
        bev_lat_min: float = -5.0,
        bev_lat_max: float = 5.0,
        min_conf: float = 0.0,
        augment: bool = False,
    ):
        self.augment = augment
        self.bev_fwd_min = bev_fwd_min
        self.bev_fwd_max = bev_fwd_max
        self.bev_lat_min = bev_lat_min
        self.bev_lat_max = bev_lat_max
        self.align_by_scene = align_by_scene
        self.cam_by_scene = cam_by_scene
        self.img_size_by_scene = img_size_by_scene

        tm = track_map[track_map["drone_track_id"] != -1]
        if "stitch_confidence" in tm.columns:
            tm = tm[tm["stitch_confidence"] >= min_conf]
        self.s2d = dict(zip(tm["street_track_id"].astype(str), tm["drone_track_id"].astype(str)))

        fc = "street_frame" if "street_frame" in drone.columns else "drone_frame"
        x1c = "bbox_x1" if "bbox_x1" in street.columns else "x1"
        y1c = x1c.replace("x1", "y1")
        x2c = x1c.replace("x1", "x2")
        y2c = x1c.replace("x1", "y2")

        self.samples: List[Dict] = []
        skipped_range = 0
        skipped_ipm = 0

        if "scene_id" not in street.columns:
            street = street.copy()
            street["scene_id"] = "scene_00"
        if "scene_id" not in drone.columns:
            drone = drone.copy()
            drone["scene_id"] = "scene_00"

        for _, sr in street.iterrows():
            scene_id = str(sr["scene_id"])
            s_tid = str(sr["track_id"])
            if s_tid not in self.s2d:
                continue
            d_tid = self.s2d[s_tid]
            frame = int(sr["frame"])

            dr_rows = drone[
                (drone["scene_id"] == scene_id) &
                (drone["track_id"] == d_tid) &
                (drone[fc].astype(int) == frame)
            ]
            if len(dr_rows) == 0:
                continue
            dr = dr_rows.iloc[0]

            cfg = self.cam_by_scene.get(scene_id, self.cam_by_scene.get("default", {}))
            img_w, img_h = self.img_size_by_scene[scene_id]
            K, R_cam, cam_h = build_K_R(cfg, actual_w=img_w, actual_h=img_h)
            align = self.align_by_scene.get(scene_id, {"offset_x": 0.0, "offset_y": 0.0, "rot_deg": 0.0})
            ox = float(align.get("offset_x", 0.0))
            oy = float(align.get("offset_y", 0.0))
            rot = float(align.get("rot_deg", 0.0))

            has_cam_bev = ("x_fwd" in dr.index) and ("y_left" in dr.index)
            if has_cam_bev:
                xf = float(dr["x_fwd"])
                yl = float(dr["y_left"])
            else:
                wx = float(dr.get("world_x", 0.0))
                wy = float(dr.get("world_y", 0.0))
                xf, yl = drone_to_cam_bev(wx, wy, ox, oy, rot)

            if not (bev_fwd_min <= xf <= bev_fwd_max and bev_lat_min <= yl <= bev_lat_max):
                skipped_range += 1
                continue

            x1 = float(sr[x1c])
            y1 = float(sr[y1c])
            x2 = float(sr[x2c])
            y2 = float(sr[y2c])
            bw = x2 - x1
            bh = y2 - y1
            px = (x1 + x2) / 2.0
            py = y2

            ipm_xy = ipm_project(px, py, K, R_cam, cam_h)
            if ipm_xy is None:
                skipped_ipm += 1
                continue
            ipm_xf, ipm_yl = ipm_xy

            self.samples.append({
                "feat": [
                    1.0 - px / img_w,
                    py / img_h,
                    bw / img_w,
                    bh / img_h,
                    math.log(max(bh / img_h, 1e-4)),
                    ipm_xf,
                    ipm_yl,
                ],
                "target": [xf, yl],
                "class": norm_class(sr.get("class_name", "car")),
                "frame": frame,
                "scene_id": scene_id,
                "track_id": s_tid,
            })

        print(f"[dataset] {len(self.samples)} samples  ({skipped_range} outside BEV range, {skipped_ipm} invalid IPM)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feat = torch.tensor(s["feat"], dtype=torch.float32)
        target = torch.tensor(s["target"], dtype=torch.float32)

        if self.augment:
            noise = torch.zeros_like(feat)
            noise[:5] = torch.randn(5, dtype=feat.dtype) * 0.01
            feat = feat + noise

        return feat, target, {
            "class": s["class"],
            "frame": s["frame"],
            "scene_id": s["scene_id"],
            "track_id": s["track_id"],
        }


# ---------------------------------------------------------------------------
# Model: tiny MLP with residual IPM correction
# ---------------------------------------------------------------------------

# Replacement: new BBoxBEVRegressor class
class BBoxBEVRegressor(nn.Module):
    """
    Lightweight MLP: geometric/IPM features (7D) → BEV position (2D).

    Architecture: 7 → 96 → 96 → 48 → 2
    Uses explicit IPM as a strong prior:
      output = [x_ipm, y_ipm] + MLP_residual(features)
    """

    def __init__(self, use_ipm_prior: bool = True):
        super().__init__()
        self.use_ipm_prior = use_ipm_prior
        self.net = nn.Sequential(
            nn.Linear(7, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.net(x)
        if self.use_ipm_prior:
            prior = x[:, 5:7]
            return prior + correction
        return correction


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Replacement: batch scene-manifest aware train()
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    with open(args.camera_cfg) as f:
        cam_cfg_all = json.load(f)

    if args.scene_csv:
        train_street, train_drone, train_track_map, train_align_by_scene, train_cam_by_scene, train_img_sizes = load_from_scene_manifest(
            args.scene_csv, cam_cfg_all, split=args.train_split, require_align_ok=not args.allow_failed_align
        )
        val_street, val_drone, val_track_map, val_align_by_scene, val_cam_by_scene, val_img_sizes = load_from_scene_manifest(
            args.scene_csv, cam_cfg_all, split=args.val_split, require_align_ok=not args.allow_failed_align
        )
        print(f"[train] batch mode from scene manifest  |  train split='{args.train_split}'  val split='{args.val_split}'")
    else:
        street = pd.read_csv(args.street_manifest)
        drone = pd.read_csv(args.drone_manifest)
        track_map = pd.read_csv(args.track_map_csv)
        street["class_name"] = street["class_name"].apply(norm_class)
        if "scene_id" not in street.columns:
            street["scene_id"] = "scene_00"
        if "scene_id" not in drone.columns:
            drone["scene_id"] = "scene_00"

        cfg = cam_cfg_all.get("scene_00", cam_cfg_all.get("default", {}))
        align_df = pd.read_csv(args.coord_align_csv)
        align_row = align_df.iloc[0]
        align = {
            "offset_x": float(align_row.get("offset_x", 0)),
            "offset_y": float(align_row.get("offset_y", 0)),
            "rot_deg": float(align_row.get("rot_deg", 0)),
        }
        x2c_img = "bbox_x2" if "bbox_x2" in street.columns else "x2"
        y2c_img = "bbox_y2" if "bbox_y2" in street.columns else "y2"
        img_w = float(int(math.ceil(float(street[x2c_img].max()))))
        img_h = float(int(math.ceil(float(street[y2c_img].max()))))

        all_frames = sorted(street["frame"].astype(int).unique())
        n_val = max(1, len(all_frames) // 5)
        val_frames = set(all_frames[-n_val:])
        train_frames = set(all_frames[:-n_val])
        train_street = street[street["frame"].astype(int).isin(train_frames)]
        val_street = street[street["frame"].astype(int).isin(val_frames)]
        train_drone = drone
        val_drone = drone
        train_track_map = track_map
        val_track_map = track_map
        train_align_by_scene = {"scene_00": align}
        val_align_by_scene = {"scene_00": align}
        train_cam_by_scene = {"scene_00": cfg, "default": cfg}
        val_cam_by_scene = {"scene_00": cfg, "default": cfg}
        train_img_sizes = {"scene_00": (img_w, img_h)}
        val_img_sizes = {"scene_00": (img_w, img_h)}
        print(f"[train] legacy frame-level split — train frames: {len(train_frames)}  val frames: {len(val_frames)}")

    train_ds = BBoxBEVDataset(
        train_street, train_drone, train_track_map,
        train_align_by_scene, train_cam_by_scene, train_img_sizes,
        augment=True,
    )
    val_ds = BBoxBEVDataset(
        val_street, val_drone, val_track_map,
        val_align_by_scene, val_cam_by_scene, val_img_sizes,
        augment=False,
    )

    if len(train_ds) == 0:
        print("[error] No training samples — check coord_align and BEV range")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = BBoxBEVRegressor(use_ipm_prior=True).to(device)
    print(f"[train] model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_err = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_errs = []
        for feat, target, _ in train_loader:
            feat, target = feat.to(device), target.to(device)
            pred = model(feat)
            loss = F.smooth_l1_loss(pred, target[:, :2])
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                errs = (pred - target[:, :2]).norm(dim=1)
                train_errs.extend(errs.cpu().tolist())
        scheduler.step()

        model.eval()
        val_errs_fwd, val_errs_lat, val_errs_l2 = [], [], []
        with torch.no_grad():
            for feat, target, _ in val_loader:
                pred = model(feat.to(device)).cpu()
                l2 = (pred - target[:, :2]).norm(dim=1)
                fwd = (pred[:, 0] - target[:, 0]).abs()
                lat = (pred[:, 1] - target[:, 1]).abs()
                val_errs_l2.extend(l2.tolist())
                val_errs_fwd.extend(fwd.tolist())
                val_errs_lat.extend(lat.tolist())

        mean_train = float(np.mean(train_errs))
        mean_val = float(np.mean(val_errs_l2))
        mean_fwd = float(np.mean(val_errs_fwd))
        mean_lat = float(np.mean(val_errs_lat))
        pck1 = sum(1 for e in val_errs_l2 if e <= 1.0) / max(1, len(val_errs_l2))
        pck2 = sum(1 for e in val_errs_l2 if e <= 2.0) / max(1, len(val_errs_l2))

        marker = ""
        if mean_val < best_val_err:
            best_val_err = mean_val
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_err_m": mean_val,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            marker = "  ✓"

        if epoch % 10 == 0 or epoch <= 5 or marker:
            print(f"Epoch {epoch:4d}/{args.epochs}  train={mean_train:.3f}m  val={mean_val:.3f}m  fwd={mean_fwd:.3f}m  lat={mean_lat:.3f}m  PCK@1m={pck1:.1%}  PCK@2m={pck2:.1%}{marker}")

        history.append({
            "epoch": epoch,
            "train_err_m": round(mean_train, 4),
            "val_err_m": round(mean_val, 4),
            "fwd_err_m": round(mean_fwd, 4),
            "lat_err_m": round(mean_lat, 4),
            "pck_1m": round(pck1, 4),
            "pck_2m": round(pck2, 4),
        })

    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[done] best val error: {best_val_err:.3f}m")
    print(f"[done] checkpoint: {os.path.join(args.out_dir, 'best.pth')}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

# Replacement: batch scene-manifest aware evaluate()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.camera_cfg) as f:
        cam_cfg_all = json.load(f)

    if args.scene_csv:
        street, drone, track_map, align_by_scene, cam_by_scene, img_size_by_scene = load_from_scene_manifest(
            args.scene_csv, cam_cfg_all, split=args.eval_split, require_align_ok=not args.allow_failed_align
        )
        print(f"[eval] batch mode from scene manifest  |  eval split='{args.eval_split}'")
    else:
        street = pd.read_csv(args.street_manifest)
        drone = pd.read_csv(args.drone_manifest)
        track_map = pd.read_csv(args.track_map_csv)
        street["class_name"] = street["class_name"].apply(norm_class)
        if "scene_id" not in street.columns:
            street["scene_id"] = "scene_00"
        if "scene_id" not in drone.columns:
            drone["scene_id"] = "scene_00"

        cfg = cam_cfg_all.get("scene_00", cam_cfg_all.get("default", {}))
        align_df = pd.read_csv(args.coord_align_csv)
        align_row = align_df.iloc[0]
        align = {
            "offset_x": float(align_row.get("offset_x", 0)),
            "offset_y": float(align_row.get("offset_y", 0)),
            "rot_deg": float(align_row.get("rot_deg", 0)),
        }
        x2c_img = "bbox_x2" if "bbox_x2" in street.columns else "x2"
        y2c_img = "bbox_y2" if "bbox_y2" in street.columns else "y2"
        img_w = float(int(math.ceil(float(street[x2c_img].max()))))
        img_h = float(int(math.ceil(float(street[y2c_img].max()))))
        align_by_scene = {"scene_00": align}
        cam_by_scene = {"scene_00": cfg, "default": cfg}
        img_size_by_scene = {"scene_00": (img_w, img_h)}

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if state["net.0.weight"].shape != (96, 7):
        raise RuntimeError(
            "Checkpoint architecture does not match the cleaned 7D residual model. Please retrain and use a new checkpoint."
        )
    model = BBoxBEVRegressor(use_ipm_prior=True).to(device)
    model.load_state_dict(state)
    model.eval()

    print(f"[eval] loaded checkpoint (epoch={ckpt.get('epoch','?')}, val_err={ckpt.get('val_err_m',-1):.3f}m)")

    tm = track_map[track_map["drone_track_id"] != -1]
    s2d = dict(zip(tm["street_track_id"].astype(str), tm["drone_track_id"].astype(str)))
    fc = "street_frame" if "street_frame" in drone.columns else "drone_frame"
    x1c = "bbox_x1" if "bbox_x1" in street.columns else "x1"
    y1c = x1c.replace("x1", "y1")
    x2c = x1c.replace("x1", "x2")
    y2c = x1c.replace("x1", "y2")

    ipm_errs, learned_errs = [], []
    fwd_errs, lat_errs = [], []
    per_class: Dict[str, List] = {}

    with torch.no_grad():
        for _, sr in street.iterrows():
            scene_id = str(sr.get("scene_id", "scene_00"))
            s_tid = str(sr["track_id"])
            if s_tid not in s2d:
                continue
            d_tid = s2d[s_tid]
            frame = int(sr["frame"])
            dr_rows = drone[
                (drone["scene_id"] == scene_id) &
                (drone["track_id"] == d_tid) &
                (drone[fc].astype(int) == frame)
            ]
            if len(dr_rows) == 0:
                continue
            dr = dr_rows.iloc[0]

            cfg = cam_by_scene.get(scene_id, cam_by_scene.get("default", {}))
            img_w, img_h = img_size_by_scene[scene_id]
            K, R_cam, cam_h = build_K_R(cfg, actual_w=img_w, actual_h=img_h)
            align = align_by_scene.get(scene_id, {"offset_x": 0.0, "offset_y": 0.0, "rot_deg": 0.0})
            ox = float(align.get("offset_x", 0.0))
            oy = float(align.get("offset_y", 0.0))
            rot = float(align.get("rot_deg", 0.0))

            has_cam_bev = ("x_fwd" in dr.index) and ("y_left" in dr.index)
            if has_cam_bev:
                xf_gt = float(dr["x_fwd"])
                yl_gt = float(dr["y_left"])
            else:
                wx = float(dr.get("world_x", 0.0))
                wy = float(dr.get("world_y", 0.0))
                xf_gt, yl_gt = drone_to_cam_bev(wx, wy, ox, oy, rot)
            if not (5 <= xf_gt <= 22 and -5 <= yl_gt <= 5):
                continue

            cname = norm_class(sr.get("class_name", "car"))
            x1, y1 = float(sr[x1c]), float(sr[y1c])
            x2, y2 = float(sr[x2c]), float(sr[y2c])
            bw = x2 - x1
            bh = y2 - y1
            px = (x1 + x2) / 2
            py = y2

            ipm = ipm_project(px, py, K, R_cam, cam_h)
            ipm_err = math.sqrt((ipm[0] - xf_gt) ** 2 + (ipm[1] - yl_gt) ** 2) if ipm else None
            if ipm is None:
                continue
            feat = torch.tensor([[1.0 - px / img_w, py / img_h, bw / img_w, bh / img_h,
                                  math.log(max(bh / img_h, 1e-4)),
                                  ipm[0], ipm[1]]], dtype=torch.float32)
            pred = model(feat.to(device)).cpu().numpy()[0]
            learned_err = math.sqrt((pred[0] - xf_gt) ** 2 + (pred[1] - yl_gt) ** 2)
            fwd_err = abs(pred[0] - xf_gt)
            lat_err = abs(pred[1] - yl_gt)

            if ipm_err is not None:
                ipm_errs.append(ipm_err)
            learned_errs.append(learned_err)
            fwd_errs.append(fwd_err)
            lat_errs.append(lat_err)
            per_class.setdefault(cname, {"ipm": [], "learned": []})
            if ipm_err is not None:
                per_class[cname]["ipm"].append(ipm_err)
            per_class[cname]["learned"].append(learned_err)

    def stats(errs):
        if not errs:
            return {"mean_m": float("nan"), "median_m": float("nan"), "pck_1m": 0.0, "pck_2m": 0.0, "n": 0}
        return {
            "mean_m": round(float(np.mean(errs)), 3),
            "median_m": round(float(np.median(errs)), 3),
            "pck_1m": round(sum(1 for e in errs if e <= 1) / len(errs), 4),
            "pck_2m": round(sum(1 for e in errs if e <= 2) / len(errs), 4),
            "n": len(errs),
        }

    ipm_s = stats(ipm_errs)
    lrn_s = stats(learned_errs)
    ade = float(np.mean(learned_errs)) if learned_errs else float("nan")
    fde = float(np.mean(learned_errs)) if learned_errs else float("nan")
    ale = float(np.mean(lat_errs)) if lat_errs else float("nan")
    alge = float(np.mean(fwd_errs)) if fwd_errs else float("nan")
    pck1 = sum(1 for e in learned_errs if e <= 1) / max(1, len(learned_errs))
    pck2 = sum(1 for e in learned_errs if e <= 2) / max(1, len(learned_errs))

    print("\n" + "=" * 60)
    print("  BBOX BEV REGRESSOR — EVALUATION")
    print("=" * 60)
    print(f"\n  Samples evaluated: {len(learned_errs)}")
    print(f"\n  {'Method':<20} {'ADE↓':>8} {'Median↓':>8} {'PCK@1m↑':>8} {'PCK@2m↑':>8}")
    print("  " + "-" * 48)
    print(f"  {'IPM (baseline)':<20} {ipm_s['mean_m']:>8.3f} {ipm_s['median_m']:>8.3f} {ipm_s['pck_1m']:>7.1%} {ipm_s['pck_2m']:>7.1%}")
    print(f"  {'BBox Regressor':<20} {lrn_s['mean_m']:>8.3f} {lrn_s['median_m']:>8.3f} {lrn_s['pck_1m']:>7.1%} {lrn_s['pck_2m']:>7.1%}")

    improvement = (ipm_s['mean_m'] - lrn_s['mean_m']) / ipm_s['mean_m'] * 100 if ipm_s['n'] > 0 and np.isfinite(ipm_s['mean_m']) and ipm_s['mean_m'] > 1e-6 else float('nan')
    print(f"\n  Improvement over IPM: {improvement:+.1f}%")
    print(f"\n  Point metrics")
    print("  " + "-" * 48)
    print(f"  ADE               : {ade:.3f}m")
    print(f"  FDE               : {fde:.3f}m")
    print(f"  ALE               : {ale:.3f}m")
    print(f"  ALgE              : {alge:.3f}m")
    print(f"  PCK@1m            : {pck1:.1%}")
    print(f"  PCK@2m            : {pck2:.1%}")

    print(f"\n  {'Class':<14} {'N':>4}  {'IPM':>7}  {'Learned':>7}  {'Improv':>7}")
    print("  " + "-" * 44)
    for cname in sorted(per_class):
        pc = per_class[cname]
        im = float(np.mean(pc['ipm'])) if pc['ipm'] else float('nan')
        lm = float(np.mean(pc['learned'])) if pc['learned'] else float('nan')
        imp = (im - lm) / im * 100 if pc['ipm'] and np.isfinite(im) and im > 1e-6 else float('nan')
        print(f"  {cname:<14} {len(pc['learned']):>4}  {im:>7.3f}  {lm:>7.3f}  {imp:>+6.1f}%")

    report = {
        "ipm": ipm_s,
        "learned": lrn_s,
        "ADE": round(ade, 3) if np.isfinite(ade) else None,
        "FDE": round(fde, 3) if np.isfinite(fde) else None,
        "ALE": round(ale, 3) if np.isfinite(ale) else None,
        "ALgE": round(alge, 3) if np.isfinite(alge) else None,
        "PCK@1m": round(pck1, 4),
        "PCK@2m": round(pck2, 4),
        "improvement_pct": round(improvement, 2) if np.isfinite(improvement) else None,
        "per_class": {k: {"ipm": stats(v['ipm']), "learned": stats(v['learned'])} for k, v in per_class.items()},
    }
    if args.out_report:
        with open(args.out_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[done] report: {args.out_report}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    def shared(p):
        p.add_argument("--scene_csv", default=None,
                       help="Processed scene manifest CSV for batch mode")
        p.add_argument("--street_manifest", default=None)
        p.add_argument("--drone_manifest", default=None)
        p.add_argument("--track_map_csv", default=None)
        p.add_argument("--coord_align_csv", default=None)
        p.add_argument("--camera_cfg", required=True)

    p_train = sub.add_parser("train")
    shared(p_train)
    p_train.add_argument("--out_dir", default="checkpoints/bbox_regressor")
    p_train.add_argument("--epochs", type=int, default=300)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--train_split", default="train")
    p_train.add_argument("--val_split", default="val")
    p_train.add_argument("--allow_failed_align", action="store_true",
                         help="Include scenes whose align_status is not 'ok'")

    p_eval = sub.add_parser("eval")
    shared(p_eval)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--out_report", default=None)
    p_eval.add_argument("--eval_split", default="test")
    p_eval.add_argument("--allow_failed_align", action="store_true",
                        help="Include scenes whose align_status is not 'ok'")

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help(); sys.exit(1)

    if args.scene_csv is None:
        if not args.street_manifest or not args.drone_manifest or not args.track_map_csv or not args.coord_align_csv:
            raise RuntimeError("Provide either --scene_csv or all of --street_manifest, --drone_manifest, --track_map_csv, --coord_align_csv")

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()