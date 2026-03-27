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

Usage — Train
-------------
  python bbox_bev_regressor.py train \
      --street_manifest  .../street_wedge_manifest.csv \
      --drone_manifest   .../drone_wedge_manifest.csv \
      --track_map_csv    .../track_mapping.csv \
      --coord_align_csv  .../coord_align.csv \
      --camera_cfg       .../camera_params.json \
      --out_dir          .../checkpoints/bbox_regressor \
      --epochs           300

Usage — Eval (compare against IPM baseline)
-------------------------------------------
  python bbox_bev_regressor.py eval \
      --street_manifest  .../street_wedge_manifest.csv \
      --drone_manifest   .../drone_wedge_manifest.csv \
      --track_map_csv    .../track_mapping.csv \
      --coord_align_csv  .../coord_align.csv \
      --camera_cfg       .../camera_params.json \
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
        align: Dict,
        img_w: float, img_h: float,
        K: np.ndarray,
        R_cam: np.ndarray,
        cam_h: float,
        bev_fwd_min: float = 5.0,
        bev_fwd_max: float = 22.0,
        bev_lat_min: float = -5.0,
        bev_lat_max: float = 5.0,
        min_conf: float = 0.0,
        augment: bool = False,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.augment = augment
        self.bev_fwd_min = bev_fwd_min
        self.bev_fwd_max = bev_fwd_max
        self.bev_lat_min = bev_lat_min
        self.bev_lat_max = bev_lat_max

        ox = float(align.get("offset_x", 0.0))
        oy = float(align.get("offset_y", 0.0))
        rot = float(align.get("rot_deg", 0.0))

        tm = track_map[track_map["drone_track_id"] != -1]
        if "stitch_confidence" in tm.columns:
            tm = tm[tm["stitch_confidence"] >= min_conf]
        s2d = dict(zip(tm["street_track_id"].astype(int),
                       tm["drone_track_id"].astype(int)))

        fc = "street_frame" if "street_frame" in drone.columns else "drone_frame"
        x1c = "bbox_x1" if "bbox_x1" in street.columns else "x1"
        y1c = x1c.replace("x1", "y1")
        x2c = x1c.replace("x1", "x2")
        y2c = x1c.replace("x1", "y2")

        self.samples: List[Dict] = []
        skipped_range = 0
        skipped_ipm = 0

        for _, sr in street.iterrows():
            s_tid = int(sr["track_id"])
            if s_tid not in s2d:
                continue
            d_tid = s2d[s_tid]
            frame = int(sr["frame"])

            dr_rows = drone[
                (drone["track_id"] == d_tid) &
                (drone[fc].astype(int) == frame)
            ]
            if len(dr_rows) == 0:
                continue
            dr = dr_rows.iloc[0]

            # GT BEV position in ego/camera frame
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

            # Bbox geometry
            x1 = float(sr[x1c])
            y1 = float(sr[y1c])
            x2 = float(sr[x2c])
            y2 = float(sr[y2c])
            bw = x2 - x1
            bh = y2 - y1
            px = (x1 + x2) / 2.0
            py = y2 # slightly above bbox bottom for more stable ground contact

            # Explicit IPM prior
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
            })

        print(f"[dataset] {len(self.samples)} samples  "
              f"({skipped_range} outside BEV range, {skipped_ipm} invalid IPM)")

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

        return feat, target, {"class": s["class"], "frame": s["frame"]}


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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # Load data
    street    = pd.read_csv(args.street_manifest)
    drone     = pd.read_csv(args.drone_manifest)
    track_map = pd.read_csv(args.track_map_csv)
    street["class_name"] = street["class_name"].apply(norm_class)

    # Camera config
    with open(args.camera_cfg) as f:
        cam_cfg = json.load(f)
    cfg = cam_cfg.get("scene_00", cam_cfg.get("default", {}))

    # Use the actual street-manifest pixel coordinate extent rather than the
    # calibration-space nominal size (2*cx, 2*cy).
    x2c_img = "bbox_x2" if "bbox_x2" in street.columns else "x2"
    y2c_img = "bbox_y2" if "bbox_y2" in street.columns else "y2"
    img_w = float(int(math.ceil(float(street[x2c_img].max()))))
    img_h = float(int(math.ceil(float(street[y2c_img].max()))))
    K, R_cam, cam_h = build_K_R(cfg, actual_w=img_w, actual_h=img_h)
    print(f"[train] using image size from street manifest: img_w={img_w}  img_h={img_h}")

    # Alignment
    align_df = pd.read_csv(args.coord_align_csv)
    align = {}
    for _, row in align_df.iterrows():
        scene = str(row["scene_id"])
        align[scene] = {
            "offset_x": float(row.get("offset_x", 0)),
            "offset_y": float(row.get("offset_y", 0)),
            "rot_deg":  float(row.get("rot_deg",  0)),
        }
    # Use first scene's align if no scene_id in street
    default_align = align.get("scene_00", list(align.values())[0] if align else {})

    if "scene_id" not in street.columns:
        street["scene_id"] = "scene_00"

    # Frame-level train/val split (last 20% → val)
    all_frames = sorted(street["frame"].astype(int).unique())
    n_val = max(1, len(all_frames) // 5)
    val_frames  = set(all_frames[-n_val:])
    train_frames = set(all_frames[:-n_val])

    train_street = street[street["frame"].astype(int).isin(train_frames)]
    val_street   = street[street["frame"].astype(int).isin(val_frames)]
    print(f"[train] frames: train={len(train_frames)}  val={len(val_frames)}")


    train_ds = BBoxBEVDataset(train_street, drone, track_map, default_align,
                               img_w, img_h, K, R_cam, cam_h, augment=True)
    val_ds   = BBoxBEVDataset(val_street,   drone, track_map, default_align,
                               img_w, img_h, K, R_cam, cam_h, augment=False)

    if len(train_ds) == 0:
        print("[error] No training samples — check coord_align and BEV range")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2)

    model = BBoxBEVRegressor(use_ipm_prior=True).to(device)
    print(f"[train] model params: {sum(p.numel() for p in model.parameters()):,}")

    opt       = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_err = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_errs = []
        for feat, target, _ in train_loader:
            feat, target = feat.to(device), target.to(device)
            pred = model(feat)
            loss = F.smooth_l1_loss(pred, target[:, :2])  # Huber loss on (x_fwd, y_lat)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                errs = (pred - target[:, :2]).norm(dim=1)
                train_errs.extend(errs.cpu().tolist())
        scheduler.step()

        # Val
        model.eval()
        val_errs_fwd, val_errs_lat, val_errs_l2 = [], [], []
        with torch.no_grad():
            for feat, target, _ in val_loader:
                pred = model(feat.to(device)).cpu()
                l2   = (pred - target[:, :2]).norm(dim=1)
                fwd  = (pred[:,0] - target[:,0]).abs()
                lat  = (pred[:,1] - target[:,1]).abs()
                val_errs_l2.extend(l2.tolist())
                val_errs_fwd.extend(fwd.tolist())
                val_errs_lat.extend(lat.tolist())

        mean_train = float(np.mean(train_errs))
        mean_val   = float(np.mean(val_errs_l2))
        mean_fwd   = float(np.mean(val_errs_fwd))
        mean_lat   = float(np.mean(val_errs_lat))
        pck1 = sum(1 for e in val_errs_l2 if e<=1.0) / max(1, len(val_errs_l2))
        pck2 = sum(1 for e in val_errs_l2 if e<=2.0) / max(1, len(val_errs_l2))

        marker = ""
        if mean_val < best_val_err:
            best_val_err = mean_val
            ckpt = {
                "epoch": epoch, "model_state": model.state_dict(),
                "val_err_m": mean_val, "img_w": img_w, "img_h": img_h,
                "align": default_align,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            marker = "  ✓"

        if epoch % 10 == 0 or epoch <= 5 or marker:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={mean_train:.3f}m  val={mean_val:.3f}m  "
                  f"fwd={mean_fwd:.3f}m  lat={mean_lat:.3f}m  "
                  f"PCK@1m={pck1:.1%}  PCK@2m={pck2:.1%}{marker}")

        history.append({
            "epoch": epoch,
            "train_err_m": round(mean_train, 4),
            "val_err_m":   round(mean_val, 4),
            "fwd_err_m":   round(mean_fwd, 4),
            "lat_err_m":   round(mean_lat, 4),
            "pck_1m":      round(pck1, 4),
            "pck_2m":      round(pck2, 4),
        })

    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[done] best val error: {best_val_err:.3f}m")
    print(f"[done] checkpoint: {os.path.join(args.out_dir, 'best.pth')}")

    # Compare against IPM on val set
    print("\n--- IPM vs Learned comparison on val set ---")
    # K, R_cam, cam_h already computed above
    x1c = "bbox_x1" if "bbox_x1" in val_street.columns else "x1"
    y1c = x1c.replace("x1","y1"); x2c = x1c.replace("x1","x2"); y2c = x1c.replace("x1","y2")
    ipm_errs, learned_errs = [], []
    model.eval()
    ox  = float(default_align.get("offset_x",0))
    oy  = float(default_align.get("offset_y",0))
    rot = float(default_align.get("rot_deg",0))
    tm2 = track_map[track_map["drone_track_id"]!=-1]
    s2d2 = dict(zip(tm2["street_track_id"].astype(int), tm2["drone_track_id"].astype(int)))
    fc = "street_frame" if "street_frame" in drone.columns else "drone_frame"

    with torch.no_grad():
        for _, sr in val_street.iterrows():
            s_tid = int(sr["track_id"])
            if s_tid not in s2d2: continue
            d_tid = s2d2[s_tid]
            frame = int(sr["frame"])
            dr_rows = drone[(drone["track_id"]==d_tid)&(drone[fc].astype(int)==frame)]
            if len(dr_rows)==0: continue
            dr = dr_rows.iloc[0]
            has_cam_bev = ("x_fwd" in dr.index) and ("y_left" in dr.index)
            if has_cam_bev:
                xf_gt = float(dr["x_fwd"])
                yl_gt = float(dr["y_left"])
            else:
                wx = float(dr.get("world_x", 0.0))
                wy = float(dr.get("world_y", 0.0))
                xf_gt, yl_gt = drone_to_cam_bev(wx, wy, ox, oy, rot)
            if not (5<=xf_gt<=22 and -5<=yl_gt<=5): continue

            x1,y1 = float(sr[x1c]),float(sr[y1c])
            x2,y2 = float(sr[x2c]),float(sr[y2c])
            bw=x2-x1; bh=y2-y1; px=(x1+x2)/2; py=y2

            # IPM
            ipm = ipm_project(px, py, K, R_cam, cam_h)
            if ipm:
                ipm_errs.append(math.sqrt((ipm[0]-xf_gt)**2+(ipm[1]-yl_gt)**2))

            # Learned: construct 7D feature
            if ipm is None:
                continue
            feat = torch.tensor([[1.0-px/img_w, py/img_h, bw/img_w, bh/img_h,
                                   math.log(max(bh/img_h,1e-4)),
                                   ipm[0], ipm[1]]], dtype=torch.float32)
            pred = model(feat.to(device)).cpu().numpy()[0]
            learned_errs.append(math.sqrt((pred[0]-xf_gt)**2+(pred[1]-yl_gt)**2))

    if ipm_errs:
        print(f"IPM:     mean={np.mean(ipm_errs):.2f}m  "
              f"PCK@1m={sum(1 for e in ipm_errs if e<=1)/len(ipm_errs):.1%}  "
              f"PCK@2m={sum(1 for e in ipm_errs if e<=2)/len(ipm_errs):.1%}")
    if learned_errs:
        print(f"Learned: mean={np.mean(learned_errs):.2f}m  "
              f"PCK@1m={sum(1 for e in learned_errs if e<=1)/len(learned_errs):.1%}  "
              f"PCK@2m={sum(1 for e in learned_errs if e<=2)/len(learned_errs):.1%}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    street    = pd.read_csv(args.street_manifest)
    drone     = pd.read_csv(args.drone_manifest)
    track_map = pd.read_csv(args.track_map_csv)
    street["class_name"] = street["class_name"].apply(norm_class)

    with open(args.camera_cfg) as f:
        cam_cfg = json.load(f)
    cfg = cam_cfg.get("scene_00", cam_cfg.get("default", {}))

    align_df = pd.read_csv(args.coord_align_csv)
    align_row = align_df.iloc[0]
    align = {
        "offset_x": float(align_row.get("offset_x",0)),
        "offset_y": float(align_row.get("offset_y",0)),
        "rot_deg":  float(align_row.get("rot_deg",0)),
    }

    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if state["net.0.weight"].shape != (96, 7):
        raise RuntimeError(
            "Checkpoint architecture does not match the cleaned 7D residual model. "
            "Please retrain and use a new checkpoint."
        )
    model = BBoxBEVRegressor(use_ipm_prior=True).to(device)
    model.load_state_dict(state)
    model.eval()

    x2c_img = "bbox_x2" if "bbox_x2" in street.columns else "x2"
    y2c_img = "bbox_y2" if "bbox_y2" in street.columns else "y2"
    img_w = float(int(math.ceil(float(street[x2c_img].max()))))
    img_h = float(int(math.ceil(float(street[y2c_img].max()))))
    print(f"[eval] loaded checkpoint (epoch={ckpt.get('epoch','?')}, val_err={ckpt.get('val_err_m',-1):.3f}m)")
    print(f"[eval] using image size from street manifest: img_w={img_w}  img_h={img_h}")

    K, R_cam, cam_h = build_K_R(cfg, actual_w=img_w, actual_h=img_h)
    ox  = float(align.get("offset_x",0))
    oy  = float(align.get("offset_y",0))
    rot = float(align.get("rot_deg",0))
    tm  = track_map[track_map["drone_track_id"]!=-1]
    s2d = dict(zip(tm["street_track_id"].astype(int), tm["drone_track_id"].astype(int)))
    fc  = "street_frame" if "street_frame" in drone.columns else "drone_frame"
    x1c = "bbox_x1" if "bbox_x1" in street.columns else "x1"
    y1c = x1c.replace("x1","y1"); x2c = x1c.replace("x1","x2"); y2c = x1c.replace("x1","y2")

    ipm_errs, learned_errs = [], []
    per_class: Dict[str, List] = {}

    with torch.no_grad():
        for _, sr in street.iterrows():
            s_tid = int(sr["track_id"])
            if s_tid not in s2d: continue
            d_tid = s2d[s_tid]; frame = int(sr["frame"])
            dr_rows = drone[(drone["track_id"]==d_tid)&(drone[fc].astype(int)==frame)]
            if len(dr_rows)==0: continue
            dr = dr_rows.iloc[0]
            has_cam_bev = ("x_fwd" in dr.index) and ("y_left" in dr.index)
            if has_cam_bev:
                xf_gt = float(dr["x_fwd"])
                yl_gt = float(dr["y_left"])
            else:
                wx = float(dr.get("world_x", 0.0))
                wy = float(dr.get("world_y", 0.0))
                xf_gt, yl_gt = drone_to_cam_bev(wx, wy, ox, oy, rot)
            if not (5<=xf_gt<=22 and -5<=yl_gt<=5): continue

            cname = norm_class(sr.get("class_name","car"))
            x1,y1=float(sr[x1c]),float(sr[y1c]); x2,y2=float(sr[x2c]),float(sr[y2c])
            bw=x2-x1; bh=y2-y1; px=(x1+x2)/2; py=y2

            ipm = ipm_project(px, py, K, R_cam, cam_h)
            ipm_err = math.sqrt((ipm[0]-xf_gt)**2+(ipm[1]-yl_gt)**2) if ipm else None
            if ipm is None:
                continue
            feat = torch.tensor([[1.0-px/img_w, py/img_h, bw/img_w, bh/img_h,
                                   math.log(max(bh/img_h,1e-4)),
                                   ipm[0], ipm[1]]], dtype=torch.float32)
            pred = model(feat.to(device)).cpu().numpy()[0]
            learned_err = math.sqrt((pred[0]-xf_gt)**2+(pred[1]-yl_gt)**2)

            if ipm_err is not None: ipm_errs.append(ipm_err)
            learned_errs.append(learned_err)
            per_class.setdefault(cname, {"ipm":[], "learned":[]})
            if ipm_err is not None: per_class[cname]["ipm"].append(ipm_err)
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

    print("\n" + "="*60)
    print("  BBOX BEV REGRESSOR — EVALUATION")
    print("="*60)
    print(f"\n  Samples evaluated: {len(learned_errs)}")
    print(f"\n  {'Method':<20} {'ADE↓':>8} {'Median↓':>8} {'PCK@1m↑':>8} {'PCK@2m↑':>8}")
    print("  " + "-"*48)
    ipm_s = stats(ipm_errs); lrn_s = stats(learned_errs)
    print(f"  {'IPM (baseline)':<20} {ipm_s['mean_m']:>8.3f} {ipm_s['median_m']:>8.3f} "
          f"{ipm_s['pck_1m']:>7.1%} {ipm_s['pck_2m']:>7.1%}")
    print(f"  {'BBox Regressor':<20} {lrn_s['mean_m']:>8.3f} {lrn_s['median_m']:>8.3f} "
          f"{lrn_s['pck_1m']:>7.1%} {lrn_s['pck_2m']:>7.1%}")
    improvement = (ipm_s['mean_m'] - lrn_s['mean_m']) / ipm_s['mean_m'] * 100 if ipm_s['n'] > 0 and np.isfinite(ipm_s['mean_m']) and ipm_s['mean_m'] > 1e-6 else float('nan')
    print(f"\n  Improvement over IPM: {improvement:+.1f}%")

    print(f"\n  {'Class':<14} {'N':>4}  {'IPM':>7}  {'Learned':>7}  {'Improv':>7}")
    print("  " + "-"*44)
    for cname in sorted(per_class):
        pc = per_class[cname]
        im = float(np.mean(pc["ipm"])) if pc["ipm"] else float("nan")
        lm = float(np.mean(pc["learned"])) if pc["learned"] else float("nan")
        imp = (im - lm) / im * 100 if pc["ipm"] and np.isfinite(im) and im > 1e-6 else float("nan")
        print(f"  {cname:<14} {len(pc['learned']):>4}  {im:>7.3f}  {lm:>7.3f}  {imp:>+6.1f}%")

    report = {
        "ipm": ipm_s, "learned": lrn_s,
        "improvement_pct": round(improvement, 2),
        "per_class": {k: {"ipm": stats(v["ipm"]), "learned": stats(v["learned"])}
                      for k,v in per_class.items()},
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
        p.add_argument("--street_manifest",  required=True)
        p.add_argument("--drone_manifest",   required=True)
        p.add_argument("--track_map_csv",    required=True)
        p.add_argument("--coord_align_csv",  required=True)
        p.add_argument("--camera_cfg",       required=True)

    p_train = sub.add_parser("train")
    shared(p_train)
    p_train.add_argument("--out_dir",  default="checkpoints/bbox_regressor")
    p_train.add_argument("--epochs",   type=int,   default=300)
    p_train.add_argument("--lr",       type=float, default=1e-3)

    p_eval = sub.add_parser("eval")
    shared(p_eval)
    p_eval.add_argument("--checkpoint",  required=True)
    p_eval.add_argument("--out_report",  default=None)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help(); sys.exit(1)

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()