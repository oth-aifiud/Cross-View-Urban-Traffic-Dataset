"""
visualize_bbox_regressor.py — BBox Regressor BEV Visualization
===============================================================
4-panel figure per frame:
  1. Street frame with detection boxes + class labels
  2. BEV comparison: IPM (circles) vs Learned (stars) vs GT (filled dots)
  3. Error scatter: per-object IPM vs Learned error
  4. Drone overhead with GT boxes + BEV grid footprint

Usage
-----
  # Single frame:
  python visualize_bbox_regressor.py \
      --street_manifest  .../street_wedge_manifest.csv \
      --drone_manifest   .../drone_wedge_manifest.csv \
      --track_map_csv    .../track_mapping.csv \
      --coord_align_csv  .../coord_align.csv \
      --camera_cfg       .../camera_params.json \
      --checkpoint       .../bbox_regressor/best.pth \
      --street_video     .../bike.mp4 \
      --drone_video      .../drone.mp4 \
      --img_dir          .../frames \
      --frame            831 \
      --out              vis_bbox/frame_831.png

  # Batch (20 richest frames):
  python visualize_bbox_regressor.py ... --n_frames 20 --out_dir vis_bbox/
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Constants — must match bbox_bev_regressor.py
# ---------------------------------------------------------------------------
BEV_FWD_MIN, BEV_FWD_MAX = 5.0, 22.0
BEV_LAT_MIN, BEV_LAT_MAX = -5.0, 5.0

CLASS_COLORS = {
    "car":        "#2196F3",
    "truck":      "#FF9800",
    "bus":        "#9C27B0",
    "person":     "#4CAF50",
    "bicycle":    "#00BCD4",
    "motorcycle": "#F44336",
}
DEFAULT_COLOR = "#607D8B"
CLASSES = ["car","truck","bus","person","bicycle","motorcycle"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_class(x):
    x = str(x).strip().lower()
    if x in {"pedestrian","person","people"}: return "person"
    if x in {"bike","bicycle","cyclist"}:     return "bicycle"
    if x in {"motorbike","motorcycle"}:       return "motorcycle"
    return x


def extract_frame(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None


def load_frame_from_dir(img_dir, scene_id, frame_num):
    for ext in [".jpg", ".png"]:
        for p in [
            os.path.join(img_dir, scene_id, f"{frame_num:06d}{ext}"),
            os.path.join(img_dir, f"{frame_num:06d}{ext}"),
        ]:
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def build_K_R(cfg, actual_w: Optional[float] = None, actual_h: Optional[float] = None):
    fx = float(cfg["fx"])
    fy = float(cfg["fy"])
    cx = float(cfg["cx"])
    cy = float(cfg["cy"])

    # Camera params are often stored for a calibration image whose nominal
    # size is approximately (2*cx, 2*cy). Rescale intrinsics to the actual
    # street-frame resolution used by the manifests / frames.
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
    a = math.radians(-float(cfg.get("pitch_deg", -8.0)))
    R_base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    Ry = np.array([
        [math.cos(a), 0, math.sin(a)],
        [0, 1, 0],
        [-math.sin(a), 0, math.cos(a)],
    ], dtype=np.float64)
    R = Ry @ R_base
    return K, R, float(cfg.get("cam_height_m", 1.1))


def ipm_project(px, py, K, R, cam_h):
    K_inv = np.linalg.inv(K)
    ray = R @ (K_inv @ np.array([px, py, 1.0]))
    if ray[2] >= -1e-6: return None
    t = -cam_h / ray[2]
    return float(t*ray[0]), float(t*ray[1])

# ---- Helper: compute IPM ray z ----
def ipm_ray_z(px, py, K, R):
    K_inv = np.linalg.inv(K)
    ray = R @ (K_inv @ np.array([px, py, 1.0]))
    return float(ray[2])


def drone_to_cam(wx, wy, ox, oy, rot_deg):
    dx, dy = wx-ox, wy-oy
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    xf =  c*dx + s*dy   # forward
    yl = -s*dx + c*dy   # lateral: positive = LEFT in camera frame
    return xf, yl


def solve_drone_affine(d_frame):
    if "x_fwd" not in d_frame.columns or "bbox_x1" not in d_frame.columns:
        return None
    rows_m, rows_px = [], []
    for _, dr in d_frame.iterrows():
        if pd.isna(dr.get("bbox_x1", float("nan"))): continue
        rows_m.append([float(dr["x_fwd"]), float(dr["y_left"]), 1.0])
        rows_px.append([(float(dr["bbox_x1"])+float(dr["bbox_x2"]))/2,
                        (float(dr["bbox_y1"])+float(dr["bbox_y2"]))/2])
    if len(rows_m) < 3: return None
    M  = np.array(rows_m,  dtype=np.float64)
    Px = np.array(rows_px, dtype=np.float64)
    A_T, _, _, _ = np.linalg.lstsq(M, Px, rcond=None)
    A = A_T.T
    pred = M @ A_T
    errs = np.linalg.norm(pred - Px, axis=1)
    print(f"[drone] affine: {len(rows_m)} pts  median={np.median(errs):.1f}px")
    return A


def apply_affine(A, x, y):
    v = np.array([x, y, 1.0])
    return int(round(float(A[0]@v))), int(round(float(A[1]@v)))


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    for name in ["bbox_bev_regressor"]:
        spec = importlib.util.find_spec(name)
        if spec is None: continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckpt  = torch.load(checkpoint_path, map_location=device)
        model = mod.BBoxBEVRegressor(use_ipm_prior=True).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[model] loaded (epoch={ckpt['epoch']}  val_err={ckpt['val_err_m']:.3f}m)")
        return model, ckpt
    print("[warn] could not import bbox_bev_regressor — learned panel will be empty")
    return None, {}


def run_model(model, px, py, bw, bh, img_w, img_h, K, R_cam, cam_h, device):
    ipm = ipm_project(px, py, K, R_cam, cam_h)
    if ipm is None:
        return None, None
    feat = torch.tensor([[
        1.0 - px/img_w,   # left of image → large → positive y_left
        py/img_h,
        bw/img_w,
        bh/img_h,
        math.log(max(bh/img_h, 1e-4)),
        ipm[0],
        ipm[1],
    ]], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(feat).cpu().numpy()[0]
    return float(pred[0]), float(pred[1])


# ---------------------------------------------------------------------------
# BEV axes helper
# ---------------------------------------------------------------------------

def setup_bev_axes(ax, title):
    # Lateral axis: positive = LEFT (camera convention)
    # Plot: left side of axes = left of camera
    ax.set_xlim(BEV_LAT_MAX + 0.5, BEV_LAT_MIN - 0.5)   # flip x: left on left
    ax.set_ylim(BEV_FWD_MIN - 0.5, BEV_FWD_MAX + 0.5)
    ax.set_xlabel("Lateral (m)  ← left | right →", fontsize=9)
    ax.set_ylabel("Forward (m)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_facecolor("#F8F8F0")
    ax.grid(True, alpha=0.25, linestyle="--")
    rect = mpatches.FancyBboxPatch(
        (BEV_LAT_MIN, BEV_FWD_MIN),
        BEV_LAT_MAX-BEV_LAT_MIN, BEV_FWD_MAX-BEV_FWD_MIN,
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor="#AAAAAA", facecolor="none"
    )
    ax.add_patch(rect)
    ax.plot(0, BEV_FWD_MIN, marker="^", color="black", markersize=9, zorder=10)
    ax.text(0, BEV_FWD_MIN-0.5, "cam", fontsize=7, ha="center")


# ---------------------------------------------------------------------------
# Main figure renderer
# ---------------------------------------------------------------------------

def render_frame(
    frame_num, scene_id,
    street, drone, track_map,
    align, K, R_cam, cam_h,
    model, ckpt_meta,
    street_video, drone_video, img_dir,
    img_w, img_h, device, out_path, dpi=150,
    ego_radius_m=20.0, ego_track_id=220,
):
    # Load images
    street_img = load_frame_from_dir(img_dir, scene_id, frame_num)
    if street_img is None and street_video:
        street_img = extract_frame(street_video, frame_num)
    if street_img is None:
        street_img = np.zeros((256,512,3), dtype=np.uint8)
    # ---- Debug: image vs camera intrinsics ----
    try:
        h_img_dbg, w_img_dbg = street_img.shape[:2]
    except Exception:
        h_img_dbg, w_img_dbg = -1, -1
    print(f"[debug:cam] frame={frame_num}  img_shape=({h_img_dbg},{w_img_dbg})  img_w={img_w}  img_h={img_h}")
    print(f"[debug:cam] K(cx,cy)=({K[0,2]:.1f},{K[1,2]:.1f})  fx,fy=({K[0,0]:.1f},{K[1,1]:.1f})  cam_h={cam_h}")

    drone_img = extract_frame(drone_video, frame_num) if drone_video else None

    # Build matched pairs for this frame
    fc   = "street_frame" if "street_frame" in drone.columns else "drone_frame"
    x1c  = "bbox_x1" if "bbox_x1" in street.columns else "x1"
    y1c  = x1c.replace("x1","y1"); x2c = x1c.replace("x1","x2"); y2c = x1c.replace("x1","y2")
    ox   = float(align.get("offset_x",0))
    oy_  = float(align.get("offset_y",0))
    rot  = float(align.get("rot_deg",0))
    tm   = track_map[track_map["drone_track_id"]!=-1]
    s2d  = dict(zip(tm["street_track_id"].astype(int), tm["drone_track_id"].astype(int)))

    s_frame = street[street["frame"].astype(int)==frame_num] if "scene_id" not in street.columns \
              else street[(street["scene_id"]==scene_id)&(street["frame"].astype(int)==frame_num)]
    d_frame = drone[drone[fc].astype(int)==frame_num]

    objects = []  # list of dicts per detected+matched object
    matched_count = 0
    gt_in_range_count = 0
    ipm_valid_count = 0
    learned_valid_count = 0
    ipm_failed_rows = []
    for _, sr in s_frame.iterrows():
        s_tid = int(sr["track_id"])
        if s_tid not in s2d: continue
        d_tid = s2d[s_tid]
        dr_rows = drone[(drone["track_id"]==d_tid)&(drone[fc].astype(int)==frame_num)]
        if len(dr_rows)==0: continue
        dr = dr_rows.iloc[0]

        matched_count += 1

        cname = norm_class(sr.get("class_name","car"))
        color = CLASS_COLORS.get(cname, DEFAULT_COLOR)

        x1,y1 = float(sr[x1c]),float(sr[y1c])
        x2,y2 = float(sr[x2c]),float(sr[y2c])
        bw,bh = x2-x1, y2-y1
        px,py = (x1+x2)/2, y2  # slightly above bbox bottom for more stable ground contact

        # GT BEV
        # Prefer ego/camera-frame coordinates directly when already present in the drone manifest.
        # Only apply coord_align/world->camera conversion when starting from world coordinates.
        has_cam_bev = ("x_fwd" in dr.index) and ("y_left" in dr.index)
        if has_cam_bev:
            xf_gt = float(dr["x_fwd"])
            yl_gt = float(dr["y_left"])
        else:
            wx = float(dr.get("world_x", 0.0))
            wy = float(dr.get("world_y", 0.0))
            xf_gt, yl_gt = drone_to_cam(wx, wy, ox, oy_, rot)

        if BEV_FWD_MIN <= xf_gt <= BEV_FWD_MAX and BEV_LAT_MIN <= yl_gt <= BEV_LAT_MAX:
            gt_in_range_count += 1

        # IPM
        ipm = ipm_project(px, py, K, R_cam, cam_h)
        ipm_xf = ipm[0] if ipm else None
        ipm_yl = ipm[1] if ipm else None

        # ---- Debug: compute ray z for this object ----
        ray_z = ipm_ray_z(px, py, K, R_cam)

        if ipm is not None:
            ipm_valid_count += 1
        else:
            ipm_failed_rows.append({
                "street_tid": s_tid,
                "drone_tid": d_tid,
                "cname": cname,
                "px": px,
                "py": py,
                "bw": bw,
                "bh": bh,
                "py_norm": py / img_h if img_h > 0 else float("nan"),
                "gt_xf": xf_gt,
                "gt_yl": yl_gt,
                "ray_z": ray_z,
            })

        # Learned
        lrn_xf, lrn_yl = None, None
        if model is not None:
            lrn_xf, lrn_yl = run_model(model, px, py, bw, bh, img_w, img_h, K, R_cam, cam_h, device)

        if lrn_xf is not None and lrn_yl is not None:
            learned_valid_count += 1

        objects.append({
            "cname": cname, "color": color,
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,"px":px,"py":py,
            "gt_xf":xf_gt,"gt_yl":yl_gt,
            "ipm_xf":ipm_xf,"ipm_yl":ipm_yl,
            "lrn_xf":lrn_xf,"lrn_yl":lrn_yl,
            "dist": float(dr["dist_m"]) if "dist_m" in dr.index else None,
        })

    print(
        f"[debug:vis] frame={frame_num}  matched={matched_count}  "
        f"gt_in_range={gt_in_range_count}  ipm_valid={ipm_valid_count}  learned_valid={learned_valid_count}"
    )
    if ipm_failed_rows:
        print(f"[debug:vis] frame={frame_num}  ipm_failed={len(ipm_failed_rows)}")
        for row in ipm_failed_rows[:8]:
            print(
                "[debug:ipm-fail] "
                f"frame={frame_num}  street_tid={row['street_tid']}  drone_tid={row['drone_tid']}  "
                f"cls={row['cname']}  px={row['px']:.1f}  py={row['py']:.1f}  "
                f"bw={row['bw']:.1f}  bh={row['bh']:.1f}  py_norm={row['py_norm']:.3f}  "
                f"ray_z={row['ray_z']:.4f}  "
                f"gt=({row['gt_xf']:.2f},{row['gt_yl']:.2f})"
            )

    # ---- Figure layout ----
    has_drone = drone_img is not None
    ncols = 4 if has_drone else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5.5*ncols, 5.5))
    fig.suptitle(
        f"CrossView BEV (BBox Regressor)  ·  {scene_id} / frame {frame_num:06d}  "
        f"·  {len(objects)} matched objects",
        fontsize=11, fontweight="bold", y=1.01
    )

    # Panel 0 — Street frame
    ax0 = axes[0]
    ax0.imshow(street_img)
    ax0.set_title("Street Frame", fontsize=10, fontweight="bold")
    ax0.axis("off")
    for obj in objects:
        color = obj["color"]
        rect = mpatches.Rectangle(
            (obj["x1"],obj["y1"]), obj["x2"]-obj["x1"], obj["y2"]-obj["y1"],
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax0.add_patch(rect)
        ax0.text(obj["x1"], obj["y1"]-4, obj["cname"],
                 fontsize=6.5, color="white",
                 bbox=dict(boxstyle="round,pad=0.1", fc=color, alpha=0.85))
        if obj["ipm_xf"] is None or obj["ipm_yl"] is None:
            ax0.plot(obj["px"], obj["py"], "x", color="red", markersize=7, markeredgewidth=2)
            ax0.text(obj["px"] + 4, obj["py"] + 4, "IPM fail",
                     fontsize=6, color="red",
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8))

    debug_text = (
        f"matched: {matched_count}\n"
        f"GT in range: {gt_in_range_count}\n"
        f"IPM valid: {ipm_valid_count}\n"
        f"IPM failed: {len(ipm_failed_rows)}\n"
        f"Learned valid: {learned_valid_count}"
    )
    ax0.text(
        0.02, 0.98, debug_text,
        transform=ax0.transAxes,
        fontsize=8,
        va="top", ha="left",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65)
    )

    # Panel 1 — BEV comparison
    ax1 = axes[1]
    setup_bev_axes(ax1, "BEV Comparison")

    # Draw GT→IPM and GT→Learned lines for each object
    for obj in objects:
        color = obj["color"]
        xf_gt, yl_gt = obj["gt_xf"], obj["gt_yl"]
        in_range = BEV_FWD_MIN <= xf_gt <= BEV_FWD_MAX and BEV_LAT_MIN <= yl_gt <= BEV_LAT_MAX

        # IPM prediction (clip to axis range for display)
        if obj["ipm_xf"] is not None:
            ixf = max(BEV_FWD_MIN-3, min(BEV_FWD_MAX+3, obj["ipm_xf"]))
            iyl = max(BEV_LAT_MIN-3, min(BEV_LAT_MAX+3, obj["ipm_yl"]))
            ax1.plot(iyl, ixf, "o", color=color, markersize=8,
                     alpha=0.5, markeredgecolor="white", markeredgewidth=1, zorder=4)
            if in_range:
                ax1.plot([yl_gt, iyl], [xf_gt, ixf], "--",
                         color=color, alpha=0.3, linewidth=0.8, zorder=3)

        # Learned prediction
        if obj["lrn_xf"] is not None:
            lxf = max(BEV_FWD_MIN-3, min(BEV_FWD_MAX+3, obj["lrn_xf"]))
            lyl = max(BEV_LAT_MIN-3, min(BEV_LAT_MAX+3, obj["lrn_yl"]))
            ax1.plot(lyl, lxf, "*", color=color, markersize=12,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)
            if in_range:
                ax1.plot([yl_gt, lyl], [xf_gt, lxf], "-",
                         color=color, alpha=0.5, linewidth=1.0, zorder=5)

        # GT
        if in_range:
            ax1.plot(yl_gt, xf_gt, "D", color=color, markersize=7,
                     markeredgecolor="black", markeredgewidth=1.2, zorder=7)
            dist_str = f"{obj['dist']:.1f}m" if obj["dist"] else ""
            if dist_str:
                ax1.text(yl_gt+0.15, xf_gt+0.15, dist_str,
                         fontsize=6.5, color="black",
                         bbox=dict(boxstyle="round,pad=0.1",fc="white",alpha=0.7))

    # Legend for BEV panel
    legend_elems = [
        plt.Line2D([0],[0], marker="D", color="gray", linestyle="none",
                   markersize=7, label="GT (drone)"),
        plt.Line2D([0],[0], marker="o", color="gray", linestyle="none",
                   markersize=7, alpha=0.6, label="IPM baseline"),
        plt.Line2D([0],[0], marker="*", color="gray", linestyle="none",
                   markersize=10, label="Learned regressor"),
    ]
    ax1.legend(handles=legend_elems, fontsize=7.5, loc="upper right")

    # Panel 2 — Error comparison bar chart per object
    ax2 = axes[2]
    if objects:
        ipm_errs, lrn_errs, labels, colors = [], [], [], []
        for i, obj in enumerate(objects):
            xf_gt, yl_gt = obj["gt_xf"], obj["gt_yl"]
            if not (BEV_FWD_MIN<=xf_gt<=BEV_FWD_MAX and BEV_LAT_MIN<=yl_gt<=BEV_LAT_MAX):
                continue
            ie = math.sqrt((obj["ipm_xf"]-xf_gt)**2+(obj["ipm_yl"]-yl_gt)**2) \
                 if obj["ipm_xf"] is not None else None
            le = math.sqrt((obj["lrn_xf"]-xf_gt)**2+(obj["lrn_yl"]-yl_gt)**2) \
                 if (obj["lrn_xf"] is not None and obj["lrn_yl"] is not None) else None
            if ie is None or le is None: continue
            ipm_errs.append(min(ie, 30.0))   # cap at 30m for display
            lrn_errs.append(min(le, 30.0))
            labels.append(f"{obj['cname'][:3]}\n{obj['dist']:.0f}m" if obj["dist"] else obj["cname"][:3])
            colors.append(obj["color"])

        if ipm_errs:
            xs = np.arange(len(ipm_errs))
            w = 0.35
            bars_ipm = ax2.bar(xs-w/2, ipm_errs, w, color="lightcoral",
                               edgecolor="white", label="IPM", alpha=0.85)
            bars_lrn = ax2.bar(xs+w/2, lrn_errs, w, color="steelblue",
                               edgecolor="white", label="Learned", alpha=0.85)
            # Color top of bars by class
            for bar, c in zip(bars_ipm, colors):
                bar.set_edgecolor(c)
                bar.set_linewidth(2)
            for bar, c in zip(bars_lrn, colors):
                bar.set_edgecolor(c)
                bar.set_linewidth(2)

            ax2.set_xticks(xs)
            ax2.set_xticklabels(labels, fontsize=7)
            ax2.set_ylabel("Error (m)", fontsize=9)
            ax2.set_title("Per-Object Error: IPM vs Learned", fontsize=10, fontweight="bold")
            ax2.axhline(2.0, color="green", linestyle="--", alpha=0.5, linewidth=1,
                        label="2m threshold")
            ax2.legend(fontsize=8)
            ax2.set_facecolor("#F8F8F0")

            # Improvement labels
            for i, (ie, le) in enumerate(zip(ipm_errs, lrn_errs)):
                pct = (ie-le)/ie*100 if ie>0 else 0
                color = "green" if pct > 0 else "red"
                ax2.text(i, max(ie,le)+0.3, f"{pct:+.0f}%",
                         ha="center", fontsize=6.5, color=color, fontweight="bold")

    # Panel 3 — Drone overhead
    if has_drone:
        ax3 = axes[3]
        ax3.imshow(drone_img)
        ax3.set_title("Drone Overhead (GT)", fontsize=10, fontweight="bold")
        ax3.axis("off")
        h_img, w_img = drone_img.shape[:2]

        A = solve_drone_affine(d_frame)

        def metric_to_px(xf, yl):
            if A is not None:
                return apply_affine(A, xf, yl)
            return int(w_img/2), int(h_img/2)

        # Draw GT boxes
        for _, dr in d_frame.iterrows():
            cname = norm_class(dr.get("class_name","car"))
            color = CLASS_COLORS.get(cname, DEFAULT_COLOR)
            if "bbox_x1" in dr.index:
                x1d,y1d=float(dr["bbox_x1"]),float(dr["bbox_y1"])
                x2d,y2d=float(dr["bbox_x2"]),float(dr["bbox_y2"])
                rect = mpatches.Rectangle((x1d,y1d),x2d-x1d,y2d-y1d,
                                           linewidth=2,edgecolor=color,facecolor="none")
                ax3.add_patch(rect)
                if "dist_m" in dr.index:
                    ax3.text(x1d,y1d-4,f"{float(dr['dist_m']):.1f}m",
                             fontsize=6.5,color="white",
                             bbox=dict(boxstyle="round,pad=0.1",fc=color,alpha=0.85))

        # Camera position and BEV grid
        # Find ego (bike) position: use track ego_track_id bbox center directly
        ego_col, ego_row = None, None
        ego_rows = d_frame[d_frame["track_id"].astype(int) == ego_track_id]
        if len(ego_rows) > 0 and "bbox_x1" in ego_rows.columns:
            er = ego_rows.iloc[0]
            x1 = float(er["bbox_x1"]); y1 = float(er["bbox_y1"])
            x2 = float(er["bbox_x2"]); y2 = float(er["bbox_y2"])
            # Handle both (x1,y1,x2,y2) and (x1,y1,w,h) formats
            # If x2 < x1 or x2 < 200, it's likely width not x2
            if x2 < x1 or x2 < 50:
                ego_col = int(x1 + x2/2)
                ego_row = int(y1 + y2/2)
            else:
                ego_col = int((x1 + x2) / 2)
                ego_row = int((y1 + y2) / 2)
            print(f"[ego] track {ego_track_id} bbox center: ({ego_col},{ego_row})")
        else:
            # Fallback: ego is the origin in ego/camera BEV coordinates.
            ego_col, ego_row = metric_to_px(0.0, 0.0)
            print(f"[ego] track {ego_track_id} not in frame {frame_num} — using BEV origin")

        if ego_col is not None:
            ax3.plot(ego_col, ego_row, "^", color="white", markersize=11,
                     markeredgecolor="black", markeredgewidth=2.0, zorder=20)
            ax3.text(ego_col+14, ego_row+14, "ego", fontsize=8, color="white",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.75), zorder=21)

            # Radius circle — convert meters to pixels using affine scale
            if A is not None:
                # Pixel scale: magnitude of affine linear part column (m→px)
                scale_px_per_m = math.sqrt(A[0,0]**2 + A[1,0]**2)
                radius_px = ego_radius_m * scale_px_per_m
            else:
                radius_px = ego_radius_m / 0.028   # fallback: nominal 60m altitude scale
            circle = plt.Circle(
                (ego_col, ego_row), radius_px,
                linewidth=2.0, edgecolor="yellow", facecolor="yellow",
                alpha=0.08, linestyle="--", zorder=15
            )
            circle_edge = plt.Circle(
                (ego_col, ego_row), radius_px,
                linewidth=2.0, edgecolor="yellow", facecolor="none",
                linestyle="--", zorder=16
            )
            ax3.add_patch(circle)
            ax3.add_patch(circle_edge)
            ax3.text(ego_col + radius_px * 0.72, ego_row - radius_px * 0.72,
                     f"r={ego_radius_m:.0f}m", fontsize=6.5, color="yellow", zorder=17,
                     bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

        ax3.text(0.02,0.02,f"frame {frame_num:06d}",
                 transform=ax3.transAxes, fontsize=7, color="white", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2",fc="black",alpha=0.6))

    # Class colour legend
    shown = set(o["cname"] for o in objects)
    handles = [mpatches.Patch(color=CLASS_COLORS.get(c,DEFAULT_COLOR), label=c.capitalize())
               for c in CLASSES if c in shown]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5,-0.04), frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def get_richest_frames(street, drone, track_map, scene_id, n):
    valid = set(track_map[track_map["drone_track_id"]!=-1]["street_track_id"].astype(int))
    s = street if "scene_id" not in street.columns else street[street["scene_id"]==scene_id]
    fc = "street_frame" if "street_frame" in drone.columns else "drone_frame"
    s_matched = s[s["track_id"].isin(valid)]
    counts = s_matched.groupby("frame")["track_id"].nunique()
    return sorted(counts.nlargest(n).index.tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street_manifest",  required=True)
    ap.add_argument("--drone_manifest",   required=True)
    ap.add_argument("--track_map_csv",    required=True)
    ap.add_argument("--coord_align_csv",  required=True)
    ap.add_argument("--camera_cfg",       required=True)
    ap.add_argument("--checkpoint",       required=True)
    ap.add_argument("--img_dir",          default="frames")
    ap.add_argument("--street_video",     default=None)
    ap.add_argument("--drone_video",      default=None)
    ap.add_argument("--scene_id",         default="scene_00")
    ap.add_argument("--frame",            type=int, default=None)
    ap.add_argument("--n_frames",         type=int, default=20)
    ap.add_argument("--out",              default=None)
    ap.add_argument("--out_dir",          default="vis_bbox_regressor")
    ap.add_argument("--ego_radius_m",     type=float, default=20.0,
                    help="Radius of circle drawn around ego in drone view (meters). Default 20m.")
    ap.add_argument("--ego_track_id",     type=int,   default=220,
                    help="Drone track ID of the ego bike. Default 220.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    street    = pd.read_csv(args.street_manifest)
    drone     = pd.read_csv(args.drone_manifest)
    track_map = pd.read_csv(args.track_map_csv)
    street["class_name"] = street["class_name"].apply(norm_class)
    if "class_name" in drone.columns: drone["class_name"] = drone["class_name"].apply(norm_class)
    if "scene_id" not in street.columns: street["scene_id"] = "scene_00"

    with open(args.camera_cfg) as f: cam_cfg = json.load(f)
    cfg = cam_cfg.get(args.scene_id, cam_cfg.get("default", {}))

    # Use the actual street-manifest pixel extent rather than calibration-space nominal size.
    x2c_img = "bbox_x2" if "bbox_x2" in street.columns else "x2"
    y2c_img = "bbox_y2" if "bbox_y2" in street.columns else "y2"
    img_w = float(int(math.ceil(float(street[x2c_img].max()))))
    img_h = float(int(math.ceil(float(street[y2c_img].max()))))
    K, R_cam, cam_h = build_K_R(cfg, actual_w=img_w, actual_h=img_h)
    print(f"[vis] using image size from street manifest: img_w={img_w}  img_h={img_h}")

    align_df = pd.read_csv(args.coord_align_csv)
    ar = align_df[align_df["scene_id"]==args.scene_id] if "scene_id" in align_df.columns \
         else align_df
    ar = ar.iloc[0]
    align = {"offset_x": float(ar.get("offset_x",0)),
             "offset_y": float(ar.get("offset_y",0)),
             "rot_deg":  float(ar.get("rot_deg",0))}

    model, ckpt_meta = load_model(args.checkpoint, device)

    frames = [args.frame] if args.frame else \
             get_richest_frames(street, drone, track_map, args.scene_id, args.n_frames)
    print(f"[info] rendering {len(frames)} frame(s): {frames[:5]}{'...' if len(frames)>5 else ''}")

    for frame_num in frames:
        if args.frame and args.out:
            out_path = args.out
        else:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"bbox_{args.scene_id}_f{frame_num:06d}.png")

        render_frame(
            frame_num=frame_num, scene_id=args.scene_id,
            street=street, drone=drone, track_map=track_map,
            align=align, K=K, R_cam=R_cam, cam_h=cam_h,
            model=model, ckpt_meta=ckpt_meta,
            street_video=args.street_video, drone_video=args.drone_video,
            img_dir=args.img_dir,
            img_w=img_w, img_h=img_h, device=device,
            out_path=out_path, dpi=150,
            ego_radius_m=args.ego_radius_m,
            ego_track_id=args.ego_track_id,
        )

    print(f"\n[done] {len(frames)} figure(s) saved to {args.out_dir or args.out}")


if __name__ == "__main__":
    main()