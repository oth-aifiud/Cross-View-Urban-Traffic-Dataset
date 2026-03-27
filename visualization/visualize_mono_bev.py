"""
render_bev_figure.py — 4-panel CrossView figure for paper
==========================================================
Panels:
  1. Street frame with detection boxes
  2. IPM BEV projection (baseline)
  3. Learned BEV (MonoLayout)
  4. Drone overhead frame with GT boxes + metric labels

Usage
-----
  python visulaize_mono_bev.py \
      --street_manifest  .../street_wedge_manifest.csv \
      --drone_manifest   .../drone_wedge_manifest.csv \
      --track_map_csv    .../track_mapping.csv \
      --coord_align_csv  .../coord_align.csv \
      --camera_cfg       .../camera_params.json \
      --checkpoint       .../best_remapped.pth \
      --street_video     .../Galgenberg_60m_bike_head.mp4 \
      --drone_video      .../Galgenberg_60m_drone.mp4 \
      --img_dir          .../frames \
      --frame            831 \
      --out              figure3_frame831.png

  # Render 20 richest frames as a batch:
  python render_bev_figure.py ... --n_frames 20 --out_dir paper_figures/
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
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "car":        "#2196F3",
    "truck":      "#FF9800",
    "bus":        "#9C27B0",
    "person":     "#4CAF50",
    "bicycle":    "#00BCD4",
    "motorcycle": "#F44336",
}
DEFAULT_COLOR = "#607D8B"

CLASSES = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

BEV_H, BEV_W = 64, 64
BEV_RANGE_M   = 25.0
BEV_Y_OFFSET  = -5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_class(x):
    x = str(x).strip().lower()
    if x in {"pedestrian", "person", "people"}: return "person"
    if x in {"bike", "bicycle", "cyclist"}:     return "bicycle"
    if x in {"motorbike", "motorcycle"}:        return "motorcycle"
    return x


def extract_frame(video_path: str, frame_num: int) -> Optional[np.ndarray]:
    """Extract a single frame (1-based) from a video. Returns RGB uint8 or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_frame_from_dir(img_dir: str, scene_id: str, frame_num: int) -> Optional[np.ndarray]:
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


def build_K_R(cfg: Dict):
    fx, fy = cfg["fx"], cfg["fy"]
    cx, cy = cfg["cx"], cfg["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    pitch = float(cfg.get("pitch_deg", -8.0))
    a = math.radians(-pitch)
    R_base = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.float64)
    Ry = np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
    R = Ry @ R_base
    cam_h = float(cfg.get("cam_height_m", 1.1))
    return K, R, cam_h


def ipm_project(px, py, K, R, cam_h):
    K_inv = np.linalg.inv(K)
    ray = R @ (K_inv @ np.array([px, py, 1.0]))
    if ray[2] >= -1e-6:
        return None
    t = -cam_h / ray[2]
    return float(t * ray[0]), float(t * ray[1])


def world_to_bev_pixel(wx, wy, ox, oy, rot_deg):
    dx, dy = wx - ox, wy - oy
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    x =  c*dx + s*dy
    y = -s*dx + c*dy
    if x < 0 or x > BEV_RANGE_M: return None
    y_rel = y - BEV_Y_OFFSET
    if y_rel < 0 or y_rel > BEV_RANGE_M: return None
    return x, y


# ---------------------------------------------------------------------------
# Model (must match bev_monolayout_v7 architecture)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load MonoLayoutBEV, handling key name mismatches between checkpoint and model."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import importlib.util
        for candidate in ["bev_monolayout_v7", "bev_monolayout"]:
            spec = importlib.util.find_spec(candidate)
            if spec is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            model = mod.MonoLayoutBEV().to(device)
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt["model_state"]

            # Auto-detect and fix key name mismatch.
            # visualize_bev uses enc/dec short names; bev_monolayout_v7 uses encoder/decoder.
            # Checkpoint may have either. Try loading as-is first, then try remapping.
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError:
                # Try remapping enc.* → encoder.*, dec.* → decoder.*
                remap_fwd = {
                    "enc.l0": "encoder.layer0",
                    "enc.l1": "encoder.layer1",
                    "enc.l2": "encoder.layer2",
                    "enc.l3": "encoder.layer3",
                    "enc.l4": "encoder.layer4",
                    "enc.c":  "encoder.compress",
                    "dec.p":  "decoder.project",
                    "dec.u1": "decoder.up1",
                    "dec.u2": "decoder.up2",
                    "dec.u3": "decoder.up3",
                    "dec.out": "decoder.out_conv",
                }
                # Also try reverse: encoder.* → enc.*
                remap_rev = {v: k for k, v in remap_fwd.items()}
                for remap in [remap_fwd, remap_rev]:
                    new_state = {}
                    for k, v in state.items():
                        new_k = k
                        for old_p, new_p in remap.items():
                            if k.startswith(old_p):
                                new_k = new_p + k[len(old_p):]
                                break
                        new_state[new_k] = v
                    try:
                        model.load_state_dict(new_state, strict=False)
                        state = new_state
                        print(f"[model] applied key remap ({list(remap.keys())[0]}→...)")
                        break
                    except Exception:
                        continue

            model.eval()
            print(f"[model] loaded from {candidate} (epoch={ckpt.get('epoch','?')})")
            return model
    except Exception as e:
        print(f"[warn] could not load model: {e}")
    return None


def run_model(model, img_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run model on a single RGB image. Returns (NUM_CLASSES, BEV_H, BEV_W) numpy."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_resized = cv2.resize(img_rgb, (512, 256))
    tensor = transform(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return pred  # (C, H, W)


# ---------------------------------------------------------------------------
# BEV heatmap rendering
# ---------------------------------------------------------------------------

def bev_to_axes_coords(x_fwd: float, y_lat: float):
    """Convert metric BEV position to matplotlib axes coords."""
    return y_lat, x_fwd   # x-axis=lateral, y-axis=forward


def draw_bev_heatmap(ax, heatmap: np.ndarray, title: str):
    """Draw a (BEV_H, BEV_W) heatmap with correct metric axes."""
    # heatmap: (H, W) — row 0 = farthest forward
    extent = [BEV_Y_OFFSET, BEV_Y_OFFSET + BEV_RANGE_M, 0, BEV_RANGE_M]
    ax.imshow(heatmap, origin="upper", extent=extent,
              cmap="YlOrRd", vmin=0, vmax=1, aspect="auto", alpha=0.85)
    ax.set_xlim(BEV_Y_OFFSET, BEV_Y_OFFSET + BEV_RANGE_M)
    ax.set_ylim(0, BEV_RANGE_M)
    ax.set_xlabel("Lateral (m)", fontsize=9)
    ax.set_ylabel("Forward (m)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_facecolor("#FAFAF0")
    # Camera marker
    ax.plot(0, 0, marker="^", color="black", markersize=8, zorder=10)
    ax.text(0, 0.5, "cam", fontsize=7, ha="center", color="black")


def plot_bev_object(ax, x_fwd, y_lat, label: str, color: str,
                    dist_label: Optional[str] = None):
    ax.plot(y_lat, x_fwd, "o", color=color, markersize=7,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    if dist_label:
        ax.text(y_lat + 0.4, x_fwd + 0.4, dist_label,
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))


# ---------------------------------------------------------------------------
# Ground scale estimation from drone altitude + FOV
# ---------------------------------------------------------------------------

def estimate_ground_scale(
    img_w: int, img_h: int,
    altitude_m: float = 60.0,
    hfov_deg: float = 84.0,   # DJI standard wide lens ~84°
) -> float:
    """Returns meters per pixel at nadir."""
    ground_w = 2 * altitude_m * math.tan(math.radians(hfov_deg / 2))
    return ground_w / img_w   # m/px


def drone_nadir_to_pixel(
    x_fwd_m: float, y_left_m: float,
    img_w: int, img_h: int,
    mpp: float,
    drone_yaw_deg: float = 0.0,
) -> Tuple[int, int]:
    """
    Convert drone-frame (x_fwd, y_left) offset from nadir → drone image pixel.

    Drone image convention (assuming north-up / nadir-centered):
      image center = drone nadir
      +x_fwd = forward in drone frame → depends on drone yaw
      +y_left = left in drone frame

    With drone_yaw_deg=0 (drone facing "up" in image):
      x_fwd → -row direction (forward = up in image)
      y_left → -col direction (left = left in image)
    """
    cx, cy = img_w / 2, img_h / 2
    # Rotate by drone yaw to get image-frame offsets
    yaw = math.radians(drone_yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    # x_fwd in drone frame → image frame
    dx_img =  x_fwd_m * (-sin_y) + y_left_m * (-cos_y)
    dy_img =  x_fwd_m * (-cos_y) + y_left_m * ( sin_y)
    col = int(cx + dx_img / mpp)
    row = int(cy + dy_img / mpp)
    return col, row   # (x, y) for matplotlib


def solve_drone_affine(
    d_frame: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Solve the exact affine transform: [col, row] = A @ [x_fwd, y_left, 1]
    using all GT objects with known metric positions and drone bbox centers.

    Returns 2×3 matrix A, or None if too few points.
    """
    if "x_fwd" not in d_frame.columns or "bbox_x1" not in d_frame.columns:
        return None

    rows_m, rows_px = [], []
    for _, dr in d_frame.iterrows():
        if pd.isna(dr.get("bbox_x1", np.nan)): continue
        wx = float(dr["x_fwd"])
        wy = float(dr["y_left"])
        gt_col = (float(dr["bbox_x1"]) + float(dr["bbox_x2"])) / 2
        gt_row = (float(dr["bbox_y1"]) + float(dr["bbox_y2"])) / 2
        rows_m.append([wx, wy, 1.0])
        rows_px.append([gt_col, gt_row])

    if len(rows_m) < 3:
        return None

    M  = np.array(rows_m,  dtype=np.float64)   # (N, 3)
    Px = np.array(rows_px, dtype=np.float64)   # (N, 2)

    # Least squares: M @ A.T = Px  →  A.T = pinv(M) @ Px
    A_T, _, _, _ = np.linalg.lstsq(M, Px, rcond=None)
    A = A_T.T   # (2, 3): [col] = A[0] @ [x,y,1],  [row] = A[1] @ [x,y,1]

    # Compute residuals
    pred = M @ A_T
    errs = np.linalg.norm(pred - Px, axis=1)
    print(f"[drone] affine solve: {len(rows_m)} pts  "
          f"median_err={np.median(errs):.1f}px  max_err={np.max(errs):.1f}px")
    return A


def apply_drone_affine(A: np.ndarray, x_fwd: float, y_left: float) -> Tuple[int, int]:
    """Apply solved affine transform to get drone image pixel coords."""
    v = np.array([x_fwd, y_left, 1.0])
    col = float(A[0] @ v)
    row = float(A[1] @ v)
    return int(round(col)), int(round(row))


# ---------------------------------------------------------------------------
# Drone overhead panel
# ---------------------------------------------------------------------------

def draw_drone_panel(ax, drone_frame_rgb: np.ndarray,
                     drone_rows: pd.DataFrame, frame_num: int,
                     cam_offset_x: float = 0.0, cam_offset_y: float = 0.0,
                     ox: float = 0.0, oy: float = 0.0, rot_deg: float = 0.0,
                     altitude_m: float = 60.0):
    """
    Show drone overhead frame with:
     - GT bounding boxes + metric labels
     - Camera position marker (bike position in drone image)
     - BEV grid footprint outline
     - Forward direction arrow from camera
    """
    h, w = drone_frame_rgb.shape[:2]
    mpp = estimate_ground_scale(w, h, altitude_m)
    print(f"[drone] image={w}x{h}  altitude={altitude_m}m  nominal_scale={mpp:.4f} m/px")

    # Solve exact affine transform from GT pairs — much more accurate than yaw grid search
    A = solve_drone_affine(drone_rows)

    def metric_to_pixel(x_fwd, y_left):
        if A is not None:
            return apply_drone_affine(A, x_fwd, y_left)
        # Fallback: image center + simple scale (no rotation)
        cx, cy = w/2, h/2
        return int(cx + y_left/mpp), int(cy - x_fwd/mpp)

    ax.imshow(drone_frame_rgb)
    ax.set_title("Drone Overhead  (GT)", fontsize=10, fontweight="bold")
    ax.axis("off")

    # Draw GT bounding boxes with distance labels
    for _, dr in drone_rows.iterrows():
        cname = norm_class(dr.get("class_name", "car"))
        color = CLASS_COLORS.get(cname, DEFAULT_COLOR)
        if "bbox_x1" in dr.index:
            x1,y1 = float(dr["bbox_x1"]), float(dr["bbox_y1"])
            x2,y2 = float(dr["bbox_x2"]), float(dr["bbox_y2"])
            rect = mpatches.Rectangle((x1,y1),x2-x1,y2-y1,
                                       linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            if "dist_m" in dr.index:
                dist = float(dr["dist_m"])
                ax.text(x1, y1-4, f"{dist:.1f}m", fontsize=6.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.1", fc=color, alpha=0.85))

    # Camera position in drone metric frame = offset from coord_align
    # (camera is at ipm=(0,0), so drone_pos = R(rot)*[0,0] + offset = offset)
    cam_col, cam_row = metric_to_pixel(cam_offset_x, cam_offset_y)
    if 0 <= cam_col < w and 0 <= cam_row < h:
        ax.plot(cam_col, cam_row, marker="^", color="white", markersize=10,
                markeredgecolor="black", markeredgewidth=1.5, zorder=20)
        ax.text(cam_col+12, cam_row+12, "cam", fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.7), zorder=21)

        # Forward direction: camera forward (ipm x=1,y=0) maps to drone frame as R(rot)*[1,0]
        if A is not None:
            r = math.radians(rot_deg)
            # camera forward unit vector in drone metric frame
            fwd_xd =  math.cos(r)   # R(rot)[0,0]
            fwd_yd =  math.sin(r)   # R(rot)[1,0]
            scale = 4.0  # draw 4m forward arrow
            # apply linear part of affine (no translation offset)
            fwd_col = cam_col + (A[0,0]*fwd_xd + A[0,1]*fwd_yd) * scale
            fwd_row = cam_row + (A[1,0]*fwd_xd + A[1,1]*fwd_yd) * scale
            ax.annotate("", xy=(fwd_col, fwd_row), xytext=(cam_col, cam_row),
                        arrowprops=dict(arrowstyle="->", color="cyan", lw=2.5), zorder=22)
            ax.text(fwd_col+8, fwd_row-8, "fwd", fontsize=7, color="cyan", zorder=23)

        # BEV grid footprint: project corners from camera BEV frame → drone metric → image pixel
        # drone_pos = R(rot_deg) * cam_pos + offset
        r = math.radians(rot_deg)
        cos_r, sin_r = math.cos(r), math.sin(r)
        corners_drone_px = []
        for (xf, yl) in [
            (0,           BEV_Y_OFFSET),
            (BEV_RANGE_M, BEV_Y_OFFSET),
            (BEV_RANGE_M, BEV_Y_OFFSET + BEV_RANGE_M),
            (0,           BEV_Y_OFFSET + BEV_RANGE_M),
        ]:
            # R(rot) * [xf, yl] + offset
            xd = cos_r * xf - sin_r * yl + cam_offset_x
            yd = sin_r * xf + cos_r * yl + cam_offset_y
            col, row = metric_to_pixel(xd, yd)
            corners_drone_px.append((col, row))

        from matplotlib.patches import Polygon as MplPolygon
        poly_fill = MplPolygon(corners_drone_px, closed=True,
                               facecolor="yellow", alpha=0.08, zorder=15)
        poly_edge = MplPolygon(corners_drone_px, closed=True, linewidth=1.5,
                               edgecolor="yellow", facecolor="none",
                               linestyle="--", zorder=16)
        ax.add_patch(poly_fill)
        ax.add_patch(poly_edge)
        # Label at far-forward corner
        far_col, far_row = corners_drone_px[1]
        ax.text(far_col+6, far_row-6, "BEV grid", fontsize=6.5,
                color="yellow", zorder=17,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    ax.text(0.02, 0.02, f"frame {frame_num:06d}",
            transform=ax.transAxes, fontsize=7, color="white", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))


# ---------------------------------------------------------------------------
# Main figure renderer
# ---------------------------------------------------------------------------

def render_figure(
    frame_num: int,
    scene_id: str,
    street: pd.DataFrame,
    drone: pd.DataFrame,
    track_map: pd.DataFrame,
    align: Dict,
    K, R_cam, cam_h,
    model,
    street_video: Optional[str],
    drone_video: Optional[str],
    img_dir: str,
    device: str,
    out_path: str,
    dpi: int = 150,
):
    # --- Load street frame ---
    street_img = load_frame_from_dir(img_dir, scene_id, frame_num)
    if street_img is None and street_video:
        street_img = extract_frame(street_video, frame_num)
    if street_img is None:
        street_img = np.zeros((256, 512, 3), dtype=np.uint8)

    # --- Load drone frame ---
    drone_frame_col = "street_frame" if "street_frame" in drone.columns else "drone_frame"
    drone_frame_num = frame_num  # 1:1 sync confirmed
    drone_img = None
    if drone_video:
        drone_img = extract_frame(drone_video, drone_frame_num)

    # --- Get matched track pairs ---
    valid = set(track_map[track_map["drone_track_id"] != -1]["street_track_id"].astype(int))
    s2d = dict(zip(
        track_map["street_track_id"].astype(int),
        track_map["drone_track_id"].astype(int)
    ))

    s_frame = street[
        (street.get("scene_id", pd.Series(["scene_00"]*len(street))) == scene_id) &
        (street["frame"].astype(int) == frame_num)
    ] if "scene_id" in street.columns else street[street["frame"].astype(int) == frame_num]

    d_frame = drone[drone[drone_frame_col].astype(int) == frame_num]

    # --- Build IPM projections ---
    ipm_objects = []
    for _, sr in s_frame.iterrows():
        s_tid = int(sr["track_id"])
        cname = norm_class(sr.get("class_name", "car"))
        x1c = "bbox_x1" if "bbox_x1" in sr.index else "x1"
        y2c = "bbox_y2" if "bbox_y2" in sr.index else "y2"
        x2c = "bbox_x2" if "bbox_x2" in sr.index else "x2"
        px = (float(sr[x1c]) + float(sr[x2c])) / 2
        py = float(sr[y2c])
        bev = ipm_project(px, py, K, R_cam, cam_h)
        if bev and 0 < bev[0] < BEV_RANGE_M:
            ipm_objects.append((bev[0], bev[1], cname, s_tid in valid))

    # --- Build drone GT BEV positions ---
    gt_objects = []
    ox = float(align.get("offset_x", 0))
    oy = float(align.get("offset_y", 0))
    rot = float(align.get("rot_deg", 0))
    for _, dr in d_frame.iterrows():
        cname = norm_class(dr.get("class_name", "car"))
        has_fwd = "x_fwd" in dr.index and "y_left" in dr.index
        wx = float(dr["x_fwd"]) if has_fwd else float(dr.get("world_x", 0))
        wy = float(dr["y_left"]) if has_fwd else float(dr.get("world_y", 0))
        pos = world_to_bev_pixel(wx, wy, ox, oy, rot)
        dist = float(dr["dist_m"]) if "dist_m" in dr.index else None
        if pos:
            gt_objects.append((pos[0], pos[1], cname, dist))

    # --- Run learned model ---
    pred_bev = None
    if model is not None:
        pred_bev = run_model(model, street_img, device)  # (C, H, W)

    # --- Figure layout ---
    has_drone_img = drone_img is not None
    n_panels = 5 if has_drone_img else 4
    fig_w = 5.5 * n_panels if has_drone_img else 22
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.2))

    n_gt  = len(gt_objects)
    n_ipm = len(ipm_objects)
    fig.suptitle(
        f"CrossView BEV  ·  {scene_id} / frame {frame_num:06d}  "
        f"·  {n_gt} GT objects  ·  {n_ipm} IPM projections",
        fontsize=11, fontweight="bold", y=1.01
    )

    # Panel 0 — Street frame
    ax_street = axes[0]
    ax_street.imshow(street_img)
    ax_street.set_title("Street Frame", fontsize=10, fontweight="bold")
    ax_street.axis("off")
    for _, sr in s_frame.iterrows():
        cname = norm_class(sr.get("class_name", "car"))
        color = CLASS_COLORS.get(cname, DEFAULT_COLOR)
        x1c = "bbox_x1" if "bbox_x1" in sr.index else "x1"
        y1c = "bbox_y1" if "bbox_y1" in sr.index else "y1"
        x2c = "bbox_x2" if "bbox_x2" in sr.index else "x2"
        y2c = "bbox_y2" if "bbox_y2" in sr.index else "y2"
        x1,y1,x2,y2 = float(sr[x1c]),float(sr[y1c]),float(sr[x2c]),float(sr[y2c])
        rect = mpatches.Rectangle((x1,y1),x2-x1,y2-y1,
                                   linewidth=1.5, edgecolor=color, facecolor="none")
        ax_street.add_patch(rect)

    # Panel 1 — IPM baseline
    ax_ipm = axes[1]
    ipm_heatmap = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    sigma_cells = 3.0 * (BEV_H / BEV_RANGE_M)  # ~7.7 cells
    for (xf, yl, cname, matched) in ipm_objects:
        row = int((BEV_RANGE_M - xf) / BEV_RANGE_M * BEV_H)
        col = int((yl - BEV_Y_OFFSET) / BEV_RANGE_M * BEV_W)
        for dr in range(-int(3*sigma_cells), int(3*sigma_cells)+1):
            for dc in range(-int(3*sigma_cells), int(3*sigma_cells)+1):
                r, c = row+dr, col+dc
                if 0 <= r < BEV_H and 0 <= c < BEV_W:
                    v = math.exp(-(dr**2+dc**2)/(2*sigma_cells**2))
                    ipm_heatmap[r,c] = max(ipm_heatmap[r,c], v)

    draw_bev_heatmap(ax_ipm, ipm_heatmap, "IPM Projection  (baseline)")
    for (xf, yl, cname, matched) in ipm_objects:
        color = CLASS_COLORS.get(cname, DEFAULT_COLOR)
        plot_bev_object(ax_ipm, xf, yl, cname, color)

    # Panel 2 — Learned BEV
    ax_ml = axes[2]
    if pred_bev is not None:
        ml_heatmap = pred_bev.max(axis=0)
    else:
        ml_heatmap = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    draw_bev_heatmap(ax_ml, ml_heatmap, "Learned BEV  (MonoLayout)")
    # Overlay GT dots on learned BEV for easy comparison
    for (xf, yl, cname, dist) in gt_objects:
        color = CLASS_COLORS.get(cname, DEFAULT_COLOR)
        ax_ml.plot(yl, xf, "o", color=color, markersize=5,
                   markeredgecolor="white", markeredgewidth=1, zorder=5, alpha=0.6)

    # Panel 3 — Drone BEV ground truth
    ax_gt = axes[3]
    gt_heatmap = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    sigma_gt = 3.0 * (BEV_H / BEV_RANGE_M)
    for (xf, yl, cname, dist) in gt_objects:
        row = int((BEV_RANGE_M - xf) / BEV_RANGE_M * BEV_H)
        col = int((yl - BEV_Y_OFFSET) / BEV_RANGE_M * BEV_W)
        for dr in range(-int(3*sigma_gt), int(3*sigma_gt)+1):
            for dc in range(-int(3*sigma_gt), int(3*sigma_gt)+1):
                r, c = row+dr, col+dc
                if 0 <= r < BEV_H and 0 <= c < BEV_W:
                    v = math.exp(-(dr**2+dc**2)/(2*sigma_gt**2))
                    gt_heatmap[r,c] = max(gt_heatmap[r,c], v)

    draw_bev_heatmap(ax_gt, gt_heatmap, "Drone Ground Truth")
    for (xf, yl, cname, dist) in gt_objects:
        color = CLASS_COLORS.get(cname, DEFAULT_COLOR)
        label = f"{dist:.1f}m" if dist is not None else ""
        plot_bev_object(ax_gt, xf, yl, cname, color, label)

    # Panel 4 — Drone overhead image (if available)
    if has_drone_img:
        ax_drone = axes[4]
        draw_drone_panel(
            ax_drone, drone_img, d_frame, frame_num,
            cam_offset_x=float(align.get("offset_x", 0)),
            cam_offset_y=float(align.get("offset_y", 0)),
            ox=float(align.get("offset_x", 0)),
            oy=float(align.get("offset_y", 0)),
            rot_deg=float(align.get("rot_deg", 0)),
        )

    # Legend
    legend_handles = []
    shown_classes = set(c for _,_,c,*_ in ipm_objects + gt_objects)
    for cname in CLASSES:
        if cname in shown_classes:
            legend_handles.append(
                mpatches.Patch(color=CLASS_COLORS[cname], label=cname.capitalize())
            )
    legend_handles.append(
        mpatches.Patch(color="black", label="Camera ▲")
    )
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Frame selection helpers
# ---------------------------------------------------------------------------

def get_richest_frames(street, drone, track_map, scene_id, n):
    """Pick frames with the most matched GT objects in BEV."""
    valid = set(track_map[track_map["drone_track_id"]!=-1]["street_track_id"].astype(int))
    s = street if "scene_id" not in street.columns else \
        street[street["scene_id"]==scene_id]
    counts = s[s["track_id"].isin(valid)].groupby("frame")["track_id"].nunique()
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
    ap.add_argument("--img_dir",          default="frames")
    ap.add_argument("--street_video",     default=None)
    ap.add_argument("--drone_video",      default=None)
    ap.add_argument("--checkpoint",       default=None)
    ap.add_argument("--scene_id",         default="scene_00")
    ap.add_argument("--frame",            type=int, default=None,
                    help="Single frame to render")
    ap.add_argument("--n_frames",         type=int, default=10,
                    help="Number of richest frames to render (ignored if --frame given)")
    ap.add_argument("--out",              default=None,
                    help="Output path for single frame")
    ap.add_argument("--out_dir",          default="bev_figures",
                    help="Output directory for batch rendering")
    ap.add_argument("--altitude_m", type=float, default=60.0,
                    help="Drone flight altitude in meters (used for ground scale). Default 60m.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    street    = pd.read_csv(args.street_manifest)
    drone     = pd.read_csv(args.drone_manifest)
    track_map = pd.read_csv(args.track_map_csv)
    street["class_name"] = street["class_name"].apply(norm_class)
    if "class_name" in drone.columns:
        drone["class_name"] = drone["class_name"].apply(norm_class)
    if "scene_id" not in street.columns:
        street["scene_id"] = "scene_00"

    # Camera
    with open(args.camera_cfg) as f:
        cam_cfg = json.load(f)
    cfg = cam_cfg.get(args.scene_id, cam_cfg.get("default", {}))
    K, R_cam, cam_h = build_K_R(cfg)

    # Alignment
    align_df = pd.read_csv(args.coord_align_csv)
    align_row = align_df[align_df["scene_id"]==args.scene_id]
    align = {}
    if len(align_row):
        r = align_row.iloc[0]
        align = {
            "offset_x": float(r.get("offset_x", 0)),
            "offset_y": float(r.get("offset_y", 0)),
            "rot_deg":  float(r.get("rot_deg",  0)),
        }

    # Model
    model = None
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
        if model is None:
            print("[warn] Could not load model — BEV panel will be blank")

    # Determine frames
    if args.frame:
        frames = [args.frame]
    else:
        frames = get_richest_frames(street, drone, track_map, args.scene_id, args.n_frames)
        print(f"[info] Rendering {len(frames)} richest frames: {frames}")

    for frame_num in frames:
        if args.frame and args.out:
            out_path = args.out
        else:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir,
                                    f"bev_{args.scene_id}_f{frame_num:06d}.png")
        render_figure(
            frame_num=frame_num,
            scene_id=args.scene_id,
            street=street,
            drone=drone,
            track_map=track_map,
            align=align,
            K=K, R_cam=R_cam, cam_h=cam_h,
            model=model,
            street_video=args.street_video,
            drone_video=args.drone_video,
            img_dir=args.img_dir,
            device=device,
            out_path=out_path,
            dpi=180,
        )

    print(f"\n[done] {len(frames)} figure(s) saved.")


if __name__ == "__main__":
    main()