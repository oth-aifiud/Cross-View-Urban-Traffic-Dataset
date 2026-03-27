"""
auto_coord_align.py — Automatic Coordinate Alignment (No Manual Setup)
========================================================================
Automatically estimates the 2D similarity transform between drone world
coordinates and street-camera BEV coordinates, per intersection/scene.

Replaces the manual coord_align_csv with an auto-solved transform using:

  1. High-confidence near-object cross-view matches
     (from match_wedge_frames.py — near pass, high stitch_confidence)
     These give:  street_track ↔ drone_track (with world_x, world_y)

  2. Street-side 3D position estimation (two options):
     (a) IPM-only         — fast, assumes flat ground, less accurate
     (b) Depth-boosted    — uses Depth Anything V2 metric depth for
                            accurate absolute depth per pixel.
                            Recommended: removes flat-ground assumption,
                            handles sloped roads and camera motion.

  3. RANSAC-robust 2D similarity transform estimation
     Solves for (offset_x, offset_y, scale, rot_deg) from matched pairs.
     Robust to ~50% outlier matches.

Why this works
--------------
  Near-object matches from match_wedge_frames.py are high-precision
  (they pass strict CLIP + angular + score thresholds).
  With just 4+ correct near correspondences per scene, RANSAC can
  reliably solve the 4-DOF similarity transform.

  Depth Anything V2 provides metric (absolute) depth without per-scene
  calibration — directly giving camera-frame 3D positions.

Output
------
  coord_align.csv:  scene_id, offset_x, offset_y, scale, rot_deg,
                    num_inliers, residual_m, method
  (Drop-in replacement for the manual coord_align_csv)

Usage
-----
  # IPM-only (no GPU needed, runs instantly):
  python auto_coord_align.py \
      --street_manifest all_street.csv \
      --drone_manifest all_drone.csv \
      --frame_matches_csv pred_frame_matches.csv \
      --track_map_csv pred_track_map.csv \
      --camera_cfg camera_params.json \
      --method ipm \
      --out_csv coord_align.csv

  # Depth-boosted (needs GPU + Depth Anything V2):
  python auto_coord_align.py \
      --street_manifest all_street.csv \
      --drone_manifest all_drone.csv \
      --frame_matches_csv pred_frame_matches.csv \
      --track_map_csv pred_track_map.csv \
      --camera_cfg camera_params.json \
      --method depth \
      --img_dir frames/ \
      --depth_model depth-anything/Depth-Anything-V2-Small-hf \
      --out_csv coord_align.csv

  # Then feed into bev_monolayout.py:
  python bev_monolayout.py train ... --coord_align_csv coord_align.csv
"""

import argparse
import json
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Camera helpers (shared with eval_bev.py)
# ---------------------------------------------------------------------------

DEFAULT_CAM = {
    "fx": 1080.0, "fy": 1080.0,
    "cx": 960.0,  "cy": 540.0,
    "img_w": 1920, "img_h": 1080,
    "cam_height_m": 1.5,
    "pitch_deg": -5.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
}


def build_K_R(cfg: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    fx, fy = float(cfg["fx"]), float(cfg["fy"])
    cx, cy = float(cfg["cx"]), float(cfg["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # pitch_deg < 0 means camera tilted DOWN (e.g. -8 = 8° below horizontal)
    pitch_deg = float(cfg.get("pitch_deg", -5.0))
    yaw_deg   = float(cfg.get("yaw_deg",   0.0))

    # -------------------------------------------------------------------
    # Correct camera → world rotation
    #
    # Camera frame:  X=right,   Y=down,    Z=forward
    # World frame:   X=forward, Y=left,    Z=up
    #
    # R_base maps camera axes to world axes for a HORIZONTAL camera:
    #   Z_cam (forward) → X_world (forward)
    #   X_cam (right)   → -Y_world (right = negative left)
    #   Y_cam (down)    → -Z_world (down  = negative up)
    #
    # Then Ry(pitch_world) tilts forward axis downward.
    # pitch_world = -pitch_deg so that pitch_deg=-8 → 8° nose-down tilt,
    # which gives ray_world[2] < 0 (pointing toward ground, Z_world=up).
    # -------------------------------------------------------------------
    R_base = np.array([
        [0,  0,  1],   # X_world ← Z_cam
        [-1, 0,  0],   # Y_world ← -X_cam
        [0, -1,  0],   # Z_world ← -Y_cam
    ], dtype=np.float64)

    # Pitch: rotate around Y_world (lateral axis)
    # -pitch_deg converts "negative=down" convention to positive tilt angle
    a = math.radians(-pitch_deg)   # e.g. pitch_deg=-8 → a=+8°
    Ry = np.array([
        [ math.cos(a), 0, math.sin(a)],
        [ 0,           1, 0          ],
        [-math.sin(a), 0, math.cos(a)],
    ], dtype=np.float64)

    # Yaw: rotate around Z_world (heading change between intersections)
    b = math.radians(yaw_deg)
    Rz = np.array([
        [ math.cos(b), -math.sin(b), 0],
        [ math.sin(b),  math.cos(b), 0],
        [ 0,            0,           1],
    ], dtype=np.float64)

    R = Rz @ Ry @ R_base
    return K, R, float(cfg.get("cam_height_m", 1.5))


def pixel_to_bev_ipm(
    px: float, py: float,
    K: np.ndarray, R: np.ndarray, cam_height: float,
) -> Optional[Tuple[float, float]]:
    """IPM: project pixel foot-point to ground plane. Returns (X_fwd, Y_lat) meters."""
    try:
        ray = np.linalg.inv(K) @ np.array([px, py, 1.0])
        ray_w = R @ ray
        if ray_w[2] >= -1e-6:
            return None
        t = -cam_height / ray_w[2]
        if t < 0:
            return None
        p = t * ray_w
        return float(p[0]), float(p[1])
    except Exception:
        return None


def pixel_to_bev_depth(
    px: float, py: float,
    depth_m: float,
    K: np.ndarray, R: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Depth-boosted: unproject pixel using metric depth, rotate to world frame.
    More accurate than IPM — does not assume flat ground.
    Returns (X_fwd, Y_lat) meters.
    """
    try:
        ray_cam = np.linalg.inv(K) @ np.array([px, py, 1.0])
        ray_cam = ray_cam / (np.linalg.norm(ray_cam) + 1e-9)
        point_cam = ray_cam * depth_m          # 3D point in camera frame
        point_world = R @ point_cam            # rotate to world frame
        return float(point_world[0]), float(point_world[1])
    except Exception:
        return None


def bbox_foot(x1, y1, x2, y2) -> Tuple[float, float]:
    return (float(x1) + float(x2)) / 2.0, float(y2)


# ---------------------------------------------------------------------------
# Depth Anything V2 loader
# ---------------------------------------------------------------------------

def load_depth_model(model_name: str):
    """
    Load Depth Anything V2 metric depth model from HuggingFace.
    Returns (model, processor, device).

    Install:
        pip install transformers torch torchvision pillow
    Model options (smallest → largest, all metric):
        depth-anything/Depth-Anything-V2-Small-hf   (~100MB)
        depth-anything/Depth-Anything-V2-Base-hf    (~400MB)
        depth-anything/Depth-Anything-V2-Large-hf   (~1.4GB)
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        raise ImportError(
            "transformers and torch required for depth mode.\n"
            "Install: pip install transformers torch torchvision pillow"
        )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[depth] loading {model_name} on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[depth] model loaded.")
    return model, processor, device


def estimate_depth_map(
    img_path: str,
    model, processor, device,
) -> Optional[np.ndarray]:
    """
    Run Depth Anything V2 on one image.
    Returns depth map (H, W) in meters (float32), or None on failure.
    """
    try:
        import torch
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Post-process: interpolate to original image size
        pred = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(img.height, img.width)],
        )[0]["predicted_depth"]
        return pred.squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"[depth] warning: failed on {img_path}: {e}")
        return None


def get_depth_at_pixel(
    depth_map: np.ndarray,
    px: float, py: float,
    window: int = 3,
) -> Optional[float]:
    """
    Sample depth map at pixel (px, py) using a small median window.
    Median is more robust than single-pixel sampling near object edges.
    """
    H, W = depth_map.shape
    r, c = int(py), int(px)
    r1, r2 = max(0, r - window), min(H, r + window + 1)
    c1, c2 = max(0, c - window), min(W, c + window + 1)
    patch = depth_map[r1:r2, c1:c2]
    vals = patch[patch > 0.1]    # exclude invalid/zero depth
    if len(vals) == 0:
        return None
    return float(np.median(vals))


# ---------------------------------------------------------------------------
# Depth map cache (avoid re-running on same frame twice)
# ---------------------------------------------------------------------------

class DepthCache:
    def __init__(self, model, processor, device, img_dir: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.img_dir = img_dir
        self._cache: Dict[str, Optional[np.ndarray]] = {}

    def get(self, scene_id: str, frame_num: int) -> Optional[np.ndarray]:
        key = f"{scene_id}/{frame_num}"
        if key not in self._cache:
            path = self._find_image(scene_id, frame_num)
            if path is None:
                self._cache[key] = None
            else:
                self._cache[key] = estimate_depth_map(
                    path, self.model, self.processor, self.device)
        return self._cache[key]

    def _find_image(self, scene_id: str, frame_num: int) -> Optional[str]:
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(self.img_dir, scene_id, f"{frame_num:06d}{ext}")
            if os.path.exists(p):
                return p
        return None


# ---------------------------------------------------------------------------
# 2D Rigid transform solver (scale locked to 1.0)
# ---------------------------------------------------------------------------
# WHY SCALE=1: IPM and drone x_fwd/y_left are both in meters. Scale should
# be 1.0 by definition. The previous similarity solver returned scale≈0.14
# because the drone hovers above the intersection center — NOT above the
# camera position. Objects near the drone nadir are far from the camera, so
# IPM (which measures camera-relative distance) gives large values while
# drone gives small values. RANSAC was fitting a spurious scale to compensate.
# The correct model: drone_pos = R(rot) * ipm_pos + offset
# where offset encodes the camera→drone_nadir displacement vector.
# ---------------------------------------------------------------------------

def solve_rigid_transform(
    src: np.ndarray,   # (N, 2) — street IPM positions (X_fwd, Y_lat)
    dst: np.ndarray,   # (N, 2) — drone x_fwd/y_left positions
) -> Tuple[float, float, float]:
    """
    Solve for 2D rigid transform (scale=1):
        dst ≈ R(rot_deg) * src + (offset_x, offset_y)

    Closed-form least-squares via SVD (Procrustes without scale).
    Returns (offset_x, offset_y, rot_deg).
    """
    assert len(src) >= 2

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R_mat = Vt.T @ np.diag([1.0, d]) @ U.T

    rot_deg = math.degrees(math.atan2(R_mat[1, 0], R_mat[0, 0]))
    offset = dst_mean - R_mat @ src_mean
    return float(offset[0]), float(offset[1]), float(rot_deg)


def _apply_rigid(points: np.ndarray, tf: Tuple) -> np.ndarray:
    """Apply (offset_x, offset_y, rot_deg) to (N,2) points."""
    offset_x, offset_y, rot_deg = tf
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    R = np.array([[c, -s], [s, c]])
    return (points @ R.T) + np.array([offset_x, offset_y])


def ransac_rigid(
    src: np.ndarray,
    dst: np.ndarray,
    n_iter: int = 1000,
    inlier_thresh_m: float = 1.5,
    min_samples: int = 3,
) -> Tuple[Optional[Tuple[float, float, float]], np.ndarray, float]:
    """
    RANSAC rigid transform (rotation + translation, scale=1).

    Returns:
        best_transform: (offset_x, offset_y, rot_deg) or None
        inlier_mask:    boolean array (N,)
        mean_residual_m: mean inlier residual in meters
    """
    N = len(src)
    if N < min_samples:
        print(f"[ransac] too few points: {N} < {min_samples}")
        return None, np.zeros(N, dtype=bool), float("inf")

    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_transform = None
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(N, size=min_samples, replace=False)
        try:
            tf = solve_rigid_transform(src[idx], dst[idx])
        except Exception:
            continue

        resid = _apply_rigid(src, tf) - dst
        dists = np.linalg.norm(resid, axis=1)
        inliers = dists < inlier_thresh_m

        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers.copy()
            best_transform = tf

    # Refit on all inliers
    if best_transform is not None and best_inliers.sum() >= min_samples:
        try:
            best_transform = solve_rigid_transform(src[best_inliers], dst[best_inliers])
        except Exception:
            pass
        resid = _apply_rigid(src[best_inliers], best_transform) - dst[best_inliers]
        residual = float(np.mean(np.linalg.norm(resid, axis=1)))
    else:
        residual = float("inf")

    return best_transform, best_inliers, residual

    if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers
            best_transform = tf

    # Refit on all inliers
    if best_transform is not None and best_inliers.sum() >= min_samples:
        try:
            best_transform = solve_similarity_transform(
                src[best_inliers], dst[best_inliers])
        except Exception:
            pass



# ---------------------------------------------------------------------------
# Per-scene alignment estimation
# ---------------------------------------------------------------------------

def estimate_scene_alignment(
    scene_id: str,
    street_scene: pd.DataFrame,
    drone_scene: pd.DataFrame,
    frame_matches: pd.DataFrame,
    track_map: pd.DataFrame,
    K: np.ndarray, R_cam: np.ndarray, cam_height: float,
    method: str = "ipm",
    depth_cache: Optional["DepthCache"] = None,
    min_conf: float = 0.6,
    near_only: bool = True,
    inlier_thresh_m: float = 1.5,
    max_ipm_fwd_m: float = 8.0,
    verbose: bool = True,
) -> Dict:
    """
    Estimate (offset_x, offset_y, scale=1, rot_deg) for one scene.

    Uses rigid transform (scale locked to 1.0). Both IPM and drone positions
    are in meters — scale should always be 1. Previous similarity solver
    returned scale≈0.14 because drone nadir ≠ camera position: far objects
    appear small in drone frame but large in IPM frame. Rigid transform
    with IPM x_fwd filter avoids this.

    max_ipm_fwd_m: only use pairs where IPM forward distance < this value.
    IPM is accurate for nearby objects (< 8m); beyond that pitch errors
    cause growing overestimation of forward distance.
    """

    # Filter frame matches for this scene's tracks
    scene_stids = set(street_scene["track_id"].astype(int).unique())

    # Use near-pass matches (more reliable) if available
    fm = frame_matches[frame_matches["street_track_id"].isin(scene_stids)].copy()
    if near_only and "match_pass" in fm.columns:
        fm_near = fm[fm["match_pass"] == "near"]
        if len(fm_near) >= 5:
            fm = fm_near
        # else fall back to all passes

    # Filter by score
    if "score" in fm.columns:
        fm = fm[fm["score"] >= min_conf]
    # Note: distance filtering now happens post-IPM (filter by ipm_xfwd, not drone_dist_m)
    # because drone_dist_m is drone-to-object 3D distance, not camera-to-object distance.

    if verbose:
        print(f"[align] scene={scene_id}  candidate frame-matches: {len(fm)}")

    if len(fm) < 3:
        print(f"[align] scene={scene_id}  insufficient matches — using identity transform")
        return _identity_result(scene_id, "insufficient_matches")

    # Build drone position lookup: (track_id, frame) → (x_fwd, y_left) in meters
    # Prefer x_fwd/y_left (drone-relative meters) over world_x/world_y (UTM).
    # UTM coordinates (~726000, ~5433000) cannot be used directly as metric offsets.
    drone_frame_col = "street_frame" if "street_frame" in drone_scene.columns else "drone_frame"
    has_fwd = "x_fwd" in drone_scene.columns and "y_left" in drone_scene.columns
    drone_pos: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for _, dr in drone_scene.iterrows():
        key = (int(dr["track_id"]), int(dr[drone_frame_col]))
        if has_fwd:
            drone_pos[key] = (float(dr["x_fwd"]), float(dr["y_left"]))
        else:
            drone_pos[key] = (float(dr["world_x"]), float(dr["world_y"]))

    # Build street detection lookup: (track_id, frame) → row
    street_idx: Dict[Tuple[int, int], pd.Series] = {}
    for _, sr in street_scene.iterrows():
        street_idx[(int(sr["track_id"]), int(sr["frame"]))] = sr

    # Collect correspondence pairs
    src_pts: List[Tuple[float, float]] = []   # street BEV positions
    dst_pts: List[Tuple[float, float]] = []   # drone world positions

    for _, row in fm.iterrows():
        s_tid = int(row["street_track_id"])
        d_tid = int(row["drone_track_id"])
        frame = int(row["street_frame"])

        s_key = (s_tid, frame)
        d_key = (d_tid, frame)

        if s_key not in street_idx or d_key not in drone_pos:
            continue

        sr = street_idx[s_key]
        wx, wy = drone_pos[d_key]

        # Street-side BEV position
        # Support both column naming conventions: x1/y1/x2/y2 and bbox_x1/bbox_y1/...
        x1_col = "bbox_x1" if "bbox_x1" in sr.index else "x1"
        y1_col = "bbox_y1" if "bbox_y1" in sr.index else "y1"
        x2_col = "bbox_x2" if "bbox_x2" in sr.index else "x2"
        y2_col = "bbox_y2" if "bbox_y2" in sr.index else "y2"
        fp_x, fp_y = bbox_foot(sr[x1_col], sr[y1_col], sr[x2_col], sr[y2_col])

        if method == "depth" and depth_cache is not None:
            depth_map = depth_cache.get(scene_id, frame)
            if depth_map is not None:
                d_val = get_depth_at_pixel(depth_map, fp_x, fp_y)
                if d_val is not None and 0.5 < d_val < 80.0:
                    bev = pixel_to_bev_depth(fp_x, fp_y, d_val, K, R_cam)
                else:
                    bev = pixel_to_bev_ipm(fp_x, fp_y, K, R_cam, cam_height)
            else:
                bev = pixel_to_bev_ipm(fp_x, fp_y, K, R_cam, cam_height)
        else:
            bev = pixel_to_bev_ipm(fp_x, fp_y, K, R_cam, cam_height)

        if bev is None:
            continue

        # DUAL FILTER: require small distance on BOTH sides.
        # IPM filter: only near camera objects (IPM accurate only close-up).
        # Drone filter: only near drone-nadir objects.
        # Wrong matches have small IPM but large drone values (object near camera,
        # far from drone nadir) — these caused the spurious RANSAC solution before.
        ipm_xfwd = bev[0]
        if ipm_xfwd <= 0 or ipm_xfwd > max_ipm_fwd_m:
            continue

        drone_dist_from_nadir = math.sqrt(wx**2 + wy**2)
        if drone_dist_from_nadir > max_ipm_fwd_m:
            continue

        src_pts.append(bev)
        dst_pts.append((wx, wy))

    if verbose:
        print(f"[align] scene={scene_id}  valid correspondence pairs: {len(src_pts)}")

    if len(src_pts) < 3:
        print(f"[align] scene={scene_id}  too few valid pairs — using identity")
        return _identity_result(scene_id, "too_few_pairs")

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)

    # Use rigid transform (scale=1). Both IPM and drone coords are in meters
    # so scale must be 1.0. Solving for scale gives spurious values because
    # drone nadir ≠ camera position (drone hovers over intersection center).
    tf, inlier_mask, residual = ransac_rigid(
        src, dst,
        inlier_thresh_m=inlier_thresh_m,
        min_samples=3,
    )

    if tf is None:
        print(f"[align] scene={scene_id}  RANSAC failed — using identity")
        return _identity_result(scene_id, "ransac_failed")

    offset_x, offset_y, rot_deg = tf
    scale = 1.0
    n_inliers = int(inlier_mask.sum())

    if verbose:
        print(f"[align] scene={scene_id}  "
              f"inliers={n_inliers}/{len(src_pts)}  "
              f"residual={residual:.2f}m  "
              f"rot={rot_deg:.1f}°  scale={scale:.3f}  "
              f"offset=({offset_x:.1f}, {offset_y:.1f})")

    return {
        "scene_id": scene_id,
        "offset_x": round(offset_x, 4),
        "offset_y": round(offset_y, 4),
        "scale": round(scale, 4),
        "rot_deg": round(rot_deg, 4),
        "num_inliers": n_inliers,
        "num_pairs": len(src_pts),
        "residual_m": round(residual, 4),
        "method": method,
        "status": "ok",
    }


def _identity_result(scene_id: str, status: str) -> Dict:
    return {
        "scene_id": scene_id,
        "offset_x": 0.0, "offset_y": 0.0,
        "scale": 1.0, "rot_deg": 0.0,
        "num_inliers": 0, "num_pairs": 0,
        "residual_m": -1.0,
        "method": "identity",
        "status": status,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Auto-estimate coordinate alignment per scene from cross-view matches."
    )
    ap.add_argument("--street_manifest", required=True)
    ap.add_argument("--drone_manifest",  required=True)
    ap.add_argument("--frame_matches_csv", required=True,
                    help="out_frame_csv from match_wedge_frames.py")
    ap.add_argument("--track_map_csv", required=True,
                    help="out_track_map_csv from match_wedge_frames.py")
    ap.add_argument("--camera_cfg", default=None,
                    help="JSON: {scene_id: {fx, fy, cx, cy, cam_height_m, pitch_deg, ...}}")
    ap.add_argument("--method", choices=["ipm", "depth"], default="ipm",
                    help="Street-side position estimation: ipm (fast) or depth (accurate)")
    ap.add_argument("--img_dir", default="frames",
                    help="Root dir of street frames (needed for depth mode): img_dir/scene_id/XXXXXX.jpg")
    ap.add_argument("--depth_model",
                    default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="HuggingFace model ID for Depth Anything V2")
    ap.add_argument("--out_csv", required=True,
                    help="Output coord_align.csv")
    ap.add_argument("--min_conf", type=float, default=0.60,
                    help="Min frame match score for correspondence")
    ap.add_argument("--max_ipm_fwd_m", type=float, default=8.0,
                    help="Only use pairs where IPM forward distance < this (meters). "
                         "IPM is reliable only for near objects. Default 8m.")
    ap.add_argument("--inlier_thresh_m", type=float, default=1.5,
                    help="RANSAC inlier threshold in meters")
    args = ap.parse_args()

    def norm_class(x):
        x = str(x).strip().lower()
        if x in {"pedestrian","person","people"}: return "person"
        if x in {"bike","bicycle","cyclist"}: return "bicycle"
        if x in {"motorbike","motorcycle"}: return "motorcycle"
        return x

    # Load
    street = pd.read_csv(args.street_manifest)
    drone  = pd.read_csv(args.drone_manifest)
    fm     = pd.read_csv(args.frame_matches_csv)
    tm     = pd.read_csv(args.track_map_csv)

    street["class_name"] = street["class_name"].apply(norm_class)
    if "class_name" in drone.columns:
        drone["class_name"] = drone["class_name"].apply(norm_class)
    if "scene_id" not in street.columns: street["scene_id"] = "scene_00"
    if "scene_id" not in drone.columns:  drone["scene_id"]  = "scene_00"

    # Camera configs
    cam_cfg_all = {}
    if args.camera_cfg and os.path.exists(args.camera_cfg):
        with open(args.camera_cfg) as f:
            cam_cfg_all = json.load(f)

    # Load depth model if needed
    depth_cache = None
    if args.method == "depth":
        model, processor, device = load_depth_model(args.depth_model)
        depth_cache = DepthCache(model, processor, device, args.img_dir)

    # Process each scene
    scenes = street["scene_id"].unique()
    results = []

    for scene_id in sorted(scenes):
        cfg = cam_cfg_all.get(str(scene_id), cam_cfg_all.get("default", DEFAULT_CAM))
        K, R_cam, cam_height = build_K_R(cfg)

        street_scene = street[street["scene_id"] == scene_id]
        drone_scene  = drone[drone["scene_id"] == scene_id]

        result = estimate_scene_alignment(
            scene_id=str(scene_id),
            street_scene=street_scene,
            drone_scene=drone_scene,
            frame_matches=fm,
            track_map=tm,
            K=K, R_cam=R_cam, cam_height=cam_height,
            method=args.method,
            depth_cache=depth_cache,
            min_conf=args.min_conf,
            max_ipm_fwd_m=args.max_ipm_fwd_m,
            inlier_thresh_m=args.inlier_thresh_m,
            verbose=True,
        )
        results.append(result)

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\n[done] wrote {args.out_csv}")

    # Summary
    ok = out_df[out_df["status"] == "ok"]
    print(f"\n  Scenes solved:      {len(ok)} / {len(results)}")
    if len(ok):
        print(f"  Mean inliers:       {ok['num_inliers'].mean():.1f}")
        print(f"  Mean residual:      {ok['residual_m'].mean():.2f} m")
        print(f"  Mean rotation:      {ok['rot_deg'].abs().mean():.1f}°")
        print(f"  Mean scale:         {ok['scale'].mean():.3f}")
    failed = out_df[out_df["status"] != "ok"]
    if len(failed):
        print(f"\n  [warn] {len(failed)} scene(s) fell back to identity: "
              f"{list(failed['scene_id'])}")
        print(f"  → Try lowering --min_conf or --max_ipm_fwd_m, "
              f"or check that near matches exist for these scenes.")


if __name__ == "__main__":
    main()