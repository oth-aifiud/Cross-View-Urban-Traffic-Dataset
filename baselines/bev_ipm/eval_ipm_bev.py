"""
eval_bev.py — BEV Projection Baseline & Evaluation
====================================================
Baseline: Inverse Perspective Mapping (IPM)
Ground truth: drone world coordinates (world_x, world_y) from drone manifest,
              back-projected to the street camera's local BEV plane via the
              cross-view track map.

Pipeline
--------
  1. For each matched street track, take the street detection bbox.
  2. Project the bbox bottom-center (foot point) to BEV using IPM
     (requires camera intrinsics K and extrinsics R, t — pitch/height/yaw).
  3. Compare projected BEV position against drone-derived GT world position
     (transformed into the same coordinate frame via a per-scene homography
     or GPS anchor if available).
  4. Report positional errors and IoU-based metrics.

Metrics
-------
  ADE  — Average Displacement Error (mean L2 in meters)
  FDE  — Final Displacement Error (last frame of track, meters)
  ALE  — Average Lateral Error
  ALgE — Average Longitudinal Error
  BEV-IoU@0.5 / @0.25 — fraction of objects with BEV box IoU > threshold
         (approximated as a circle-disc IoU using estimated object size)
  PCK@1m / PCK@2m — Percentage of Correct Keypoints at distance threshold

  All per-class and overall.

Inputs
------
  --street_manifest   Street detections CSV
                      Columns: scene_id, frame, track_id, class_name,
                               x1, y1, x2, y2

  --drone_manifest    Drone detections CSV
                      Columns: scene_id, street_frame, drone_frame,
                               track_id, class_name,
                               world_x, world_y, dist_m, theta_rad

  --track_map_csv     Cross-view track map (from match_wedge_frames.py)
                      Columns: street_track_id, drone_track_id,
                               stitch_confidence

  --camera_cfg        YAML/JSON file with camera parameters per scene_id:
                      {
                        "scene_00": {
                          "fx": 1080.0, "fy": 1080.0,
                          "cx": 960.0,  "cy": 540.0,
                          "img_w": 1920, "img_h": 1080,
                          "cam_height_m": 1.5,
                          "pitch_deg": -5.0,
                          "yaw_deg": 0.0,
                          "roll_deg": 0.0
                        }
                      }
                      If not provided, default GoPro-on-bike params are used.

  --coord_align_csv   (Optional) Per-scene coordinate alignment:
                      maps drone world_x/world_y to street IPM frame.
                      Columns: scene_id, offset_x, offset_y, scale, rot_deg
                      If not provided, drone world coords are used as-is
                      (assumes they share the same origin as IPM output).

  --out_report        Output JSON report path
  --out_csv           Output per-track BEV error CSV
  --min_conf          Minimum stitch_confidence to use a track pair
  --classes           Comma-separated class list

Usage
-----
  python eval_bev.py \
      --street_manifest all_street.csv \
      --drone_manifest all_drone.csv \
      --track_map_csv all_track_map.csv \
      --camera_cfg camera_params.json \
      --out_report bev_eval.json \
      --out_csv bev_per_track.csv
"""

import argparse
import json
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Default camera parameters (GoPro Hero on bike, approximate)
# ---------------------------------------------------------------------------

DEFAULT_CAM = {
    "fx": 1080.0, "fy": 1080.0,
    "cx": 960.0, "cy": 540.0,
    "img_w": 1920, "img_h": 1080,
    "cam_height_m": 1.5,    # camera height above ground (meters)
    "pitch_deg": -5.0,      # negative = tilted down
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
}

# Approximate object dimensions for BEV disc IoU (meters)
OBJECT_RADIUS = {
    "car": 2.5,
    "truck": 4.0,
    "bus": 5.0,
    "person": 0.4,
    "bicycle": 0.8,
    "motorcycle": 1.0,
}
DEFAULT_RADIUS = 1.5


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


def safe_div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def rot_matrix_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_matrix_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_matrix_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# IPM: pixel foot-point → ground plane (X_fwd, Y_lat) in meters
# ---------------------------------------------------------------------------

def build_ipm(cfg: Dict):
    """
    Build the camera matrix K and the rotation R_cam_to_world.

    Camera convention:
      - Z forward (optical axis)
      - X right
      - Y down
    World convention (BEV plane):
      - X_w forward (away from camera along ground)
      - Y_w lateral (left positive)
      - Z_w up (ground = Z_w = 0)

    Returns K (3x3), R (3x3 cam→world), cam_height (float)
    """
    fx = float(cfg["fx"])
    fy = float(cfg["fy"])
    cx = float(cfg["cx"])
    cy = float(cfg["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    pitch_deg = float(cfg.get("pitch_deg", -5.0))
    yaw_deg   = float(cfg.get("yaw_deg",   0.0))

    # -------------------------------------------------------------------
    # Correct camera → world rotation
    #
    # Camera frame:  X=right,   Y=down,    Z=forward
    # World frame:   X=forward, Y=left,    Z=up
    #
    # R_base maps camera axes to world axes for a horizontal camera:
    #   Z_cam → X_world,  X_cam → -Y_world,  Y_cam → -Z_world
    #
    # Ry(a) then tilts the forward axis downward.
    # a = -pitch_deg so that pitch_deg=-8 (8° below horizontal)
    # correctly gives ray_world[2] < 0 (ray heading toward ground).
    # -------------------------------------------------------------------
    R_base = np.array([
        [0,  0,  1],
        [-1, 0,  0],
        [0, -1,  0],
    ], dtype=np.float64)

    a = math.radians(-pitch_deg)
    Ry = np.array([
        [ math.cos(a), 0, math.sin(a)],
        [ 0,           1, 0          ],
        [-math.sin(a), 0, math.cos(a)],
    ], dtype=np.float64)

    b = math.radians(yaw_deg)
    Rz_yaw = np.array([
        [ math.cos(b), -math.sin(b), 0],
        [ math.sin(b),  math.cos(b), 0],
        [ 0,            0,           1],
    ], dtype=np.float64)

    R = Rz_yaw @ Ry @ R_base
    return K, R, float(cfg.get("cam_height_m", 1.5))


def pixel_to_bev(
    px: float, py: float,
    K: np.ndarray,
    R: np.ndarray,
    cam_height: float,
) -> Optional[Tuple[float, float]]:
    """
    Project a single image pixel (px, py) — typically bbox bottom-center —
    onto the flat ground plane Z_w = 0.

    Returns (X_fwd_meters, Y_lat_meters) or None if behind camera / no intersection.

    The IPM equation:
      ray_cam = K^-1 * [px, py, 1]^T
      ray_world = R * ray_cam
      t_intersect = -cam_height / ray_world[2]   (solve for Z_w=0)
      point_world = t_intersect * ray_world
    """
    try:
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ np.array([px, py, 1.0], dtype=np.float64)
        ray_world = R @ ray_cam

        # ray_world[2] is the Z component in world frame
        # For the ray to hit the ground (Z_w=0) below the camera,
        # ray_world[2] must be negative (pointing downward)
        rz = ray_world[2]
        if rz >= -1e-6:
            return None  # ray parallel to ground or pointing upward

        t = -cam_height / rz
        if t < 0:
            return None  # intersection behind camera

        point = t * ray_world
        x_fwd = float(point[0])   # forward
        y_lat = float(point[1])   # lateral
        return x_fwd, y_lat
    except Exception:
        return None


def bbox_foot_point(x1, y1, x2, y2) -> Tuple[float, float]:
    """Bottom-center of bounding box — best IPM projection point."""
    cx = (float(x1) + float(x2)) / 2.0
    by = float(y2)
    return cx, by


# ---------------------------------------------------------------------------
# Coordinate alignment: drone world → street IPM frame
# ---------------------------------------------------------------------------

def align_drone_to_ipm(
    world_x: float,
    world_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale: float = 1.0,
    rot_deg: float = 0.0,
) -> Tuple[float, float]:
    """
    Transform drone x_fwd/y_left into street IPM coordinate frame.

    RANSAC solved:  drone = R(rot) * ipm + offset
    Inverse is:     ipm   = R(-rot) * (drone - offset)

    R(-rot) = [[cos, +sin], [-sin, cos]]  (note: +sin not -sin)
    """
    dx = world_x - offset_x
    dy = world_y - offset_y
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    # Inverse rotation: transpose of R(rot)
    x_aligned = c * dx + s * dy
    y_aligned = -s * dx + c * dy
    return x_aligned, y_aligned


# ---------------------------------------------------------------------------
# Per-track error computation
# ---------------------------------------------------------------------------

def compute_track_errors(
    street_df: pd.DataFrame,
    drone_df: pd.DataFrame,
    K: np.ndarray,
    R: np.ndarray,
    cam_height: float,
    align_params: Dict,
    street_tid: int,
    drone_tid: int,
    cname: str,
) -> Optional[Dict]:
    """
    For one matched track pair, compute BEV projection errors
    across all co-visible frames.

    Returns dict with per-frame errors and aggregate stats.
    """
    s_track = street_df[street_df["track_id"] == street_tid].sort_values("frame")
    d_track = drone_df[drone_df["track_id"] == drone_tid]

    if len(s_track) == 0 or len(d_track) == 0:
        return None

    # Index drone by the synchronized street_frame
    drone_by_frame: Dict[int, Tuple[float, float]] = {}
    frame_col = "street_frame" if "street_frame" in d_track.columns else "drone_frame"
    has_fwd = "x_fwd" in d_track.columns and "y_left" in d_track.columns
    for _, dr in d_track.iterrows():
        f = int(dr[frame_col])
        if has_fwd:
            wx, wy = float(dr["x_fwd"]), float(dr["y_left"])
        else:
            wx, wy = float(dr["world_x"]), float(dr["world_y"])
        ax, ay = align_drone_to_ipm(wx, wy, **align_params)
        drone_by_frame[f] = (ax, ay)

    frame_errors: List[Dict] = []
    lateral_errs: List[float] = []
    longitudinal_errs: List[float] = []
    l2_errs: List[float] = []

    # Only evaluate frames where camera is near intersection center.
    # Static coord_align is only valid when the bike is close to where
    # the alignment was solved (near intersection). Frames far from the
    # intersection have growing static-transform error unrelated to IPM quality.
    MAX_EVAL_FWD_M = 20.0   # only evaluate when IPM forward < 20m

    for _, sr in s_track.iterrows():
        f = int(sr["frame"])
        if f not in drone_by_frame:
            continue

        x1_col = "bbox_x1" if "bbox_x1" in sr.index else "x1"
        y1_col = "bbox_y1" if "bbox_y1" in sr.index else "y1"
        x2_col = "bbox_x2" if "bbox_x2" in sr.index else "x2"
        y2_col = "bbox_y2" if "bbox_y2" in sr.index else "y2"
        fp_x, fp_y = bbox_foot_point(sr[x1_col], sr[y1_col], sr[x2_col], sr[y2_col])
        bev = pixel_to_bev(fp_x, fp_y, K, R, cam_height)
        if bev is None:
            continue

        # Skip frames where IPM forward distance is large — static transform invalid
        if bev[0] > MAX_EVAL_FWD_M or bev[0] <= 0:
            continue

        pred_x, pred_y = bev
        gt_x, gt_y = drone_by_frame[f]

        # L2 displacement error (meters)
        l2 = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        # Decompose into longitudinal (forward) and lateral errors
        lon_err = abs(pred_x - gt_x)
        lat_err = abs(pred_y - gt_y)

        l2_errs.append(l2)
        lateral_errs.append(lat_err)
        longitudinal_errs.append(lon_err)
        frame_errors.append({
            "frame": f,
            "pred_x": round(pred_x, 3),
            "pred_y": round(pred_y, 3),
            "gt_x": round(gt_x, 3),
            "gt_y": round(gt_y, 3),
            "l2_m": round(l2, 3),
            "lat_err_m": round(lat_err, 3),
            "lon_err_m": round(lon_err, 3),
        })

    if not l2_errs:
        return None

    # ADE / FDE
    ade = float(np.mean(l2_errs))
    fde = float(l2_errs[-1])

    # PCK
    pck_1m = safe_div(sum(1 for e in l2_errs if e <= 1.0), len(l2_errs))
    pck_2m = safe_div(sum(1 for e in l2_errs if e <= 2.0), len(l2_errs))

    # BEV disc IoU (approximate, using fixed object radius)
    radius = OBJECT_RADIUS.get(cname, DEFAULT_RADIUS)
    bev_iou_per_frame = [disc_iou(e, radius) for e in l2_errs]
    iou_50 = safe_div(sum(1 for iou in bev_iou_per_frame if iou >= 0.5), len(bev_iou_per_frame))
    iou_25 = safe_div(sum(1 for iou in bev_iou_per_frame if iou >= 0.25), len(bev_iou_per_frame))

    return {
        "street_track_id": int(street_tid),
        "drone_track_id": int(drone_tid),
        "class_name": cname,
        "num_frames": len(l2_errs),
        "ADE_m": round(ade, 3),
        "FDE_m": round(fde, 3),
        "ALE_m": round(float(np.mean(lateral_errs)), 3),
        "ALgE_m": round(float(np.mean(longitudinal_errs)), 3),
        "PCK_1m": round(pck_1m, 3),
        "PCK_2m": round(pck_2m, 3),
        "BEV_IoU_50": round(iou_50, 3),
        "BEV_IoU_25": round(iou_25, 3),
        "frame_errors": frame_errors,
    }


def disc_iou(center_dist: float, radius: float) -> float:
    """
    Approximate IoU of two identical discs separated by center_dist.
    IoU = intersection_area / union_area.
    Returns 0 if discs don't overlap.
    """
    r = float(radius)
    d = float(center_dist)
    if d >= 2 * r:
        return 0.0
    if d <= 0:
        return 1.0
    # Lens-shaped intersection of two circles radius r, centers d apart
    alpha = 2.0 * math.acos(d / (2.0 * r))
    intersection = r * r * (alpha - math.sin(alpha))
    union = 2.0 * math.pi * r * r - intersection
    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(track_results: List[Dict], classes: List[str]) -> Dict:
    def stats_from(vals):
        if not vals:
            return {}
        return {
            "mean": round(float(np.mean(vals)), 3),
            "median": round(float(np.median(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "p75": round(float(np.percentile(vals, 75)), 3),
            "p90": round(float(np.percentile(vals, 90)), 3),
        }

    def agg_tracks(subset):
        if not subset:
            return {}
        ade = [t["ADE_m"] for t in subset]
        fde = [t["FDE_m"] for t in subset]
        ale = [t["ALE_m"] for t in subset]
        alge = [t["ALgE_m"] for t in subset]
        pck1 = [t["PCK_1m"] for t in subset]
        pck2 = [t["PCK_2m"] for t in subset]
        iou50 = [t["BEV_IoU_50"] for t in subset]
        iou25 = [t["BEV_IoU_25"] for t in subset]
        return {
            "num_tracks": len(subset),
            "ADE_m": stats_from(ade),
            "FDE_m": stats_from(fde),
            "ALE_m": stats_from(ale),
            "ALgE_m": stats_from(alge),
            "PCK_1m": round(float(np.mean(pck1)), 3),
            "PCK_2m": round(float(np.mean(pck2)), 3),
            "BEV_IoU_50": round(float(np.mean(iou50)), 3),
            "BEV_IoU_25": round(float(np.mean(iou25)), 3),
        }

    result = {"overall": agg_tracks(track_results), "per_class": {}}
    for cname in classes:
        subset = [t for t in track_results if t["class_name"] == cname]
        result["per_class"][cname] = agg_tracks(subset)

    return result


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(agg: Dict, classes: List[str]):
    print()
    print("=" * 65)
    print("  BEV PROJECTION EVALUATION  (IPM Baseline)")
    print("=" * 65)

    ov = agg["overall"]
    if not ov:
        print("  No results.")
        return

    print(f"\n  Tracks evaluated  : {ov['num_tracks']}")
    print(f"  ADE  (mean ± std) : {ov['ADE_m']['mean']:.2f} ± {ov['ADE_m']['std']:.2f} m")
    print(f"  FDE  (mean ± std) : {ov['FDE_m']['mean']:.2f} ± {ov['FDE_m']['std']:.2f} m")
    print(f"  ALE  (lateral)    : {ov['ALE_m']['mean']:.2f} m")
    print(f"  ALgE (longitud.)  : {ov['ALgE_m']['mean']:.2f} m")
    print(f"  PCK@1m            : {ov['PCK_1m']:.1%}")
    print(f"  PCK@2m            : {ov['PCK_2m']:.1%}")
    print(f"  BEV-IoU@0.5       : {ov['BEV_IoU_50']:.1%}")
    print(f"  BEV-IoU@0.25      : {ov['BEV_IoU_25']:.1%}")

    print()
    print(f"  {'Class':<14} {'N':>4} {'ADE':>6} {'FDE':>6} {'PCK@1m':>7} {'IoU@.5':>7}")
    print("  " + "-" * 44)
    for cname in classes:
        pc = agg["per_class"].get(cname, {})
        if not pc or pc.get("num_tracks", 0) == 0:
            continue
        print(f"  {cname:<14} {pc['num_tracks']:>4} "
              f"{pc['ADE_m']['mean']:>6.2f} "
              f"{pc['FDE_m']['mean']:>6.2f} "
              f"{pc['PCK_1m']:>6.1%} "
              f"{pc['BEV_IoU_50']:>6.1%}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="BEV projection baseline and evaluation.")
    ap.add_argument("--street_manifest", required=True)
    ap.add_argument("--drone_manifest", required=True)
    ap.add_argument("--track_map_csv", required=True)
    ap.add_argument("--camera_cfg", default=None,
                    help="JSON file: {scene_id: {fx, fy, cx, cy, cam_height_m, pitch_deg, ...}}")
    ap.add_argument("--coord_align_csv", default=None,
                    help="CSV: scene_id, offset_x, offset_y, scale, rot_deg")
    ap.add_argument("--out_report", default=None)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--classes", type=str,
                    default="car,truck,bus,person,bicycle,motorcycle")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

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

    # Camera configs
    cam_cfg_all: Dict = {}
    if args.camera_cfg:
        with open(args.camera_cfg) as f:
            cam_cfg_all = json.load(f)

    # Coordinate alignment per scene
    # auto_coord_align_v3.py outputs: scene_id, offset_x, offset_y, scale, rot_deg
    align_all: Dict = {}
    if args.coord_align_csv:
        align_df = pd.read_csv(args.coord_align_csv)
        for _, row in align_df.iterrows():
            # Only use successfully solved scenes
            if str(row.get("status", "ok")) not in ("ok", "nan", ""):
                if str(row.get("status", "ok")) != "ok":
                    print(f"[warn] scene {row['scene_id']} alignment status={row.get('status')} — skipping")
                    continue
            align_all[str(row["scene_id"])] = {
                "offset_x": float(row.get("offset_x", 0.0)),
                "offset_y": float(row.get("offset_y", 0.0)),
                "rot_deg":  float(row.get("rot_deg", 0.0)),
                # scale intentionally omitted — rigid transform only
            }

    # Filter track map by confidence
    tm = track_map[track_map["drone_track_id"] != -1].copy()
    if "stitch_confidence" in tm.columns:
        tm = tm[tm["stitch_confidence"] >= args.min_conf]

    # Build track→class lookup from street manifest
    track_class: Dict[int, str] = {}
    for _, row in street.drop_duplicates("track_id").iterrows():
        track_class[int(row["track_id"])] = norm_class(row["class_name"])

    # Build track→scene lookup
    track_scene: Dict[int, str] = {}
    for _, row in street.drop_duplicates("track_id").iterrows():
        track_scene[int(row["track_id"])] = str(row.get("scene_id", "scene_00"))

    all_track_results: List[Dict] = []

    for _, tm_row in tm.iterrows():
        s_tid = int(tm_row["street_track_id"])
        d_tid = int(tm_row["drone_track_id"])
        cname = track_class.get(s_tid, "unknown")
        if cname not in classes:
            continue

        scene_id = track_scene.get(s_tid, "scene_00")

        # Camera params for this scene
        cfg = cam_cfg_all.get(scene_id, cam_cfg_all.get("default", DEFAULT_CAM))
        K, R, cam_h = build_ipm(cfg)

        # Alignment params
        align_params = align_all.get(scene_id, {
            "offset_x": 0.0, "offset_y": 0.0, "rot_deg": 0.0
        })

        # Scene-specific subsets
        s_scene = street[street["scene_id"] == scene_id]
        d_scene = drone[drone["scene_id"] == scene_id]

        result = compute_track_errors(
            s_scene, d_scene, K, R, cam_h,
            align_params, s_tid, d_tid, cname
        )
        if result is not None:
            result["scene_id"] = scene_id
            all_track_results.append(result)

    print(f"[info] Evaluated {len(all_track_results)} matched track pairs.")

    # Aggregate
    agg = aggregate(all_track_results, classes)
    print_summary(agg, classes)

    # Output report
    report = {"aggregated": agg, "per_track": all_track_results}
    if args.out_report:
        with open(args.out_report, "w") as f:
            def jsonify(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return obj
            json.dump(report, f, indent=2, default=jsonify)
        print(f"[done] report: {args.out_report}")

    # Output flat CSV (no per-frame details)
    if args.out_csv:
        flat = [
            {k: v for k, v in t.items() if k != "frame_errors"}
            for t in all_track_results
        ]
        pd.DataFrame(flat).to_csv(args.out_csv, index=False)
        print(f"[done] per-track CSV: {args.out_csv}")


if __name__ == "__main__":
    main()