"""
dataset_stats.py — Dataset Statistics for Paper
================================================
Generates all statistics needed for the dataset paper's
"Dataset Statistics" section and Table 1.

Outputs
-------
  Console: formatted summary table (copy-paste ready for paper)
  JSON:    full statistics for all scenes/intersections
  CSV:     per-intersection breakdown table

Required inputs (CSVs from your annotation pipeline)
---------------------
  --street_manifest   Street detections/tracks CSV
                      Columns: scene_id, frame, track_id, class_name,
                               x1, y1, x2, y2, [area], [bearing_rad]

  --drone_manifest    Drone detections/tracks CSV
                      Columns: scene_id, street_frame, drone_frame,
                               track_id, class_name, dist_m,
                               world_x, world_y, [theta_rad]

  --track_map_csv     Cross-view matched track map CSV (from match_wedge_frames.py)
                      Columns: street_track_id, drone_track_id,
                               stitch_confidence, best_votes, mean_score

  --scene_meta_csv    (Optional) Per-scene metadata CSV
                      Columns: scene_id, intersection_name, location,
                               date, time_of_day, weather, duration_sec,
                               fps_street, fps_drone

  --out_json          Output JSON path
  --out_csv           Output per-scene CSV path

Usage
-----
  python dataset_stats.py \
      --street_manifest /data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold/outputs/wedge_export/street_wedge_manifest.csv \
      --drone_manifest /data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold/outputs/wedge_export/drone_wedge_manifest.csv \
      --track_map_csv /data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold/outputs/wedge_export/track_mapping.csv \
      --scene_meta_csv /data/home/bhp39283/disk/detection-tracking/dataset/scene_manifest.csv \
      --out_json dataset_stats.json \
      --out_csv per_scene_stats.csv

Batch usage
-----------
  python dataset_stats.py \
      --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
      --out_json dataset_stats.json \
      --out_csv per_scene_stats.csv
"""

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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


def bbox_area(row) -> float:
    """Compute bbox area if not already present."""
    try:
        return float(row.get("area", (row["x2"] - row["x1"]) * (row["y2"] - row["y1"])))
    except Exception:
        return float("nan")


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def maybe_read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    path = str(path).strip()
    if path == "" or path.lower() == "nan":
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
# ---------------------------------------------------------------------------
# Per-scene statistics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Batch loading from processed_scene_manifest.csv
# ---------------------------------------------------------------------------

def load_from_scene_manifest(scene_csv: str):
    sm = pd.read_csv(scene_csv, skipinitialspace=True)

    required = ["scene_id", "street_manifest_csv", "drone_manifest_csv", "track_mapping_csv"]
    missing = [c for c in required if c not in sm.columns]
    if missing:
        raise RuntimeError(f"scene manifest missing required columns: {missing}")

    street_parts = []
    drone_parts = []
    track_parts = []
    meta_rows = []

    for _, row in sm.iterrows():
        sid = str(row["scene_id"]).strip()

        street_path = str(row["street_manifest_csv"]).strip()
        drone_path = str(row["drone_manifest_csv"]).strip()
        track_path = str(row["track_mapping_csv"]).strip()

        S = pd.read_csv(street_path)
        D = pd.read_csv(drone_path)
        T = pd.read_csv(track_path)

        S = S.copy()
        D = D.copy()
        T = T.copy()
        S["scene_id"] = sid
        D["scene_id"] = sid

        if "track_id" in S.columns:
            S["track_id_orig"] = S["track_id"]
            S["track_id"] = S["track_id"].apply(lambda x: f"{sid}::{x}")

        if "track_id" in D.columns:
            D["track_id_orig"] = D["track_id"]
            D["track_id"] = D["track_id"].apply(lambda x: f"{sid}::{x}")

        if "street_track_id" in T.columns:
            T["street_track_id_orig"] = T["street_track_id"]
            T["street_track_id"] = T["street_track_id"].apply(lambda x: f"{sid}::{x}")
        if "drone_track_id" in T.columns:
            T["drone_track_id_orig"] = T["drone_track_id"]
            T["drone_track_id"] = T["drone_track_id"].apply(lambda x: -1 if str(x) == "-1" or x == -1 else f"{sid}::{x}")

        street_parts.append(S)
        drone_parts.append(D)
        track_parts.append(T)
        meta_rows.append(row.to_dict())

    street = pd.concat(street_parts, ignore_index=True) if street_parts else pd.DataFrame()
    drone = pd.concat(drone_parts, ignore_index=True) if drone_parts else pd.DataFrame()
    track_map = pd.concat(track_parts, ignore_index=True) if track_parts else pd.DataFrame()
    meta = pd.DataFrame(meta_rows) if meta_rows else None

    return street, drone, track_map, meta


# ---------------------------------------------------------------------------
# Per-scene statistics
# ---------------------------------------------------------------------------

def per_scene_stats(
    street: pd.DataFrame,
    drone: pd.DataFrame,
    track_map: pd.DataFrame,
    meta: Optional[pd.DataFrame],
    classes: List[str],
) -> List[Dict]:
    """Compute stats for each scene_id independently."""
    scenes = sorted(street["scene_id"].unique())
    rows = []

    for sid in scenes:
        S = street[street["scene_id"] == sid]
        D = drone[drone["scene_id"] == sid] if "scene_id" in drone.columns else drone

        # Frames
        street_frames = S["frame"].nunique()
        drone_frames = D["drone_frame"].nunique() if "drone_frame" in D.columns else 0

        # Tracks
        street_tracks = S["track_id"].nunique()
        drone_tracks = D["track_id"].nunique()

        # Matched tracks for this scene
        # In batch mode, track ids may already be scene-safe strings like `scene_xx::123`.
        scene_stids = set(S["track_id"].astype(str).unique())
        scene_tm = track_map[track_map["street_track_id"].astype(str).isin(scene_stids)]
        matched_tracks = (~scene_tm["drone_track_id"].astype(str).isin(["-1", "-1.0", "nan", "None"])).sum()

        # Detections
        total_street_dets = len(S)
        total_drone_dets = len(D)

        # Per-class street track counts
        class_track_counts = {}
        for cname in classes:
            n = S[S["class_name"] == cname]["track_id"].nunique()
            class_track_counts[cname] = int(n)

        # Density: detections per frame
        street_det_per_frame = round(total_street_dets / max(1, street_frames), 2)
        drone_det_per_frame = round(total_drone_dets / max(1, drone_frames), 2)

        # Drone distance stats (if available)
        dist_stats = {}
        if "dist_m" in D.columns:
            dists = D["dist_m"].dropna()
            if len(dists):
                dist_stats = {
                    "min_dist_m": round(float(dists.min()), 1),
                    "mean_dist_m": round(float(dists.mean()), 1),
                    "max_dist_m": round(float(dists.max()), 1),
                    "p50_dist_m": round(float(np.percentile(dists, 50)), 1),
                }

        # Track length stats (frames per track)
        track_lens = S.groupby("track_id")["frame"].nunique()
        tl_stats = {}
        if len(track_lens):
            tl_stats = {
                "mean_track_len_frames": round(float(track_lens.mean()), 1),
                "median_track_len_frames": round(float(track_lens.median()), 1),
                "min_track_len_frames": int(track_lens.min()),
                "max_track_len_frames": int(track_lens.max()),
            }

        # Scene metadata
        meta_row = {}
        if meta is not None and "scene_id" in meta.columns:
            m = meta[meta["scene_id"] == sid]
            if len(m):
                meta_row = m.iloc[0].to_dict()
                meta_row.pop("scene_id", None)

        row = {
            "scene_id": sid,
            "street_frames": street_frames,
            "drone_frames": drone_frames,
            "street_tracks": street_tracks,
            "drone_tracks": drone_tracks,
            "matched_cross_view_tracks": int(matched_tracks),
            "match_rate": round(safe_div(matched_tracks, street_tracks), 3),
            "street_detections": total_street_dets,
            "drone_detections": total_drone_dets,
            "street_det_per_frame": street_det_per_frame,
            "drone_det_per_frame": drone_det_per_frame,
            "per_class_street_tracks": class_track_counts,
            **dist_stats,
            **tl_stats,
            **{k: v for k, v in meta_row.items() if not isinstance(v, float) or not np.isnan(v)},
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------

def global_stats(
    street: pd.DataFrame,
    drone: pd.DataFrame,
    track_map: pd.DataFrame,
    scene_rows: List[Dict],
    classes: List[str],
) -> Dict:

    total_street_frames = int(sum(r.get("street_frames", 0) for r in scene_rows))
    total_drone_frames = int(sum(r.get("drone_frames", 0) for r in scene_rows))

    total_street_tracks = int(sum(r.get("street_tracks", 0) for r in scene_rows))
    total_drone_tracks = int(sum(r.get("drone_tracks", 0) for r in scene_rows))
    total_matched = int(sum(r.get("matched_cross_view_tracks", 0) for r in scene_rows))
    match_rate = safe_div(total_matched, total_street_tracks)

    # Stitch confidence distribution
    conf_stats = {}
    valid_tm = track_map[track_map["drone_track_id"] != -1]
    if "stitch_confidence" in valid_tm.columns and len(valid_tm):
        conf = valid_tm["stitch_confidence"].dropna()
        conf_stats = {
            "mean_stitch_conf": round(float(conf.mean()), 3),
            "median_stitch_conf": round(float(conf.median()), 3),
            "p90_stitch_conf": round(float(np.percentile(conf, 90)), 3),
        }

    # Per-class global track counts
    per_class = {}
    for cname in classes:
        s_n = street[street["class_name"] == cname]["track_id"].nunique()
        d_n = drone[drone["class_name"] == cname]["track_id"].nunique() \
            if "class_name" in drone.columns else 0
        per_class[cname] = {
            "street_tracks": int(s_n),
            "drone_tracks": int(d_n),
        }

    # Annotation density
    total_street_dets = len(street)
    total_drone_dets = len(drone)
    avg_density_street = round(total_street_dets / max(1, total_street_frames), 2)
    avg_density_drone = round(total_drone_dets / max(1, total_drone_frames), 2)

    # Scene-level meta summary
    num_scenes = len(scene_rows)
    conditions_summary: Dict = {}
    for key in ["weather", "time_of_day", "location"]:
        vals = [r[key] for r in scene_rows if key in r and r[key]]
        if vals:
            from collections import Counter
            conditions_summary[key] = dict(Counter(vals))

    return {
        "num_scenes": num_scenes,
        "total_street_frames": total_street_frames,
        "total_drone_frames": total_drone_frames,
        "total_street_tracks": int(total_street_tracks),
        "total_drone_tracks": int(total_drone_tracks),
        "total_matched_cross_view_tracks": int(total_matched),
        "overall_match_rate": round(match_rate, 3),
        "total_street_detections": int(total_street_dets),
        "total_drone_detections": int(total_drone_dets),
        "avg_street_det_per_frame": avg_density_street,
        "avg_drone_det_per_frame": avg_density_drone,
        "stitch_confidence": conf_stats,
        "per_class": per_class,
        "conditions": conditions_summary,
    }


def safe_div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


# ---------------------------------------------------------------------------
# Print summary for paper
# ---------------------------------------------------------------------------

def print_summary(g: Dict, scene_rows: List[Dict], classes: List[str]):
    print()
    print("=" * 65)
    print("  DATASET STATISTICS SUMMARY")
    print("=" * 65)
    print(f"  Scenes (intersections)     : {g['num_scenes']}")
    print(f"  Street frames (total)      : {g['total_street_frames']:,}")
    print(f"  Drone frames (total)       : {g['total_drone_frames']:,}")
    print(f"  Street tracks              : {g['total_street_tracks']:,}")
    print(f"  Drone tracks               : {g['total_drone_tracks']:,}")
    print(f"  Matched cross-view tracks  : {g['total_matched_cross_view_tracks']:,}")
    print(f"  Overall match rate         : {g['overall_match_rate']:.1%}")
    print(f"  Street detections          : {g['total_street_detections']:,}")
    print(f"  Drone detections           : {g['total_drone_detections']:,}")
    print(f"  Avg street det/frame       : {g['avg_street_det_per_frame']}")
    print(f"  Avg drone det/frame        : {g['avg_drone_det_per_frame']}")

    if g.get("stitch_confidence"):
        sc = g["stitch_confidence"]
        print(f"  Mean stitch confidence     : {sc.get('mean_stitch_conf', 'n/a')}")

    print()
    print("  Class breakdown (street tracks):")
    print(f"  {'Class':<14} {'Street':>8} {'Drone':>8}")
    print("  " + "-" * 32)
    for cname in classes:
        pc = g["per_class"].get(cname, {})
        print(f"  {cname:<14} {pc.get('street_tracks', 0):>8,} {pc.get('drone_tracks', 0):>8,}")

    if g.get("conditions"):
        print()
        for key, val in g["conditions"].items():
            print(f"  {key.capitalize():<14}: {', '.join(f'{k}({v})' for k, v in val.items())}")

    print()
    print("  Per-scene breakdown:")
    print(f"  {'Scene':<16} {'S.Frames':>9} {'S.Tracks':>9} {'D.Tracks':>9} {'Matched':>8} {'Rate':>6}")
    print("  " + "-" * 62)
    for r in scene_rows:
        print(f"  {str(r['scene_id']):<16} {r['street_frames']:>9,} {r['street_tracks']:>9,} "
              f"{r['drone_tracks']:>9,} {r['matched_cross_view_tracks']:>8,} "
              f"{r.get('match_rate', 0):>5.1%}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate dataset statistics for paper.")
    ap.add_argument("--scene_csv", default=None,
                    help="Processed scene manifest CSV with columns: scene_id, street_manifest_csv, drone_manifest_csv, track_mapping_csv")
    ap.add_argument("--street_manifest", default=None,
                    help="Street detections CSV: scene_id, frame, track_id, class_name, x1, y1, x2, y2")
    ap.add_argument("--drone_manifest", default=None,
                    help="Drone detections CSV: scene_id, drone_frame, track_id, class_name, dist_m, world_x, world_y")
    ap.add_argument("--track_map_csv", default=None,
                    help="Cross-view track map CSV from match_wedge_frames.py")
    ap.add_argument("--scene_meta_csv", default=None,
                    help="(Optional) Per-scene metadata: scene_id, intersection_name, weather, time_of_day, ...")
    ap.add_argument("--out_json", default=None, help="Output JSON path")
    ap.add_argument("--out_csv", default=None, help="Output per-scene stats CSV path")
    ap.add_argument("--classes", type=str,
                    default="car,truck,bus,person,bicycle,motorcycle")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # Load
    if args.scene_csv:
        street, drone, track_map, meta = load_from_scene_manifest(args.scene_csv)
    else:
        if not args.street_manifest or not args.drone_manifest or not args.track_map_csv:
            raise RuntimeError("Either --scene_csv or all of --street_manifest, --drone_manifest, --track_map_csv must be provided.")
        street = pd.read_csv(args.street_manifest)
        drone = pd.read_csv(args.drone_manifest)
        track_map = pd.read_csv(args.track_map_csv)
        meta = pd.read_csv(args.scene_meta_csv) if args.scene_meta_csv else None

    street["class_name"] = street["class_name"].apply(norm_class)
    if "class_name" in drone.columns:
        drone["class_name"] = drone["class_name"].apply(norm_class)

    # Ensure scene_id exists
    if "scene_id" not in street.columns:
        print("[warn] 'scene_id' not found in street manifest — treating as single scene")
        street["scene_id"] = "scene_00"
    if "scene_id" not in drone.columns:
        drone["scene_id"] = "scene_00"

    # Compute
    scene_rows = per_scene_stats(street, drone, track_map, meta, classes)
    g = global_stats(street, drone, track_map, scene_rows, classes)

    # Print
    print_summary(g, scene_rows, classes)

    # Save JSON
    report = {"global": g, "per_scene": scene_rows}
    if args.out_json:
        with open(args.out_json, "w") as f:
            def jsonify(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return obj
            json.dump(report, f, indent=2, default=jsonify)
        print(f"[done] JSON report: {args.out_json}")

    # Save per-scene CSV (flatten per_class dict for tabular format)
    if args.out_csv:
        flat_rows = []
        for r in scene_rows:
            flat = {k: v for k, v in r.items() if k != "per_class_street_tracks"}
            for cname, count in r.get("per_class_street_tracks", {}).items():
                flat[f"tracks_{cname}"] = count
            flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(args.out_csv, index=False)
        print(f"[done] per-scene CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
