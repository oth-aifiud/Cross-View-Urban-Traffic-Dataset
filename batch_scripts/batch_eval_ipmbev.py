#!/usr/bin/env python3
import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd, dry_run=False):
    print("\n[cmd]", " ".join(shlex.quote(str(x)) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def safe_read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def to_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def to_int(x):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except Exception:
        return None


def weighted_mean(values, weights):
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        if v is None or w is None:
            continue
        num += float(v) * float(w)
        den += float(w)
    return None if den == 0 else num / den


def extract_scene_metrics(scene_id: str, report: dict):
    summary = report.get("summary", report)
    agg = report.get("aggregated", {})
    overall = agg.get("overall", {}) if isinstance(agg, dict) else {}

    def pick(*names):
        for n in names:
            if isinstance(summary, dict) and n in summary:
                return summary[n]
            if isinstance(overall, dict) and n in overall:
                return overall[n]
        return None

    tracks_evaluated = to_int(pick("tracks_evaluated", "num_tracks", "n_tracks"))
    if tracks_evaluated is None and isinstance(report.get("per_track"), list):
        tracks_evaluated = len(report["per_track"])

    row = {
        "scene_id": scene_id,
        "tracks_evaluated": tracks_evaluated,
        "ade": to_float(pick("ade", "ADE", "ADE_m", "mean_ADE")),
        "fde": to_float(pick("fde", "FDE", "FDE_m", "mean_FDE")),
        "ale": to_float(pick("ale", "ALE", "ALE_m", "mean_ALE")),
        "alge": to_float(pick("alge", "ALgE", "ALgE_m", "mean_ALgE", "longitudinal_error")),
        "pck_1m": to_float(pick("pck_1m", "PCK@1m", "pck1m", "PCK_1m")),
        "pck_2m": to_float(pick("pck_2m", "PCK@2m", "pck2m", "PCK_2m")),
        "iou_05": to_float(pick("bev_iou_05", "iou_05", "IoU@0.5", "BEV-IoU@0.5", "BEV_IoU_50")),
        "iou_025": to_float(pick("bev_iou_025", "iou_025", "IoU@0.25", "BEV-IoU@0.25", "BEV_IoU_25")),
    }

    row["pck_1m_num"] = to_int(pick("pck_1m_num", "pck1m_num"))
    row["pck_1m_den"] = to_int(pick("pck_1m_den", "pck1m_den"))
    row["pck_2m_num"] = to_int(pick("pck_2m_num", "pck2m_num"))
    row["pck_2m_den"] = to_int(pick("pck_2m_den", "pck2m_den"))
    row["iou_05_num"] = to_int(pick("iou_05_num", "bev_iou_05_num"))
    row["iou_05_den"] = to_int(pick("iou_05_den", "bev_iou_05_den"))
    row["iou_025_num"] = to_int(pick("iou_025_num", "bev_iou_025_num"))
    row["iou_025_den"] = to_int(pick("iou_025_den", "bev_iou_025_den"))

    return row


def main():
    ap = argparse.ArgumentParser(description="Batch IPM BEV evaluation over all scenes")
    ap.add_argument("--scene_csv", required=True, help="processed_scene_manifest.csv")
    ap.add_argument("--eval_script", required=True, help="path to eval_ipm_bev.py")
    ap.add_argument("--camera_cfg", required=True, help="camera_params.json")
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--coord_align_name", default="coord_align.csv",
                    help="filename to use if coord_align path is not explicit")
    ap.add_argument("--min_conf", type=float, default=None)
    ap.add_argument("--only_scene", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sm = pd.read_csv(args.scene_csv, skipinitialspace=True)

    required = ["scene_id", "street_manifest_csv", "drone_manifest_csv", "track_mapping_csv"]
    missing = [c for c in required if c not in sm.columns]
    if missing:
        raise RuntimeError(f"scene manifest missing required columns: {missing}")

    rows = []

    for _, r in sm.iterrows():
        sid = str(r["scene_id"]).strip()
        if args.only_scene and sid != args.only_scene:
            continue

        street_manifest = Path(str(r["street_manifest_csv"]).strip())
        drone_manifest = Path(str(r["drone_manifest_csv"]).strip())
        track_map_csv = Path(str(r["track_mapping_csv"]).strip())

        if "coord_align_csv" in sm.columns and pd.notna(r.get("coord_align_csv", None)) and str(r["coord_align_csv"]).strip() != "":
            coord_align_csv = Path(str(r["coord_align_csv"]).strip())
        else:
            coord_align_csv = track_map_csv.parent / args.coord_align_name

        needed = [street_manifest, drone_manifest, track_map_csv, coord_align_csv, Path(args.camera_cfg)]
        missing_files = [str(p) for p in needed if not p.exists()]
        if missing_files:
            print(f"[skip] {sid}: missing files:")
            for p in missing_files:
                print("   ", p)
            continue

        scene_out_dir = out_dir / sid
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        out_report = scene_out_dir / "bev_ipm_eval.json"
        out_csv = scene_out_dir / "bev_ipm_per_track.csv"

        cmd = [
            args.python_bin,
            args.eval_script,
            "--street_manifest", str(street_manifest),
            "--drone_manifest", str(drone_manifest),
            "--track_map_csv", str(track_map_csv),
            "--camera_cfg", str(args.camera_cfg),
            "--coord_align_csv", str(coord_align_csv),
            "--out_report", str(out_report),
            "--out_csv", str(out_csv),
        ]
        if args.min_conf is not None:
            cmd += ["--min_conf", str(args.min_conf)]

        run_cmd(cmd, dry_run=args.dry_run)
        if args.dry_run:
            continue

        report = safe_read_json(out_report)
        if report is None:
            print(f"[warn] {sid}: missing report {out_report}")
            continue

        rows.append(extract_scene_metrics(sid, report))

    if args.dry_run:
        print("\n[dry-run] no outputs written.")
        return

    if not rows:
        print("[done] no scene reports collected.")
        return

    per_scene = pd.DataFrame(rows).sort_values("scene_id")
    per_scene_csv = out_dir / "per_scene_ipm_bev.csv"
    per_scene.to_csv(per_scene_csv, index=False)

    macro = {
        "num_scenes_evaluated": int(len(per_scene)),
        "ade_macro": pd.to_numeric(per_scene["ade"], errors="coerce").mean(),
        "fde_macro": pd.to_numeric(per_scene["fde"], errors="coerce").mean(),
        "ale_macro": pd.to_numeric(per_scene["ale"], errors="coerce").mean(),
        "alge_macro": pd.to_numeric(per_scene["alge"], errors="coerce").mean(),
        "pck_1m_macro": pd.to_numeric(per_scene["pck_1m"], errors="coerce").mean(),
        "pck_2m_macro": pd.to_numeric(per_scene["pck_2m"], errors="coerce").mean(),
        "iou_05_macro": pd.to_numeric(per_scene["iou_05"], errors="coerce").mean(),
        "iou_025_macro": pd.to_numeric(per_scene["iou_025"], errors="coerce").mean(),
    }

    w = pd.to_numeric(per_scene["tracks_evaluated"], errors="coerce").fillna(0).tolist()

    micro = {
        "num_scenes_evaluated": int(len(per_scene)),
        "tracks_evaluated_total": int(pd.to_numeric(per_scene["tracks_evaluated"], errors="coerce").fillna(0).sum()),
        "ade_micro": weighted_mean(per_scene["ade"].tolist(), w),
        "fde_micro": weighted_mean(per_scene["fde"].tolist(), w),
        "ale_micro": weighted_mean(per_scene["ale"].tolist(), w),
        "alge_micro": weighted_mean(per_scene["alge"].tolist(), w),
    }

    for metric in [("pck_1m", "pck_1m_num", "pck_1m_den"),
                   ("pck_2m", "pck_2m_num", "pck_2m_den"),
                   ("iou_05", "iou_05_num", "iou_05_den"),
                   ("iou_025", "iou_025_num", "iou_025_den")]:
        mname, ncol, dcol = metric
        num = pd.to_numeric(per_scene[ncol], errors="coerce").fillna(0).sum()
        den = pd.to_numeric(per_scene[dcol], errors="coerce").fillna(0).sum()
        if den > 0:
            micro[f"{mname}_micro"] = float(num) / float(den)
        else:
            micro[f"{mname}_micro"] = weighted_mean(per_scene[mname].tolist(), w)

    macro_json = out_dir / "overall_ipm_bev_macro.json"
    micro_json = out_dir / "overall_ipm_bev_micro.json"
    macro_csv = out_dir / "overall_ipm_bev_macro.csv"
    micro_csv = out_dir / "overall_ipm_bev_micro.csv"

    with open(macro_json, "w") as f:
        json.dump(macro, f, indent=2)
    with open(micro_json, "w") as f:
        json.dump(micro, f, indent=2)

    pd.DataFrame([macro]).to_csv(macro_csv, index=False)
    pd.DataFrame([micro]).to_csv(micro_csv, index=False)

    print("\n[done] wrote:")
    print(" ", per_scene_csv)
    print(" ", macro_csv)
    print(" ", micro_csv)
    print(" ", macro_json)
    print(" ", micro_json)


if __name__ == "__main__":
    main()
