#!/usr/bin/env python3
import argparse
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


def main():
    ap = argparse.ArgumentParser(description="Batch auto coordinate alignment over all scenes")
    ap.add_argument("--scene_csv", required=True, help="processed_scene_manifest.csv")
    ap.add_argument("--align_script", required=True, help="path to auto_coord_align.py")
    ap.add_argument("--camera_cfg", required=True, help="camera_params.json")
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--method", default="ipm", choices=["ipm", "depth"])
    ap.add_argument("--img_dir_name", default="frames", help="fallback frames directory name next to manifests")
    ap.add_argument("--only_scene", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    sm = pd.read_csv(args.scene_csv, skipinitialspace=True)

    required = ["scene_id", "street_manifest_csv", "drone_manifest_csv", "frame_matches_csv", "track_mapping_csv"]
    missing = [c for c in required if c not in sm.columns]
    if missing:
        raise RuntimeError(f"scene manifest missing required columns: {missing}")

    for _, r in sm.iterrows():
        sid = str(r["scene_id"]).strip()
        if args.only_scene and sid != args.only_scene:
            continue

        street_manifest = Path(str(r["street_manifest_csv"]).strip())
        drone_manifest = Path(str(r["drone_manifest_csv"]).strip())
        frame_matches_csv = Path(str(r["frame_matches_csv"]).strip())
        track_mapping_csv = Path(str(r["track_mapping_csv"]).strip())

        if "frames_dir" in sm.columns and pd.notna(r.get("frames_dir", None)) and str(r["frames_dir"]).strip() != "":
            img_dir = Path(str(r["frames_dir"]).strip())
        else:
            img_dir = street_manifest.parent / args.img_dir_name

        if "coord_align_csv" in sm.columns and pd.notna(r.get("coord_align_csv", None)) and str(r["coord_align_csv"]).strip() != "":
            out_csv = Path(str(r["coord_align_csv"]).strip())
        else:
            out_csv = track_mapping_csv.parent / "coord_align.csv"

        needed = [street_manifest, drone_manifest, frame_matches_csv, track_mapping_csv, Path(args.camera_cfg)]
        missing_files = [str(p) for p in needed if not p.exists()]
        if missing_files:
            print(f"[skip] {sid}: missing files:")
            for p in missing_files:
                print("   ", p)
            continue

        cmd = [
            args.python_bin,
            args.align_script,
            "--street_manifest", str(street_manifest),
            "--drone_manifest", str(drone_manifest),
            "--frame_matches_csv", str(frame_matches_csv),
            "--track_map_csv", str(track_mapping_csv),
            "--camera_cfg", str(args.camera_cfg),
            "--method", args.method,
            "--out_csv", str(out_csv),
        ]

        if img_dir.exists():
            cmd += ["--img_dir", str(img_dir)]

        run_cmd(cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run] no files written.")


if __name__ == "__main__":
    main()
