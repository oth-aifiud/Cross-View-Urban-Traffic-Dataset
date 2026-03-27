#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n[cmd]", " ".join(shlex.quote(str(x)) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def scene_output_paths(scene_dir: Path) -> dict[str, Path]:
    out_root = scene_dir / "outputs"
    wedge_dir = out_root / "wedge_export"
    ensure_dir(out_root)
    ensure_dir(wedge_dir)

    return {
        "out_root": out_root,
        "wedge_dir": wedge_dir,
        "street_manifest": wedge_dir / "street_wedge_manifest.csv",
        "drone_manifest": wedge_dir / "drone_wedge_manifest.csv",
        "street_emb": wedge_dir / "street_wedge_emb.npz",
        "drone_emb": wedge_dir / "drone_wedge_emb.npz",
        "frame_matches": wedge_dir / "frame_matches.csv",
        "track_mapping": wedge_dir / "track_mapping.csv",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch pipeline: wedge export -> embeddings -> matching")

    ap.add_argument("--scene_csv", required=True, help="CSV with scene_id,street_video,drone_video,street_csv,drone_csv,ego_track_id")
    ap.add_argument("--python_bin", default=sys.executable)

    # script paths
    ap.add_argument("--export_wedge_script", required=True)
    ap.add_argument("--embed_script", required=True)
    ap.add_argument("--match_script", required=True)

    # wedge export params
    ap.add_argument("--fov_deg", type=float, default=120.0)
    ap.add_argument("--max_range", type=float, default=70.0)
    ap.add_argument("--topN_street", type=int, default=20)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1, help="-1 means till video end")
    ap.add_argument("--stride", type=int, default=1)

    # embedding params
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")

    # matching params
    ap.add_argument("--max_per_class", type=int, default=30)
    ap.add_argument("--near_k_vehicle", type=int, default=8)
    ap.add_argument("--near_k_person", type=int, default=4)

    ap.add_argument("--min_clip_near", type=float, default=0.25)
    ap.add_argument("--min_score_near", type=float, default=0.55)
    ap.add_argument("--min_clip_far", type=float, default=0.35)
    ap.add_argument("--min_score_far", type=float, default=0.65)

    ap.add_argument("--far_min_angle_sim", type=float, default=0.75)
    ap.add_argument("--far_max_abs_dtheta_deg", type=float, default=15.0)

    ap.add_argument("--min_conf_near", type=float, default=0.65)
    ap.add_argument("--min_conf_far", type=float, default=0.75)

    ap.add_argument("--margin_m0_near", type=float, default=0.04)
    ap.add_argument("--margin_tau_near", type=float, default=0.02)
    ap.add_argument("--margin_m0_far", type=float, default=0.08)
    ap.add_argument("--margin_tau_far", type=float, default=0.03)

    ap.add_argument("--sticky_bonus_near", type=float, default=0.02)
    ap.add_argument("--sticky_bonus_far", type=float, default=0.06)

    ap.add_argument("--min_votes", type=int, default=5)
    ap.add_argument("--vote_ratio", type=float, default=1.5)
    ap.add_argument("--min_mean_score", type=float, default=0.65)
    ap.add_argument("--min_total_votes", type=int, default=8)
    ap.add_argument("--min_dom_frac", type=float, default=0.65)
    ap.add_argument("--min_best_run", type=int, default=4)

    # execution control
    ap.add_argument("--do_export", action="store_true")
    ap.add_argument("--do_embed", action="store_true")
    ap.add_argument("--do_match", action="store_true")
    ap.add_argument("--only_scene", default=None)
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    if not any([args.do_export, args.do_embed, args.do_match]):
        args.do_export = args.do_embed = args.do_match = True

    df = pd.read_csv(args.scene_csv, skipinitialspace=True)
    required = ["scene_id", "street_video", "drone_video", "street_csv", "drone_csv", "ego_track_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"scene_csv missing columns: {missing}")

    for _, row in df.iterrows():
        scene_id = str(row["scene_id"]).strip()
        if args.only_scene and scene_id != args.only_scene:
            continue

        street_video = Path(str(row["street_video"]).strip())
        drone_video = Path(str(row["drone_video"]).strip())
        street_csv = Path(str(row["street_csv"]).strip())
        drone_csv = Path(str(row["drone_csv"]).strip())
        ego_track_id = int(row["ego_track_id"])

        scene_dir = street_video.parent
        paths = scene_output_paths(scene_dir)

        print("\n" + "=" * 80)
        print(f"[scene] {scene_id}")
        print("=" * 80)

        if args.do_export:
            cmd = [
                args.python_bin,
                args.export_wedge_script,
                "--street_video", str(street_video),
                "--drone_video", str(drone_video),
                "--street_csv", str(street_csv),
                "--drone_csv", str(drone_csv),
                "--out_dir", str(paths["wedge_dir"]),
                "--ego_track_id", str(ego_track_id),
                "--fov_deg", str(args.fov_deg),
                "--max_range", str(args.max_range),
                "--topN_street", str(args.topN_street),
                "--start_frame", str(args.start_frame),
                "--stride", str(args.stride),
            ]
            if args.end_frame >= 0:
                cmd += ["--end_frame", str(args.end_frame)]
            run(cmd, dry_run=args.dry_run)

        if args.do_embed:
            for manifest_path, out_npz, view in [
                (paths["street_manifest"], paths["street_emb"], "street"),
                (paths["drone_manifest"], paths["drone_emb"], "drone"),
            ]:
                cmd = [
                    args.python_bin,
                    args.embed_script,
                    "--manifest_csv", str(manifest_path),
                    "--view", view,
                    "--out_npz", str(out_npz),
                    "--model", args.clip_model,
                    "--batch_size", str(args.batch_size),
                    "--device", args.device,
                ]
                run(cmd, dry_run=args.dry_run)

        if args.do_match:
            cmd = [
                args.python_bin,
                args.match_script,
                "--street_manifest", str(paths["street_manifest"]),
                "--drone_manifest", str(paths["drone_manifest"]),
                "--street_emb_npz", str(paths["street_emb"]),
                "--drone_emb_npz", str(paths["drone_emb"]),
                "--out_frame_csv", str(paths["frame_matches"]),
                "--out_track_map_csv", str(paths["track_mapping"]),
                "--max_per_class", str(args.max_per_class),
                "--near_k_vehicle", str(args.near_k_vehicle),
                "--near_k_person", str(args.near_k_person),
                "--min_clip_near", str(args.min_clip_near),
                "--min_score_near", str(args.min_score_near),
                "--min_clip_far", str(args.min_clip_far),
                "--min_score_far", str(args.min_score_far),
                "--far_min_angle_sim", str(args.far_min_angle_sim),
                "--far_max_abs_dtheta_deg", str(args.far_max_abs_dtheta_deg),
                "--min_conf_near", str(args.min_conf_near),
                "--min_conf_far", str(args.min_conf_far),
                "--margin_m0_near", str(args.margin_m0_near),
                "--margin_tau_near", str(args.margin_tau_near),
                "--margin_m0_far", str(args.margin_m0_far),
                "--margin_tau_far", str(args.margin_tau_far),
                "--sticky_bonus_near", str(args.sticky_bonus_near),
                "--sticky_bonus_far", str(args.sticky_bonus_far),
                "--min_votes", str(args.min_votes),
                "--vote_ratio", str(args.vote_ratio),
                "--min_mean_score", str(args.min_mean_score),
                "--min_total_votes", str(args.min_total_votes),
                "--min_dom_frac", str(args.min_dom_frac),
                "--min_best_run", str(args.min_best_run),
            ]
            run(cmd, dry_run=args.dry_run)

    print("\n[done] pipeline finished up to matching.")


if __name__ == "__main__":
    main()