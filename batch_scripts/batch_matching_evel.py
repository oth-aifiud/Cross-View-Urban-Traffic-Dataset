#!/usr/bin/env python3
import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("\n[cmd]", " ".join(shlex.quote(str(x)) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def safe_read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def infer_scene_root(row: pd.Series) -> Path:
    street_video = Path(str(row["street_video"]).strip())
    return street_video.parent


def to_float(x):
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
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
        if isinstance(x, str) and x.strip() == "":
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except Exception:
        return None


def ratio(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def f1_from_pr(p, r):
    if p is None or r is None or (p + r) == 0:
        return None
    return 2.0 * p * r / (p + r)


def nanmean(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return None
    return float(s.mean())


def extract_metrics(scene_id: str, report: dict) -> dict:
    out = {"scene_id": scene_id}

    # Preferred schema: flat summary block written by the per-scene evaluator
    summary = report.get("summary", {})

    # Backward-compatible fallback to older nested schemas
    tl = report.get("track_level", {})
    fl = report.get("frame_level", report.get("frame_benchmark", {}))
    st = report.get("stability", report.get("stability_vs_gt", {}))
    cs = report.get("consistency", {})
    nf = report.get("near_far", report.get("near_far_breakdown", {}))
    pc = report.get("per_class", {})

    if not pc and isinstance(tl, dict):
        pc = tl.get("per_class", {})

    # If the report already contains a flat summary, use it directly
    if summary:
        out.update({
            "track_precision": to_float(summary.get("track_precision")),
            "track_recall": to_float(summary.get("track_recall")),
            "track_f1": to_float(summary.get("track_f1")),
            "track_tp": to_int(summary.get("track_tp")),
            "track_fp": to_int(summary.get("track_fp")),
            "track_fn": to_int(summary.get("track_fn")),
            "mean_score_tp": to_float(summary.get("mean_score_tp")),
            "mean_conf_tp": to_float(summary.get("mean_conf_tp")),

            "idp": to_float(summary.get("idp")),
            "idr": to_float(summary.get("idr")),
            "idf1": to_float(summary.get("idf1")),
            "frame_acc": to_float(summary.get("frame_acc")),
            "gt_visible_frames": to_int(summary.get("gt_visible_frames")),
            "frame_idtp": to_int(summary.get("frame_idtp")),
            "frame_idfp": to_int(summary.get("frame_idfp")),
            "frame_idfn": to_int(summary.get("frame_idfn")),
            "frame_correct_assignments": to_int(summary.get("frame_correct_assignments")),

            "near_f1": to_float(summary.get("near_f1")),
            "far_f1": to_float(summary.get("far_f1")),
            "near_idtp": to_int(summary.get("near_idtp")),
            "near_idfp": to_int(summary.get("near_idfp")),
            "near_idfn": to_int(summary.get("near_idfn")),
            "far_idtp": to_int(summary.get("far_idtp")),
            "far_idfp": to_int(summary.get("far_idfp")),
            "far_idfn": to_int(summary.get("far_idfn")),

            "stability_mean": to_float(summary.get("stability_mean")),
            "stability_median": to_float(summary.get("stability_median")),
            "id_switches_mean": to_float(summary.get("id_switches_mean")),
            "stability_track_count": to_int(summary.get("stability_track_count")),
            "stability_sum": to_float(summary.get("stability_sum")),

            "consistency_mean": to_float(summary.get("consistency_mean")),
            "consistency_median": to_float(summary.get("consistency_median")),
            "consistency_perfect_num": to_int(summary.get("consistency_perfect_num")),
            "consistency_perfect_den": to_int(summary.get("consistency_perfect_den")),
            "consistency_track_count": to_int(summary.get("consistency_track_count")),
            "consistency_sum": to_float(summary.get("consistency_sum")),
        })

        for cname in ["car", "truck", "bus", "person", "bicycle", "motorcycle"]:
            out[f"{cname}_p"] = to_float(summary.get(f"{cname}_p"))
            out[f"{cname}_r"] = to_float(summary.get(f"{cname}_r"))
            out[f"{cname}_f1"] = to_float(summary.get(f"{cname}_f1"))
            out[f"{cname}_tp"] = to_int(summary.get(f"{cname}_tp"))
            out[f"{cname}_fp"] = to_int(summary.get(f"{cname}_fp"))
            out[f"{cname}_fn"] = to_int(summary.get(f"{cname}_fn"))

        return out

    # Track-level fallback
    out.update({
        "track_precision": to_float(tl.get("precision")),
        "track_recall": to_float(tl.get("recall")),
        "track_f1": to_float(tl.get("f1")),
        "track_tp": to_int(tl.get("tp")),
        "track_fp": to_int(tl.get("fp")),
        "track_fn": to_int(tl.get("fn")),
        "mean_score_tp": to_float(tl.get("mean_score_tp")),
        "mean_conf_tp": to_float(tl.get("mean_conf_tp")),
    })

    # Frame-level fallback (older schema)
    fl_overall = fl.get("overall", {}) if isinstance(fl, dict) else {}
    out.update({
        "idp": to_float(fl_overall.get("id_precision", fl.get("idp"))),
        "idr": to_float(fl_overall.get("id_recall", fl.get("idr"))),
        "idf1": to_float(fl_overall.get("id_f1", fl.get("idf1"))),
        "frame_acc": to_float(fl_overall.get("frame_acc", fl.get("frame_accuracy"))),
        "gt_visible_frames": to_int(fl_overall.get("n_gt_visible", fl.get("gt_visible_frames"))),
        "frame_idtp": to_int(fl_overall.get("IDTP", fl.get("idtp"))),
        "frame_idfp": to_int(fl_overall.get("IDFP", fl.get("idfp"))),
        "frame_idfn": to_int(fl_overall.get("IDFN", fl.get("idfn"))),
        "frame_correct_assignments": to_int(fl.get("correct_assignments")),
    })

    # Near/far fallback (older schema)
    nf_near = nf.get("near", {}) if isinstance(nf, dict) else {}
    nf_far = nf.get("far", {}) if isinstance(nf, dict) else {}
    out.update({
        "near_f1": to_float(nf_near.get("id_f1", nf.get("near_f1"))),
        "far_f1": to_float(nf_far.get("id_f1", nf.get("far_f1"))),
        "near_idtp": to_int(nf_near.get("IDTP", nf.get("near_idtp"))),
        "near_idfp": to_int(nf_near.get("IDFP", nf.get("near_idfp"))),
        "near_idfn": to_int(nf_near.get("IDFN", nf.get("near_idfn"))),
        "far_idtp": to_int(nf_far.get("IDTP", nf.get("far_idtp"))),
        "far_idfp": to_int(nf_far.get("IDFP", nf.get("far_idfp"))),
        "far_idfn": to_int(nf_far.get("IDFN", nf.get("far_idfn"))),
    })

    # Temporal fallback (older schema)
    out.update({
        "stability_mean": to_float(st.get("mean_stability", st.get("mean"))),
        "stability_median": to_float(st.get("median_stability", st.get("median"))),
        "id_switches_mean": to_float(st.get("mean_id_switches")),
        "stability_track_count": to_int(st.get("tracks", st.get("track_count"))),
        "stability_sum": to_float(st.get("stability_sum", st.get("sum"))),

        "consistency_mean": to_float(cs.get("mean_consistency", cs.get("mean"))),
        "consistency_median": to_float(cs.get("median_consistency", cs.get("median"))),
        "consistency_perfect_num": to_int(cs.get("tracks_perfect_consistency", cs.get("perfect_num"))),
        "consistency_perfect_den": to_int(cs.get("tracks_evaluated", cs.get("perfect_den"))),
        "consistency_track_count": to_int(cs.get("tracks_evaluated", cs.get("track_count"))),
        "consistency_sum": to_float(cs.get("consistency_sum", cs.get("sum"))),
    })

    # Per-class fallback
    for cname in ["car", "truck", "bus", "person", "bicycle", "motorcycle"]:
        c = pc.get(cname, {}) if isinstance(pc, dict) else {}
        out[f"{cname}_p"] = to_float(c.get("precision"))
        out[f"{cname}_r"] = to_float(c.get("recall"))
        out[f"{cname}_f1"] = to_float(c.get("f1"))
        out[f"{cname}_tp"] = to_int(c.get("TP", c.get("tp")))
        out[f"{cname}_fp"] = to_int(c.get("FP", c.get("fp")))
        out[f"{cname}_fn"] = to_int(c.get("FN", c.get("fn")))

    return out


def compute_macro(per_scene_df: pd.DataFrame) -> dict:
    return {
        "num_scenes_evaluated": int(len(per_scene_df)),
        "track_precision_macro": nanmean(per_scene_df["track_precision"]),
        "track_recall_macro": nanmean(per_scene_df["track_recall"]),
        "track_f1_macro": nanmean(per_scene_df["track_f1"]),
        "idp_macro": nanmean(per_scene_df["idp"]),
        "idr_macro": nanmean(per_scene_df["idr"]),
        "idf1_macro": nanmean(per_scene_df["idf1"]),
        "frame_acc_macro": nanmean(per_scene_df["frame_acc"]),
        "near_f1_macro": nanmean(per_scene_df["near_f1"]),
        "far_f1_macro": nanmean(per_scene_df["far_f1"]),
        "stability_mean_macro": nanmean(per_scene_df["stability_mean"]),
        "id_switches_mean_macro": nanmean(per_scene_df["id_switches_mean"]),
        "consistency_mean_macro": nanmean(per_scene_df["consistency_mean"]),
        "mean_score_tp_macro": nanmean(per_scene_df["mean_score_tp"]),
        "mean_conf_tp_macro": nanmean(per_scene_df["mean_conf_tp"]),
    }


def compute_micro(per_scene_df: pd.DataFrame) -> dict:
    # Track-level micro
    track_tp = pd.to_numeric(per_scene_df["track_tp"], errors="coerce").fillna(0).sum()
    track_fp = pd.to_numeric(per_scene_df["track_fp"], errors="coerce").fillna(0).sum()
    track_fn = pd.to_numeric(per_scene_df["track_fn"], errors="coerce").fillna(0).sum()

    track_p = ratio(track_tp, track_tp + track_fp)
    track_r = ratio(track_tp, track_tp + track_fn)
    track_f1 = f1_from_pr(track_p, track_r)

    # Frame-level micro
    frame_idtp = pd.to_numeric(per_scene_df["frame_idtp"], errors="coerce").fillna(0).sum()
    frame_idfp = pd.to_numeric(per_scene_df["frame_idfp"], errors="coerce").fillna(0).sum()
    frame_idfn = pd.to_numeric(per_scene_df["frame_idfn"], errors="coerce").fillna(0).sum()
    gt_visible_frames = pd.to_numeric(per_scene_df["gt_visible_frames"], errors="coerce").fillna(0).sum()
    frame_correct_assignments = pd.to_numeric(per_scene_df["frame_correct_assignments"], errors="coerce").fillna(0).sum()

    idp = ratio(frame_idtp, frame_idtp + frame_idfp)
    idr = ratio(frame_idtp, frame_idtp + frame_idfn)
    idf1 = f1_from_pr(idp, idr)
    frame_acc = ratio(frame_correct_assignments, gt_visible_frames)

    # Near/far micro
    near_idtp = pd.to_numeric(per_scene_df["near_idtp"], errors="coerce").fillna(0).sum()
    near_idfp = pd.to_numeric(per_scene_df["near_idfp"], errors="coerce").fillna(0).sum()
    near_idfn = pd.to_numeric(per_scene_df["near_idfn"], errors="coerce").fillna(0).sum()
    far_idtp = pd.to_numeric(per_scene_df["far_idtp"], errors="coerce").fillna(0).sum()
    far_idfp = pd.to_numeric(per_scene_df["far_idfp"], errors="coerce").fillna(0).sum()
    far_idfn = pd.to_numeric(per_scene_df["far_idfn"], errors="coerce").fillna(0).sum()

    near_p = ratio(near_idtp, near_idtp + near_idfp)
    near_r = ratio(near_idtp, near_idtp + near_idfn)
    near_f1 = f1_from_pr(near_p, near_r)

    far_p = ratio(far_idtp, far_idtp + far_idfp)
    far_r = ratio(far_idtp, far_idtp + far_idfn)
    far_f1 = f1_from_pr(far_p, far_r)

    # Temporal micro-like weighted means if sums/counts exist, otherwise fallback to mean of means
    stability_sum = pd.to_numeric(per_scene_df["stability_sum"], errors="coerce").fillna(0).sum()
    stability_track_count = pd.to_numeric(per_scene_df["stability_track_count"], errors="coerce").fillna(0).sum()
    stability_mean = ratio(stability_sum, stability_track_count)
    if stability_mean is None:
        stability_mean = nanmean(per_scene_df["stability_mean"])

    consistency_sum = pd.to_numeric(per_scene_df["consistency_sum"], errors="coerce").fillna(0).sum()
    consistency_track_count = pd.to_numeric(per_scene_df["consistency_track_count"], errors="coerce").fillna(0).sum()
    consistency_mean = ratio(consistency_sum, consistency_track_count)
    if consistency_mean is None:
        consistency_mean = nanmean(per_scene_df["consistency_mean"])

    perfect_num = pd.to_numeric(per_scene_df["consistency_perfect_num"], errors="coerce").fillna(0).sum()
    perfect_den = pd.to_numeric(per_scene_df["consistency_perfect_den"], errors="coerce").fillna(0).sum()
    consistency_perfect_rate = ratio(perfect_num, perfect_den)

    id_switches_mean = nanmean(per_scene_df["id_switches_mean"])

    out = {
        "num_scenes_evaluated": int(len(per_scene_df)),

        "track_tp_micro": int(track_tp),
        "track_fp_micro": int(track_fp),
        "track_fn_micro": int(track_fn),
        "track_precision_micro": track_p,
        "track_recall_micro": track_r,
        "track_f1_micro": track_f1,

        "frame_idtp_micro": int(frame_idtp),
        "frame_idfp_micro": int(frame_idfp),
        "frame_idfn_micro": int(frame_idfn),
        "gt_visible_frames_micro": int(gt_visible_frames),
        "frame_correct_assignments_micro": int(frame_correct_assignments),
        "idp_micro": idp,
        "idr_micro": idr,
        "idf1_micro": idf1,
        "frame_acc_micro": frame_acc,

        "near_idtp_micro": int(near_idtp),
        "near_idfp_micro": int(near_idfp),
        "near_idfn_micro": int(near_idfn),
        "near_f1_micro": near_f1,

        "far_idtp_micro": int(far_idtp),
        "far_idfp_micro": int(far_idfp),
        "far_idfn_micro": int(far_idfn),
        "far_f1_micro": far_f1,

        "stability_mean_micro": stability_mean,
        "id_switches_mean_micro": id_switches_mean,
        "consistency_mean_micro": consistency_mean,
        "consistency_perfect_rate_micro": consistency_perfect_rate,
    }

    # Per-class micro
    for cname in ["car", "truck", "bus", "person", "bicycle", "motorcycle"]:
        tp = pd.to_numeric(per_scene_df[f"{cname}_tp"], errors="coerce").fillna(0).sum()
        fp = pd.to_numeric(per_scene_df[f"{cname}_fp"], errors="coerce").fillna(0).sum()
        fn = pd.to_numeric(per_scene_df[f"{cname}_fn"], errors="coerce").fillna(0).sum()
        p = ratio(tp, tp + fp)
        r = ratio(tp, tp + fn)
        f1 = f1_from_pr(p, r)
        out[f"{cname}_tp_micro"] = int(tp)
        out[f"{cname}_fp_micro"] = int(fp)
        out[f"{cname}_fn_micro"] = int(fn)
        out[f"{cname}_p_micro"] = p
        out[f"{cname}_r_micro"] = r
        out[f"{cname}_f1_micro"] = f1

    return out


def main():
    ap = argparse.ArgumentParser(description="Batch evaluation for all scenes with macro and micro aggregation")
    ap.add_argument("--scene_csv", required=True, help="scene_manifest.csv")
    ap.add_argument("--eval_script", required=True, help="path to eval_matching.py")
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--near_dist_m", type=float, default=30.0)
    ap.add_argument("--out_dir", required=True, help="directory for aggregated outputs")
    ap.add_argument("--only_scene", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scene_csv, skipinitialspace=True)

    required = ["scene_id", "street_video", "gt_pairs_csv"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"scene manifest missing required columns: {missing}")

    rows = []

    for _, row in df.iterrows():
        scene_id = str(row["scene_id"]).strip()
        if args.only_scene and scene_id != args.only_scene:
            continue

        gt_csv = Path(str(row["gt_pairs_csv"]).strip())
        if not gt_csv.exists():
            print(f"[skip] {scene_id}: gt_pairs.csv not found -> {gt_csv}")
            continue

        scene_root = infer_scene_root(row)

        def row_path(colname: str, fallback: Path) -> Path:
            if colname in row.index and pd.notna(row[colname]) and str(row[colname]).strip() != "":
                return Path(str(row[colname]).strip())
            return fallback

        pred_track_csv = row_path(
            "track_mapping_csv",
            scene_root / "outputs" / "wedge_export" / "track_mapping.csv",
        )
        pred_frame_csv = row_path(
            "frame_matches_csv",
            scene_root / "outputs" / "wedge_export" / "frame_matches.csv",
        )
        street_manifest = row_path(
            "street_manifest_csv",
            scene_root / "outputs" / "wedge_export" / "street_wedge_manifest.csv",
        )
        drone_manifest = row_path(
            "drone_manifest_csv",
            scene_root / "outputs" / "wedge_export" / "drone_wedge_manifest.csv",
        )
        out_report = pred_track_csv.parent / "eval_report.json"

        needed = [pred_track_csv, pred_frame_csv, street_manifest, drone_manifest]
        missing_files = [str(p) for p in needed if not p.exists()]
        if missing_files:
            print(f"[skip] {scene_id}: missing files:")
            for p in missing_files:
                print("   ", p)
            continue

        cmd = [
            args.python_bin,
            args.eval_script,
            "--gt_csv", str(gt_csv),
            "--pred_track_csv", str(pred_track_csv),
            "--pred_frame_csv", str(pred_frame_csv),
            "--street_manifest", str(street_manifest),
            "--drone_manifest", str(drone_manifest),
            "--near_dist_m", str(args.near_dist_m),
            "--out_report", str(out_report),
        ]
        run_cmd(cmd, dry_run=args.dry_run)

        if args.dry_run:
            continue

        report = safe_read_json(out_report)
        if report is None:
            print(f"[warn] {scene_id}: report not found after eval -> {out_report}")
            continue

        rows.append(extract_metrics(scene_id, report))

    if args.dry_run:
        print("\n[dry-run] no aggregation written.")
        return

    if not rows:
        print("[done] no scene reports collected.")
        return

    per_scene_df = pd.DataFrame(rows).sort_values("scene_id")
    per_scene_csv = out_dir / "per_scene_eval.csv"
    per_scene_df.to_csv(per_scene_csv, index=False)

    macro = compute_macro(per_scene_df)
    micro = compute_micro(per_scene_df)

    macro_json = out_dir / "overall_eval_macro.json"
    micro_json = out_dir / "overall_eval_micro.json"
    macro_csv = out_dir / "overall_eval_macro.csv"
    micro_csv = out_dir / "overall_eval_micro.csv"

    with open(macro_json, "w") as f:
        json.dump(macro, f, indent=2)
    with open(micro_json, "w") as f:
        json.dump(micro, f, indent=2)

    pd.DataFrame([macro]).to_csv(macro_csv, index=False)
    pd.DataFrame([micro]).to_csv(micro_csv, index=False)

    print("\n[done] wrote:")
    print(" ", per_scene_csv)
    print(" ", macro_json)
    print(" ", macro_csv)
    print(" ", micro_json)
    print(" ", micro_csv)


if __name__ == "__main__":
    main()