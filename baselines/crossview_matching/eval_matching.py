"""
eval_matching.py — Cross-View ID Matching Evaluation
=====================================================
Evaluates the output of match_wedge_frames.py against ground-truth
track-ID correspondences (human-verified or annotation-derived).

Metrics reported
----------------
  Per-class and overall:
    Precision   = TP / (TP + FP)
    Recall      = TP / (TP + FN)
    F1          = harmonic mean of P / R

  Track-level:
    Track Precision / Track Recall / Track F1
    Consistency    = fraction of matched frames that agree with voted ID
    Mean Match Score (MMS) — average score of TP frame matches

  Frame-level:
    Frame-level P/R/F1 (each frame pair treated independently)

  Unmatched analysis:
    Unmatched street tracks (FN)
    Spurious drone matches (FP)

Inputs
------
  --gt_csv          Ground-truth track map CSV with columns:
                      street_track_id, drone_track_id, class_name
                    (one row per verified correspondence)

  --pred_track_csv  Output of match_wedge_frames.py: out_track_map_csv
                    Columns: street_track_id, drone_track_id,
                             stitch_confidence, best_votes, total_votes,
                             vote_ratio, mean_score, second_best_votes

  --pred_frame_csv  Output of match_wedge_frames.py: out_frame_csv
                    Columns: street_frame, drone_frame, class_name,
                             street_track_id, drone_track_id, score,
                             clip_cos, angle_sim, rank_sim, ...

  --street_manifest (optional) street detections CSV — used to count
                    total street tracks per class for recall denominator

  --out_report      Path to write the text/JSON report (default: stdout)
  --min_conf        Minimum stitch_confidence to consider a prediction
                    (default: 0.0, i.e. use all rows with drone_track_id != -1)
  --classes         Comma-separated list of classes to evaluate
                    (default: car,truck,bus,person,bicycle,motorcycle)

Usage
-----
  python eval_matching.py \
      --gt_csv gt_track_map.csv \
      --pred_track_csv pred_track_map.csv \
      --pred_frame_csv pred_frame_matches.csv \
      --out_report eval_report.json
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def f1(p: float, r: float) -> float:
    return safe_div(2 * p * r, p + r)


# ---------------------------------------------------------------------------
# Track-level evaluation
# ---------------------------------------------------------------------------

def eval_tracks(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    classes: List[str],
    min_conf: float = 0.0,
) -> Dict:
    """
    gt   : columns [street_track_id, drone_track_id, class_name]
    pred : columns [street_track_id, drone_track_id, stitch_confidence,
                    best_votes, mean_score, ...]
    """
    # Filter predictions by confidence and valid drone id
    pred = pred[pred["drone_track_id"] != -1].copy()
    if "stitch_confidence" in pred.columns:
        pred = pred[pred["stitch_confidence"] >= min_conf]

    # Build lookup dicts
    # gt_map:   street_tid -> drone_tid  (ground truth)
    # pred_map: street_tid -> drone_tid  (prediction)
    gt_map: Dict[int, int] = {}
    gt_class: Dict[int, str] = {}
    for _, row in gt.iterrows():
        s = int(row["street_track_id"])
        d = int(row["drone_track_id"])
        gt_map[s] = d
        if "class_name" in gt.columns:
            gt_class[s] = norm_class(row["class_name"])

    pred_map: Dict[int, int] = {}
    pred_conf: Dict[int, float] = {}
    pred_score: Dict[int, float] = {}
    for _, row in pred.iterrows():
        s = int(row["street_track_id"])
        pred_map[s] = int(row["drone_track_id"])
        pred_conf[s] = float(row.get("stitch_confidence", 1.0))
        pred_score[s] = float(row.get("mean_score", 0.0))

    # All street track IDs present in GT
    gt_ids = set(gt_map.keys())
    # All predicted IDs (that are accepted)
    pred_ids = set(pred_map.keys())

    overall = {
        "TP": 0, "FP": 0, "FN": 0,
        "TP_ids": [], "FP_ids": [], "FN_ids": [],
        "mean_conf_TP": [], "mean_score_TP": [],
    }
    per_class: Dict[str, Dict] = {c: {"TP": 0, "FP": 0, "FN": 0,
                                      "mean_score_TP": []} for c in classes}
    per_class["__other__"] = {"TP": 0, "FP": 0, "FN": 0, "mean_score_TP": []}

    for s_tid in gt_ids:
        gt_d = gt_map[s_tid]
        cname = gt_class.get(s_tid, "__other__")
        if cname not in per_class:
            cname = "__other__"

        if s_tid in pred_map:
            if pred_map[s_tid] == gt_d:
                # Correct match
                overall["TP"] += 1
                overall["TP_ids"].append(s_tid)
                overall["mean_conf_TP"].append(pred_conf.get(s_tid, 1.0))
                overall["mean_score_TP"].append(pred_score.get(s_tid, 0.0))
                per_class[cname]["TP"] += 1
                per_class[cname]["mean_score_TP"].append(pred_score.get(s_tid, 0.0))
            else:
                # Wrong drone ID predicted
                overall["FP"] += 1
                overall["FP_ids"].append((s_tid, pred_map[s_tid], gt_d))
                per_class[cname]["FP"] += 1
        else:
            # Not predicted at all
            overall["FN"] += 1
            overall["FN_ids"].append(s_tid)
            per_class[cname]["FN"] += 1

    # Spurious predictions (predicted but not in GT)
    spurious = pred_ids - gt_ids
    overall["FP"] += len(spurious)
    overall["spurious_ids"] = list(spurious)

    # Compute metrics
    def metrics(tp, fp, fn, scores):
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        return {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1(p, r), 4),
            "mean_score_TP": round(float(np.mean(scores)), 4) if scores else 0.0,
        }

    result = {}
    tp, fp, fn = overall["TP"], overall["FP"], overall["FN"]
    result["overall"] = metrics(tp, fp, fn, overall["mean_score_TP"])
    result["overall"]["mean_conf_TP"] = round(
        float(np.mean(overall["mean_conf_TP"])) if overall["mean_conf_TP"] else 0.0, 4)
    result["overall"]["unmatched_street_tracks"] = overall["FN_ids"]
    result["overall"]["wrong_drone_predictions"] = [(s, p, g) for s, p, g in overall["FP_ids"]]
    result["overall"]["spurious_predictions"] = overall["spurious_ids"]

    result["per_class"] = {}
    for cname in classes:
        pc = per_class[cname]
        result["per_class"][cname] = metrics(
            pc["TP"], pc["FP"], pc["FN"], pc["mean_score_TP"])

    return result


# ---------------------------------------------------------------------------
# Frame-level evaluation
# ---------------------------------------------------------------------------

def eval_frames(
    gt: pd.DataFrame,
    pred_frames: pd.DataFrame,
    classes: List[str],
) -> Dict:
    """
    Frame-level: each (street_frame, street_track_id, drone_track_id) triple
    is evaluated independently.

    gt columns:    street_track_id, drone_track_id
    pred columns:  street_frame, street_track_id, drone_track_id, score, class_name
    """
    gt_map: Dict[int, int] = {
        int(r["street_track_id"]): int(r["drone_track_id"])
        for _, r in gt.iterrows()
    }

    tp_frames = fp_frames = fn_frames = 0
    scores_tp: List[float] = []

    # For each predicted frame-match, check if it aligns with GT track map
    for _, row in pred_frames.iterrows():
        s_tid = int(row["street_track_id"])
        d_tid = int(row["drone_track_id"])
        score = float(row.get("score", 0.0))

        if s_tid not in gt_map:
            # No GT for this track — skip (can't penalize)
            continue

        if gt_map[s_tid] == d_tid:
            tp_frames += 1
            scores_tp.append(score)
        else:
            fp_frames += 1

    # FN frames: GT tracks that never appear in pred_frames
    pred_stids = set(pred_frames["street_track_id"].astype(int).unique())
    for s_tid in gt_map:
        if s_tid not in pred_stids:
            fn_frames += 1  # whole track never matched

    p = safe_div(tp_frames, tp_frames + fp_frames)
    r = safe_div(tp_frames, tp_frames + fn_frames)

    return {
        "TP_frames": tp_frames,
        "FP_frames": fp_frames,
        "FN_tracks_never_seen": fn_frames,
        "frame_precision": round(p, 4),
        "frame_recall": round(r, 4),
        "frame_f1": round(f1(p, r), 4),
        "mean_score_TP_frames": round(float(np.mean(scores_tp)) if scores_tp else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Consistency: how often does a track's frame matches agree with its voted ID
# ---------------------------------------------------------------------------

def eval_consistency(pred_frames: pd.DataFrame, pred_tracks: pd.DataFrame) -> Dict:
    """
    For each street track that has a valid voted drone_track_id,
    compute what fraction of its frame-level matches agree with that vote.
    """
    voted = pred_tracks[pred_tracks["drone_track_id"] != -1]
    voted_map = {int(r["street_track_id"]): int(r["drone_track_id"])
                 for _, r in voted.iterrows()}

    consistencies: List[float] = []
    per_track: List[Dict] = []

    for s_tid, voted_d in voted_map.items():
        frames_for_track = pred_frames[pred_frames["street_track_id"] == s_tid]
        if len(frames_for_track) == 0:
            continue
        agree = (frames_for_track["drone_track_id"].astype(int) == voted_d).sum()
        total = len(frames_for_track)
        c = safe_div(agree, total)
        consistencies.append(c)
        per_track.append({
            "street_track_id": s_tid,
            "voted_drone_id": voted_d,
            "consistent_frames": int(agree),
            "total_frames": total,
            "consistency": round(c, 4),
        })

    return {
        "mean_consistency": round(float(np.mean(consistencies)) if consistencies else 0.0, 4),
        "median_consistency": round(float(np.median(consistencies)) if consistencies else 0.0, 4),
        "tracks_evaluated": len(consistencies),
        "tracks_perfect_consistency": sum(1 for c in consistencies if c == 1.0),
        "per_track": per_track,
    }


# ---------------------------------------------------------------------------
# Score distribution analysis
# ---------------------------------------------------------------------------

def score_distribution(pred_frames: pd.DataFrame, gt_map: Dict[int, int]) -> Dict:
    """Analyze score distributions for TP vs FP frame matches."""
    tp_scores, fp_scores = [], []
    for _, row in pred_frames.iterrows():
        s_tid = int(row["street_track_id"])
        d_tid = int(row["drone_track_id"])
        score = float(row.get("score", 0.0))
        if s_tid not in gt_map:
            continue
        if gt_map[s_tid] == d_tid:
            tp_scores.append(score)
        else:
            fp_scores.append(score)

    def stats(arr):
        if not arr:
            return {}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "median": round(float(np.median(arr)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "max": round(float(np.max(arr)), 4),
            "n": len(arr),
        }

    return {
        "TP_score_dist": stats(tp_scores),
        "FP_score_dist": stats(fp_scores),
        "separability_delta_mean": round(
            float(np.mean(tp_scores) - np.mean(fp_scores))
            if tp_scores and fp_scores else 0.0, 4
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate cross-view track matching against ground truth."
    )
    ap.add_argument("--gt_csv", required=True,
                    help="Ground-truth track map CSV: street_track_id, drone_track_id[, class_name]")
    ap.add_argument("--pred_track_csv", required=True,
                    help="Predicted track map (out_track_map_csv from match_wedge_frames.py)")
    ap.add_argument("--pred_frame_csv", required=True,
                    help="Predicted frame matches (out_frame_csv from match_wedge_frames.py)")
    ap.add_argument("--street_manifest", default=None,
                    help="(Optional) Street detections CSV to count total unique tracks per class")
    ap.add_argument("--out_report", default=None,
                    help="Output JSON report path (default: print to stdout)")
    ap.add_argument("--min_conf", type=float, default=0.0,
                    help="Minimum stitch_confidence to accept a prediction (default: 0.0)")
    ap.add_argument("--classes", type=str,
                    default="car,truck,bus,person,bicycle,motorcycle",
                    help="Comma-separated class names to evaluate")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # Load data
    gt = pd.read_csv(args.gt_csv)
    if "class_name" in gt.columns:
        gt["class_name"] = gt["class_name"].apply(norm_class)

    pred_tracks = pd.read_csv(args.pred_track_csv)
    pred_frames = pd.read_csv(args.pred_frame_csv)
    pred_frames["class_name"] = pred_frames["class_name"].apply(norm_class)

    gt_map: Dict[int, int] = {
        int(r["street_track_id"]): int(r["drone_track_id"])
        for _, r in gt.iterrows()
    }

    # ---- Track-level ----
    track_results = eval_tracks(gt, pred_tracks, classes, min_conf=args.min_conf)

    # ---- Frame-level ----
    frame_results = eval_frames(gt, pred_frames, classes)

    # ---- Consistency ----
    consistency_results = eval_consistency(pred_frames, pred_tracks)

    # ---- Score distributions ----
    score_dist = score_distribution(pred_frames, gt_map)

    # ---- Near vs Far pass breakdown ----
    near_far: Dict = {}
    if "match_pass" in pred_frames.columns:
        for pass_name in ["near", "far"]:
            pf = pred_frames[pred_frames["match_pass"] == pass_name]
            tp = sum(
                1 for _, r in pf.iterrows()
                if int(r["street_track_id"]) in gt_map
                and gt_map[int(r["street_track_id"])] == int(r["drone_track_id"])
            )
            total = len(pf)
            near_far[pass_name] = {
                "total_predictions": total,
                "TP": tp,
                "precision": round(safe_div(tp, total), 4),
                "mean_score": round(float(pf["score"].mean()) if len(pf) else 0.0, 4),
            }

    # ---- Optional: total track counts from manifest ----
    total_tracks_per_class: Dict = {}
    if args.street_manifest:
        sm = pd.read_csv(args.street_manifest)
        sm["class_name"] = sm["class_name"].apply(norm_class)
        for cname in classes:
            n = sm[sm["class_name"] == cname]["track_id"].nunique()
            total_tracks_per_class[cname] = int(n)

    # ---- Assemble report ----
    report = {
        "config": {
            "min_conf": args.min_conf,
            "classes": classes,
        },
        "track_level": track_results,
        "frame_level": frame_results,
        "consistency": consistency_results,
        "score_distributions": score_dist,
        "near_far_breakdown": near_far,
        "total_tracks_in_manifest": total_tracks_per_class,
    }

    # ---- Print summary table ----
    print("\n" + "=" * 60)
    print("CROSS-VIEW MATCHING EVALUATION")
    print("=" * 60)
    ov = track_results["overall"]
    print(f"\n  Track-level  |  P={ov['precision']:.3f}  R={ov['recall']:.3f}  F1={ov['f1']:.3f}")
    print(f"               |  TP={ov['TP']}  FP={ov['FP']}  FN={ov['FN']}")
    print(f"               |  Mean score (TP): {ov['mean_score_TP']:.3f}")
    print(f"               |  Mean conf  (TP): {ov['mean_conf_TP']:.3f}")

    fl = frame_results
    print(f"\n  Frame-level  |  P={fl['frame_precision']:.3f}  R={fl['frame_recall']:.3f}  F1={fl['frame_f1']:.3f}")
    print(f"               |  TP={fl['TP_frames']}  FP={fl['FP_frames']}")
    print(f"               |  Mean score (TP): {fl['mean_score_TP_frames']:.3f}")

    cons = consistency_results
    print(f"\n  Consistency  |  Mean={cons['mean_consistency']:.3f}  "
          f"Median={cons['median_consistency']:.3f}  "
          f"Perfect={cons['tracks_perfect_consistency']}/{cons['tracks_evaluated']}")

    if near_far:
        print("\n  Near/Far     |  " + "  |  ".join(
            f"{k}: P={v['precision']:.3f} n={v['total_predictions']}"
            for k, v in near_far.items()
        ))

    print("\n  Per-class:")
    print(f"  {'Class':<12} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 46)
    for cname in classes:
        pc = track_results["per_class"].get(cname, {})
        if not pc:
            continue
        print(f"  {cname:<12} {pc['precision']:>6.3f} {pc['recall']:>6.3f} "
              f"{pc['f1']:>6.3f} {pc['TP']:>5} {pc['FP']:>5} {pc['FN']:>5}")
    print()

    # ---- Output ----
    if args.out_report:
        # Convert sets/tuples to lists for JSON serialization
        def jsonify(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        with open(args.out_report, "w") as f:
            json.dump(report, f, indent=2, default=jsonify)
        print(f"[done] report written to: {args.out_report}")
    else:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
