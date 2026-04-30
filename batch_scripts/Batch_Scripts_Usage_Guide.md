# Batch Scripts Usage Guide

This page documents the **batch-mode commands** used to run the full benchmark across all scenes listed in a processed scene manifest.

It is intended for the GitHub repository and assumes you already have a valid `processed_scene_manifest.csv` with per-scene file paths and split metadata.

---

## Overview

The repository supports three main benchmark components in batch mode:

1. **Cross-view matching**
2. **BEV baselines**
   - IPM baseline
   - MonoLayout-style learned BEV baseline
   - BBox BEV regressor baseline
3. **Dataset statistics and evaluation aggregation**

All batch scripts operate from a single scene manifest and write outputs per scene or into a shared `results/` directory.

---

## 1. Scene Manifest

Typical batch processing uses a manifest like:

```csv
scene_id,street_video,drone_video,street_csv,drone_csv,ego_track_id,gt_pairs_csv,street_manifest_csv,drone_manifest_csv,frame_matches_csv,track_mapping_csv,coord_align_csv,frames_dir,split,align_status
scene_01_01,/path/street.mp4,/path/drone.mp4,/path/street.csv,/path/drone.csv,220,/path/gt_pairs.csv,/path/street_wedge_manifest.csv,/path/drone_wedge_manifest.csv,/path/frame_matches.csv,/path/track_mapping.csv,/path/coord_align.csv,/path/frames/scene_00/,train,ok
```

### Important fields

- `scene_id`: unique scene name
- `street_csv`, `drone_csv`: raw tracked detections
- `gt_pairs_csv`: GT cross-view correspondences
- `street_manifest_csv`, `drone_manifest_csv`: wedge-exported manifests
- `frame_matches_csv`, `track_mapping_csv`: matching outputs
- `coord_align_csv`: per-scene coordinate alignment
- `frames_dir`: extracted street frames for learned BEV
- `split`: `train`, `val`, or `test`
- `align_status`: usually `ok` or `failed`

---

## 2. Cross-view Matching Pipeline

Run the matching pipeline over all scenes in the manifest.

```bash
python matching_pipeline.py \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --export_wedge_script /data/home/bhp39283/disk/detection-tracking/wedge_export/export_wedge_crops.py \
  --embed_script /data/home/bhp39283/disk/detection-tracking/wedge_export/clip_wedge_embeddings.py \
  --match_script /data/home/bhp39283/disk/detection-tracking/wedge_export/match_wedge_frames.py
```

### Outputs per scene

- `street_wedge_manifest.csv`
- `drone_wedge_manifest.csv`
- `street_wedge_emb.npz`
- `drone_wedge_emb.npz`
- `frame_matches.csv`
- `track_mapping.csv`

---

## 3. Batch Matching Evaluation

Evaluate cross-view matching across all scenes with GT.

```bash
python /data/home/bhp39283/disk/detection-tracking/batch_matching_evel.py \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --eval_script /data/home/bhp39283/disk/detection-tracking/wedge_export/eval_matching.py \
  --out_dir /data/home/bhp39283/disk/detection-tracking/results/matching
```

### Outputs

- `per_scene_eval.csv`
- `overall_eval_macro.csv`
- `overall_eval_macro.json`
- `overall_eval_micro.csv`
- `overall_eval_micro.json`

These contain track-level, frame-level, temporal, and per-class matching metrics aggregated across scenes.

---

## 4. Batch Dataset Statistics

Compute dataset-wide statistics from the processed manifest.

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/dataset_stats.py \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --out_json /data/home/bhp39283/disk/detection-tracking/results/dataset_stats.json \
  --out_csv /data/home/bhp39283/disk/detection-tracking/results/per_scene_datastats.csv
```

### Outputs

- `dataset_stats.json`
- `per_scene_datastats.csv`

Use these files to populate dataset summary tables in the paper.

---

## 5. Batch Coordinate Alignment

Before BEV evaluation, run scene-wise coordinate alignment.

### IPM-based alignment

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/batch_auto_coord_align.py \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --align_script /data/home/bhp39283/disk/detection-tracking/baseline/auto_coord_align.py \
  --camera_cfg /data/home/bhp39283/disk/detection-tracking/camera_params.json \
  --method ipm
```

### Depth-based alignment

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/batch_auto_coord_align.py \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --align_script /data/home/bhp39283/disk/detection-tracking/baseline/auto_coord_align.py \
  --camera_cfg /data/home/bhp39283/disk/detection-tracking/camera_params.json \
  --method depth
```

### Notes

- Scenes with unreliable alignment should be marked `align_status=failed` in the manifest.
- BEV learned baselines should usually train only on `align_status=ok` scenes.

---

## 6. Batch IPM BEV Evaluation

Run the IPM BEV baseline on all aligned scenes.

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/batch_eval_ipm_bev.py \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --eval_script /data/home/bhp39283/disk/detection-tracking/baseline/eval_ipm_bev.py \
  --camera_cfg /data/home/bhp39283/disk/detection-tracking/camera_params.json \
  --out_dir /data/home/bhp39283/disk/detection-tracking/results/bev_ipm
```

### Outputs

- per-scene IPM reports
- aggregated BEV metric summaries

These metrics are used for the geometric baseline row in the paper.

---

## 7. Batch MonoLayout-Style BEV Baseline

### Train

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/bev_monolayout.py train \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --train_split train \
  --val_split val \
  --out_dir /data/home/bhp39283/disk/detection-tracking/results/bev_monolayout/run2 \
  --epochs 15 \
  --batch_size 4 \
  --lr 3e-5
```

### Eval

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/bev_monolayout.py eval \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --eval_split test \
  --checkpoint /data/home/bhp39283/disk/detection-tracking/results/bev_monolayout/run2/best.pth \
  --point_decode_thresh 0.25 \
  --out_report /data/home/bhp39283/disk/detection-tracking/results/bev_monolayout/eval_test.json \
  --vis_dir /data/home/bhp39283/disk/detection-tracking/results/bev_monolayout/vis
```

### Reported metrics

- `ADE`
- `FDE`
- `ALE`
- `ALgE`
- `PCK@1m`
- `PCK@2m`
- `mIoU@0.25`
- `mIoU@0.5`

---

## 8. Batch BBox BEV Regressor Baseline

### Train

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/bbox_bev_regressor.py train \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --camera_cfg /data/home/bhp39283/disk/detection-tracking/camera_params.json \
  --train_split train \
  --val_split val \
  --out_dir /data/home/bhp39283/disk/detection-tracking/results/bbox_bev_regressor/run1 \
  --epochs 300
```

### Eval

```bash
python /data/home/bhp39283/disk/detection-tracking/baseline/bbox_bev_regressor.py eval \
  --scene_csv /data/home/bhp39283/disk/detection-tracking/dataset/processed_scene_manifest.csv \
  --camera_cfg /data/home/bhp39283/disk/detection-tracking/camera_params.json \
  --eval_split test \
  --checkpoint /data/home/bhp39283/disk/detection-tracking/results/bbox_bev_regressor/run1/best.pth \
  --out_report /data/home/bhp39283/disk/detection-tracking/results/bbox_bev_regressor/eval_test.json
```

### Reported metrics

- `ADE`
- `FDE`
- `ALE`
- `ALgE`
- `PCK@1m`
- `PCK@2m`
- improvement over IPM baseline

---

## 9. Recommended Split Policy

Use **scene-level splits**, not frame-level splits, for benchmark reporting.

Recommended manifest values:

- training scenes: `split=train`
- validation scenes: `split=val`
- held-out scenes: `split=test`

For BEV baselines, only scenes with reliable alignment should be included by default:

- `align_status=ok` → usable
- `align_status=failed` → skipped unless explicitly allowed

---

## 10. Typical End-to-End Workflow

```text
1. Prepare processed_scene_manifest.csv
2. Run matching_pipeline.py
3. Annotate GT and finalize gt_pairs.csv
4. Run batch_matching_evel.py
5. Run dataset_stats.py
6. Run batch_auto_coord_align.py
7. Run batch_eval_ipm_bev.py
8. Train/eval bev_monolayout.py
9. Train/eval bbox_bev_regressor.py
10. Export tables for the paper
```

---

## 11. Output Directories

Recommended result layout:

```text
results/
  matching/
    per_scene_eval.csv
    overall_eval_macro.csv
    overall_eval_micro.csv
  bev_ipm/
    ...
  bev_monolayout/
    run1/
      best.pth
      training_history.json
    eval_test.json
    vis/
  bbox_bev_regressor/
    run1/
      best.pth
      training_history.json
    eval_test.json
```

---

## 12. Notes

- `scene_csv` batch mode is the recommended way to run all final experiments.
- Keep raw data paths and generated output paths separate.
- Scene-level manifests make the benchmark reproducible and easy to extend.
- For paper reporting, always state whether a result uses:
  - all evaluated scenes
  - only aligned scenes
  - train/val/test scene-level splits

---

## 13. Summary

The batch scripts provide a reproducible way to:

- process all scenes consistently
- evaluate cross-view matching across the whole benchmark
- train and test BEV baselines at scene level
- generate aggregate numbers for the paper

This is the recommended setup for final benchmark reporting and future public GitHub usage.
