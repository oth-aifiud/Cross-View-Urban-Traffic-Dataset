# Cross-View Urban Traffic Dataset

**Synchronized street-level and drone footage for Bird's-Eye View (BEV) perception research.**

A dataset of urban intersections recorded simultaneously from a bike-mounted GoPro camera and a drone flying at 60m altitude. Drone detections provide accurate metric ground truth for training and evaluating BEV projection models — a capability not available in any existing monocular dataset.

[![Dataset](https://img.shields.io/badge/Dataset-Download-8BC34A?style=for-the-badge)](https://huggingface.co/datasets/prakharbh/CrossViewUrbanTrafficDataset)


---
## Overview

| | |
|---|---|
| **Modalities** | Street-view video (GoPro Hero 11, 4K) + Drone overhead video (4K, 60m altitude) |
| **Sync** | Frame-synchronized 1:1 at 30 fps |
| **Scenes** | Urban intersections, Regensburg, Germany |
| **GT source** | Drone object detections with metric positions (x_fwd, y_left in meters) |
| **Classes** | Car, Truck, Bus, Person, Bicycle, Motorcycle |
| **Task** | Cross-View feature matching and Monocular BEV object localization |


This pipeline processes synchronized street-view (GoPro on bike) and drone videos
to produce a cross-view dataset with ground-truth BEV supervision.

```
Raw Videos (street + drone) with annotations
        │
        ▼
  [scene_manifest.csv]
        │
        ▼
  Street Manifest + Drone Manifest CSVs
        │
        ├──► SCRIPT 1: matching_pipeline.py   ── (Cross-view ID matching pipeline)
        │           │
        │           ▼
        │    frame_matches.csv + track_map.csv + (wedge manifests & CLIP embeddings)
        │           │
        │     ┌─────┴──────────────────────┐
        │     ▼                            ▼
        │  SCRIPT 1a: eval_matching.py    SCRIPT 3: dataset_stats.py
        │  (benchmark: P/R/F1)           
        │
        └──► SCRIPT 2: auto_coord_align.py  ── Auto coordinate alignment
                   │   (uses near matches + optional Depth Anything V2)
                   ▼
            coord_align.csv  (replaces manual per-scene measurement)
                   │
             ┌─────┴──────────────────────┐───────────────────────────────────────┐
             ▼                            ▼                                       ▼
          SCRIPT 4: eval_ipm_bev.py      SCRIPT 5: bev_monolayout.py.          SCRIPT 6: bbox_bev_regressor.py
          (IPM baseline benchmark)       (learning-based BEV model).           (Detection-Conditioned BEV Position Regressor)
                                         (train + eval)                        (train + eval) 
                                          │                                      │
                                          ▼                                      ▼
                                        SCRIPT 7: visualize_mono_bev.py        SCRIPT 8: visualize_bbox_regressor.py

```

---
## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cross-view-urban-traffic.git
cd cross-view-urban-traffic
# Option: full reproducibility
conda env create -f environment.yml
conda activate crossview

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Pipeline

The full pipeline takes raw videos and produces BEV-evaluated resuts.

```bash
# 1. Cross-view matching pipeline
 Purpose: Core cross-view ID matching. Matches street tracks to drone tracks
 using CLIP embeddings + angular geometry + distance ranking.
 Runs two passes: near objects (appearance-dominant) and far objects
 (geometry-dominant). Outputs frame-level matches and voted track-level map.

python matching_pipeline.py \
  --scene_csv dataset/scene_manifest.csv \
  --export_wedge_script export_wedge_crops.py \
  --embed_script clip_wedge_embeddings.py \
  --match_script match_wedge_frames.py

# 1a. Cross-view matching evaluation
python eval_matching.py \
    --gt_csv          annotations/gt_track_map.csv \
    --pred_track_csv  outputs/track_map.csv \
    --pred_frame_csv  outputs/frame_matches.csv \
    --street_manifest data/street_manifest.csv \
    --drone_manifest data/drone_manifest.csv \
    --near_dist_m 30 \
    --out_report      results/matching_eval.json

# 2. Auto coordinate alignment (depth mode recommended)
python auto_coord_align.py \
    --street_manifest   data/street_manifest.csv \
    --drone_manifest    data/drone_manifest.csv \
    --frame_matches_csv outputs/frame_matches.csv \
    --track_map_csv     outputs/track_map.csv \
    --camera_cfg        data/camera_params.json \
    --method            depth or ipm \
    --img_dir           data/frames/ \
    --out_csv           outputs/coord_align.csv

# 3. Dataset statistics 
python dataset_stats.py \
    --street_manifest data/street_manifest.csv \
    --drone_manifest  data/drone_manifest.csv \
    --track_map_csv   outputs/track_map.csv \
    --scene_meta_csv  data/scene_metadata.csv \
    --out_json        results/dataset_stats.json \
    --out_csv         results/per_scene_stats.csv

# 4. IPM baseline 
python eval_bev.py \
    --street_manifest data/street_manifest.csv \
    --drone_manifest  data/drone_manifest.csv \
    --track_map_csv   outputs/track_map.csv \
    --camera_cfg      data/camera_params.json \
    --coord_align_csv outputs/coord_align.csv \
    --out_report      results/bev_ipm_eval.json \
    --out_csv         results/bev_ipm_per_track.csv

# 5a. Train learned BEV model
python bev_monolayout.py train \
    --street_manifest data/street_manifest.csv \
    --drone_manifest  data/drone_manifest.csv \
    --track_map_csv   outputs/track_map.csv \
    --coord_align_csv outputs/coord_align.csv \
    --img_dir         data/frames/ \
    --out_dir         checkpoints/bev_run1 \
    --epochs          50 --batch_size 4

# 5b. Evaluate learned BEV model
python bev_monolayout.py eval \
    --street_manifest data/val_street.csv \
    --drone_manifest  data/val_drone.csv \
    --track_map_csv   outputs/val_track_map.csv \
    --coord_align_csv outputs/coord_align.csv \
    --img_dir         data/frames/ \
    --checkpoint      checkpoints/bev_run1/best.pth \
    --out_report      results/bev_learned_eval.json \
    --vis_dir         results/bev_vis/

# 6a. Train BEV Regressor(< 5 minutes on CPU, < 1 minute on GPU)
python pipeline/06_bbox_bev_regressor.py train \
    --street_manifest street_wedge_manifest.csv \
    --drone_manifest  drone_wedge_manifest.csv \
    --track_map_csv   track_mapping.csv \
    --coord_align_csv coord_align.csv \
    --camera_cfg      camera_params.json \
    --out_dir         checkpoints/bbox_regressor \
    --epochs          300

# 6b. Evaluate BEV Regressor
python pipeline/06_bbox_bev_regressor.py eval \
    --street_manifest street_wedge_manifest.csv \
    --drone_manifest  drone_wedge_manifest.csv \
    --track_map_csv   track_mapping.csv \
    --coord_align_csv coord_align.csv \
    --camera_cfg      camera_params.json \
    --checkpoint      checkpoints/bbox_regressor/best.pth \
    --out_report      results/bbox_regressor_eval.json

# 7. Visialize MonoLayout BEV
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

# 8. BBox Regressor BEV Visualization
python visualize_bbox_regressor.py \
      --street_manifest  .../street_wedge_manifest.csv \
      --drone_manifest   .../drone_wedge_manifest.csv \
      --track_map_csv    .../track_mapping.csv \
      --coord_align_csv  .../coord_align.csv \
      --camera_cfg       .../camera_params.json \
      --checkpoint       .../bbox_regressor/best.pth \
      --street_video     .../bike.mp4 \
      --drone_video      .../drone.mp4 \
      --img_dir          .../frames \
      --frame            831 \
      --out              vis_bbox/frame_831.png

# For whole-dataset processing and evaluation across multiple scenes, use the batch scripts in the batch_scripts/ directory
```
---
## Matches Ground Truth Annotation framework
We provide a Streamlit UI for fast annotation.
```
streamlit run annotate_web_frames.py --server.port 8501 --server.address 0.0.0.0
```
Open:
```
http://localhost:8501
```
Outputs:

 gt_pairs.csv

Ground truth correspondences:
```
scene_id,street_track_id,drone_track_id,class_name
```
 gt_audit.csv

Annotation history (optional)

Then run evaluation as above (4. Matching evaluation )

**to extract video frames needed by the pipeline, use /scripts/extract_frames.py (usuge given in the script file)**

---

## Data Format

Data is **not included in this repository**. See [Data Access](#data-access) below.

### street_wedge_manifest.csv

Per-detection records from the street camera.

| Column | Type | Description |
|---|---|---|
| `frame` | int | Frame number (1-based, matches video timestamp) |
| `track_id` | int | Street-side track ID |
| `class_name` | str | Object class |
| `bbox_x1/y1/x2/y2` | float | Bounding box in pixels |
| `scene_id` | str | Scene identifier (e.g. `scene_00`) |
| `crop_path` | str | Path to cropped detection image |

### drone_wedge_manifest.csv

Per-detection records from the drone, with metric positions.

| Column | Type | Description |
|---|---|---|
| `street_frame` | int | Synchronized street frame number |
| `track_id` | int | Drone-side track ID |
| `class_name` | str | Object class |
| `bbox_x1/y1/x2/y2` | float | Bounding box in drone image pixels |
| `x_fwd` | float | Metric distance forward from drone nadir (m) |
| `y_left` | float | Metric distance left from drone nadir (m) |
| `dist_m` | float | Euclidean distance from drone nadir (m) |

### coord_align.csv

Per-scene rigid transform: `drone_pos = R(rot_deg) × ipm_pos + offset`

| Column | Description |
|---|---|
| `scene_id` | Scene identifier |
| `offset_x/y` | Translation from IPM origin to drone nadir (m) |
| `rot_deg` | Rotation between camera forward and drone x_fwd axis |
| `residual_m` | Mean RANSAC inlier residual (quality indicator) |


---

### Directory structure after download and wedge postprocessing

```
dataset/
│
├── scene_manifest.csv              # MASTER file (all scenes)
│
├── scenes/
│   ├── scene_00/
│   │   ├── raw/
│   │   │   ├── street.mp4
│   │   │   ├── drone.mp4
│   │   │   
│   │   │   ├── street_annotations.csv
│   │   │   ├── drone_annotations.csv
│   │   │
│   │   │   └── gt_pairs.csv        # Cross-view GT (manual)
│   │   │
│   │   ├── processed/
│   │      ├── wedge/
│   │      │   ├── street_wedge_manifest.csv
│   │      │   ├── drone_wedge_manifest.csv
│   │      │   ├── frame_matches.csv
│   │      │   └── track_mapping.csv
│   │      │    
│   │      │
│   │      └── embeddings/
│   │          ├── street_emb.npz
│   │          └── drone_emb.npz 
│   │   
│   ├── scene_01/
│   │   └── ...
│   │
│   └── ...

```

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{crossview_urban_traffic_2026,
  title     = {Cross-View Urban Traffic Dataset},
  author    = {TODO},
  year      = {2026},
  note      = {NeurIPS 2026 submission},
  url       = {https://gitlab.oth-regensburg.de/IM/labor_ai_iud/street_level_bev_projection/cross-view-dataset/}
}
```

---

## License

The pipeline code in this repository is licensed under the **MIT License**.

The dataset (videos, manifests, annotations) is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International). You may use it for research and educational purposes with attribution. Commercial use is not permitted.

See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Detection and tracking: [TODO — add trackers used]
- BEV architecture inspired by [MonoLayout](https://github.com/hbutsuak95/monolayout)
- Recorded in Regensburg, Bavaria, Germany
