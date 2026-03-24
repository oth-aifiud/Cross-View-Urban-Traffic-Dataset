# Cross-View Urban Traffic Dataset

**Synchronized street-level and drone footage for Bird's-Eye View (BEV) perception research.**

A dataset of urban intersections recorded simultaneously from a bike-mounted GoPro camera and a drone flying at 60m altitude. Drone detections provide accurate metric ground truth for training and evaluating BEV projection models — a capability not available in any existing monocular dataset.

---

## Overview

| | |
|---|---|
| **Modalities** | Street-view video (GoPro Hero 11, 4K) + Drone overhead video (4K, 60m altitude) |
| **Sync** | Frame-synchronized 1:1 at 30 fps |
| **Scenes** | Urban intersections, Regensburg, Germany |
| **GT source** | Drone object detections with metric positions (x_fwd, y_left in meters) |
| **Classes** | Car, Truck, Bus, Person, Bicycle, Motorcycle |
| **Task** | Monocular BEV object localization |

---

## Repository Structure

```
cross-view-urban-traffic/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── pipeline/                        # Full processing pipeline (6 stages)
│   ├── 01_generate_camera_params.py     # GoPro intrinsics generation / calibration
│   ├── 02_extract_frames.py             # Extract frames from video by manifest
│   ├── 03_match_wedge_frames.py         # Cross-view object matching (street↔drone)
│   ├── 04_auto_coord_align.py           # Automatic coordinate alignment (IPM-based)
│   ├── 05_eval_bev.py                   # IPM baseline evaluation
│   └── 06_bbox_bev_regressor.py         # Detection-conditioned BEV regressor
│
├── models/
│   ├── bev_monolayout.py                # MonoLayout-style image→BEV model
│   └── bbox_bev_regressor.py            # Lightweight bbox→BEV MLP (recommended)
│
├── visualization/
│   ├── visualize_bbox_regressor.py      # 4-panel BEV figure renderer
│   └── render_bev_figure.py             # Full pipeline figure with drone overhead
│
├── configs/
│   └── camera_params_example.json       # Example GoPro Hero 11 camera config
│
└── docs/
    ├── PIPELINE.md                      # Step-by-step pipeline guide
    ├── DATA_FORMAT.md                   # CSV schema documentation
    └── COORDINATE_SYSTEMS.md            # Coordinate frame definitions
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cross-view-urban-traffic.git
cd cross-view-urban-traffic
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0
torchvision>=0.15
opencv-python>=4.8
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

---

## Pipeline

The full pipeline takes raw videos and produces BEV-evaluated results in 6 steps.

### Step 1 — Camera Parameters

```bash
# Use published GoPro specs (instant, ±1% accuracy for Linear mode)
python pipeline/01_generate_camera_params.py spec \
    --model hero11 --fov linear --resolution 4k \
    --scenes scene_00 scene_01 \
    --out camera_params.json

# Or use your own calibrated intrinsics
# Edit camera_params.json directly — see configs/camera_params_example.json
```

### Step 2 — Extract Frames

```bash
python pipeline/02_extract_frames.py \
    --video    /path/to/street_video.mp4 \
    --manifest /path/to/street_wedge_manifest.csv \
    --out_dir  frames/scene_00 \
    --scene_id scene_00
```

### Step 3 — Cross-View Matching

Matches street detections to drone detections across synchronized frames using appearance embeddings + geometric constraints.

```bash
python pipeline/03_match_wedge_frames.py \
    --street_manifest street_wedge_manifest.csv \
    --drone_manifest  drone_wedge_manifest.csv \
    --out_csv         frame_matches.csv \
    --track_map_csv   track_mapping.csv
```

### Step 4 — Coordinate Alignment

Estimates the rigid transform between the drone coordinate frame and the camera BEV frame using RANSAC on matched object pairs.

```bash
python pipeline/04_auto_coord_align.py \
    --street_manifest   street_wedge_manifest.csv \
    --drone_manifest    drone_wedge_manifest.csv \
    --frame_matches_csv frame_matches.csv \
    --track_map_csv     track_mapping.csv \
    --camera_cfg        camera_params.json \
    --out_csv           coord_align.csv
```

Expected output: `scale=1.000  residual<1.0m  rot=<scene_heading_diff>`

### Step 5 — IPM Baseline Evaluation

```bash
python pipeline/05_eval_bev.py \
    --street_manifest street_wedge_manifest.csv \
    --drone_manifest  drone_wedge_manifest.csv \
    --track_map_csv   track_mapping.csv \
    --camera_cfg      camera_params.json \
    --coord_align_csv coord_align.csv \
    --out_report      results/ipm_eval.json
```

### Step 6 — Train & Evaluate BEV Regressor

```bash
# Train (< 5 minutes on CPU, < 1 minute on GPU)
python pipeline/06_bbox_bev_regressor.py train \
    --street_manifest street_wedge_manifest.csv \
    --drone_manifest  drone_wedge_manifest.csv \
    --track_map_csv   track_mapping.csv \
    --coord_align_csv coord_align.csv \
    --camera_cfg      camera_params.json \
    --out_dir         checkpoints/bbox_regressor \
    --epochs          300

# Evaluate
python pipeline/06_bbox_bev_regressor.py eval \
    --street_manifest street_wedge_manifest.csv \
    --drone_manifest  drone_wedge_manifest.csv \
    --track_map_csv   track_mapping.csv \
    --coord_align_csv coord_align.csv \
    --camera_cfg      camera_params.json \
    --checkpoint      checkpoints/bbox_regressor/best.pth \
    --out_report      results/bbox_regressor_eval.json
```

---

## Results

Evaluated on 2,707 matched detection pairs across 2 intersections.

| Method | ADE↓ | Median↓ | PCK@1m↑ | PCK@2m↑ |
|---|---|---|---|---|
| IPM (geometric baseline) | 62.9m | 13.8m | 0.0% | 0.0% |
| **BBox Regressor (ours)** | **4.4m** | **3.5m** | **20.5%** | **32.6%** |

**+92.9% improvement over IPM.** The model is trained with drone-provided metric ground truth — a supervision signal that monocular methods cannot access without this dataset.

Per-class breakdown:

| Class | N | IPM ADE | Regressor ADE | Δ |
|---|---|---|---|---|
| Car | 2225 | 61.3m | 4.6m | +92.5% |
| Person | 163 | 5.3m | 4.1m | +23.3% |
| Bicycle | 93 | 575.2m | 5.7m | +99.0% |
| Bus | 185 | — | 2.5m | — |
| Truck | 41 | — | 2.8m | — |

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

### Coordinate Systems

See `docs/COORDINATE_SYSTEMS.md` for full definitions. In brief:

- **Street IPM frame:** origin at camera, x=forward, y=left, z=up
- **Drone frame:** origin at drone nadir projection on ground, x_fwd/y_left
- **Alignment:** `drone = R(rot) × ipm + offset` — solved per scene by `auto_coord_align.py`

---

## Data Access

The dataset videos and manifests are hosted separately due to size.

> **Download:** [TODO — add link when uploaded]
>
> **Request access:** [TODO — add form/email]

### Directory structure after download

```
data/
├── scene_00/                        # Intersection 1 (Galgenberg, Regensburg)
│   ├── street_wedge_manifest.csv
│   ├── drone_wedge_manifest.csv
│   ├── track_mapping.csv
│   ├── frame_matches.csv
│   ├── coord_align.csv
│   └── frames/                      # Extracted street frames (run extract_frames.py)
│       └── scene_00/
│           ├── 000001.jpg
│           └── ...
│
└── scene_01/                        # Intersection 2 (Dr.-Martin-Luther-Str., Regensburg)
    ├── street_wedge_manifest.csv
    └── ...
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
  url       = {https://github.com/YOUR_USERNAME/cross-view-urban-traffic}
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