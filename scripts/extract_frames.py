"""
extract_frames.py — Extract only the frames needed by the pipeline
===================================================================
Reads your street manifest to find exactly which frame numbers are needed,
then extracts only those frames from the video. Much faster than dumping
the entire video to disk.

Usage
-----
  python extract_frames.py \
      --video     /data/home/bhp39283/disk/detection-tracking/Galgenberg_60m_bike_head.mp4 \
      --manifest  /data/home/bhp39283/disk/detection-tracking/wedge_export/street_wedge_manifest.csv \
      --out_dir   /data/home/bhp39283/disk/detection-tracking/wedge_export/frames/scene_00 \
      --scene_id  scene_00

  # If you also need drone frames (for any future use):
  python extract_frames.py \
      --video     /data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold/Dr_MartinLuther_Luitpold_50m_drone.mp4 \
      --manifest  /data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold/outputs/wedge_export/drone_wedge_manifest.csv \
      --out_dir   /data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold/outputs/wedge_export/frames/drone_scene_01 \
      --scene_id  scene_01_01 \
      --frame_col street_frame
"""

import argparse
import os
import sys

import cv2
import pandas as pd


def extract_frames(
    video_path: str,
    out_dir: str,
    frame_numbers: list,
    verbose: bool = True,
) -> int:
    """
    Extract specific frame numbers from a video file.
    Frames are saved as XXXXXX.jpg (zero-padded 6 digits).
    Returns number of frames successfully saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Cannot open video: {video_path}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[info] Video: {os.path.basename(video_path)}")
    print(f"       {width}x{height}  {fps:.2f}fps  {total_video_frames} total frames")
    print(f"[info] Frames to extract: {len(frame_numbers)}")
    print(f"[info] Output dir: {out_dir}")
    print()

    # Sort frame numbers for sequential seek (much faster than random seeks)
    frame_set = sorted(set(frame_numbers))

    # Check for already-extracted frames
    already_done = set()
    for fn in frame_set:
        out_path = os.path.join(out_dir, f"{fn:06d}.jpg")
        if os.path.exists(out_path):
            already_done.add(fn)

    to_extract = [fn for fn in frame_set if fn not in already_done]

    if already_done:
        print(f"[info] {len(already_done)} frames already extracted — skipping")
    if not to_extract:
        print(f"[done] All {len(frame_set)} frames already exist in {out_dir}")
        return len(frame_set)

    print(f"[info] Extracting {len(to_extract)} frames...")

    # NOTE: OpenCV uses 0-based frame indexing.
    # Your manifest uses 1-based frame numbers (frame 1 = first frame of video).
    # We subtract 1 when seeking, add 1 when reading.

    saved = 0
    failed = 0
    current_pos = -1

    for i, frame_num in enumerate(to_extract):
        # 0-based index in the video
        video_idx = frame_num - 1

        if video_idx < 0 or video_idx >= total_video_frames:
            print(f"[warn] frame {frame_num} out of range (video has {total_video_frames} frames)")
            failed += 1
            continue

        # Sequential read is faster — only seek if we'd go backwards or skip far ahead
        if video_idx != current_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)

        ret, frame = cap.read()
        current_pos = video_idx

        if not ret:
            print(f"[warn] failed to read frame {frame_num}")
            failed += 1
            continue

        out_path = os.path.join(out_dir, f"{frame_num:06d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if verbose and (i + 1) % 200 == 0:
            pct = (i + 1) / len(to_extract) * 100
            print(f"  [{pct:5.1f}%] {i+1}/{len(to_extract)} frames extracted...")

    cap.release()

    print()
    print(f"[done] Extracted {saved} frames  ({failed} failed)")
    if failed > 0:
        print(f"[warn] {failed} frames failed — check that your manifest frame numbers")
        print(f"       match the video length ({total_video_frames} frames total)")

    # Verify a sample
    sample = sorted(os.listdir(out_dir))[:3]
    print(f"[info] Sample output files: {sample}")

    return saved


def main():
    ap = argparse.ArgumentParser(description="Extract frames from video for BEV training")
    ap.add_argument("--video",      required=True, help="Path to video file (.mp4, .mov, etc)")
    ap.add_argument("--manifest",   required=True, help="street_wedge_manifest.csv or drone_wedge_manifest.csv")
    ap.add_argument("--out_dir",    required=True, help="Output directory for frames (e.g. frames/scene_00)")
    ap.add_argument("--scene_id",   default=None,
                    help="If manifest has scene_id column, only extract frames for this scene. "
                         "Leave empty to use all rows.")
    ap.add_argument("--frame_col",  default="frame",
                    help="Column name for frame numbers in manifest. "
                         "Use 'street_frame' for drone manifests. Default: 'frame'")
    ap.add_argument("--quality",    type=int, default=95,
                    help="JPEG quality 0-100. 95 is lossless-quality, 85 saves ~40%% disk space.")
    args = ap.parse_args()

    # Load manifest
    df = pd.read_csv(args.manifest)
    print(f"[info] Loaded manifest: {len(df)} rows")

    # Filter by scene if requested
    if args.scene_id and "scene_id" in df.columns:
        df = df[df["scene_id"] == args.scene_id]
        print(f"[info] Filtered to scene '{args.scene_id}': {len(df)} rows")

    # Get unique frame numbers
    if args.frame_col not in df.columns:
        print(f"[error] Column '{args.frame_col}' not found in manifest.")
        print(f"        Available columns: {df.columns.tolist()}")
        sys.exit(1)

    frame_numbers = sorted(df[args.frame_col].dropna().astype(int).unique().tolist())
    print(f"[info] Unique frames to extract: {len(frame_numbers)}")
    print(f"       Range: {frame_numbers[0]} – {frame_numbers[-1]}")

    extract_frames(args.video, args.out_dir, frame_numbers)


if __name__ == "__main__":
    main()