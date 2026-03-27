import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from collections import defaultdict
 
# -----------------------------
# Basic helpers
# -----------------------------
def norm_class_name(x: str) -> str:
    x = str(x).strip().lower()
    if x in ["pedestrian", "person", "people"]:
        return "person"
    if x in ["bike", "bicycle", "cyclist"]:
        return "bicycle"
    if x in ["motorbike", "motorcycle"]:
        return "motorcycle"
    return x


def rolling_median_angle(theta: np.ndarray, k: int = 11) -> np.ndarray:
    un = np.unwrap(theta)
    out = un.copy()
    r = k // 2
    for i in range(len(un)):
        lo = max(0, i - r)
        hi = min(len(un), i + r + 1)
        out[i] = np.median(un[lo:hi])
    return (out + np.pi) % (2 * np.pi) - np.pi


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.last_idx = None
        self.last_frame = None

    def read(self, frame_idx: int):
        # simple random access
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


# -----------------------------
# Ego pose from drone world coords
# -----------------------------
class EgoPose:
    def __init__(self, X, Y, psi):
        self.X = X
        self.Y = Y
        self.psi = psi


def compute_ego_pose(drone_df: pd.DataFrame, ego_track_id: int, n_frames: int,
                     delta: int = 5, smooth_k: int = 11) -> EgoPose:
    ego = drone_df[drone_df["track_id"].astype(int) == int(ego_track_id)].copy()
    ego["frame_id"] = ego["frame_id"].astype(int)
    ego = ego.sort_values("frame_id")

    X = np.full(n_frames, np.nan, dtype=np.float64)
    Y = np.full(n_frames, np.nan, dtype=np.float64)

    for _, r in ego.iterrows():
        t = int(r["frame_id"])
        if 0 <= t < n_frames:
            X[t] = float(r["world_x"])
            Y[t] = float(r["world_y"])

    # fill gaps
    for arr in [X, Y]:
        s = pd.Series(arr).interpolate(limit_direction="both")
        arr[:] = s.to_numpy()

    # heading with stationary hold
    psi = np.zeros(n_frames, dtype=np.float64)
    prev = 0.0
    min_disp = 0.20  # meters across baseline; holds heading if almost stationary

    for t in range(n_frames):
        t0 = max(0, t - delta)
        t1 = min(n_frames - 1, t + delta)
        dx = X[t1] - X[t0]
        dy = Y[t1] - Y[t0]
        disp = math.hypot(dx, dy)
        if disp < min_disp and t > 0:
            psi[t] = prev
        else:
            psi[t] = math.atan2(dy, dx)
            prev = psi[t]

    psi = rolling_median_angle(psi, k=smooth_k)
    return EgoPose(X=X, Y=Y, psi=psi)


def ego_relative(wx, wy, ex, ey, epsi):
    rx = wx - ex
    ry = wy - ey
    c = math.cos(epsi)
    s = math.sin(epsi)
    x_fwd = c * rx + s * ry
    y_left = -s * rx + c * ry
    dist = math.hypot(x_fwd, y_left)
    theta = math.atan2(y_left, x_fwd)  # 0 forward
    return x_fwd, y_left, dist, theta


# -----------------------------
# bbox parsing
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def street_bbox_xyxy(r, W, H) -> Optional[Tuple[int,int,int,int]]:
    # expects pixel x_center/y_center/width/height
    xc = float(r["x_center"]); yc = float(r["y_center"])
    bw = float(r["width"]);   bh = float(r["height"])
    x1 = int(round(xc - bw/2)); y1 = int(round(yc - bh/2))
    x2 = int(round(xc + bw/2)); y2 = int(round(yc + bh/2))
    x1 = clamp(x1, 0, W-1); y1 = clamp(y1, 0, H-1)
    x2 = clamp(x2, 1, W);   y2 = clamp(y2, 1, H)
    if x2 <= x1+2 or y2 <= y1+2:
        return None
    return x1,y1,x2,y2


def drone_bbox_xyxy(r, W, H) -> Optional[Tuple[int,int,int,int]]:
    # supports either pixel_x/pixel_y or normalized x_center/y_center
    if "pixel_x" in r and "pixel_y" in r:
        xc = float(r["pixel_x"]); yc = float(r["pixel_y"])
        bw = float(r["width"]);   bh = float(r["height"])
    else:
        xc = float(r["x_center"]); yc = float(r["y_center"])
        bw = float(r["width"]);   bh = float(r["height"])
        # normalize -> pixels if needed
        if abs(xc) <= 1.5 and abs(yc) <= 1.5:
            xc *= W; yc *= H
        if abs(bw) <= 1.5 and abs(bh) <= 1.5:
            bw *= W; bh *= H

    x1 = int(round(xc - bw/2)); y1 = int(round(yc - bh/2))
    x2 = int(round(xc + bw/2)); y2 = int(round(yc + bh/2))
    x1 = clamp(x1, 0, W-1); y1 = clamp(y1, 0, H-1)
    x2 = clamp(x2, 1, W);   y2 = clamp(y2, 1, H)
    if x2 <= x1+2 or y2 <= y1+2:
        return None
    return x1,y1,x2,y2


def infer_street_width(street: pd.DataFrame) -> Optional[int]:
    try:
        xc_max = float(street["x_center"].max())
        if xc_max > 2.0:
            right_edge = street["x_center"].astype(float) + 0.5 * street["width"].astype(float)
            est = float(np.percentile(right_edge.to_numpy(), 99))
            return int(max(100, min(10000, math.ceil(est * 1.05))))
    except Exception:
        pass
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street_video", required=True)
    ap.add_argument("--drone_video", required=True)
    ap.add_argument("--street_csv", required=True)
    ap.add_argument("--drone_csv", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--ego_track_id", type=int, default=24)
    ap.add_argument("--fov_deg", type=float, default=120.0)
    ap.add_argument("--max_range", type=float, default=70.0)

    ap.add_argument("--frame_offset", type=int, default=0, help="drone_frame = street_frame + offset")
    ap.add_argument("--yaw_offset_deg", type=float, default=0.0, help="ego heading + yaw offset")
    ap.add_argument("--include_classes", type=str, default="car,truck,bus,person,bicycle,motorcycle")

    ap.add_argument("--topN_street", type=int, default=20, help="keep top-N largest street boxes per class per frame")
    ap.add_argument("--min_area_street", type=float, default=0.0, help="optional hard min area filter")

    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    street_out = out_dir / "street_wedge_crops"
    drone_out = out_dir / "drone_wedge_crops"
    street_out.mkdir(parents=True, exist_ok=True)
    drone_out.mkdir(parents=True, exist_ok=True)

    street = pd.read_csv(args.street_csv)
    drone = pd.read_csv(args.drone_csv)

    street["class_name"] = street["class_name"].apply(norm_class_name)

    if "class_name" not in drone.columns:
        drone["class_name"] = np.nan
    if "cls" in drone.columns:
        label_to_class = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle", 4: "person", 5: "bicycle"}
        mapped = drone["cls"].map(label_to_class)
        drone["class_name"] = drone["class_name"].where(~drone["class_name"].isna(), mapped)
    drone["class_name"] = drone["class_name"].apply(norm_class_name)

    include = set(norm_class_name(x) for x in args.include_classes.split(",") if x.strip())

    half = math.radians(args.fov_deg) / 2.0
    yaw_off = math.radians(args.yaw_offset_deg)

    # frame bounds
    max_street = int(street["frame"].max())
    max_drone = int(drone["frame_id"].max())
    end_frame = args.end_frame if args.end_frame >= 0 else min(max_street, max_drone - args.frame_offset)
    start_frame = max(0, args.start_frame)
    end_frame = min(end_frame, max_street)

    # street width inference (for bearing)
    street_W = infer_street_width(street) or 1920
    print(f"[info] street width used={street_W}")

    # prepare ego pose length for drone indices we may access
    n_ego = max_drone + 2
    ego = compute_ego_pose(drone, args.ego_track_id, n_ego)

    # group by frame
    street_by_frame = {int(f): g for f, g in street.groupby(street["frame"].astype(int))}
    drone_by_frame = {int(f): g for f, g in drone.groupby(drone["frame_id"].astype(int))}

    # open videos
    street_v = VideoReader(args.street_video)
    drone_v = VideoReader(args.drone_video)

    street_manifest = []
    drone_manifest = []

    # We need actual frame sizes for cropping
    # (read one frame)
    sf0 = street_v.read(start_frame)
    df0 = drone_v.read(max(0, start_frame + args.frame_offset))
    if sf0 is None or df0 is None:
        raise RuntimeError("Could not read initial frames from videos. Check paths and frame indices.")
    sh, sw = sf0.shape[:2]
    dh, dw = df0.shape[:2]
    street_W = sw
    print(f"[info] overriding street width from video: {street_W}")
    print(f"[info] street video size: {sw}x{sh}")
    print(f"[info] drone  video size: {dw}x{dh}")
    print(f"[info] exporting frames {start_frame}..{end_frame} stride={args.stride}")

    for ts in range(start_frame, end_frame + 1, args.stride):
        td = ts + args.frame_offset
        if td < 0 or td > max_drone:
            continue

        sF = street_by_frame.get(ts)
        dF = drone_by_frame.get(td)
        if sF is None and dF is None:
            continue

        street_frame = street_v.read(ts) if sF is not None else None
        drone_frame  = drone_v.read(td) if dF is not None else None

        # --- Street wedge ---
        if sF is not None and street_frame is not None:
            # build candidates per class
            cand = defaultdict(list)
            for _, r in sF.iterrows():
                cname = norm_class_name(r.get("class_name",""))
                if cname not in include:
                    continue

                xc = float(r["x_center"])
                bw = float(r["width"]); bh = float(r["height"])
                area = max(1.0, bw * bh)
                if area < args.min_area_street:
                    continue

                # bearing from x
                x_norm = (xc / float(street_W)) * 2.0 - 1.0
                bearing = x_norm * half
                if abs(bearing) > half:
                    continue

                cand[cname].append((area, r, bearing))

            for cname in list(cand.keys()):
                cand[cname].sort(key=lambda x: x[0], reverse=True)
                cand[cname] = cand[cname][:args.topN_street]

            for cname, lst in cand.items():
                for rank_i, (area, r, bearing) in enumerate(lst):
                    bb = street_bbox_xyxy(r, sw, sh)
                    if bb is None:
                        continue
                    x1,y1,x2,y2 = bb
                    crop = street_frame[y1:y2, x1:x2]
                    s_tid = int(r["track_id"])
                    fname = f"street_f{ts:06d}_tid{s_tid}_c{cname}_r{rank_i:02d}.jpg"
                    out_path = street_out / fname
                    cv2.imwrite(str(out_path), crop)

                    street_manifest.append({
                        "view": "street",
                        "frame": ts,
                        "track_id": s_tid,
                        "class_name": cname,
                        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                        "bearing_rad": float(bearing),
                        "area": float(area),
                        "crop_path": str(out_path)
                    })

        # --- Drone wedge ---
        if dF is not None and drone_frame is not None:
            for _, r in dF.iterrows():
                cname = norm_class_name(r.get("class_name",""))
                if cname not in include:
                    continue

                if "world_x" not in r or "world_y" not in r or pd.isna(r["world_x"]) or pd.isna(r["world_y"]):
                    continue

                wx = float(r["world_x"]); wy = float(r["world_y"])
                x_fwd, y_left, dist, theta = ego_relative(wx, wy, ego.X[td], ego.Y[td], ego.psi[td] + yaw_off)

                if x_fwd <= 0:
                    continue
                if dist > args.max_range:
                    continue
                if abs(theta) > half:
                    continue

                bb = drone_bbox_xyxy(r, dw, dh)
                if bb is None:
                    continue
                x1,y1,x2,y2 = bb
                crop = drone_frame[y1:y2, x1:x2]

                d_tid = int(r["track_id"])
                fname = f"drone_f{td:06d}_tid{d_tid:06d}_c{cname}.jpg"
                out_path = drone_out / fname
                cv2.imwrite(str(out_path), crop)

                drone_manifest.append({
                    "view": "drone",
                    "street_frame": ts,
                    "drone_frame": td,
                    "track_id": d_tid,
                    "class_name": cname,
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "world_x": wx, "world_y": wy,
                    "x_fwd": float(x_fwd), "y_left": float(y_left),
                    "dist_m": float(dist),
                    "theta_rad": float(theta),
                    "crop_path": str(out_path)
                })

    # save manifests
    street_csv = out_dir / "street_wedge_manifest.csv"
    drone_csv  = out_dir / "drone_wedge_manifest.csv"
    pd.DataFrame(street_manifest).to_csv(street_csv, index=False)
    pd.DataFrame(drone_manifest).to_csv(drone_csv, index=False)

    street_v.release()
    drone_v.release()

    print(f"[done] street crops: {len(street_manifest)} -> {street_csv}")
    print(f"[done] drone  crops: {len(drone_manifest)} -> {drone_csv}")
    print(f"[done] out_dir: {out_dir}")


if __name__ == "__main__":
    main()
