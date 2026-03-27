import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel


def make_key(frame: int, track_id: int) -> str:
    # frame refers to STREET frame index for both views (drone manifest already has street_frame)
    return f"{int(frame)}:{int(track_id)}"


def load_manifest(manifest_csv: str, view: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if "crop_path" not in df.columns:
        raise RuntimeError(f"{manifest_csv} missing crop_path column")
    if view == "street":
        if "frame" not in df.columns or "track_id" not in df.columns:
            raise RuntimeError("street manifest must have columns: frame, track_id")
        df["key_frame"] = df["frame"].astype(int)
    elif view == "drone":
        # we will key drone crops by street_frame (so matching is on street time)
        if "street_frame" not in df.columns or "track_id" not in df.columns:
            raise RuntimeError("drone manifest must have columns: street_frame, track_id")
        df["key_frame"] = df["street_frame"].astype(int)
    else:
        raise ValueError("view must be street or drone")

    df["track_id"] = df["track_id"].astype(int)
    df["key"] = [make_key(f, tid) for f, tid in zip(df["key_frame"], df["track_id"])]
    return df


def batched(iterable, n: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True, help="street_wedge_manifest.csv or drone_wedge_manifest.csv")
    ap.add_argument("--view", required=True, choices=["street", "drone"])
    ap.add_argument("--out_npz", required=True)

    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_items", type=int, default=-1, help="debug: limit number of crops (-1 = all)")
    ap.add_argument("--use_fast_processor", action="store_true", help="use fast processor (default is HF behavior)")
    args = ap.parse_args()

    df = load_manifest(args.manifest_csv, args.view)

    if args.max_items and args.max_items > 0:
        df = df.iloc[: args.max_items].copy()

    # drop duplicates per (frame, track) in case you saved multiple ranks etc.
    # keep the largest area crop if available
    if "area" in df.columns:
        df = df.sort_values("area", ascending=False)
    df = df.drop_duplicates(subset=["key"], keep="first").reset_index(drop=True)

    print(f"[info] items to embed: {len(df)} from {args.manifest_csv}")
    print(f"[info] device={args.device} batch_size={args.batch_size} model={args.model}")

    device = torch.device(args.device)

    model = CLIPModel.from_pretrained(args.model)
    model.eval()
    model.to(device)

    processor = CLIPProcessor.from_pretrained(args.model, use_fast=args.use_fast_processor)

    keys: List[str] = df["key"].tolist()
    paths: List[str] = df["crop_path"].tolist()

    out: Dict[str, np.ndarray] = {}

    for batch_idx, batch in enumerate(tqdm(list(batched(list(zip(keys, paths)), args.batch_size)), desc="Embedding")):
        bkeys = [k for k, _ in batch]
        bpaths = [p for _, p in batch]

        images = []
        valid_keys = []
        for k, p in zip(bkeys, bpaths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_keys.append(k)
            except Exception:
                # skip unreadable images
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out.pooler_output
        feats = model.visual_projection(pooled)
        feats = feats / (torch.linalg.vector_norm(feats, dim=-1, keepdim=True) + 1e-9)


        feats_np = feats.detach().cpu().numpy().astype(np.float32)

        for k, f in zip(valid_keys, feats_np):
            out[k] = f

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_npz), **out)
    print(f"[done] wrote embeddings: {out_npz}  keys={len(out)}")
    print("[note] keys are 'street_frame:track_id' for both views")


if __name__ == "__main__":
    main()
