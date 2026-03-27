import os
import pandas as pd
import numpy as np
import streamlit as st
import cv2

# -----------------------------
# Utils
# -----------------------------

def norm_class(x: str) -> str:
    x = str(x).strip().lower()
    if x in ["pedestrian", "person", "people"]:
        return "person"
    if x in ["bike", "bicycle", "cyclist"]:
        return "bicycle"
    if x in ["motorbike", "motorcycle"]:
        return "motorcycle"
    return x


@st.cache_resource
def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def get_frame(cap, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame  # BGR


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_or_init_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)


def append_rows_csv(existing: pd.DataFrame, rows: list[dict], path: str) -> pd.DataFrame:
    if len(rows) == 0:
        return existing
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def ensure_cols(df, req, name):
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def draw_boxes(
    img_bgr,
    df,
    id_col,
    cls_col="class_name",
    color=(0, 200, 0),
    thickness=2,
    font_scale=0.6,
    alpha_fill=0.0,
    highlight_ids=None,
    highlight_color=(0, 0, 255),
):
    """Draw bboxes with labels on image.

    df must have bbox_x1,bbox_y1,bbox_x2,bbox_y2.
    highlight_ids: set of ids to draw in highlight_color.
    """
    if img_bgr is None:
        return None

    img = img_bgr.copy()
    overlay = img.copy()
    highlight_ids = set(highlight_ids or [])

    for _, r in df.iterrows():
        x1 = int(r["bbox_x1"])
        y1 = int(r["bbox_y1"])
        x2 = int(r["bbox_x2"])
        y2 = int(r["bbox_y2"])
        tid = int(r[id_col])
        cname = str(r.get(cls_col, "")).strip()

        col = highlight_color if tid in highlight_ids else color

        if alpha_fill > 0:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)

        cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness)

        label = f"{tid}"
        if cname:
            label = f"{tid}|{cname}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        y_text = max(0, y1 - 5)
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text), col, -1)
        cv2.putText(
            img,
            label,
            (x1 + 3, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if alpha_fill > 0:
        img = cv2.addWeighted(overlay, alpha_fill, img, 1 - alpha_fill, 0)

    return img


def fit_max_side(img_rgb, max_side=1100):
    h, w = img_rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_rgb
    s = max_side / float(m)
    return cv2.resize(img_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# -----------------------------
# Main App
# -----------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Cross-view GT Pair Annotator (Frame Context)")

    with st.sidebar:
        st.header("Paths & Settings")

        scene_id = st.text_input("scene_id", value="scene01")

        track_map_csv = st.text_input("track_map_csv (suggestions)", value="")
        street_manifest_path = st.text_input("street_manifest.csv", value="")
        drone_manifest_path = st.text_input("drone_manifest.csv", value="")

        street_video = st.text_input("street_video.mp4", value="")
        drone_video = st.text_input("drone_video.mp4", value="")

        out_gt_pairs = st.text_input("out_gt_pairs.csv", value="gt_pairs.csv")
        out_audit = st.text_input("out_audit.csv", value="gt_audit.csv")

        mode = st.selectbox("annotation_mode", ["frame_batch", "single_track"], index=0)
        frame_batch_strategy = st.selectbox(
            "frame_batch_strategy",
            ["densest_first", "chronological"],
            index=0,
            help="densest_first shows frames with the most unresolved suggested tracks first",
        )
        min_conf = st.slider("min_conf (filter suggestions)", 0.0, 1.0, 0.2, 0.01)
        sort_by = st.selectbox("sort_by", ["low_conf_first", "high_conf_first"])
        show_all_boxes = st.checkbox("Show ALL wedge boxes in frame", value=True)
        max_boxes_draw = st.slider("Max boxes to draw per view (speed)", 10, 200, 80, 5)
        alpha_fill = st.slider("Box fill alpha", 0.0, 0.8, 0.0, 0.05)
        max_side = st.slider("Max image side (render)", 600, 2000, 1100, 50)

        st.markdown("---")
        st.subheader("Wedge filter (Drone)")
        apply_drone_wedge = st.checkbox("Apply drone wedge filter (dist/theta)", value=True)
        max_range_m = st.slider("max_range_m", 10.0, 150.0, 70.0, 1.0)
        fov_deg = st.slider("fov_deg", 30, 180, 120, 5)

        st.markdown("---")
        st.caption("Tip: Use 'best frame (max area)' and confirm using full-frame context.")

    # Validate paths
    if not (track_map_csv and os.path.exists(track_map_csv)):
        st.warning("Set a valid track_map_csv in sidebar.")
        return
    if not (street_manifest_path and os.path.exists(street_manifest_path)):
        st.warning("Set a valid street_manifest_path.")
        return
    if not (drone_manifest_path and os.path.exists(drone_manifest_path)):
        st.warning("Set a valid drone_manifest_path.")
        return
    if not (street_video and os.path.exists(street_video)):
        st.warning("Set a valid street_video path.")
        return
    if not (drone_video and os.path.exists(drone_video)):
        st.warning("Set a valid drone_video path.")
        return

    tm = pd.read_csv(track_map_csv)
    if "stitch_confidence" in tm.columns:
        tm = tm.rename(columns={"stitch_confidence": "confidence"})
    if "confidence" not in tm.columns:
        tm["confidence"] = 0.0

    tm = tm[tm["confidence"].astype(float) >= float(min_conf)].copy()
    tm = tm.sort_values("confidence", ascending=(sort_by == "low_conf_first"))
    tm = tm.reset_index(drop=True)

    # one suggestion row per street track
    tm["street_track_id"] = tm["street_track_id"].astype(int)
    tm_lookup = tm.drop_duplicates(["street_track_id"], keep="first").set_index("street_track_id", drop=False)

    S = pd.read_csv(street_manifest_path)
    D = pd.read_csv(drone_manifest_path)

    ensure_cols(
        S,
        [
            "view",
            "frame",
            "track_id",
            "class_name",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "area",
        ],
        "Street manifest",
    )
    ensure_cols(
        D,
        [
            "view",
            "street_frame",
            "drone_frame",
            "track_id",
            "class_name",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ],
        "Drone manifest",
    )

    S["class_name"] = S["class_name"].apply(norm_class)
    D["class_name"] = D["class_name"].apply(norm_class)

    total_unique_street_tracks = int(S["track_id"].astype(int).nunique())
    total_unique_drone_tracks = int(D["track_id"].astype(int).nunique())

    # half FOV in radians
    half_fov_rad = np.deg2rad(float(fov_deg) / 2.0)

    # Load existing decisions so we can resume
    gt_pairs = load_or_init_csv(out_gt_pairs, ["scene_id", "street_track_id", "drone_track_id", "class_name"])
    audit = load_or_init_csv(
        out_audit,
        [
            "scene_id",
            "street_track_id",
            "suggested_drone_track_id",
            "final_drone_track_id",
            "decision",
            "confidence",
            "note",
        ],
    )

    done = set((str(r["scene_id"]), int(r["street_track_id"])) for _, r in gt_pairs.iterrows())
    done |= set((str(r["scene_id"]), int(r["street_track_id"])) for _, r in audit.iterrows())

    done_track_ids_scene = set([k[1] for k in done if k[0] == scene_id])
    remaining_street_tracks_scene = max(0, total_unique_street_tracks - len(done_track_ids_scene))

    with st.sidebar:
        st.markdown("---")
        st.subheader("Scene summary")
        st.write(f"Unique street tracks in whole video: {total_unique_street_tracks}")
        st.write(f"Unique drone tracks in whole video: {total_unique_drone_tracks}")
        st.write(f"Annotated street tracks in this scene: {len(done_track_ids_scene)}")
        st.write(f"Remaining street tracks in this scene: {remaining_street_tracks_scene}")

    if mode == "frame_batch":
        # unresolved tracks = suggestions not yet accepted/rejected
        unresolved_tm = tm[~tm["street_track_id"].astype(int).isin([k[1] for k in done if k[0] == scene_id])].copy()
        unresolved_track_ids = set(unresolved_tm["street_track_id"].astype(int).tolist())

        # candidate frames: frames containing at least one unresolved suggested street track
        frame_df = S[S["track_id"].astype(int).isin(unresolved_track_ids)].copy()
        frame_counts = (
            frame_df.groupby("frame")["track_id"]
            .nunique()
            .reset_index(name="n_unresolved")
        )
        if frame_batch_strategy == "densest_first":
            frame_counts = frame_counts.sort_values(["n_unresolved", "frame"], ascending=[False, True])
        else:
            frame_counts = frame_counts.sort_values(["frame"], ascending=[True])
        candidate_frames = frame_counts["frame"].astype(int).tolist()
        frame_count_map = {int(r["frame"]): int(r["n_unresolved"]) for _, r in frame_counts.iterrows()}

        if len(candidate_frames) == 0:
            st.success("All suggested tracks for this scene are already annotated ✅")
            return

        if "frame_batch_idx" not in st.session_state:
            st.session_state.frame_batch_idx = 0

        # keep index valid
        st.session_state.frame_batch_idx = max(0, min(st.session_state.frame_batch_idx, len(candidate_frames) - 1))
        current_frame = candidate_frames[st.session_state.frame_batch_idx]

        top_a, top_b, top_c = st.columns([2, 3, 3])
        with top_a:
            if st.button("⬅️ Prev frame", use_container_width=True):
                st.session_state.frame_batch_idx = max(0, st.session_state.frame_batch_idx - 1)
                st.rerun()
        with top_b:
            st.write(f"Frame batch {st.session_state.frame_batch_idx + 1}/{len(candidate_frames)}")
            st.write(f"Current street frame: {current_frame}")
            st.write(f"Unresolved suggestions in this frame: {frame_count_map.get(int(current_frame), 0)}")
        with top_c:
            frame_options = [f"{fr}  ({frame_count_map.get(int(fr), 0)} tracks)" for fr in candidate_frames]
            selected_label = st.selectbox(
                "Jump to frame",
                options=frame_options,
                index=candidate_frames.index(current_frame),
            )
            selected_frame = int(str(selected_label).split()[0])
            if int(selected_frame) != int(current_frame):
                st.session_state.frame_batch_idx = candidate_frames.index(int(selected_frame))
                st.rerun()

        t = int(candidate_frames[st.session_state.frame_batch_idx])

        # visible unresolved street tracks in this frame
        S_t_all = S[S["frame"].astype(int) == t].copy()
        S_batch = S_t_all[S_t_all["track_id"].astype(int).isin(unresolved_track_ids)].copy()
        if len(S_batch) == 0:
            st.warning("No unresolved suggested street tracks in this frame. Moving to next frame.")
            if st.session_state.frame_batch_idx < len(candidate_frames) - 1:
                st.session_state.frame_batch_idx += 1
                st.rerun()
            return

        # attach suggestion info from track map
        sugg_rows = []
        seen_sids = set()
        for _, sr in S_batch.sort_values("area", ascending=False).iterrows():
            s_tid = int(sr["track_id"])
            if s_tid in seen_sids:
                continue
            if s_tid not in tm_lookup.index:
                continue
            seen_sids.add(s_tid)
            tr = tm_lookup.loc[s_tid]
            sugg_rows.append({
                "street_track_id": s_tid,
                "class_name": str(sr["class_name"]),
                "street_area": float(sr.get("area", 0.0)),
                "suggested_drone_track_id": int(tr.get("drone_track_id", -1)),
                "confidence": float(tr.get("confidence", 0.0)),
                "decision": "accept",
                "final_drone_track_id": int(tr.get("drone_track_id", -1)),
            })

        batch_df = pd.DataFrame(sugg_rows)
        if len(batch_df) == 0:
            st.warning("No suggestion rows available for unresolved tracks in this frame.")
            return

        # drone frame and wedge-filtered detections for this street frame
        D_t = D[D["street_frame"].astype(int) == t].copy()
        if apply_drone_wedge and len(D_t) > 0 and ("dist_m" in D_t.columns) and ("theta_rad" in D_t.columns):
            D_t = D_t[
                (D_t["dist_m"].astype(float) <= float(max_range_m))
                & (np.abs(D_t["theta_rad"].astype(float)) <= float(half_fov_rad))
            ].copy()
        drone_frame = int(D_t["drone_frame"].iloc[0]) if len(D_t) else t

        # read and draw frames
        street_cap = open_video(street_video)
        drone_cap = open_video(drone_video)
        street_img = get_frame(street_cap, t)
        drone_img = get_frame(drone_cap, drone_frame)
        if street_img is None:
            st.error(f"Could not read street frame {t}")
            return
        if drone_img is None:
            st.error(f"Could not read drone frame {drone_frame}")
            return

        visible_s_ids = set(batch_df["street_track_id"].astype(int).tolist())
        visible_d_ids = set([int(x) for x in batch_df["suggested_drone_track_id"].tolist() if int(x) >= 0])

        if show_all_boxes:
            S_draw = S_t_all.sort_values("area", ascending=False).head(max_boxes_draw)
            if "dist_m" in D_t.columns:
                D_draw = D_t.sort_values("dist_m", ascending=True).head(max_boxes_draw)
            else:
                D_draw = D_t.head(max_boxes_draw)
        else:
            S_draw = S_t_all[S_t_all["track_id"].astype(int).isin(visible_s_ids)]
            D_draw = D_t[D_t["track_id"].astype(int).isin(visible_d_ids)]

        street_ov = draw_boxes(
            street_img,
            S_draw,
            id_col="track_id",
            cls_col="class_name",
            color=(0, 200, 0),
            highlight_ids=visible_s_ids,
            highlight_color=(0, 0, 255),
            alpha_fill=alpha_fill,
        )
        drone_ov = draw_boxes(
            drone_img,
            D_draw,
            id_col="track_id",
            cls_col="class_name",
            color=(0, 200, 0),
            highlight_ids=visible_d_ids,
            highlight_color=(0, 0, 255),
            alpha_fill=alpha_fill,
        )

        street_rgb = fit_max_side(bgr_to_rgb(street_ov), max_side=max_side)
        drone_rgb = fit_max_side(bgr_to_rgb(drone_ov), max_side=max_side)

        colL, colR = st.columns(2)
        with colL:
            st.subheader(f"Street frame {t}")
            st.image(street_rgb, use_container_width=True)
        with colR:
            st.subheader(f"Drone frame {drone_frame} (street_frame={t})")
            st.image(drone_rgb, use_container_width=True)

        st.markdown("---")
        st.subheader("Batch decisions for all visible unresolved suggestions in this frame")
        st.caption("Edit decision/final_drone_track_id for multiple rows, then save all at once.")
        if len(batch_df) <= 2:
            st.info("This frame is sparse. Use frame_batch_strategy='densest_first' or jump to a frame with more visible unresolved tracks.")

        edited = st.data_editor(
            batch_df[["street_track_id", "class_name", "street_area", "suggested_drone_track_id", "confidence", "decision", "final_drone_track_id"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "street_track_id": st.column_config.NumberColumn(disabled=True),
                "class_name": st.column_config.TextColumn(disabled=True),
                "street_area": st.column_config.NumberColumn(disabled=True),
                "suggested_drone_track_id": st.column_config.NumberColumn(disabled=True),
                "confidence": st.column_config.NumberColumn(disabled=True, format="%.3f"),
                "decision": st.column_config.SelectboxColumn(options=["accept", "reject"]),
                "final_drone_track_id": st.column_config.NumberColumn(step=1),
            },
            key=f"batch_editor_{scene_id}_{t}",
        )

        save_c1, save_c2, save_c3 = st.columns([2, 2, 3])
        with save_c1:
            save_clicked = st.button("💾 Save all decisions in this frame", use_container_width=True)
        with save_c2:
            next_clicked = st.button("➡️ Save and go to next frame", use_container_width=True)
        with save_c3:
            st.write(f"Rows in this frame: {len(edited)}")

        if save_clicked or next_clicked:
            gt_rows = []
            audit_rows = []
            for _, r in edited.iterrows():
                s_tid = int(r["street_track_id"])

                d_sug_raw = r["suggested_drone_track_id"]
                d_sug = int(d_sug_raw) if pd.notna(d_sug_raw) else -1

                final_raw = r["final_drone_track_id"]
                final_d = int(final_raw) if pd.notna(final_raw) else -1

                decision = str(r["decision"]).strip().lower()
                cname = str(r["class_name"])
                conf = float(r["confidence"]) if pd.notna(r["confidence"]) else 0.0

                if decision == "accept" and final_d >= 0:
                    gt_rows.append({
                        "scene_id": scene_id,
                        "street_track_id": s_tid,
                        "drone_track_id": final_d,
                        "class_name": cname,
                    })
                    audit_rows.append({
                        "scene_id": scene_id,
                        "street_track_id": s_tid,
                        "suggested_drone_track_id": d_sug,
                        "final_drone_track_id": final_d,
                        "decision": "accept",
                        "confidence": conf,
                        "note": f"frame_batch frame={t}",
                    })
                else:
                    audit_rows.append({
                        "scene_id": scene_id,
                        "street_track_id": s_tid,
                        "suggested_drone_track_id": d_sug,
                        "final_drone_track_id": -1,
                        "decision": "reject",
                        "confidence": conf,
                        "note": f"frame_batch frame={t}",
                    })

            gt_pairs = append_rows_csv(gt_pairs, gt_rows, out_gt_pairs)
            audit = append_rows_csv(audit, audit_rows, out_audit)

            if next_clicked and st.session_state.frame_batch_idx < len(candidate_frames) - 1:
                st.session_state.frame_batch_idx += 1
            st.rerun()

        return

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    # Skip already-done tracks in single-track mode
    while st.session_state.idx < len(tm):
        rr_tmp = tm.iloc[st.session_state.idx]
        key_tmp = (scene_id, int(rr_tmp["street_track_id"]))
        if key_tmp in done:
            st.session_state.idx += 1
        else:
            break

    if st.session_state.idx >= len(tm):
        st.success("All suggestions in this filtered set are already annotated ✅")
        return

    # ------------------------------
    # Original single-track mode
    # ------------------------------
    rr = tm.iloc[st.session_state.idx]
    s_tid = int(rr["street_track_id"])
    d_sug = int(rr["drone_track_id"])
    conf = float(rr.get("confidence", 0.0))

    # All frames where this street track appears
    S_tid = S[S["track_id"].astype(int) == s_tid].copy()
    if len(S_tid) == 0:
        st.error(f"Street track {s_tid} not found in street manifest.")
        st.session_state.idx += 1
        st.rerun()

    # Best frame = max area
    best_row = S_tid.sort_values("area", ascending=False).iloc[0]
    best_frame = int(best_row["frame"])
    cname = str(best_row["class_name"])

    frames = sorted(S_tid["frame"].astype(int).unique().tolist())

    st.write(
        f"Whole video: {total_unique_street_tracks} unique street tracks | {total_unique_drone_tracks} unique drone tracks"
    )
    st.caption(
        f"{scene_id} | suggestion {st.session_state.idx+1}/{len(tm)} | class={cname} | street={s_tid} | suggested drone={d_sug} | conf={conf:.3f}"
    )

    ctop1, ctop2, ctop3 = st.columns([2, 2, 3])
    with ctop1:
        if st.button("⏩ Jump to best frame (max area)", use_container_width=True):
            st.session_state.sel_frame = best_frame
            st.rerun()
    with ctop2:
        st.write(f"Track visible frames: {len(frames)}")
        st.write(f"Best frame: {best_frame}")
    with ctop3:
        if "sel_frame" not in st.session_state or st.session_state.sel_frame not in frames:
            st.session_state.sel_frame = best_frame
        st.session_state.sel_frame = st.selectbox(
            "Select street frame",
            options=frames,
            index=frames.index(st.session_state.sel_frame),
        )

    t = int(st.session_state.sel_frame)

    # Filter per-frame detections
    S_t = S[S["frame"].astype(int) == t].copy()
    D_t = D[D["street_frame"].astype(int) == t].copy()

    # Drone wedge filter (dist/theta)
    if apply_drone_wedge and len(D_t) > 0 and ("dist_m" in D_t.columns) and ("theta_rad" in D_t.columns):
        D_t = D_t[
            (D_t["dist_m"].astype(float) <= float(max_range_m))
            & (np.abs(D_t["theta_rad"].astype(float)) <= float(half_fov_rad))
        ].copy()

    # Drone frame index (still works even if D_t is empty after filtering)
    drone_frame = int(D_t["drone_frame"].iloc[0]) if len(D_t) else t

    # Draw subsets for speed/clarity
    if show_all_boxes:
        S_draw = S_t.sort_values("area", ascending=False).head(max_boxes_draw)
        if "dist_m" in D_t.columns:
            D_draw = D_t.sort_values("dist_m", ascending=True).head(max_boxes_draw)
        else:
            D_draw = D_t.head(max_boxes_draw)
    else:
        S_draw = S_t[S_t["track_id"].astype(int) == s_tid]
        D_draw = D_t[D_t["track_id"].astype(int) == d_sug]

    # Read video frames
    street_cap = open_video(street_video)
    drone_cap = open_video(drone_video)

    street_img = get_frame(street_cap, t)
    drone_img = get_frame(drone_cap, drone_frame)

    if street_img is None:
        st.error(f"Could not read street frame {t}")
        return
    if drone_img is None:
        st.error(f"Could not read drone frame {drone_frame}")
        return

    street_ov = draw_boxes(
        street_img,
        S_draw,
        id_col="track_id",
        cls_col="class_name",
        color=(0, 200, 0),
        highlight_ids={s_tid},
        highlight_color=(0, 0, 255),
        alpha_fill=alpha_fill,
    )
    drone_ov = draw_boxes(
        drone_img,
        D_draw,
        id_col="track_id",
        cls_col="class_name",
        color=(0, 200, 0),
        highlight_ids={d_sug},
        highlight_color=(0, 0, 255),
        alpha_fill=alpha_fill,
    )

    street_rgb = fit_max_side(bgr_to_rgb(street_ov), max_side=max_side)
    drone_rgb = fit_max_side(bgr_to_rgb(drone_ov), max_side=max_side)

    colL, colR = st.columns(2)
    with colL:
        st.subheader(f"Street frame {t}")
        st.image(street_rgb, use_container_width=True)
    with colR:
        st.subheader(f"Drone frame {drone_frame} (street_frame={t})")
        st.image(drone_rgb, use_container_width=True)

    st.markdown("---")

    def append_audit(decision, final_d, note=""):
        nonlocal audit
        audit = pd.concat(
            [
                audit,
                pd.DataFrame(
                    [
                        {
                            "scene_id": scene_id,
                            "street_track_id": s_tid,
                            "suggested_drone_track_id": d_sug,
                            "final_drone_track_id": int(final_d),
                            "decision": decision,
                            "confidence": conf,
                            "note": note,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        audit.to_csv(out_audit, index=False)

    def append_gt(final_d):
        nonlocal gt_pairs
        gt_pairs = pd.concat(
            [
                gt_pairs,
                pd.DataFrame(
                    [
                        {
                            "scene_id": scene_id,
                            "street_track_id": s_tid,
                            "drone_track_id": int(final_d),
                            "class_name": cname,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        gt_pairs.to_csv(out_gt_pairs, index=False)

    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])

    with c1:
        if st.button("✅ Accept suggested", use_container_width=True):
            append_gt(d_sug)
            append_audit("accept", d_sug, note=f"frame={t}")
            st.session_state.idx += 1
            st.rerun()

    with c2:
        if st.button("❌ Reject / No match", use_container_width=True):
            append_audit("reject", -1, note=f"frame={t}")
            st.session_state.idx += 1
            st.rerun()

    with c3:
        new_id = st.text_input("Correct drone_track_id (or -1)", value=str(d_sug))
        if st.button("✏️ Save correction", use_container_width=True):
            try:
                final_d = int(new_id)
            except Exception:
                st.error("Invalid integer")
                st.stop()
            if final_d >= 0:
                append_gt(final_d)
                append_audit("accept", final_d, note=f"manual_correction frame={t}")
            else:
                append_audit("reject", -1, note=f"manual_reject frame={t}")
            st.session_state.idx += 1
            st.rerun()

    with c4:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("⬅️ Prev", use_container_width=True):
                st.session_state.idx = max(0, st.session_state.idx - 1)
                st.rerun()
        with b2:
            if st.button("➡️ Next", use_container_width=True):
                st.session_state.idx += 1
                st.rerun()


if __name__ == "__main__":
    main()
