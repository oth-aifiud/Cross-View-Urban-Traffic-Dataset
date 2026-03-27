from collections import defaultdict
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------
# Config
# -----------------------
video_path = "/data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold_50m/Dr_MartinLuther_Luitpold_50m_bike_blurred.mp4"           # input video
output_video_path = "/data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold_50m/01_Dr_MartinLuther_Luitpold_50m_bike_annotated.mp4"    # output video
output_csv_path = "/data/home/bhp39283/disk/detection-tracking/dataset/01_Dr_MartinLuther_Luitpold_50m/01_Dr_MartinLuther_Luitpold_50m_bike_detections.csv"              # output CSV file

# -----------------------
# Load the YOLO11 model
# -----------------------
model = YOLO("yolo11l.pt")

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Error opening video file: {video_path}")

# Get video properties for the output writer
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Fallback FPS if not available
if fps == 0 or np.isnan(fps):
    fps = 25.0

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "XVID"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Store the track history for drawing trails
track_history = defaultdict(lambda: [])

# -----------------------
# Prepare CSV file
# -----------------------
csv_file = open(output_csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

# CSV header
csv_writer.writerow([
    "frame",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x_center",
    "y_center",
    "width",
    "height"
])

# -----------------------
# Process video frames
# -----------------------
frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # end of video

    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, tracker= 'bytetrack.yaml',persist=True)
    if not results:
        out.write(frame)
        frame_idx += 1
        continue

    result = results[0]

    # Get the boxes and track IDs
    if result.boxes is not None and result.boxes.id is not None:
        # (cx, cy, w, h) in pixels
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        # classes and confidences
        if result.boxes.cls is not None:
            cls_ids = result.boxes.cls.int().cpu().tolist()
        else:
            cls_ids = [None] * len(track_ids)

        if result.boxes.conf is not None:
            confs = result.boxes.conf.cpu().numpy().tolist()
        else:
            confs = [None] * len(track_ids)

        names = result.names  # class id -> label name dict

        # Visualize YOLO predictions on the frame
        frame = result.plot(font_size=0.4 , line_width=1)

        # Plot the tracks + write CSV rows
        for (cx, cy, w, h), track_id, cls_id, conf in zip(
            boxes_xywh, track_ids, cls_ids, confs
        ):
            # -----------------------
            # Track history for trail drawing
            # -----------------------
            track = track_history[track_id]
            track.append((float(cx), float(cy)))  # x, y center point

            # Keep only last 30 positions for each id
            if len(track) > 30:
                track.pop(0)

            # Draw the polyline for this track
            if len(track) > 1:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame,
                    [points],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=2,
                )

            # -----------------------
            # Write detection info to CSV
            # -----------------------
            if cls_id is not None and int(cls_id) in names:
                class_name = names[int(cls_id)]
            else:
                class_name = "object"
            csv_writer.writerow([
                frame_idx,
                int(track_id),
                int(cls_id) if cls_id is not None else -1,
                class_name,
                float(conf) if conf is not None else -1.0,
                float(cx),
                float(cy),
                float(w),
                float(h),
            ])

    # Write the annotated frame to the output video
    out.write(frame)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx} frames...")

# -----------------------
# Cleanup
# -----------------------
cap.release()
out.release()
csv_file.close()

print(f"Done. Saved tracked video to: {output_video_path}")
print(f"Saved detection CSV to: {output_csv_path}")
