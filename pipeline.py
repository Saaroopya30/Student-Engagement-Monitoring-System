import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from engagement import classify_engagement, extract_features
from reporting import generate_reports
from storage import EngagementStorage
from tracking import MultiObjectTracker, bbox_center

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".runtime_cache"
ULTRALYTICS_DIR = CACHE_DIR / "ultralytics"
MPL_DIR = CACHE_DIR / "matplotlib"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

FRAME_SIZE = (960, 540)
PERSON_CLASS_ID = 0
MODEL = YOLO("yolov8n-pose.pt")


class ProcessingCancelled(Exception):
    pass


def create_video_writer(output_path: Path, fps: float):
    codecs = ["avc1", "mp4v"]
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            FRAME_SIZE,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("Could not initialize a video writer for annotated output.")


def process_video(
    input_path,
    output_path,
    report_dir,
    display=False,
    max_frames=None,
    progress_callback=None,
    should_cancel=None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    tracker = MultiObjectTracker(max_missed_frames=25, iou_threshold=0.15)
    storage = EngagementStorage(report_dir)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and total_frames:
        total_frames = min(total_frames, max_frames)
    writer, selected_codec = create_video_writer(output_path, fps)

    if progress_callback:
        progress_callback(
            {
                "stage": "processing",
                "processed_frames": 0,
                "total_frames": total_frames,
                "percent": 0,
                "eta_seconds": None,
                "message": "Video loaded. Analysis has started.",
            }
        )

    start_time = time.time()

    engaged_count = 0
    partial_count = 0
    not_engaged_count = 0
    processed_frames = 0
    previous_centers = {}
    latest_feedback = "System is monitoring classroom engagement."
    written_records = []

    while True:
        if should_cancel and should_cancel():
            raise ProcessingCancelled("Analysis cancelled by user.")

        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1
        if max_frames is not None and processed_frames > max_frames:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = MODEL(rgb, verbose=False)
        detections = []
        result = results[0]
        if result.boxes is not None and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy()

            for box, cls, confidence, pose_points in zip(
                boxes, classes, confidences, keypoints
            ):
                if int(cls) != PERSON_CLASS_ID:
                    continue
                detections.append(
                    {
                        "bbox": tuple(map(int, box)),
                        "confidence": float(confidence),
                        "keypoints": pose_points,
                    }
                )

        active_tracks = tracker.update(detections)
        frame_records = []
        frame_counts = {"Engaged": 0, "Partially Engaged": 0, "Not Engaged": 0}

        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            center = bbox_center(track.bbox)
            movement = 0.0
            if track.track_id in previous_centers:
                movement = float(
                    np.linalg.norm(center - previous_centers[track.track_id])
                )
            previous_centers[track.track_id] = center

            features = extract_features(track.keypoints, track.bbox, movement, gray)
            label, latest_feedback = classify_engagement(features)
            track.history.append(label)

            if label == "Engaged":
                engaged_count += 1
                frame_counts[label] += 1
                color = (0, 255, 0)
            elif label == "Partially Engaged":
                partial_count += 1
                frame_counts[label] += 1
                color = (0, 255, 255)
            else:
                not_engaged_count += 1
                frame_counts[label] += 1
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track.track_id} - {label}",
                (x1, max(y1 - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            frame_records.append(
                {
                    "frame_index": processed_frames,
                    "timestamp_seconds": round((processed_frames - 1) / fps, 2),
                    "track_id": track.track_id,
                    "label": label,
                    **features.as_dict(),
                }
            )

        storage.write_records(frame_records)
        written_records.extend(frame_records)

        total = engaged_count + partial_count + not_engaged_count
        if total > 0:
            percent = (engaged_count / total) * 100
            cv2.putText(
                frame,
                f"Engagement: {percent:.2f}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        cv2.putText(
            frame,
            (
                f"Frame E/P/N: {frame_counts['Engaged']}/"
                f"{frame_counts['Partially Engaged']}/"
                f"{frame_counts['Not Engaged']}"
            ),
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 40, 255),
            2,
        )
        cv2.putText(
            frame,
            latest_feedback[:90],
            (20, FRAME_SIZE[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

        if progress_callback and (processed_frames == 1 or processed_frames % 10 == 0):
            percent = 0
            eta_seconds = None
            if total_frames:
                percent = min(100, round((processed_frames / total_frames) * 100))
                elapsed = time.time() - start_time
                if processed_frames > 0 and elapsed > 0:
                    seconds_per_frame = elapsed / processed_frames
                    eta_seconds = max(
                        0, int((total_frames - processed_frames) * seconds_per_frame)
                    )
            progress_callback(
                {
                    "stage": "processing",
                    "processed_frames": processed_frames,
                    "total_frames": total_frames,
                    "percent": percent,
                    "eta_seconds": eta_seconds,
                    "message": f"Analyzing frame {processed_frames}" + (
                        f" of {total_frames}" if total_frames else ""
                    ),
                }
            )

        if display:
            cv2.imshow("Student Engagement System", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    storage.close()
    cv2.destroyAllWindows()

    if progress_callback:
        progress_callback(
            {
                "stage": "reporting",
                "processed_frames": processed_frames,
                "total_frames": total_frames,
                "percent": 97,
                "eta_seconds": 3,
                "message": "Generating charts, summaries, and downloadable reports.",
            }
        )

    if should_cancel and should_cancel():
        raise ProcessingCancelled("Analysis cancelled by user.")

    report_files = generate_reports(written_records, report_dir)
    results = {
        "processed_frames": processed_frames,
        "output_video": output_path,
        "video_codec": selected_codec,
        "database": report_dir / "engagement_results.db",
        "csv_log": report_dir / "engagement_results.csv",
        "summary_json": report_files["summary_json"],
        "student_summary_csv": report_files["student_summary_csv"],
        "trend_chart": report_files["trend_chart"],
        "distribution_chart": report_files["distribution_chart"],
        "dashboard_html": report_files["dashboard_html"],
    }
    if progress_callback:
        progress_callback(
            {
                "stage": "completed",
                "processed_frames": processed_frames,
                "total_frames": total_frames,
                "percent": 100,
                "eta_seconds": 0,
                "message": "Analysis complete. Your reports are ready.",
            }
        )
    return results
