from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


BoundingBox = Tuple[int, int, int, int]


def bbox_center(box: BoundingBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def bbox_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    track_id: int
    bbox: BoundingBox
    confidence: float
    keypoints: np.ndarray
    missed_frames: int = 0
    history: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        center = bbox_center(self.bbox)
        self.kalman.statePre = np.array(
            [[center[0]], [center[1]], [0], [0]], dtype=np.float32
        )
        self.kalman.statePost = self.kalman.statePre.copy()
        self.last_center = center
        self.predicted_center = center

    def predict(self) -> np.ndarray:
        prediction = self.kalman.predict()
        self.predicted_center = prediction[:2].reshape(2)
        return self.predicted_center

    def update(
        self,
        bbox: BoundingBox,
        confidence: float,
        keypoints: np.ndarray,
        label: Optional[str] = None,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.keypoints = keypoints
        measurement = bbox_center(bbox).reshape(2, 1)
        self.kalman.correct(measurement.astype(np.float32))
        self.last_center = measurement.reshape(2)
        self.missed_frames = 0
        if label:
            self.history.append(label)


class MultiObjectTracker:
    def __init__(self, max_missed_frames: int = 20, iou_threshold: float = 0.2):
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def _create_track(
        self, bbox: BoundingBox, confidence: float, keypoints: np.ndarray
    ) -> Track:
        track = Track(
            track_id=self.next_track_id,
            bbox=bbox,
            confidence=confidence,
            keypoints=keypoints,
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        return track

    def update(self, detections: List[dict]) -> List[Track]:
        track_ids = list(self.tracks.keys())
        for track in self.tracks.values():
            track.predict()

        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(track_ids)
        matches: List[Tuple[int, int]] = []

        candidates: List[Tuple[float, int, int]] = []
        for track_id in track_ids:
            for det_idx, detection in enumerate(detections):
                score = bbox_iou(self.tracks[track_id].bbox, detection["bbox"])
                if score >= self.iou_threshold:
                    candidates.append((score, track_id, det_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])
        for _, track_id, det_idx in candidates:
            if track_id in unmatched_tracks and det_idx in unmatched_detections:
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(det_idx)
                matches.append((track_id, det_idx))

        for track_id, det_idx in matches:
            detection = detections[det_idx]
            self.tracks[track_id].update(
                detection["bbox"], detection["confidence"], detection["keypoints"]
            )

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self.tracks[track_id]

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            self._create_track(
                detection["bbox"], detection["confidence"], detection["keypoints"]
            )

        return list(self.tracks.values())
