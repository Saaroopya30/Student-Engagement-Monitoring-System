from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


KEYPOINT_NAMES = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}


@dataclass
class EngagementFeatures:
    head_centered: bool
    hand_raised: bool
    upright_posture: bool
    face_detected: bool
    excessive_movement: bool
    movement_score: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "head_centered": int(self.head_centered),
            "hand_raised": int(self.hand_raised),
            "upright_posture": int(self.upright_posture),
            "face_detected": int(self.face_detected),
            "excessive_movement": int(self.excessive_movement),
            "movement_score": float(self.movement_score),
        }


def safe_point(keypoints: np.ndarray, name: str, conf_threshold: float = 0.3) -> Optional[np.ndarray]:
    idx = KEYPOINT_NAMES[name]
    point = keypoints[idx]
    if point[2] < conf_threshold:
        return None
    return point[:2]


def detect_face(gray_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    roi = gray_frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if roi.size == 0:
        return False

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    return len(faces) > 0


def extract_features(
    keypoints: np.ndarray,
    bbox: Tuple[int, int, int, int],
    movement_score: float,
    gray_frame: np.ndarray,
) -> EngagementFeatures:
    nose = safe_point(keypoints, "nose")
    left_shoulder = safe_point(keypoints, "left_shoulder")
    right_shoulder = safe_point(keypoints, "right_shoulder")
    left_wrist = safe_point(keypoints, "left_wrist")
    right_wrist = safe_point(keypoints, "right_wrist")
    left_hip = safe_point(keypoints, "left_hip")
    right_hip = safe_point(keypoints, "right_hip")

    head_centered = False
    if nose is not None and left_shoulder is not None and right_shoulder is not None:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_span = max(abs(left_shoulder[0] - right_shoulder[0]), 1.0)
        head_centered = abs(nose[0] - shoulder_center_x) <= shoulder_span * 0.35

    hand_raised = False
    shoulder_line = None
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_line = min(left_shoulder[1], right_shoulder[1])
    if shoulder_line is not None:
        wrists = [point for point in (left_wrist, right_wrist) if point is not None]
        hand_raised = any(wrist[1] < shoulder_line for wrist in wrists)

    upright_posture = False
    if (
        left_shoulder is not None
        and right_shoulder is not None
        and left_hip is not None
        and right_hip is not None
    ):
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        hip_mid = (left_hip + right_hip) / 2.0
        torso_height = abs(hip_mid[1] - shoulder_mid[1])
        torso_shift = abs(hip_mid[0] - shoulder_mid[0])
        upright_posture = torso_height > 30 and torso_shift <= torso_height * 0.35

    face_detected = detect_face(gray_frame, bbox)
    excessive_movement = movement_score > 35.0

    return EngagementFeatures(
        head_centered=head_centered,
        hand_raised=hand_raised,
        upright_posture=upright_posture,
        face_detected=face_detected,
        excessive_movement=excessive_movement,
        movement_score=movement_score,
    )


def classify_engagement(features: EngagementFeatures) -> Tuple[str, str]:
    score = 0

    if features.face_detected:
        score += 1
    if features.head_centered:
        score += 2
    if features.upright_posture:
        score += 2
    if features.hand_raised:
        score += 2
    if features.excessive_movement:
        score -= 2

    if score >= 4:
        label = "Engaged"
        feedback = "Students appear attentive. Continue with the current pace."
    elif score >= 2:
        label = "Partially Engaged"
        feedback = "Mixed attention signals detected. Consider asking a quick question."
    else:
        label = "Not Engaged"
        feedback = "Low engagement detected. Consider intervention or interaction."

    return label, feedback
