# capsule_vision/detection/capsule_detector.py
"""
Capsule Orientation Detector (YOLOv8 + OpenCV)

Pipeline per frame:
  1. YOLOv8 Model -> Finds perfect Axis-Aligned Bounding Box (AABB) of the capsule.
  2. Crop -> Isolate the capsule perfectly.
  3. OpenCV -> Find the exact angle inside the crop (ellipse fit).
  4. Straighten -> Rotate crop.
  5. Seam -> Absolute color difference gradient.
  6. Orientation -> Determine Cap vs Body.
"""

from __future__ import annotations

import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


log = logging.getLogger("capsule_detector")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class OrientationResult:
    detected:       bool
    capsule_type:   str   = "UNKNOWN"      # COLOURED | TRANSPARENT
    angle_deg:      float = 0.0            # major-axis angle in frame (°)
    long_body_side: str   = "UNKNOWN"      # LEFT | RIGHT | TOP | BOTTOM
    seam_ratio:     float = 0.5            # seam position along major axis (0–1)
    body_ratio:     float = 0.0            # fraction of capsule that is body
    confidence:     float = 0.0            # 0–1
    center:         Tuple[int, int] = field(default_factory=lambda: (0, 0))
    ellipse:        Optional[tuple] = None
    contour:        Optional[np.ndarray] = None
    bbox:           Optional[tuple] = None # x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class CapsuleDetector:
    def __init__(
        self,
        model_path:    str   = "models/capsule_best.pt",
        conf_thresh:   float = 0.50,
        sat_threshold: int   = 35,
    ) -> None:
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is not installed. Please pip install ultralytics.")

        self.conf_thresh   = conf_thresh
        self.sat_threshold = sat_threshold

        # Load YOLO Model
        if not os.path.exists(model_path):
            log.warning(f"YOLO model not found at {model_path}! AI Detection disabled until model is downloaded.")
            self.model = None
        else:
            log.info(f"Loading YOLO model from {model_path}...")
            self.model = YOLO(model_path)
            # Warmup
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            log.info("YOLO Model loaded and warmed up.")


    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> OrientationResult:
        if self.model is None:
            return OrientationResult(detected=False)

        # 1. AI Inference
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return OrientationResult(detected=False)

        # Get highest confidence box
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        ai_conf = float(box.conf[0].cpu().numpy())

        # ===================================================================
        # USER REQUEST: JUST DETECT THE CAPSULE FIRST!
        # Return immediately right after YOLO finds the box, skipping all OpenCV
        # ===================================================================
        return OrientationResult(
            detected=True,
            capsule_type="AI DETECTED",
            confidence=ai_conf,
            bbox=(x1, y1, x2, y2)
        )

    # ------------------------------------------------------------------
    # Step 3 – classify type (runs on the isolated crop)
    # ------------------------------------------------------------------
    def _classify_type(self, crop: np.ndarray, contour_crop: np.ndarray) -> str:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour_crop], -1, 255, -1)
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_sat = float(cv2.mean(hsv[:, :, 1], mask=mask)[0])
        return "COLOURED" if mean_sat > self.sat_threshold else "TRANSPARENT"

    # ------------------------------------------------------------------
    # Step 4 – straighten
    # ------------------------------------------------------------------
    def _straighten(self, frame: np.ndarray, global_ellipse: tuple) -> Optional[np.ndarray]:
        (cx, cy), (minor_ax, major_ax), angle = global_ellipse
        M       = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        pad_w = int(major_ax / 2) + 12
        pad_h = int(minor_ax / 2) + 12
        x1    = max(0, int(cx) - pad_w)
        y1    = max(0, int(cy) - pad_h)
        x2    = min(frame.shape[1], int(cx) + pad_w)
        y2    = min(frame.shape[0], int(cy) + pad_h)
        return rotated[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Step 5 – seam detection
    # ------------------------------------------------------------------
    def _find_seam(
        self, patch: np.ndarray, cap_type: str
    ) -> Tuple[float, float]:
        h, w = patch.shape[:2]
        if h < 5 or w < 5:
            return 0.5, 0.0

        # Universal Seam Detector: Sum of absolute differences along the X-axis
        patch_float = patch.astype(float)
        diff = np.abs(np.diff(patch_float, axis=1))
        signal = diff.mean(axis=(0, 2))

        signal = self._smooth(signal, sigma=max(3, w // 20))

        margin = int(w * 0.20)
        region = signal[margin: w - margin]

        if region.size == 0:
            return 0.5, 0.0

        peak_idx   = int(np.argmax(region)) + margin
        seam_ratio = peak_idx / w

        mean_val   = float(region.mean()) + 1e-6
        confidence = min(1.0, float(region.max()) / mean_val / 5.0)

        return seam_ratio, confidence

    # ------------------------------------------------------------------
    # Step 6 – orientation
    # ------------------------------------------------------------------
    @staticmethod
    def _orientation(seam_ratio: float, angle_deg: float):
        cap_on_start = seam_ratio < 0.5
        body_ratio   = max(seam_ratio, 1.0 - seam_ratio)

        norm_angle   = angle_deg % 180
        is_horizontal = (norm_angle < 45) or (norm_angle >= 135)

        if is_horizontal:
            long_body_side = "RIGHT" if cap_on_start else "LEFT"
        else:
            long_body_side = "BOTTOM" if cap_on_start else "TOP"

        return long_body_side, body_ratio

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth(signal: np.ndarray, sigma: int = 3) -> np.ndarray:
        radius = int(3 * sigma)
        x      = np.arange(-radius, radius + 1)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        padded = np.pad(signal, radius, mode="edge")
        return np.convolve(padded, kernel, mode="valid")
