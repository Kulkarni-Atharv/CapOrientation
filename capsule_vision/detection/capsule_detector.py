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

        # Ensure bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
             return OrientationResult(detected=False)

        # 2. Angle calculation inside the tight AI Crop
        # Because the crop is perfectly surrounding the capsule, finding the ellipse is trivial
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        _, mask   = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Cleanup mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return OrientationResult(detected=False)

        largest_cnt = max(contours, key=cv2.contourArea)
        if len(largest_cnt) < 5:
            return OrientationResult(detected=False)

        ellipse_crop = cv2.fitEllipse(largest_cnt)
        
        # Translate ellipse coordinates back to full frame
        (cx_crop, cy_crop), axes, angle = ellipse_crop
        global_cx = cx_crop + x1
        global_cy = cy_crop + y1
        global_ellipse = ((global_cx, global_cy), axes, angle)
        
        # Translate contour to full frame for visualization
        global_contour = largest_cnt + np.array([[x1, y1]])

        # 3. Type
        cap_type = self._classify_type(crop, largest_cnt)

        # 4. Straighten using global ellipse but we can just warp the whole frame
        patch = self._straighten(frame, global_ellipse)
        if patch is None or patch.size == 0:
            return OrientationResult(detected=False)

        # 5. Seam
        seam_ratio, seam_conf = self._find_seam(patch, cap_type)

        # 6. Orientation
        long_body_side, body_ratio = self._orientation(seam_ratio, global_ellipse[2])

        return OrientationResult(
            detected       = True,
            capsule_type   = cap_type,
            angle_deg      = float(angle),
            long_body_side = long_body_side,
            seam_ratio     = seam_ratio,
            body_ratio     = body_ratio,
            confidence     = float(seam_conf * ai_conf),
            center         = (int(global_cx), int(global_cy)),
            ellipse        = global_ellipse,
            contour        = global_contour,
            bbox           = (x1, y1, x2, y2)
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
