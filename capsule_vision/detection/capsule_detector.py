# capsule_vision/detection/capsule_detector.py
"""
Capsule Orientation Detector (YOLOv8 + OpenCV)

Pipeline per frame:
  1. YOLOv8 Model -> Finds the Axis-Aligned Bounding Box (AABB) of the capsule.
  2. Crop        -> Isolate the capsule region.
  3. OpenCV      -> Find the exact angle inside the crop (ellipse fit).
  4. Straighten  -> Rotate crop so capsule is horizontal.
  5. Seam        -> Absolute colour-difference gradient to find cap/body junction.
  6. Orientation -> Determine which side is Cap vs Body.
"""

from __future__ import annotations

import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from pathlib import Path
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
    capsule_type:   str   = "UNKNOWN"    # COLOURED | TRANSPARENT
    angle_deg:      float = 0.0          # major-axis angle in frame (°)
    long_body_side: str   = "UNKNOWN"    # LEFT | RIGHT | TOP | BOTTOM
    seam_ratio:     float = 0.5          # seam position along major axis (0–1)
    body_ratio:     float = 0.0          # fraction of capsule that is body
    confidence:     float = 0.0          # 0–1
    center:         Tuple[int, int] = field(default_factory=lambda: (0, 0))
    ellipse:        Optional[tuple] = None
    contour:        Optional[np.ndarray] = None
    bbox:           Optional[tuple] = None   # x1, y1, x2, y2
    ai_conf:        float = 0.0              # raw YOLO confidence (for HUD)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class CapsuleDetector:
    def __init__(
        self,
        model_path:    str   = "models/capsule_best.pt",
        conf_thresh:   float = 0.05,   # very sensitive — trust YOLO box first
        sat_threshold: int   = 35,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO is not installed. Run: pip install ultralytics"
            )

        self.conf_thresh   = conf_thresh
        self.sat_threshold = sat_threshold

        # Resolve to absolute path so CWD doesn't matter
        if not os.path.isabs(model_path):
            project_root = Path(__file__).resolve().parents[2]
            model_path   = str(project_root / model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found at: {model_path}\n"
                "Please ensure models/capsule_best.pt exists in the project root."
            )

        log.info("Loading YOLO model from %s …", model_path)
        self.model = YOLO(model_path)

        # Warmup pass so first real frame isn't slow
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=640, verbose=False)
        log.info("YOLO model loaded and warmed up.  conf_thresh=%.3f", conf_thresh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> OrientationResult:
        h, w = frame.shape[:2]

        # ── 1. YOLO inference ──────────────────────────────────────────────
        # Pass frame as-is (BGR numpy). Ultralytics converts BGR→RGB internally.
        results = self.model.predict(
            frame,
            imgsz=640,
            conf=self.conf_thresh,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            log.debug("YOLO: 0 detections on frame (%dx%d)", w, h)
            return OrientationResult(detected=False)

        boxes = results[0].boxes

        # Log every detected box so we can diagnose the model output
        for b in boxes:
            cls_id = int(b.cls[0])
            cls_name = (results[0].names or {}).get(cls_id, str(cls_id))
            log.debug(
                "YOLO box  class=%s  conf=%.3f  xyxy=%s",
                cls_name,
                float(b.conf[0]),
                [round(v) for v in b.xyxy[0].cpu().tolist()],
            )

        # Pick the highest-confidence box
        best_box = max(boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
        ai_conf = float(best_box.conf[0])

        log.debug("YOLO best: conf=%.3f  box=(%d,%d,%d,%d)", ai_conf, x1, y1, x2, y2)

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            log.debug("Crop too small (%dx%d) — returning bbox only.", crop.shape[1], crop.shape[0])
            return OrientationResult(detected=False, bbox=(x1, y1, x2, y2), ai_conf=ai_conf)

        # ── 2. Angle via ellipse fit inside the YOLO crop ─────────────────
        ellipse_result = self._fit_ellipse(crop)
        if ellipse_result is None:
            log.debug("Ellipse fit failed — returning bbox-only result.")
            return OrientationResult(detected=False, bbox=(x1, y1, x2, y2), ai_conf=ai_conf)

        ellipse_crop, largest_cnt = ellipse_result

        # Translate ellipse & contour back to full-frame coordinates
        (cx_crop, cy_crop), axes, angle = ellipse_crop
        global_cx      = cx_crop + x1
        global_cy      = cy_crop + y1
        global_ellipse = ((global_cx, global_cy), axes, angle)
        global_contour = largest_cnt + np.array([[x1, y1]])

        # ── 3. Type classification ─────────────────────────────────────────
        cap_type = self._classify_type(crop, largest_cnt)

        # ── 4. Straighten ─────────────────────────────────────────────────
        patch = self._straighten(frame, global_ellipse)
        if patch is None or patch.size == 0:
            log.debug("Straighten produced empty patch.")
            return OrientationResult(detected=False, bbox=(x1, y1, x2, y2), ai_conf=ai_conf)

        # ── 5. Seam (junction point) ───────────────────────────────────────
        seam_ratio, seam_conf = self._find_seam(patch, cap_type)

        # ── 6. Orientation ─────────────────────────────────────────────────
        long_body_side, body_ratio = self._orientation(seam_ratio, angle)

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
            bbox           = (x1, y1, x2, y2),
            ai_conf        = ai_conf,
        )

    # ------------------------------------------------------------------
    # Step 2 – fit an ellipse to the capsule inside the crop
    # ------------------------------------------------------------------
    def _fit_ellipse(
        self, crop: np.ndarray
    ) -> Optional[Tuple[tuple, np.ndarray]]:
        """
        Returns (ellipse_in_crop_coords, largest_contour) or None on failure.

        Uses an ADAPTIVE threshold strategy: tries both BINARY_INV and BINARY
        (Otsu) and keeps whichever produces the larger single blob — this
        correctly handles both dark-capsule-on-light and light-capsule-on-dark
        setups without manual tuning.
        """
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        def _threshold_and_clean(flag: int):
            _, m = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return cnts

        cnts_inv = _threshold_and_clean(cv2.THRESH_BINARY_INV)
        cnts_reg = _threshold_and_clean(cv2.THRESH_BINARY)

        area_inv = max((cv2.contourArea(c) for c in cnts_inv), default=0)
        area_reg = max((cv2.contourArea(c) for c in cnts_reg), default=0)

        # Pick the threshold direction that isolates the bigger blob
        contours = cnts_inv if area_inv >= area_reg else cnts_reg
        log.debug(
            "Threshold: INV_area=%.0f  REG_area=%.0f  → using %s",
            area_inv, area_reg,
            "INV" if area_inv >= area_reg else "REG",
        )

        if not contours:
            log.debug("No contours found after thresholding.")
            return None

        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            log.debug("Largest contour has < 5 points — cannot fit ellipse.")
            return None

        ellipse = cv2.fitEllipse(largest)
        return ellipse, largest

    # ------------------------------------------------------------------
    # Step 3 – classify capsule type (runs on the isolated crop)
    # ------------------------------------------------------------------
    def _classify_type(self, crop: np.ndarray, contour_crop: np.ndarray) -> str:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour_crop], -1, 255, -1)
        hsv      = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_sat = float(cv2.mean(hsv[:, :, 1], mask=mask)[0])
        return "COLOURED" if mean_sat > self.sat_threshold else "TRANSPARENT"

    # ------------------------------------------------------------------
    # Step 4 – straighten the capsule for seam detection
    # ------------------------------------------------------------------
    def _straighten(
        self, frame: np.ndarray, global_ellipse: tuple
    ) -> Optional[np.ndarray]:
        (cx, cy), (ax0, ax1), angle = global_ellipse
        major_ax = max(ax0, ax1)
        minor_ax = min(ax0, ax1)

        M       = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        pad_w = int(major_ax / 2) + 15
        pad_h = int(minor_ax / 2) + 15
        rx1   = max(0, int(cx) - pad_w)
        ry1   = max(0, int(cy) - pad_h)
        rx2   = min(frame.shape[1], int(cx) + pad_w)
        ry2   = min(frame.shape[0], int(cy) + pad_h)
        return rotated[ry1:ry2, rx1:rx2]

    # ------------------------------------------------------------------
    # Step 5 – seam detection (colour-difference gradient along X)
    # ------------------------------------------------------------------
    def _find_seam(
        self, patch: np.ndarray, cap_type: str
    ) -> Tuple[float, float]:
        h, w = patch.shape[:2]
        if h < 5 or w < 5:
            return 0.5, 0.0

        patch_f = patch.astype(float)
        diff    = np.abs(np.diff(patch_f, axis=1))
        signal  = diff.mean(axis=(0, 2))
        signal  = self._smooth(signal, sigma=max(3, w // 20))

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
    # Step 6 – determine orientation from seam position + ellipse angle
    # ------------------------------------------------------------------
    @staticmethod
    def _orientation(seam_ratio: float, angle_deg: float):
        cap_on_start   = seam_ratio < 0.5
        body_ratio     = max(seam_ratio, 1.0 - seam_ratio)
        norm_angle     = angle_deg % 180
        is_horizontal  = (norm_angle < 45) or (norm_angle >= 135)

        if is_horizontal:
            long_body_side = "RIGHT" if cap_on_start else "LEFT"
        else:
            long_body_side = "BOTTOM" if cap_on_start else "TOP"

        return long_body_side, body_ratio

    # ------------------------------------------------------------------
    # Utility – Gaussian smoothing of 1-D signal
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth(signal: np.ndarray, sigma: int = 3) -> np.ndarray:
        radius = int(3 * sigma)
        x      = np.arange(-radius, radius + 1)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        padded = np.pad(signal, radius, mode="edge")
        return np.convolve(padded, kernel, mode="valid")
