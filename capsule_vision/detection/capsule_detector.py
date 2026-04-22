# capsule_vision/detection/capsule_detector.py
"""
Capsule Orientation Detector

Detects capsule orientation for two types:
  - COLOURED  : one half has pigment, seam found via hue/saturation boundary
  - TRANSPARENT: both halves clear,   seam found via edge/gradient boundary

Pipeline per frame:
  1. Threshold → find capsule mask
  2. Fit ellipse → get major axis, angle, center
  3. Classify type (coloured vs transparent)
  4. Straighten capsule along major axis
  5. Build 1-D intensity profile → locate seam
  6. Compare left/right lengths → determine orientation
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


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


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class CapsuleDetector:
    """
    Parameters
    ----------
    min_area        : minimum contour area (px²) to be considered a capsule
    max_area        : maximum contour area (px²)
    aspect_ratio_min: minimum major/minor ratio (capsules are elongated)
    sat_threshold   : HSV saturation threshold to distinguish coloured vs transparent
    """

    def __init__(
        self,
        min_area:         int   = 2000,
        max_area:         int   = 500_000,
        aspect_ratio_min: float = 1.8,
        sat_threshold:    int   = 35,
    ) -> None:
        self.min_area         = min_area
        self.max_area         = max_area
        self.aspect_ratio_min = aspect_ratio_min
        self.sat_threshold    = sat_threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> OrientationResult:
        """Run the full orientation detection pipeline on one BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Mask
        mask = self._build_mask(gray)

        # 2. Contour + ellipse
        contour, ellipse = self._find_capsule_contour(mask)
        if contour is None or ellipse is None:
            return OrientationResult(detected=False)

        # 3. Type
        cap_type = self._classify_type(frame, contour)

        # 4. Straighten
        patch = self._straighten(frame, ellipse)
        if patch is None or patch.size == 0:
            return OrientationResult(detected=False)

        # 5. Seam
        seam_ratio, confidence = self._find_seam(patch, cap_type)

        # 6. Orientation
        long_body_side, body_ratio = self._orientation(seam_ratio, ellipse[2])

        cx, cy = int(ellipse[0][0]), int(ellipse[0][1])

        return OrientationResult(
            detected       = True,
            capsule_type   = cap_type,
            angle_deg      = float(ellipse[2]),
            long_body_side = long_body_side,
            seam_ratio     = seam_ratio,
            body_ratio     = body_ratio,
            confidence     = confidence,
            center         = (cx, cy),
            ellipse        = ellipse,
            contour        = contour,
        )

    # ------------------------------------------------------------------
    # Step 1 – mask
    # ------------------------------------------------------------------
    def _build_mask(self, gray: np.ndarray) -> np.ndarray:
        # 1. Smooth to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 2. Adaptive Canny Edge Detection (ignores shiny reflections inside capsule)
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(blurred, lower, upper)
        
        # 3. Close the edge loops with a MUCH smaller kernel to prevent bloating/rounding
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
        
        # 4. Fill the external contours to create a solid mask
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_area * 0.5:
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
                
        # Final cleanup to remove small noise without expanding the contour
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    # ------------------------------------------------------------------
    # Step 2 – contour + ellipse
    # ------------------------------------------------------------------
    def _find_capsule_contour(
        self, mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue
            if len(cnt) < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)
            _, (minor, major), _ = ellipse
            if (major / (minor + 1e-6)) < self.aspect_ratio_min:
                continue
            if best is None or area > cv2.contourArea(best[0]):
                best = (cnt, ellipse)

        return (best[0], best[1]) if best else (None, None)

    # ------------------------------------------------------------------
    # Step 3 – classify type
    # ------------------------------------------------------------------
    def _classify_type(self, frame: np.ndarray, contour: np.ndarray) -> str:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_sat = float(cv2.mean(hsv[:, :, 1], mask=mask)[0])
        return "COLOURED" if mean_sat > self.sat_threshold else "TRANSPARENT"

    # ------------------------------------------------------------------
    # Step 4 – straighten
    # ------------------------------------------------------------------
    def _straighten(self, frame: np.ndarray, ellipse: tuple) -> Optional[np.ndarray]:
        (cx, cy), (minor_ax, major_ax), angle = ellipse
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
        # This perfectly catches both physical ridges (transparent) and color splits (coloured)
        patch_float = patch.astype(float)
        
        # Calculate horizontal color gradient
        diff = np.abs(np.diff(patch_float, axis=1))
        
        # Average across the height of the capsule and across the 3 color channels
        signal = diff.mean(axis=(0, 2))

        # Smooth the signal to remove noise
        signal = self._smooth(signal, sigma=max(3, w // 20))

        # Ignore the extreme edges (the rounded tips of the capsule)
        margin = int(w * 0.20)
        region = signal[margin: w - margin]

        if region.size == 0:
            return 0.5, 0.0

        # The seam is the absolute maximum color/gradient change in the middle region
        peak_idx   = int(np.argmax(region)) + margin
        seam_ratio = peak_idx / w

        # Confidence: how prominent is the peak vs background
        mean_val   = float(region.mean()) + 1e-6
        confidence = min(1.0, float(region.max()) / mean_val / 5.0)

        return seam_ratio, confidence

    # ------------------------------------------------------------------
    # Step 6 – orientation
    # ------------------------------------------------------------------
    @staticmethod
    def _orientation(seam_ratio: float, angle_deg: float):
        """
        seam_ratio < 0.5 → seam is in the left half → left part is the CAP
                                                      → right part is the BODY
        seam_ratio > 0.5 → left part is the BODY
        angle_deg classifies horizontal vs vertical.
        """
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
        """Gaussian smoothing using numpy convolution (no scipy needed)."""
        radius = int(3 * sigma)
        x      = np.arange(-radius, radius + 1)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        padded = np.pad(signal, radius, mode="edge")
        return np.convolve(padded, kernel, mode="valid")
