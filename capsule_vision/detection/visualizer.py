# capsule_vision/detection/visualizer.py
"""
Draws orientation results onto a BGR frame for display / debugging.
"""

import cv2
import numpy as np
from capsule_vision.detection.capsule_detector import OrientationResult

# ── Colour palette ──────────────────────────────────────────────────────────
_GREEN  = (0, 220, 80)
_RED    = (0, 60, 220)
_YELLOW = (0, 200, 255)
_WHITE  = (240, 240, 240)
_BLACK  = (10, 10, 10)
_CYAN   = (220, 200, 0)
_PURPLE = (255, 50, 255)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 1


def annotate(frame: np.ndarray, result: OrientationResult) -> np.ndarray:
    """Return a copy of frame with orientation annotations drawn."""
    out = frame.copy()

    if not result.detected:
        cv2.putText(out, "No capsule detected (OpenCV Failed)", (20, 40),
                    _FONT, 0.7, _RED, 2, cv2.LINE_AA)
        
        # If AI found it, but OpenCV failed, still draw the AI box!
        if result.bbox is not None:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), _PURPLE, 2)
            cv2.putText(out, "AI Box (Angle Failed)", (x1, y1 - 10), _FONT, 0.5, _PURPLE, 1, cv2.LINE_AA)
            
        return out

    # ── 1. Draw YOLO Bounding Box ───────────────────────────────────────────
    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), _PURPLE, 2)
        cv2.putText(out, "YOLO AI Box", (x1, y1 - 10), _FONT, 0.5, _PURPLE, 1, cv2.LINE_AA)

    # ── 2. Draw contour ─────────────────────────────────────────────────────
    if result.contour is not None:
        cv2.drawContours(out, [result.contour], -1, _GREEN, 2)

    # ── 3. Draw fitted ellipse ───────────────────────────────────────────────
    if result.ellipse is not None:
        cv2.ellipse(out, result.ellipse, _CYAN, 2)

    # ── 4. Draw major axis line + seam marker ───────────────────────────────
    if result.ellipse is not None:
        (cx, cy), (minor_ax, major_ax), angle = result.ellipse
        half = int(major_ax / 2)
        rad  = np.deg2rad(angle)

        dx = int(half * np.cos(rad))
        dy = int(half * np.sin(rad))

        # Major axis endpoints
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        cv2.line(out, p1, p2, _WHITE, 1, cv2.LINE_AA)

        # Seam position along the axis
        seam_offset = result.seam_ratio - 0.5           # -0.5 … +0.5
        sx = int(cx + seam_offset * major_ax * np.cos(rad))
        sy = int(cy + seam_offset * major_ax * np.sin(rad))

        # Perpendicular direction for the seam tick
        perp_dx = int(12 * np.sin(rad))
        perp_dy = int(12 * np.cos(rad))
        cv2.line(out,
                 (sx - perp_dx, sy + perp_dy),
                 (sx + perp_dx, sy - perp_dy),
                 _YELLOW, 2, cv2.LINE_AA)

        # Label ends
        cv2.putText(out, "CAP",  p1, _FONT, 0.45, _YELLOW,  1, cv2.LINE_AA)
        cv2.putText(out, "BODY", p2, _FONT, 0.45, _GREEN,   1, cv2.LINE_AA)

    # ── 5. Info panel (top-left) ─────────────────────────────────────────────
    lines = [
        f"Type       : {result.capsule_type}",
        f"Body side  : {result.long_body_side}",
        f"Angle      : {result.angle_deg:.1f} deg",
        f"Body ratio : {result.body_ratio:.2f}",
        f"Confidence : {result.confidence:.2f}",
    ]
    panel_h = len(lines) * 22 + 16
    panel_w = 270
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    for i, txt in enumerate(lines):
        y   = 18 + i * 22
        col = _GREEN if i < 2 else _WHITE
        cv2.putText(out, txt, (8, y), _FONT, _FONT_SCALE, col, _THICKNESS, cv2.LINE_AA)

    return out
