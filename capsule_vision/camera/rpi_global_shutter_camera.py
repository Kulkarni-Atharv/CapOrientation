# capsule_vision/camera/rpi_global_shutter_camera.py

from __future__ import annotations
import time
import numpy as np
import cv2
from typing import Optional

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None
    _PICAMERA2_AVAILABLE = False

from capsule_vision.config import CameraConfig
from capsule_vision.utils  import get_logger

log = get_logger(__name__)

class RPiGlobalShutterCamera:
    def __init__(self, config: CameraConfig) -> None:
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError("Picamera2 not found.")
        self.cfg   = config
        self.picam2 = None
        self._meta:   dict = {}
        self._is_open:   bool = False

    def open(self) -> None:
        log.info("Opening camera.")
        self.picam2 = Picamera2()

        cfg = self.picam2.create_video_configuration(
            main={
                "size":   (self.cfg.width, self.cfg.height),
                "format": "RGB888",   # Native RGB from sensor
            }
        )
        self.picam2.configure(cfg)
        self.picam2.start()

        # ── Step 1: Enable AE + AWB and set framerate ─────────────────────
        self.picam2.set_controls({
            "AeEnable":  True,
            "AwbEnable": True,
            "FrameRate": float(self.cfg.framerate),
        })

        # ── Step 2: Wait for AWB to settle (IMX296 needs ~2s) ────────────
        log.info("Waiting for AWB to settle...")
        time.sleep(2.0)

        # ── Step 3: Read the converged colour gains ───────────────────────
        metadata      = self.picam2.capture_metadata()
        colour_gains  = metadata.get("ColourGains")
        log.info("AWB settled — ColourGains: %s", colour_gains)

        # ── Step 4: Lock gains so every frame is colour-consistent ────────
        if colour_gains:
            self.picam2.set_controls({
                "AwbEnable":   False,           # Stop AWB from drifting
                "ColourGains": colour_gains,    # Lock the settled values
            })
        else:
            # Fallback: known-good gains for indoor LED lighting
            log.warning("ColourGains not available — using fallback gains.")
            self.picam2.set_controls({
                "AwbEnable":   False,
                "ColourGains": (2.2, 1.5),      # (R_gain, B_gain) — tune if needed
            })

        self._is_open = True
        log.info("Camera ready — AWB locked.")

    def release(self) -> None:
        if self.picam2 is not None and self._is_open:
            self.picam2.stop()
            self._is_open = False
            self.picam2 = None

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._is_open or self.picam2 is None:
            return None
        try:
            # 2. Take the feed from the camera (Camera Original Feed is RGB)
            frame_rgb = self.picam2.capture_array("main")
            
            # 3. Convert it into BGR for OpenCV processing and live display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            return frame_bgr
        except Exception as exc:
            log.error("Frame capture failed: %s", exc)
            return None

    def get_metadata(self) -> dict:
        return dict(self._meta)

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.cfg.width, self.cfg.height)

    @property
    def framerate(self) -> float:
        return float(self.cfg.framerate)
