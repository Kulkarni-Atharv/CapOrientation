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

        # Using the exact configuration from the user snippet
        cfg = self.picam2.create_preview_configuration(
            main={"size": (1456, 1088)},
            lores={"size": (640, 480)},
            display="main"
        )
        
        self.picam2.configure(cfg)
        self.picam2.start()

        self._is_open = True
        log.info("Camera ready.")

    def release(self) -> None:
        if self.picam2 is not None and self._is_open:
            self.picam2.stop()
            self._is_open = False
            self.picam2 = None

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._is_open or self.picam2 is None:
            return None
        try:
            frame = self.picam2.capture_array()
            
            # ✅ FIX: Convert RGB → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
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
