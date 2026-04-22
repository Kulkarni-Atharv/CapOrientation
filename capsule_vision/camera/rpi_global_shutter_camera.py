# capsule_vision/camera/rpi_global_shutter_camera.py
"""
RPi Global Shutter Camera driver (IMX296) via Picamera2.

Camera strategy (industry-grade stable feed):
  1. Start in full auto (AE + AWB ON)
  2. Wait 2 s for sensor to settle
  3. Read back the stable exposure / gain / colour-gains
  4. Lock those values — every frame is now identical in brightness + colour
  5. Feed locked BGR888 frames directly to the AI pipeline (no conversion needed)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import cv2

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None            # type: ignore[assignment,misc]
    _PICAMERA2_AVAILABLE = False

from capsule_vision.config import CameraConfig
from capsule_vision.utils  import get_logger

log = get_logger(__name__)


class RPiGlobalShutterCamera:
    """
    Concrete driver for the Raspberry Pi Global Shutter Camera (IMX296).

    Lifecycle
    ---------
    ::
        cam = RPiGlobalShutterCamera(cfg)
        cam.open()          # auto-expose → lock → ready
        frame = cam.read_frame()   # BGR888 ndarray, no conversion needed
        cam.release()
    """

    def __init__(self, config: CameraConfig) -> None:
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "Picamera2 not found. Install it with:\n"
                "  sudo apt install python3-picamera2"
            )
        self.cfg   = config
        self.picam2: Optional[Picamera2] = None
        self._meta:   dict = {}
        self._is_open:   bool = False

    def open(self) -> None:
        log.info("Opening camera %dx%d @ %d fps",
                 self.cfg.width, self.cfg.height, self.cfg.framerate)

        self.picam2 = Picamera2()

        cfg = self.picam2.create_video_configuration(
            main={
                "size":   (self.cfg.width, self.cfg.height),
                "format": "BGR888",
            }
        )
        self.picam2.configure(cfg)
        self.picam2.start()

        self.picam2.set_controls({
            "AeEnable":  True,
            "AwbEnable": True,
            "FrameRate": float(self.cfg.framerate),
        })

        self._is_open = True
        log.info("Camera ready (Auto Mode).")

    def release(self) -> None:
        if self.picam2 is not None and self._is_open:
            log.info("Releasing camera.")
            try:
                self.picam2.stop()
            except Exception as exc:   # noqa: BLE001
                log.warning("Error stopping camera: %s", exc)
            finally:
                self._is_open   = False
                self.picam2 = None
            log.info("Camera released.")

    def read_frame(self) -> Optional[np.ndarray]:
        """Return one BGR888 frame — ready for OpenCV / AI pipeline."""
        if not self._is_open or self.picam2 is None:
            log.error("read_frame() called on closed camera.")
            return None
        try:
            # ✅ Already BGR888 — NO cv2.cvtColor needed
            frame = self.picam2.capture_array("main")
            return frame
        except Exception as exc:   # noqa: BLE001
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
