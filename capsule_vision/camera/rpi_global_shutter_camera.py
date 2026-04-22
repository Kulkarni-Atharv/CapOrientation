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
        self._cfg   = config
        self._picam2: Optional[Picamera2] = None
        self._meta:   dict = {}
        self._open:   bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        log.info("Opening camera %dx%d @ %d fps",
                 self._cfg.width, self._cfg.height, self._cfg.framerate)

        self._picam2 = Picamera2()

        # ── Video Config: Optimized for continuous headless capture ──────
        cfg = self._picam2.create_video_configuration(
            main={
                "size":   (self._cfg.width, self._cfg.height),
                "format": "RGB888",
            }
        )
        cfg["main"]["framerate"] = float(self._cfg.framerate)
        
        self._picam2.configure(cfg)

        # ── Step 1: full auto to let sensor settle ───────────────────────
        self._picam2.set_controls({
            "AeEnable":  True,
            "AwbEnable": True,
        })
        self._picam2.start()
        log.info("Auto-exposure running — stabilising for 2 s …")
        time.sleep(2)

        # ── Step 2: read the stable values ──────────────────────────────
        metadata    = self._picam2.capture_metadata()
        exposure    = metadata["ExposureTime"]
        gain        = metadata["AnalogueGain"]
        awb_gains   = metadata["ColourGains"]
        log.info("Locked  exposure=%d µs  gain=%.2f  awb_gains=%s",
                 exposure, gain, awb_gains)

        # ── Step 3: lock — every frame identical for AI ─────────────────
        self._picam2.set_controls({
            "AeEnable":   False,
            "AwbEnable":  False,
            "ExposureTime": exposure,
            "AnalogueGain": gain,
            "ColourGains":  awb_gains,
        })

        self._open = True
        log.info("Camera ready (locked).")

    def release(self) -> None:
        if self._picam2 is not None and self._open:
            log.info("Releasing camera.")
            try:
                self._picam2.stop()
            except Exception as exc:   # noqa: BLE001
                log.warning("Error stopping camera: %s", exc)
            finally:
                self._open   = False
                self._picam2 = None
            log.info("Camera released.")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------
    def read_frame(self) -> Optional[np.ndarray]:
        """Return one BGR888 frame — ready for OpenCV / AI pipeline."""
        if not self._open or self._picam2 is None:
            log.error("read_frame() called on closed camera.")
            return None
        try:
            # request_image("main") is safer and non-blocking compared to capture_array()
            frame = self._picam2.capture_array("main")
            # Fast NumPy slicing to flip RGB to BGR
            return frame[:, :, ::-1]
        except Exception as exc:   # noqa: BLE001
            log.error("Frame capture failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Metadata / properties
    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return dict(self._meta)

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._cfg.width, self._cfg.height)

    @property
    def framerate(self) -> float:
        return float(self._cfg.framerate)
