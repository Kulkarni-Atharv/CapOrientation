# capsule_vision/camera/rpi_global_shutter_camera.py
"""
RPi Global Shutter Camera driver (IMX296 sensor) via Picamera2.

Hardware
--------
  - Board          : Raspberry Pi CM5 / Pi 5
  - Camera module  : Raspberry Pi Global Shutter Camera (IMX296, 1.58 MP)
  - Interface      : MIPI CSI-2 via Picamera2 / libcamera

Key properties of the IMX296 global shutter sensor
---------------------------------------------------
  - Native resolution : 1456 × 1088
  - Max framerate     : ~60 fps (full resolution)
  - No rolling-shutter artefacts – ideal for high-speed inspection lines
  - Monochrome or colour variants available
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

# Picamera2 is only available on Raspberry Pi OS.
# Guard the import so the module can be imported (for type-checking /
# unit-testing) on a development machine without crashing.
try:
    from picamera2 import Picamera2
    from picamera2.controls import Controls
    _PICAMERA2_AVAILABLE = True
except ImportError:  # pragma: no cover
    Picamera2 = None          # type: ignore[assignment,misc]
    Controls  = None          # type: ignore[assignment,misc]
    _PICAMERA2_AVAILABLE = False

from capsule_vision.camera.camera_interface import CameraInterface
from capsule_vision.config import CameraConfig
from capsule_vision.utils import get_logger

log = get_logger(__name__)


class RPiGlobalShutterCamera(CameraInterface):
    """
    Concrete driver for the Raspberry Pi Global Shutter Camera
    using the Picamera2 library.

    Parameters
    ----------
    config : CameraConfig
        Camera configuration dataclass (see config/settings.py).

    Example
    -------
    ::

        from capsule_vision.config import CameraConfig
        from capsule_vision.camera import RPiGlobalShutterCamera

        cfg = CameraConfig(width=1456, height=1088, framerate=60)
        cam = RPiGlobalShutterCamera(cfg)
        cam.open()

        try:
            while True:
                frame = cam.read_frame()
                if frame is not None:
                    # frame is a (H, W, 3) BGR uint8 ndarray
                    cv2.imshow("Preview", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
        finally:
            cam.release()
            cv2.destroyAllWindows()
    """

    def __init__(self, config: CameraConfig) -> None:
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "Picamera2 is not installed or not running on a Raspberry Pi. "
                "Install it with:  sudo apt install python3-picamera2"
            )
        self._cfg    = config
        self._picam2: Optional[Picamera2] = None
        self._meta:   dict = {}
        self._open:   bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Initialise Picamera2 and start the capture pipeline."""
        log.info(
            "Opening RPi Global Shutter Camera | "
            "index=%d  res=%dx%d  fps=%d",
            self._cfg.camera_index,
            self._cfg.width,
            self._cfg.height,
            self._cfg.framerate,
        )

        self._picam2 = Picamera2(self._cfg.camera_index)

        # ---------------------------------------------------------------
        # Build the still/video configuration
        # Picamera2 uses sensor_mode + video_configuration for streaming.
        # ---------------------------------------------------------------
        video_cfg = self._picam2.create_video_configuration(
            main={
                "size":   (self._cfg.width, self._cfg.height),
                "format": self._cfg.format,
            },
            controls={
                "FrameRate":      float(self._cfg.framerate),
                "ExposureTime":   self._cfg.exposure_us,
                "AnalogueGain":   self._cfg.analogue_gain,
            },
            buffer_count=self._cfg.buffer_count,
        )

        # Apply H/V flip if requested
        if self._cfg.hflip or self._cfg.vflip:
            video_cfg["transform"] = {
                "hflip": int(self._cfg.hflip),
                "vflip": int(self._cfg.vflip),
            }

        self._picam2.configure(video_cfg)

        # Optional on-screen preview (only useful when a display is attached)
        if self._cfg.preview:
            self._picam2.start_preview()

        self._picam2.start()
        self._open = True

        log.info("Camera started successfully.")

    def release(self) -> None:
        """Stop the camera and release all resources."""
        if self._picam2 is not None and self._open:
            log.info("Releasing camera resources.")
            try:
                self._picam2.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("Error stopping camera: %s", exc)
            finally:
                self._open    = False
                self._picam2  = None
                log.info("Camera released.")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Capture one frame from the global shutter sensor.

        Returns
        -------
        np.ndarray or None
            BGR image (H × W × 3, uint8) or None on failure.
        """
        if not self._open or self._picam2 is None:
            log.error("read_frame() called on a closed camera.")
            return None

        try:
            # capture_array returns an ndarray; metadata is a dict
            frame, metadata = self._picam2.capture_array("main", wait=True), {}
            # Picamera2 ≥ 0.3.12 — preferred two-tuple API
            # If your version returns only the array, comment next line
            # frame, metadata = self._picam2.capture_arrays(["main"])[0], {}
            self._meta = metadata if isinstance(metadata, dict) else {}
        except Exception as exc:  # noqa: BLE001
            log.error("Frame capture failed: %s", exc)
            return None

        # Convert from the configured format to BGR for OpenCV compatibility
        frame = self._to_bgr(frame)
        return frame

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        """Return per-frame metadata from the last captured frame."""
        return dict(self._meta)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._cfg.width, self._cfg.height)

    @property
    def framerate(self) -> float:
        return float(self._cfg.framerate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_bgr(frame: np.ndarray) -> np.ndarray:
        """
        Normalise any Picamera2 pixel format to a standard BGR uint8 array.

        Picamera2 can return:
          - RGB888  → convert to BGR
          - BGR888  → already compatible with OpenCV
          - XRGB8888 (4-channel RGBA-like) → strip alpha, convert
        """
        if frame is None:
            raise ValueError("Received None frame from Picamera2.")

        if frame.ndim == 2:
            # Monochrome → duplicate to 3-channel BGR
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        channels = frame.shape[2] if frame.ndim == 3 else 1

        if channels == 3:
            # Assume RGB888 (Picamera2 default for "RGB888")
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if channels == 4:
            # XRGB8888 – strip the X channel (alpha-like) and convert
            return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)

        log.warning("Unexpected frame shape: %s — returning as-is.", frame.shape)
        return frame
