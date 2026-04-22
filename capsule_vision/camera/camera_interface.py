# capsule_vision/camera/camera_interface.py
"""
Abstract base class for all camera back-ends.

Any concrete camera driver must implement this interface so that the rest of
the pipeline remains hardware-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CameraInterface(ABC):
    """
    Minimal contract every camera driver must fulfil.

    Lifecycle
    ---------
        cam = MyCameraDriver(cfg)
        cam.open()
        try:
            frame = cam.read_frame()   # -> np.ndarray (H, W, C) BGR uint8
            meta  = cam.get_metadata() # -> dict with exposure, gain, ts …
        finally:
            cam.release()
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def open(self) -> None:
        """Initialise the hardware and start the acquisition pipeline."""

    @abstractmethod
    def release(self) -> None:
        """Stop acquisition and free all resources."""

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Capture and return one frame.

        Returns
        -------
        np.ndarray
            BGR image array with shape (H, W, 3) and dtype uint8.
            Returns ``None`` if a frame could not be acquired.
        """

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Return per-frame metadata from the camera driver.

        The dict SHOULD contain at minimum:
            - ``timestamp_us``  (int)   : sensor timestamp in microseconds
            - ``exposure_us``   (int)   : actual exposure used
            - ``analogue_gain`` (float) : actual analogue gain used
            - ``colour_temp``   (int)   : colour temperature (K), if available
        """

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """True if the camera is currently open and streaming."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) of the active capture mode."""

    @property
    @abstractmethod
    def framerate(self) -> float:
        """Return the configured capture framerate."""
