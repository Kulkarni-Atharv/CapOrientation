# capsule_vision/display/live_viewer.py
"""
LiveViewer – OpenCV-based debug display for the camera feed.

Renders the raw camera frame with an on-screen HUD showing:
  - Frame ID
  - Capture FPS
  - Resolution
  - Timestamp

Press  Q  or  ESC  to quit.
"""

from __future__ import annotations

import queue
import time

import cv2
import numpy as np

from capsule_vision.pipeline import FramePacket
from capsule_vision.utils    import get_logger

log = get_logger(__name__)

# HUD styling constants
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 1
_LINE_TYPE  = cv2.LINE_AA
_HUD_COLOR  = (0, 255, 128)      # Bright green
_BG_COLOR   = (20, 20, 20)       # Near-black
_PADDING    = 10
_LINE_H     = 22                  # Pixels between HUD lines


class LiveViewer:
    """
    Pulls packets from a frame queue and renders them in an OpenCV window.

    Parameters
    ----------
    in_queue  : queue.Queue[FramePacket]
        Frame queue produced by ``FrameProducer``.
    window_name : str
        OpenCV window title.
    scale  : float
        Display scale factor (e.g. 0.5 to halve the window size).
    """

    def __init__(
        self,
        in_queue:    queue.Queue,
        window_name: str   = "CapsuleVision – Live Feed",
        scale:       float = 0.5,
    ) -> None:
        self._queue  = in_queue
        self._title  = window_name
        self._scale  = scale

    # ------------------------------------------------------------------
    # Main display loop  (blocking – call from the main thread)
    # ------------------------------------------------------------------
    def run(self, fps_ref: "callable[[], float] | None" = None) -> None:
        """
        Start the viewer loop.

        Parameters
        ----------
        fps_ref : callable, optional
            A zero-argument callable that returns the current producer FPS.
            If provided, the value is shown on the HUD.
        """
        log.info("LiveViewer starting.  Press Q or ESC to quit.")
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)

        while True:
            try:
                packet: FramePacket = self._queue.get(timeout=1.0)
            except queue.Empty:
                log.debug("Viewer: no frame in 1 s – waiting.")
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            # Optional resize for the display window
            frame = packet.frame
            if self._scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(
                    frame,
                    (int(w * self._scale), int(h * self._scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow(self._title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                log.info("Quit key pressed – closing viewer.")
                break

        cv2.destroyAllWindows()
        log.info("LiveViewer stopped.")

