# capsule_vision/pipeline/frame_producer.py
"""
FrameProducer – runs the camera in a dedicated background thread and pushes
frames into a shared queue consumed by the processing pipeline.

Design rationale
----------------
* Camera I/O and vision processing are decoupled across a thread boundary.
* A bounded ``queue.Queue`` acts as the back-pressure mechanism: if the
  consumer is slower than the producer the oldest frames are dropped rather
  than accumulating unbounded memory.
* The producer honours a ``threading.Event`` for clean shutdown.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from capsule_vision.camera import RPiGlobalShutterCamera
from capsule_vision.utils  import get_logger, FPSCounter

log = get_logger(__name__)


@dataclass
class FramePacket:
    """
    One unit of data passed from producer to consumer.

    Attributes
    ----------
    frame       : BGR image array (H, W, 3) uint8
    frame_id    : Monotonically increasing integer
    timestamp   : Wall-clock time (seconds) when the frame was captured
    camera_meta : Raw metadata dict from the camera driver
    """
    frame:       np.ndarray
    frame_id:    int
    timestamp:   float
    camera_meta: dict = field(default_factory=dict)


class FrameProducer(threading.Thread):
    """
    Background thread that continuously reads frames from the camera and
    enqueues ``FramePacket`` objects.

    Parameters
    ----------
    camera    : CameraInterface
        An already-opened camera driver instance.
    out_queue : queue.Queue[FramePacket]
        Queue to push packets into (shared with the consumer).
    warmup_frames : int
        Number of frames to discard before enqueuing (sensor stabilisation).
    """

    def __init__(
        self,
        camera:       RPiGlobalShutterCamera,
        out_queue:    queue.Queue,
        warmup_frames: int = 30,
    ) -> None:
        super().__init__(name="FrameProducer", daemon=True)
        self._camera        = camera
        self._queue         = out_queue
        self._warmup_frames = warmup_frames
        self._stop_event    = threading.Event()
        self._fps           = FPSCounter(window=60)
        self._frame_id      = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal the producer thread to stop gracefully."""
        log.info("FrameProducer stop requested.")
        self._stop_event.set()

    @property
    def fps(self) -> float:
        """Current capture FPS (rolling average over last 60 frames)."""
        return self._fps.get()

    # ------------------------------------------------------------------
    # Thread entry-point
    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info(
            "FrameProducer started (warmup=%d frames).", self._warmup_frames
        )

        warmup_remaining = self._warmup_frames

        while not self._stop_event.is_set():
            frame: Optional[np.ndarray] = self._camera.read_frame()

            if frame is None:
                log.warning("Null frame received – retrying in 10 ms.")
                time.sleep(0.01)
                continue

            # Discard warm-up frames
            if warmup_remaining > 0:
                warmup_remaining -= 1
                if warmup_remaining == 0:
                    log.info("Warm-up complete – starting frame production.")
                continue

            self._frame_id += 1
            self._fps.tick()

            packet = FramePacket(
                frame       = frame,
                frame_id    = self._frame_id,
                timestamp   = time.monotonic(),
                camera_meta = self._camera.get_metadata(),
            )

            # Non-blocking put: drop oldest frame if the consumer is lagging
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                try:
                    self._queue.get_nowait()   # evict oldest
                except queue.Empty:
                    pass
                self._queue.put_nowait(packet)
                log.debug("Queue full – oldest frame dropped (id=%d).", self._frame_id)

        log.info("FrameProducer thread exiting.")
