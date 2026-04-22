# capsule_vision/utils/fps_counter.py
"""
Thread-safe FPS (frames per second) counter.

Usage
-----
    from capsule_vision.utils.fps_counter import FPSCounter

    fps = FPSCounter(window=60)
    while True:
        fps.tick()
        print(f"FPS: {fps.get():.1f}")
"""

import time
from collections import deque
from threading import Lock


class FPSCounter:
    """
    Rolling-window FPS counter.

    Parameters
    ----------
    window : int
        Number of recent frame timestamps to keep for the rolling average.
    """

    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._timestamps: deque[float] = deque(maxlen=window)
        self._lock = Lock()

    def tick(self) -> None:
        """Register a new frame arrival."""
        with self._lock:
            self._timestamps.append(time.monotonic())

    def get(self) -> float:
        """Return the current rolling-average FPS (0.0 if < 2 frames seen)."""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear internal state."""
        with self._lock:
            self._timestamps.clear()
