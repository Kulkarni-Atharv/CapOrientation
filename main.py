#!/usr/bin/env python3
# main.py
"""
CapsuleVision – application entry-point.

Usage
-----
    python main.py                         # live display (default)
    python main.py --no-display            # headless / SSH mode
    python main.py --width 1280 --height 720 --fps 30
    python main.py --log-level DEBUG

Camera strategy
---------------
  1. Start in full auto (AE + AWB)
  2. Wait 2 s for sensor to settle
  3. Lock exposure, gain, and colour-gains
  4. Stream stable BGR888 frames into the AI pipeline
"""

import argparse
import queue
import signal
import sys

from capsule_vision.camera   import RPiGlobalShutterCamera
from capsule_vision.config   import DEFAULT_CONFIG
from capsule_vision.display  import LiveViewer
from capsule_vision.pipeline import FrameProducer
from capsule_vision.utils    import setup_root_logger, get_logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="capsule_vision",
        description="CapsuleVision – RPi Global Shutter Camera feed",
    )
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode – no OpenCV window")
    p.add_argument("--scale",  type=float, default=1.0,
                   help="Display window scale factor (default: 1.0)")
    p.add_argument("--width",  type=int,   default=None,
                   help="Capture width  (default: 1280)")
    p.add_argument("--height", type=int,   default=None,
                   help="Capture height (default: 720)")
    p.add_argument("--fps",    type=int,   default=None,
                   help="Capture framerate (default: 30)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    cfg = DEFAULT_CONFIG
    if args.width:  cfg.camera.width     = args.width
    if args.height: cfg.camera.height    = args.height
    if args.fps:    cfg.camera.framerate = args.fps
    cfg.logging.level = args.log_level

    # ── Logging ─────────────────────────────────────────────────────────────
    setup_root_logger(cfg.logging)
    log = get_logger("main")

    log.info("=" * 55)
    log.info("CapsuleVision  –  starting up")
    log.info("Resolution : %dx%d @ %d fps",
             cfg.camera.width, cfg.camera.height, cfg.camera.framerate)
    log.info("Display    : %s",
             "headless" if args.no_display else f"on (scale={args.scale})")
    log.info("=" * 55)

    # ── Camera (auto-expose → lock → stream) ────────────────────────────────
    camera = RPiGlobalShutterCamera(cfg.camera)
    camera.open()

    # ── Frame queue ─────────────────────────────────────────────────────────
    frame_queue: queue.Queue = queue.Queue(maxsize=cfg.pipeline.queue_size)

    # ── Producer thread ─────────────────────────────────────────────────────
    producer = FrameProducer(
        camera        = camera,
        out_queue     = frame_queue,
        warmup_frames = cfg.pipeline.warmup_frames,
    )
    producer.start()

    # ── Graceful shutdown ───────────────────────────────────────────────────
    def _shutdown(sig, _frame):
        log.info("Signal %s – shutting down.", sig)
        producer.stop()
        camera.release()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Display / headless loop ─────────────────────────────────────────────
    try:
        if args.no_display:
            log.info("Headless mode.  Ctrl-C to stop.")
            producer.join()
        else:
            viewer = LiveViewer(
                in_queue    = frame_queue,
                window_name = "CapsuleVision – Live Feed",
                scale       = args.scale,
            )
            viewer.run()
    finally:
        log.info("Stopping …")
        producer.stop()
        producer.join(timeout=5.0)
        camera.release()
        log.info("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
