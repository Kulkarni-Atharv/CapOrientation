#!/usr/bin/env python3
# main.py
"""
CapsuleVision – application entry-point.

Usage
-----
    python main.py              # default config, live display
    python main.py --no-display # headless mode (logging only)
    python main.py --scale 0.5  # display at 50% of sensor resolution
    python main.py --fps 30     # override framerate
    python main.py --exposure 3000  # override exposure (µs)

Architecture
------------
                        ┌──────────────────────┐
                        │   RPi Global Shutter  │
                        │   Camera (IMX296)     │
                        └──────────┬───────────┘
                                   │  Picamera2
                        ┌──────────▼───────────┐
                        │   FrameProducer       │  ← background thread
                        │   (camera I/O loop)   │
                        └──────────┬───────────┘
                                   │  queue.Queue[FramePacket]
              ┌────────────────────▼────────────────────┐
              │              Main Thread                 │
              │                                          │
              │   LiveViewer (OpenCV HUD display)        │
              │       │                                  │
              │       └─── [Future] Orientation Detector │
              └──────────────────────────────────────────┘
"""

import argparse
import queue
import signal
import sys

from capsule_vision.camera    import RPiGlobalShutterCamera
from capsule_vision.config    import DEFAULT_CONFIG
from capsule_vision.display   import LiveViewer
from capsule_vision.pipeline  import FrameProducer
from capsule_vision.utils     import setup_root_logger, get_logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="capsule_vision",
        description="CapsuleVision – RPi Global Shutter Camera feed viewer",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Run headless (no OpenCV window – logging only)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Display window scale factor (default: 0.5)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        metavar="INT",
        help="Override camera framerate",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=None,
        metavar="INT",
        help="Override exposure time in microseconds",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override analogue gain (1.0 – 16.0)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()

    # ── 1. Build config ────────────────────────────────────────────────
    cfg = DEFAULT_CONFIG

    # Apply CLI overrides
    if args.fps is not None:
        cfg.camera.framerate = args.fps
    if args.exposure is not None:
        cfg.camera.exposure_us = args.exposure
    if args.gain is not None:
        cfg.camera.analogue_gain = args.gain
    cfg.logging.level = args.log_level

    # ── 2. Configure logging ───────────────────────────────────────────
    setup_root_logger(cfg.logging)
    log = get_logger("main")

    log.info("=" * 60)
    log.info("CapsuleVision  –  starting up")
    log.info("Camera    : %dx%d @ %d fps", cfg.camera.width, cfg.camera.height, cfg.camera.framerate)
    log.info("Exposure  : %d µs    Gain: %.1f", cfg.camera.exposure_us, cfg.camera.analogue_gain)
    log.info("Display   : %s", "disabled (headless)" if args.no_display else f"enabled (scale={args.scale})")
    log.info("=" * 60)

    # ── 3. Initialise hardware ─────────────────────────────────────────
    camera   = RPiGlobalShutterCamera(cfg.camera)
    camera.open()

    # ── 4. Build the shared frame queue ───────────────────────────────
    frame_queue: queue.Queue = queue.Queue(maxsize=cfg.pipeline.queue_size)

    # ── 5. Start the producer thread ──────────────────────────────────
    producer = FrameProducer(
        camera        = camera,
        out_queue     = frame_queue,
        warmup_frames = cfg.pipeline.warmup_frames,
    )
    producer.start()

    # ── 6. Graceful shutdown on SIGINT / SIGTERM ───────────────────────
    def _shutdown(sig, frame_ref):  # noqa: ANN001
        log.info("Signal %s received – initiating shutdown.", sig)
        producer.stop()
        camera.release()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 7. Run display / headless loop ─────────────────────────────────
    try:
        if args.no_display:
            log.info("Headless mode active.  Press Ctrl-C to stop.")
            producer.join()   # Block until producer exits (or Ctrl-C)
        else:
            viewer = LiveViewer(
                in_queue    = frame_queue,
                window_name = "CapsuleVision – Live Feed",
                scale       = args.scale,
            )
            viewer.run(fps_ref=lambda: producer.fps)
    finally:
        log.info("Shutting down…")
        producer.stop()
        producer.join(timeout=5.0)
        camera.release()
        log.info("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
