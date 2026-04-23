#!/usr/bin/env python3
# main.py  –  CapsuleVision entry-point
"""
Usage
-----
    python main.py                        # live display
    python main.py --no-display           # headless / SSH
    python main.py --width 1280 --height 720 --fps 30
    python main.py --log-level DEBUG      # show raw YOLO box confidences
"""

import argparse
import queue
import signal
import sys
from pathlib import Path

import cv2

from capsule_vision.camera    import RPiGlobalShutterCamera
from capsule_vision.config    import DEFAULT_CONFIG
from capsule_vision.detection import CapsuleDetector, annotate
from capsule_vision.pipeline  import FrameProducer
from capsule_vision.utils     import setup_root_logger, get_logger

# Project root is one directory above this file
_PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="capsule_vision")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--scale",  type=float, default=1.0)
    p.add_argument("--width",  type=int,   default=None)
    p.add_argument("--height", type=int,   default=None)
    p.add_argument("--fps",    type=int,   default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cfg = DEFAULT_CONFIG
    if args.width:  cfg.camera.width     = args.width
    if args.height: cfg.camera.height    = args.height
    if args.fps:    cfg.camera.framerate = args.fps
    cfg.logging.level = args.log_level

    setup_root_logger(cfg.logging)
    log = get_logger("main")
    log.info("CapsuleVision starting  %dx%d @ %d fps",
             cfg.camera.width, cfg.camera.height, cfg.camera.framerate)

    # ── Camera ──────────────────────────────────────────────────────────────
    camera = RPiGlobalShutterCamera(cfg.camera)
    camera.open()

    # ── Frame queue + producer thread ───────────────────────────────────────
    frame_queue: queue.Queue = queue.Queue(maxsize=cfg.pipeline.queue_size)
    producer = FrameProducer(
        camera        = camera,
        out_queue     = frame_queue,
        warmup_frames = cfg.pipeline.warmup_frames,
    )
    producer.start()

    # ── Model selection (absolute path so CWD never matters) ────────────────
    capsule_model = _PROJECT_ROOT / "models" / "capsule_best.pt"
    if not capsule_model.exists():
        log.error(
            "Custom capsule model NOT FOUND at %s.\n"
            "The generic yolov8n.pt is NOT able to detect capsules.\n"
            "Please train / copy the model to models/capsule_best.pt and restart.",
            capsule_model,
        )
        producer.stop()
        camera.release()
        return 1

    log.info("Using capsule model: %s", capsule_model)

    # ── Detector ────────────────────────────────────────────────────────────
    detector = CapsuleDetector(
        model_path    = str(capsule_model),
        conf_thresh   = 0.05,   # low threshold — model confidence can be weak
        sat_threshold = 35,
    )

    # ── Shutdown handler ─────────────────────────────────────────────────────
    def _shutdown(sig, _):
        log.info("Shutting down …")
        producer.stop()
        camera.release()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main loop ────────────────────────────────────────────────────────────
    log.info("Running.  Press ESC / Q to quit.  Use --log-level DEBUG to see raw YOLO scores.")
    try:
        while True:
            try:
                packet = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frame  = packet.frame
            result = detector.detect(frame)

            log.debug(
                "Frame %d | detected=%s | type=%s | body=%s | ai_conf=%.3f | conf=%.3f",
                packet.frame_id,
                result.detected,
                result.capsule_type,
                result.long_body_side,
                result.ai_conf,
                result.confidence,
            )

            if not args.no_display:
                display = annotate(frame, result)

                if args.scale != 1.0:
                    dh, dw = display.shape[:2]
                    display = cv2.resize(
                        display,
                        (int(dw * args.scale), int(dh * args.scale)),
                    )

                cv2.imshow("CapsuleVision", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        log.info("Stopping …")
        producer.stop()
        producer.join(timeout=5.0)
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        log.info("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
