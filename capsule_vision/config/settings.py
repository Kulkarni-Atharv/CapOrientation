# capsule_vision/config/settings.py
"""
Central configuration for the CapsuleVision system.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
LOG_DIR     = BASE_DIR / "logs"
CAPTURE_DIR = BASE_DIR / "captures"
MODEL_DIR   = BASE_DIR / "models"

for _dir in (LOG_DIR, CAPTURE_DIR, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Camera settings – RPi Global Shutter Camera (IMX296)
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    width:        int   = 1280   # capture width  (px)
    height:       int   = 720    # capture height (px)
    framerate:    int   = 30     # fps (30 = more light per frame)
    camera_index: int   = 0      # 0 for single-camera setups


# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    queue_size:    int = 8    # frame queue depth between producer and consumer
    warmup_frames: int = 0   # no warmup needed (open() already stabilises)


# ---------------------------------------------------------------------------
# Logging settings
# ---------------------------------------------------------------------------
@dataclass
class LoggingConfig:
    level:        str  = "INFO"
    log_to_file:  bool = True
    log_file:     Path = LOG_DIR / "capsule_vision.log"
    max_bytes:    int  = 10 * 1024 * 1024   # 10 MB
    backup_count: int  = 5


# ---------------------------------------------------------------------------
# Application config
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    camera:   CameraConfig   = field(default_factory=CameraConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging:  LoggingConfig  = field(default_factory=LoggingConfig)


DEFAULT_CONFIG = AppConfig()
