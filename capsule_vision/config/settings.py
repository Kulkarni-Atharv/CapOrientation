# capsule_vision/config/settings.py
"""
Central configuration for the CapsuleVision system.
All hardware, camera, logging, and pipeline parameters live here.
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

# Ensure directories exist
for _dir in (LOG_DIR, CAPTURE_DIR, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Camera settings – RPi Global Shutter Camera (IMX296)
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    # Sensor resolution: 1456 x 1088 native for IMX296
    width: int           = 1456
    height: int          = 1088
    framerate: int       = 60          # Global shutter supports up to 60 fps
    exposure_us: int     = 5000        # Exposure time in microseconds
    analogue_gain: float = 1.0         # 1.0 – 16.0
    awb_mode: str        = "auto"      # White balance mode
    colour_space: str    = "Rec709"
    hflip: bool          = False
    vflip: bool          = False
    # Preview window (set to None to disable)
    preview: bool        = False
    # Camera index (usually 0 for single-camera CM5 setups)
    camera_index: int    = 0
    # Capture format passed to Picamera2
    format: str          = "RGB888"    # BGR888 / RGB888 / XRGB8888
    # Buffer count – higher = smoother at cost of latency
    buffer_count: int    = 4


# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    # Processing resolution (resize after capture for performance)
    proc_width: int  = 1456
    proc_height: int = 1088
    # Frame queue depth between producer and consumer threads
    queue_size: int  = 8
    # Warmup frames to skip before analysis begins
    warmup_frames: int = 30


# ---------------------------------------------------------------------------
# Logging settings
# ---------------------------------------------------------------------------
@dataclass
class LoggingConfig:
    level: str       = "INFO"          # DEBUG | INFO | WARNING | ERROR
    log_to_file: bool = True
    log_file: Path   = LOG_DIR / "capsule_vision.log"
    max_bytes: int   = 10 * 1024 * 1024   # 10 MB per file
    backup_count: int = 5


# ---------------------------------------------------------------------------
# Application-level config (aggregates all sections)
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    camera:   CameraConfig   = field(default_factory=CameraConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging:  LoggingConfig  = field(default_factory=LoggingConfig)


# Singleton-style default config
DEFAULT_CONFIG = AppConfig()
