# capsule_vision/utils/logger.py
"""
Centralised logging factory.

Usage
-----
    from capsule_vision.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Camera initialised")
"""

import logging
import logging.handlers
from pathlib import Path

from capsule_vision.config import LoggingConfig


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def setup_root_logger(cfg: LoggingConfig) -> None:
    """
    Configure the root logger once at application startup.
    Call this from main.py before any other module imports loggers.
    """
    root = logging.getLogger("capsule_vision")
    root.setLevel(_level(cfg.level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_FORMATTER)
    root.addHandler(console_handler)

    # Rotating file handler
    if cfg.log_to_file:
        Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=cfg.log_file,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_FORMATTER)
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'capsule_vision' namespace."""
    return logging.getLogger(f"capsule_vision.{name}")
