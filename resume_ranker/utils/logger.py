"""Centralized logging using loguru."""

import sys
from pathlib import Path
from loguru import logger
import config


def setup_logger(log_level: str = None, log_file: Path = None):
    """Configure loguru logger for the project."""
    level = log_level or config.LOG_LEVEL
    file_path = log_file or config.LOG_FILE

    logger.remove()

    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        str(file_path),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )

    return logger


log = setup_logger()
