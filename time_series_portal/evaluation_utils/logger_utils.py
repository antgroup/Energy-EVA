import os
import sys

from loguru import logger


def setup_logger(_log_file):
    """Global logging configuration"""
    # Remove the default stderr handler
    logger.remove()
    if os.path.exists(_log_file):
        os.remove(_log_file)
    logger.add(
        _log_file,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.add(
        sys.stdout,
        level="DEBUG",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    return logger
