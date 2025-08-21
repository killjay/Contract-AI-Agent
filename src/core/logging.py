"""
Logging configuration for the Legal Document Review AI Agent.
"""

import sys
import os
from typing import Optional
from loguru import logger
from src.core.config import get_config


def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    config = get_config()
    level = log_level or config.log_level
    
    # Remove default logger
    logger.remove()
    
    # Console logging with colors
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File logging if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    # Create logs directory if it doesn't exist
    logs_dir = "./logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Add daily rotating file handler
    logger.add(
        "./logs/legal_agent_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip"
    )
    
    # Error file handler
    logger.add(
        "./logs/errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="60 days"
    )


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)


# Set up logging on import
setup_logging()
