"""
Centralized logging configuration for feature generation pipeline.
All operations logged to both console and logs/pipeline.log
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[0]
LOG_DIR = BASE / 'logs'
LOG_FILE = LOG_DIR / 'pipeline.log'


def setup_logging():
    """Configure logging to file and console with timestamp."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('feature_pipeline')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler - detailed
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - simpler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str):
    """Get logger instance for a module."""
    return logging.getLogger(f'feature_pipeline.{name}')
