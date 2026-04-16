# Data_pull/logging_config.py
"""
Logging configuration for data-ingest project.

Log Levels:
  - DEBUG (10): Detailed information for diagnosing problems (rare)
  - INFO (20): General informational messages (major operations)
  - WARNING (30): Warning messages for potentially harmful situations
  - ERROR (40): Error messages for serious problems (job failed)
  - CRITICAL (50): Critical errors that need immediate attention

Usage:
    from Data_pull.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing file...")
    logger.error("Failed to process: %s", error)
"""

import logging
import logging.handlers
import os
from pathlib import Path


def configure_logging(
    logs_dir: str = None,
    app_name: str = "data-ingest",
    level: int = logging.INFO,
) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        logs_dir: Directory to store log files. Defaults to env var LOGS_DIR or ./logs
        app_name: Application name for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if logs_dir is None:
        logs_dir = os.getenv("LOGS_DIR", "/logs/ingest")
    
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Log file path
    log_file = logs_path / f"{app_name}.log"
    
    # ===== FILE HANDLER (All levels) =====
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,  # Keep 5 backup files
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    
    # ===== CONSOLE HANDLER (INFO and above) =====
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # ===== FORMATTER =====
    # File: detailed format with timestamps
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console: simpler format
    console_formatter = logging.Formatter(
        fmt='%(levelname)-8s | %(name)-20s | %(message)s'
    )
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        logging.Logger instance
    """
    return logging.getLogger(name)


# Initialize logging when module is imported
if not logging.getLogger().handlers:
    configure_logging()
