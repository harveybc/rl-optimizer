# app/logger.py

import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(
    log_level=logging.DEBUG,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format="%Y-%m-%d %H:%M:%S",
    log_file="app.log",
    max_bytes=5*1024*1024,  # 5 MB
    backup_count=3
):
    """
    Sets up the logging configuration for the application.

    Parameters:
    ----------
    log_level : int, optional
        The threshold for the logger. Defaults to logging.DEBUG.
    log_format : str, optional
        The format string for log messages. Defaults to a detailed format.
    date_format : str, optional
        The format string for timestamps. Defaults to "%Y-%m-%d %H:%M:%S".
    log_file : str, optional
        The file path for the log file. Defaults to "app.log".
    max_bytes : int, optional
        The maximum size in bytes before a log file is rotated. Defaults to 5 MB.
    backup_count : int, optional
        The number of backup log files to keep. Defaults to 3.
    """
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Define formatter
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger with the specified name.

    Parameters:
    ----------
    name : str
        The name of the logger.

    Returns:
    -------
    logging.Logger
        Configured logger instance.
    """
    return logging.getLogger(name)
