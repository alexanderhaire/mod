"""
Centralized Logging Configuration

Provides consistent logging setup across all modules in the application.
Import and call setup_logging() once at application startup.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# Default configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = Path(__file__).parent / "logs"


class AppLogFilter(logging.Filter):
    """Filter to add contextual information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add default values for custom fields if not present
        if not hasattr(record, 'user'):
            record.user = 'system'
        if not hasattr(record, 'request_id'):
            record.request_id = '-'
        return True


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional custom log file path
        enable_console: Whether to log to console
        enable_file: Whether to log to file
    
    Returns:
        The root logger configured for the application
    """
    # Ensure log directory exists
    if enable_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(AppLogFilter())
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        log_path = log_file or (LOG_DIR / "app.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(AppLogFilter())
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    
    # Log startup
    root_logger.info("Logging initialized successfully")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Usage:
        from logging_config import get_logger
        LOGGER = get_logger(__name__)
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding temporary context to logs."""
    
    def __init__(self, user: str = None, request_id: str = None):
        self.user = user
        self.request_id = request_id
        self._old_factory = None
    
    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        
        user = self.user
        request_id = self.request_id
        old_factory = self._old_factory
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.user = user or getattr(record, 'user', 'system')
            record.request_id = request_id or getattr(record, 'request_id', '-')
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self._old_factory)
        return False
