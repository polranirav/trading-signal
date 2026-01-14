"""
Structured logging configuration using structlog.

Provides JSON-formatted logs for production and human-readable logs for development.
Usage: from src.logging_config import get_logger
"""

import logging
import sys
import structlog
from src.config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Determine if we're in development or production
    is_development = settings.ENVIRONMENT == "development"
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )
    
    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if is_development:
        # Human-readable output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add file handler for persistent logs
    file_handler = logging.FileHandler(
        settings.LOGS_DIR / "trading_signals.log"
    )
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger for a module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Structured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", symbol="AAPL", count=100)
    """
    return structlog.get_logger(name)


# Configure logging on import
configure_logging()
