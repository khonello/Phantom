"""
Structured logging for the Phantom pipeline.

Provides a logger factory that emits both to stdout and to the event bus,
enabling both local and remote (e.g., desktop GUI) log visibility.

Replaces manual print() + globals.update() patterns from pipeline/core.py.
"""

import logging
import sys
from typing import Optional

from pipeline.events import BUS, STATUS_CHANGED, ERROR, WARNING


# Configure root logger once
_configured = False


def _configure_logging() -> None:
    """Set up root logger with stdout handler (call once)."""
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt='[%(name)s] %(levelname)s: %(message)s',
    ))
    root_logger.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    _configure_logging()
    return logging.getLogger(name)


def emit_status(
    message: str,
    scope: str = 'PHANTOM',
    level: str = 'info',
) -> None:
    """
    Emit a status message to both logger and event bus.

    This is the main way to communicate status changes to the desktop UI
    and other listeners.

    Args:
        message: Status message text
        scope: Component name (e.g., 'CORE', 'STREAM', 'API')
        level: Log level ('info', 'warning', 'error')
    """
    logger = get_logger(scope)

    # Log to stdout
    if level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    else:
        logger.debug(message)

    # Emit to event bus for remote listeners (desktop UI, etc.)
    BUS.emit(STATUS_CHANGED, message=message, scope=scope, level=level)


def emit_error(
    message: str,
    exception: Optional[Exception] = None,
    scope: str = 'PHANTOM',
) -> None:
    """
    Emit an error message with optional exception details.

    Args:
        message: Error description
        exception: Optional exception object for debugging
        scope: Component name (e.g., 'CORE', 'STREAM')
    """
    logger = get_logger(scope)
    if exception:
        logger.exception(message)
    else:
        logger.error(message)

    BUS.emit(ERROR, message=message, exception=exception, scope=scope)


def emit_warning(message: str, scope: str = 'PHANTOM') -> None:
    """
    Emit a warning message to logger and event bus.

    Args:
        message: Warning description
        scope: Component name
    """
    logger = get_logger(scope)
    logger.warning(message)
    BUS.emit(WARNING, message=message, scope=scope)
