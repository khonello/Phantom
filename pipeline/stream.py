"""
Streaming pipeline for real-time face swapping.

Refactored for Phase 6 to use ProcessingPipeline instead of monolithic loop.
Provides start_pipeline() and stop_pipeline() interface for backward compatibility.
"""

import threading
from typing import Optional

from pipeline.config import CONFIG
from pipeline.events import BUS
from pipeline.processing.pipeline import ProcessingPipeline

NAME = 'PHANTOM.STREAM'

_pipeline: Optional[ProcessingPipeline] = None
_running: bool = False


def start_pipeline() -> None:
    """
    Start the streaming (realtime) pipeline.

    Launches ProcessingPipeline in a background thread.
    Uses CONFIG for all settings (source, tracker, enhancement, etc.).
    """
    global _pipeline, _running

    if _running:
        return

    _running = True

    try:
        # Create and start pipeline in background
        _pipeline = ProcessingPipeline(CONFIG, BUS)
        thread = threading.Thread(target=_pipeline.run_stream, daemon=True)
        thread.start()

    except Exception as e:
        _running = False
        raise


def stop_pipeline() -> None:
    """Stop the streaming pipeline."""
    global _pipeline, _running

    if _pipeline is not None:
        _pipeline.stop()

    _running = False


def is_running() -> bool:
    """Check if pipeline is currently running."""
    return _running
