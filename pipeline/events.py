"""
Event system for the Phantom pipeline.

Provides a lightweight, synchronous pub/sub mechanism for inter-component
communication without requiring external dependencies.

Events are identified by string constants to prevent typos and improve IDE
autocomplete.
"""

from typing import Callable, Dict, List, Any, Optional


class EventBus:
    """
    Lightweight pub/sub event emitter within a single process.

    This is synchronous (blocking) - subscribers are called immediately
    during emit(). For async patterns, use asyncio.Event instead.

    Example:
        bus = EventBus()
        bus.on('frame_ready', lambda frame: print(f"Got frame {frame.shape}"))
        bus.emit('frame_ready', frame=my_frame)
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[..., None]]] = {}

    def on(self, event: str, handler: Callable[..., None]) -> None:
        """
        Register a handler for an event.

        Args:
            event: Event type name
            handler: Callable that will be invoked with event data as kwargs
        """
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Callable[..., None]) -> None:
        """
        Unregister a handler for an event.

        Args:
            event: Event type name
            handler: The handler to remove
        """
        if event in self._handlers:
            self._handlers[event] = [h for h in self._handlers[event] if h != handler]

    def emit(self, event: str, **data: Any) -> None:
        """
        Emit an event, calling all registered handlers synchronously.

        Args:
            event: Event type name
            **data: Keyword arguments passed to all handlers
        """
        for handler in self._handlers.get(event, []):
            try:
                handler(**data)
            except Exception as e:
                # Log but don't crash if a handler fails
                import sys
                print(f"Warning: event handler for '{event}' failed: {e}", file=sys.stderr)

    def once(self, event: str, handler: Callable[..., None]) -> None:
        """
        Register a handler that fires only once, then auto-unregisters.

        Args:
            event: Event type name
            handler: Callable that will be invoked once
        """

        def wrapper(**data: Any) -> None:
            try:
                handler(**data)
            finally:
                self.off(event, wrapper)

        self.on(event, wrapper)

    def clear(self, event: Optional[str] = None) -> None:
        """
        Clear registered handlers.

        Args:
            event: If provided, clear only that event's handlers.
                   If None, clear all handlers.
        """
        if event is None:
            self._handlers.clear()
        elif event in self._handlers:
            del self._handlers[event]


# Global event bus singleton
BUS: EventBus = EventBus()


# ============================================================================
# Event Type Constants
# ============================================================================
# Use these instead of string literals to avoid typos and enable autocomplete.
# Events are emitted by ProcessingPipeline and listened to by API server,
# desktop bridge, and other components.

# Pipeline lifecycle
PIPELINE_STARTED = 'pipeline_started'
PIPELINE_STOPPED = 'pipeline_stopped'

# Frame processing
FRAME_READY = 'frame_ready'  # kwargs: frame, seq (sequence number)
DETECTION = 'detection'  # kwargs: detection, frame_seq
FACE_LOST = 'face_lost'  # kwargs: reason
DROP_RATE = 'drop_rate'  # kwargs: dropped, total, percent

# Status updates
STATUS_CHANGED = 'status_changed'  # kwargs: message, scope (optional)
CONFIG_CHANGED = 'config_changed'  # kwargs: field, value

# ML operations
EMBEDDING_READY = 'embedding_ready'  # kwargs: paths
SWAP_COMPLETE = 'swap_complete'  # kwargs: frame, detection

# Error handling
ERROR = 'error'  # kwargs: message, exception (optional), scope
WARNING = 'warning'  # kwargs: message, scope (optional)
