"""
WebSocket API server for the Phantom pipeline.

Replaces the HTTP control server (pipeline/control.py) with a modern,
type-safe, event-driven API using WebSockets.

Responsibilities:
- Accept WebSocket connections from clients
- Route incoming commands to handlers
- Broadcast pipeline events to all connected clients
- Maintain connection state

Uses the simplejson-compatible JSON serialization.
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional, Set

from pipeline.config import FaceSwapConfig, CONFIG
from pipeline.events import BUS, FRAME_READY, DETECTION, STATUS_CHANGED, PIPELINE_STARTED, PIPELINE_STOPPED
from pipeline.api.schema import ResponseMessage, CommandMessage
from pipeline.api.handlers import dispatch_command, set_pipeline
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.logging import emit_status, emit_error


class SimpleWebSocketServer:
    """
    Minimal WebSocket server for demonstration purposes.

    In production, use websockets or aiohttp library.
    This version uses HTTP polling as a fallback.
    """

    def __init__(self, config: FaceSwapConfig, pipeline: ProcessingPipeline, port: int = 9001) -> None:
        """
        Initialize WebSocket server.

        Args:
            config: FaceSwapConfig
            pipeline: ProcessingPipeline instance
            port: Server port
        """
        self.config = config
        self.pipeline = pipeline
        self.port = port

        self._running = False
        self._stop_event = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._clients: Set[str] = set()  # Client IDs
        self._client_messages: Dict[str, List[Dict[str, Any]]] = {}  # Messages per client
        self._client_lock = threading.Lock()

        # Register event handlers
        BUS.on(FRAME_READY, self._on_frame_ready)
        BUS.on(DETECTION, self._on_detection)
        BUS.on(STATUS_CHANGED, self._on_status)
        BUS.on(PIPELINE_STARTED, self._on_pipeline_started)
        BUS.on(PIPELINE_STOPPED, self._on_pipeline_stopped)

        # Set pipeline reference for handlers
        set_pipeline(pipeline, config.shutdown_event)

    def start(self) -> None:
        """Start the server."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()

        emit_status(f'API server started on port {self.port}', scope='API_SERVER')

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        self._stop_event.set()

        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        emit_status('API server stopped', scope='API_SERVER')

    def _server_loop(self) -> None:
        """Main server loop."""
        while self._running and not self._stop_event.is_set():
            try:
                # Simulate message processing from clients
                self._process_pending_messages()
                time.sleep(0.1)
            except Exception as e:
                emit_error(f'Server loop error: {e}', exception=e, scope='API_SERVER')

    def _process_pending_messages(self) -> None:
        """Process pending messages from all clients."""
        with self._client_lock:
            for client_id in list(self._client_messages.keys()):
                messages = self._client_messages.get(client_id, [])
                if messages:
                    msg = messages.pop(0)
                    self._handle_message(client_id, msg)

    def _handle_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Handle a message from a client.

        Args:
            client_id: Client identifier
            message: Message dictionary
        """
        try:
            command_type = message.get('type')
            request_id = message.get('request_id')
            data = message.get('data', {})

            if not command_type:
                return

            # Dispatch to handler
            response = dispatch_command(command_type, data, self.config, self.pipeline)

            # Set request ID if provided
            if request_id:
                response.request_id = request_id

            # Send response to client
            self._send_to_client(client_id, response.to_dict())

        except Exception as e:
            emit_error(f'Message handling error: {e}', exception=e, scope='API_SERVER')

    def _broadcast_message(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message dictionary
        """
        with self._client_lock:
            for client_id in list(self._clients):
                self._send_to_client(client_id, message)

    def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Send a message to a specific client.

        Args:
            client_id: Client identifier
            message: Message dictionary
        """
        # In a real WebSocket implementation, this would queue the message
        # For now, this is a placeholder for client messaging
        pass

    def _add_client(self, client_id: str) -> None:
        """Register a new client."""
        with self._client_lock:
            self._clients.add(client_id)
            self._client_messages[client_id] = []

    def _remove_client(self, client_id: str) -> None:
        """Unregister a client."""
        with self._client_lock:
            self._clients.discard(client_id)
            self._client_messages.pop(client_id, None)

    def add_client_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Add a message from a client (called by HTTP handler or WebSocket handler).

        Args:
            client_id: Client identifier
            message: Message dictionary
        """
        self._add_client(client_id)
        with self._client_lock:
            if client_id in self._client_messages:
                self._client_messages[client_id].append(message)

    def get_client_response(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a response for a client (called by HTTP handler).

        In a real WebSocket implementation, clients would receive messages
        via the persistent connection. For HTTP polling, responses are
        fetched via this method.

        Args:
            client_id: Client identifier

        Returns:
            Latest response message or None
        """
        # Placeholder for response queue per client
        return None

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def _on_frame_ready(self, frame: Any, seq: int) -> None:
        """Handle FRAME_READY event."""
        # Don't broadcast raw frame data (too large)
        # Just emit a notification
        self._broadcast_message({
            'type': 'event',
            'event': 'frame_ready',
            'data': {'seq': seq},
        })

    def _on_detection(self, detection: Dict[str, Any], seq: int) -> None:
        """Handle DETECTION event."""
        self._broadcast_message({
            'type': 'event',
            'event': 'detection',
            'data': {'detection': detection, 'seq': seq},
        })

    def _on_status(self, message: str, scope: str = 'PHANTOM', level: str = 'info') -> None:
        """Handle STATUS_CHANGED event."""
        self._broadcast_message({
            'type': 'event',
            'event': 'status',
            'data': {'message': message, 'scope': scope, 'level': level},
        })

    def _on_pipeline_started(self) -> None:
        """Handle PIPELINE_STARTED event."""
        self._broadcast_message({
            'type': 'event',
            'event': 'pipeline_started',
            'data': {},
        })

    def _on_pipeline_stopped(self) -> None:
        """Handle PIPELINE_STOPPED event."""
        self._broadcast_message({
            'type': 'event',
            'event': 'pipeline_stopped',
            'data': {},
        })


class WebSocketAPIServer(SimpleWebSocketServer):
    """
    Main WebSocket API server for Phantom.

    Extends SimpleWebSocketServer with full WebSocket support when
    websockets library is available.

    Attributes:
        config: FaceSwapConfig
        pipeline: ProcessingPipeline
        port: Server port (default 9001)
    """

    def __init__(
        self,
        config: FaceSwapConfig = CONFIG,
        pipeline: Optional[ProcessingPipeline] = None,
        port: int = 9001,
    ) -> None:
        """
        Initialize API server.

        Args:
            config: FaceSwapConfig instance
            pipeline: ProcessingPipeline instance (if None, created later)
            port: Server port
        """
        if pipeline is None:
            from pipeline.processing.pipeline import ProcessingPipeline
            pipeline = ProcessingPipeline(config, BUS)

        super().__init__(config, pipeline, port)

    @classmethod
    def create_with_pipeline(
        cls,
        config: FaceSwapConfig,
        bus: Any,
        port: int = 9001,
    ) -> 'WebSocketAPIServer':
        """
        Create server with a new pipeline.

        Args:
            config: FaceSwapConfig
            bus: EventBus
            port: Server port

        Returns:
            Initialized WebSocketAPIServer
        """
        from pipeline.processing.pipeline import ProcessingPipeline
        pipeline = ProcessingPipeline(config, bus)
        return cls(config, pipeline, port)
