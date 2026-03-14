"""
WebSocket API server for the Phantom pipeline.

Provides a real WebSocket server for push-based frame delivery and event streaming.
Replaces the HTTP-based implementation with:
- Text frames: JSON messages for commands and events
- Binary frames: JPEG-encoded video frames pushed to all clients

Protocol:
  - Client sends: {"action": "<command>", "data": {...}}
  - Server pushes events: {"type": "event", "event": "<name>", "data": {...}}
  - Server pushes frames: raw JPEG bytes (binary frames)

Server listens on ws://host:9000/ws
Health check: send {"action": "health"}, receive {"status": "healthy", "uptime": <seconds>}
"""

import json
import queue
import threading
import time
from typing import Any, Dict, Optional, Set

from pipeline.config import FaceSwapConfig, CONFIG
from pipeline.events import BUS, FRAME_READY, DETECTION, STATUS_CHANGED, PIPELINE_STARTED, PIPELINE_STOPPED, WARNING
from pipeline.api.schema import ResponseMessage
from pipeline.api.handlers import dispatch_command, HandlerContext
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.logging import emit_status, emit_error


class WebSocketAPIServer:
    """
    WebSocket API server for Phantom pipeline.

    Accepts WebSocket connections at ws://host:9000/ws.
    Pushes JPEG frames (binary) and JSON events (text) to all connected clients.
    Receives commands as JSON text frames.

    Supports:
    - Frame streaming (FRAME_READY event → binary JPEG push)
    - Status updates (STATUS_CHANGED event → JSON text push)
    - Command dispatch (JSON text received → handler response)
    - Health check ({"action": "health"} command)
    - Heartbeat ping/pong every 30 seconds

    Attributes:
        config: FaceSwapConfig
        pipeline: ProcessingPipeline
        port: Server port (default 9000)
    """

    def __init__(
        self,
        config: FaceSwapConfig = CONFIG,
        pipeline: Optional[ProcessingPipeline] = None,
        port: int = 9000,
    ) -> None:
        """
        Initialize WebSocket API server.

        Args:
            config: FaceSwapConfig instance
            pipeline: ProcessingPipeline instance (created if None)
            port: Server port (default 9000)
        """
        self.config = config
        self.port = port

        if pipeline is None:
            pipeline = ProcessingPipeline(config, BUS)
        self.pipeline = pipeline

        # Frame queue for WebSocket push mode — desktop sends JPEG frames here,
        # pipeline reads from it instead of opening a local VideoCapture.
        self.pipeline.frame_queue: queue.Queue = queue.Queue(maxsize=10)  # type: ignore[assignment]

        self._running = False
        self._stop_event = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._ws_server: Optional[Any] = None

        # Connected WebSocket clients (set of websocket objects)
        self._clients: Set[Any] = set()
        self._clients_lock = threading.Lock()

        # Start time for uptime reporting
        self._start_time = time.time()

        # Handler context (dependency injection — no globals)
        self._ctx = HandlerContext(
            pipeline=self.pipeline,
            shutdown_event=self.config.shutdown_event,
        )

        # Register event handlers
        BUS.on(FRAME_READY, self._on_frame_ready)
        BUS.on(STATUS_CHANGED, self._on_status_changed)
        BUS.on(DETECTION, self._on_detection)
        BUS.on(PIPELINE_STARTED, self._on_pipeline_started)
        BUS.on(PIPELINE_STOPPED, self._on_pipeline_stopped)
        BUS.on(WARNING, self._on_warning)

    @classmethod
    def create_with_pipeline(
        cls,
        config: FaceSwapConfig,
        bus: Any,
        port: int = 9000,
    ) -> 'WebSocketAPIServer':
        """
        Create server with a new pipeline.

        Args:
            config: FaceSwapConfig
            bus: EventBus (unused, kept for API compatibility)
            port: Server port

        Returns:
            Initialized WebSocketAPIServer
        """
        pipeline = ProcessingPipeline(config, BUS)
        return cls(config, pipeline, port)

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._start_time = time.time()

        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        emit_status(f'WebSocket API server started on port {self.port}', scope='API_SERVER')

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        self._stop_event.set()

        # Shutdown the websockets server (unblocks serve_forever())
        if self._ws_server is not None:
            try:
                self._ws_server.shutdown()
            except Exception:
                pass
            self._ws_server = None

        # Close all client connections
        with self._clients_lock:
            for ws in list(self._clients):
                try:
                    ws.close()
                except Exception:
                    pass
            self._clients.clear()

        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
            self._server_thread = None

        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=3.0)
            self._heartbeat_thread = None

        emit_status('WebSocket API server stopped', scope='API_SERVER')

    # ── Server loop ──────────────────────────────────────────────────────────

    def _server_loop(self) -> None:
        """Main WebSocket server loop using websockets.sync.server."""
        try:
            from websockets.sync.server import serve as ws_serve

            def handler(websocket: Any) -> None:
                """Handle a new WebSocket connection."""
                with self._clients_lock:
                    self._clients.add(websocket)

                emit_status(f'Client connected: {websocket.remote_address}', scope='API_SERVER')

                try:
                    for message in websocket:
                        if self._stop_event.is_set():
                            break
                        if isinstance(message, str):
                            self._handle_text_message(websocket, message)
                        elif isinstance(message, bytes):
                            self._handle_binary_frame(message)
                except Exception as e:
                    if self._running:
                        emit_error(
                            f'Client connection error: {e}',
                            exception=e,
                            scope='API_SERVER',
                        )
                finally:
                    with self._clients_lock:
                        self._clients.discard(websocket)
                    emit_status(
                        f'Client disconnected: {websocket.remote_address}',
                        scope='API_SERVER',
                    )

            with ws_serve(
                handler,
                '0.0.0.0',
                self.port,
                max_size=64 * 1024 * 1024,  # 64 MB max message (for file transfers)
            ) as server:
                self._ws_server = server
                emit_status(
                    f'WebSocket server listening on ws://0.0.0.0:{self.port}/ws',
                    scope='API_SERVER',
                )
                # serve_forever() drives the accept loop — without it connections
                # queue but handshakes never complete. Shutdown from stop().
                server.serve_forever()

        except OSError as e:
            if 'Address already in use' in str(e):
                emit_error(f'Port {self.port} already in use', scope='API_SERVER')
            else:
                emit_error(f'Server error: {e}', exception=e, scope='API_SERVER')
        except Exception as e:
            emit_error(f'Server loop error: {e}', exception=e, scope='API_SERVER')

    # ── Message handling ──────────────────────────────────────────────────────

    def _handle_text_message(self, websocket: Any, message: str) -> None:
        """
        Handle a JSON text message from a client.

        Args:
            websocket: The WebSocket connection that sent the message
            message: JSON string
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            self._send_json(websocket, {
                'type': 'error',
                'error': f'Invalid JSON: {e}',
            })
            return

        action = data.get('action')
        if not action:
            self._send_json(websocket, {
                'type': 'error',
                'error': 'Missing "action" field',
            })
            return

        # Health check — fast path
        if action == 'health':
            self._send_json(websocket, {
                'type': 'response',
                'action': 'health',
                'status': 'healthy',
                'uptime': time.time() - self._start_time,
            })
            return

        # Dispatch to handler
        try:
            response = dispatch_command(
                action,
                data,
                self.config,
                self._ctx,
            )
            self._send_json(websocket, response.to_dict())
        except Exception as e:
            emit_error(f'Command dispatch error: {e}', exception=e, scope='API_SERVER')
            self._send_json(websocket, {
                'type': 'response',
                'action': action,
                'success': False,
                'error': str(e),
            })

    def _handle_binary_frame(self, data: bytes) -> None:
        """
        Handle an inbound binary WebSocket message (JPEG frame from desktop).

        Puts the raw JPEG bytes into the pipeline's frame_queue so the stream
        loop can decode and process them. Drops the frame if the queue is full
        (pipeline is falling behind) to avoid unbounded buffering.

        Args:
            data: Raw JPEG bytes sent by the desktop webcam thread
        """
        fq = getattr(self.pipeline, 'frame_queue', None)
        if fq is not None:
            try:
                fq.put_nowait(data)
            except queue.Full:
                pass  # drop — pipeline is behind, keep latency low

    # ── Push helpers ──────────────────────────────────────────────────────────

    def _broadcast_text(self, payload: Dict[str, Any]) -> None:
        """
        Broadcast a JSON message to all connected clients.

        Args:
            payload: Dictionary to serialize and send as JSON text
        """
        message = json.dumps(payload)
        with self._clients_lock:
            disconnected = set()
            for ws in self._clients:
                try:
                    ws.send(message)
                except Exception:
                    disconnected.add(ws)
            for ws in disconnected:
                self._clients.discard(ws)

    def _broadcast_binary(self, data: bytes) -> None:
        """
        Broadcast binary data (JPEG frame) to all connected clients.

        Args:
            data: Raw bytes to send (JPEG-encoded frame)
        """
        with self._clients_lock:
            disconnected = set()
            for ws in self._clients:
                try:
                    ws.send(data)
                except Exception:
                    disconnected.add(ws)
            for ws in disconnected:
                self._clients.discard(ws)

    def _send_json(self, websocket: Any, payload: Dict[str, Any]) -> None:
        """
        Send JSON to a single client.

        Args:
            websocket: Target WebSocket connection
            payload: Dictionary to serialize and send
        """
        try:
            websocket.send(json.dumps(payload))
        except Exception as e:
            emit_error(f'Failed to send response: {e}', scope='API_SERVER')

    # ── Heartbeat ────────────────────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """Send WebSocket ping to all clients every 30 seconds."""
        while self._running and not self._stop_event.is_set():
            self._stop_event.wait(timeout=30.0)
            if not self._running:
                break
            with self._clients_lock:
                disconnected = set()
                for ws in self._clients:
                    try:
                        ws.ping()
                    except Exception:
                        disconnected.add(ws)
                for ws in disconnected:
                    self._clients.discard(ws)

    # ── Event handlers ───────────────────────────────────────────────────────

    def _on_frame_ready(self, frame: Any, seq: int) -> None:
        """
        Handle FRAME_READY event — encode frame as JPEG and broadcast to clients.

        Args:
            frame: numpy frame array
            seq: Sequence number
        """
        import cv2
        try:
            success, jpeg_data = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if success:
                self._broadcast_binary(jpeg_data.tobytes())
        except Exception as e:
            emit_error(f'Frame encoding error: {e}', exception=e, scope='API_SERVER')

    def _on_status_changed(
        self,
        message: str,
        scope: str = 'PHANTOM',
        level: str = 'info',
    ) -> None:
        """
        Handle STATUS_CHANGED event — push JSON text to all clients.

        Args:
            message: Status message
            scope: Source scope
            level: Log level
        """
        self._broadcast_text({
            'type': 'event',
            'event': 'STATUS_CHANGED',
            'message': message,
            'scope': scope,
            'level': level,
        })
        # Also update config status message
        if self.config:
            self.config.status_message = message

    def _on_warning(self, message: str, scope: str = 'PHANTOM') -> None:
        """
        Handle WARNING event — push to all clients as STATUS_CHANGED with warning level.

        Args:
            message: Warning message
            scope: Source scope
        """
        self._broadcast_text({
            'type': 'event',
            'event': 'STATUS_CHANGED',
            'message': message,
            'scope': scope,
            'level': 'warning',
        })

    def _on_detection(self, detection: Any, seq: int) -> None:
        """
        Handle DETECTION event — push JSON text to all clients.

        Args:
            detection: Detection dictionary
            seq: Sequence number
        """
        self._broadcast_text({
            'type': 'event',
            'event': 'DETECTION',
            'detection': detection,
            'seq': seq,
        })

    def _on_pipeline_started(self) -> None:
        """Handle PIPELINE_STARTED event."""
        self._broadcast_text({
            'type': 'event',
            'event': 'PIPELINE_STARTED',
        })

    def _on_pipeline_stopped(self) -> None:
        """Handle PIPELINE_STOPPED event."""
        self._broadcast_text({
            'type': 'event',
            'event': 'PIPELINE_STOPPED',
        })
