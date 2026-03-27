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
import os
import queue
import struct
import sys
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

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
    - Frame streaming (FRAME_READY event → binary JPEG push via dedicated sender thread)
    - Status updates (STATUS_CHANGED event → JSON text push)
    - Command dispatch (JSON text received → handler response)
    - Health check ({"action": "health"} command)
    - Built-in WebSocket ping/pong (30s interval, 120s timeout)

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
        self._ws_server: Optional[Any] = None

        # Connected WebSocket clients (set of websocket objects)
        self._clients: Set[Any] = set()
        self._clients_lock = threading.Lock()

        # Frame broadcast queue — decouples pipeline thread from network I/O.
        # Pipeline thread puts encoded frames here; a dedicated sender thread
        # drains and broadcasts them so slow clients never stall processing.
        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self._frame_sender_thread: Optional[threading.Thread] = None

        # Start time for uptime reporting
        self._start_time = time.time()

        # Auto-stop timer — stops the RunPod pod after RUNPOD_MAX_UPTIME minutes
        # to prevent billing overruns. Configurable via env vars; disabled if
        # RUNPOD_MAX_UPTIME is 0 or unset.
        self._auto_stop_max = int(os.getenv('RUNPOD_MAX_UPTIME', '0')) * 60  # seconds
        self._auto_stop_warning = int(os.getenv('RUNPOD_STOP_WARNING', '5')) * 60  # seconds
        self._auto_stop_deadline = 0.0  # set on start()
        self._auto_stop_thread: Optional[threading.Thread] = None

        # Handler context (dependency injection — no globals)
        self._ctx = HandlerContext(
            pipeline=self.pipeline,
            shutdown_event=self.config.shutdown_event,
            reset_auto_stop=self._reset_auto_stop if self._auto_stop_max > 0 else None,
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

        self._frame_sender_thread = threading.Thread(
            target=self._frame_sender_loop, daemon=True,
        )
        self._frame_sender_thread.start()

        # Start auto-stop timer if configured
        if self._auto_stop_max > 0:
            self._auto_stop_deadline = time.time() + self._auto_stop_max
            self._auto_stop_thread = threading.Thread(
                target=self._auto_stop_loop, daemon=True,
            )
            self._auto_stop_thread.start()
            emit_status(
                f'Auto-stop enabled: pod will stop after '
                f'{self._auto_stop_max // 60}m (warning at '
                f'{self._auto_stop_warning // 60}m before)',
                scope='API_SERVER',
            )

        emit_status(f'WebSocket API server started on port {self.port}', scope='API_SERVER')

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        self._stop_event.set()

        # Shutdown the websockets server (unblocks serve_forever())
        if self._ws_server is not None:
            try:
                self._ws_server.shutdown()
            except Exception as e:
                emit_error(f'Server shutdown error: {type(e).__name__}: {e}', scope='API_SERVER')
            self._ws_server = None

        # Close all client connections
        with self._clients_lock:
            for ws in list(self._clients):
                try:
                    ws.close()
                except Exception as e:
                    emit_error(f'Client close error: {type(e).__name__}: {e}', scope='API_SERVER')
            self._clients.clear()

        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
            self._server_thread = None

        if self._frame_sender_thread is not None:
            self._frame_sender_thread.join(timeout=3.0)
            self._frame_sender_thread = None

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

            def process_request(connection: Any, request: Any) -> Any:
                """Respond to plain HTTP requests (RunPod proxy health probes).

                WebSocket upgrades pass through unchanged. Plain HTTP GETs
                get a 200 OK so the RunPod proxy considers the port alive.
                """
                from http import HTTPStatus
                from websockets.datastructures import Headers
                from websockets.http11 import Response
                upgrade = (request.headers.get('Upgrade') or '').lower()
                if upgrade == 'websocket':
                    return None  # proceed with WebSocket handshake
                # Plain HTTP — return 200 OK for proxy health checks
                return Response(
                    HTTPStatus.OK, 'OK',
                    headers=Headers([('Content-Type', 'text/plain')]),
                    body=b'OK\n',
                )

            with ws_serve(
                handler,
                '0.0.0.0',
                self.port,
                max_size=64 * 1024 * 1024,  # 64 MB max message (for file transfers)
                ping_interval=30,
                ping_timeout=120,  # generous timeout for high-latency / saturated links
                process_request=process_request,
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

    # Size of the capture_ts header prepended to binary frames (int64 nanoseconds)
    _TS_HEADER_SIZE = 8

    def _handle_binary_frame(self, data: bytes) -> None:
        """
        Handle an inbound binary WebSocket message (JPEG frame from desktop).

        Expected format: [8 bytes int64 capture_ts_ns] [N bytes JPEG data].
        If the message is shorter than the header (legacy client), treats the
        entire payload as JPEG with capture_ts = 0.

        Puts (capture_ts, jpeg_bytes) into the pipeline's frame_queue so the
        stream loop can decode and process them. Drops the frame if the queue
        is full (pipeline is falling behind) to avoid unbounded buffering.

        Args:
            data: Binary message from the desktop webcam thread
        """
        if len(data) > self._TS_HEADER_SIZE:
            capture_ts = struct.unpack('<q', data[:self._TS_HEADER_SIZE])[0]
            jpeg_bytes = data[self._TS_HEADER_SIZE:]
        else:
            capture_ts = 0
            jpeg_bytes = data

        fq = getattr(self.pipeline, 'frame_queue', None)
        if fq is not None:
            try:
                fq.put_nowait((capture_ts, jpeg_bytes))
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
                except Exception as e:
                    emit_error(f'Broadcast text failed ({ws.remote_address}): {type(e).__name__}: {e}', scope='API_SERVER')
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
                except Exception as e:
                    emit_error(f'Broadcast binary failed ({ws.remote_address}): {type(e).__name__}: {e}', scope='API_SERVER')
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

    # ── Auto-stop timer ─────────────────────────────────────────────────────

    def _reset_auto_stop(self) -> None:
        """Reset the auto-stop deadline, extending uptime by the full duration."""
        self._auto_stop_deadline = time.time() + self._auto_stop_max
        remaining = self._auto_stop_max // 60
        emit_status(f'Auto-stop reset — {remaining}m remaining', scope='AUTO_STOP')

    def _auto_stop_loop(self) -> None:
        """Background thread that enforces the pod uptime limit.

        Runs a check every 10 seconds. When the warning threshold is reached,
        broadcasts an auto_stop_warning event to all clients. If no keep_alive
        command resets the deadline before it expires, stops the pod.
        """
        warning_sent = False

        while not self._stop_event.is_set():
            time.sleep(10)
            now = time.time()
            remaining = self._auto_stop_deadline - now

            # Warning threshold reached
            if not warning_sent and remaining <= self._auto_stop_warning:
                warning_sent = True
                mins_left = max(1, int(remaining / 60))
                emit_status(
                    f'Pod will auto-stop in {mins_left} minute(s). '
                    f'Send keep_alive to extend.',
                    scope='AUTO_STOP',
                )
                self._broadcast_text({
                    'type': 'event',
                    'event': 'auto_stop_warning',
                    'data': {
                        'minutes_remaining': mins_left,
                        'deadline': self._auto_stop_deadline,
                    },
                })

            # Reset warning flag if deadline was extended past warning threshold
            if warning_sent and remaining > self._auto_stop_warning:
                warning_sent = False

            # Deadline reached — stop the pod
            if remaining <= 0:
                self._stop_pod()
                return

    def _stop_pod(self) -> None:
        """Stop the RunPod pod to halt billing. Falls back to sys.exit if API fails."""
        pod_id = os.getenv('RUNPOD_POD_ID', '')
        api_key = os.getenv('RUNPOD_API_KEY', '')

        emit_status('Auto-stop deadline reached — stopping pod...', scope='AUTO_STOP')
        self._broadcast_text({
            'type': 'event',
            'event': 'auto_stop',
            'data': {'reason': 'uptime limit reached'},
        })

        # Give clients a moment to receive the final event
        time.sleep(1)

        if pod_id and api_key:
            try:
                import runpod
                runpod.api_key = api_key
                runpod.stop_pod(pod_id)
                print(
                    f'[AUTO_STOP] Pod {pod_id} stopped via RunPod API.',
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f'[AUTO_STOP] RunPod API stop failed: {e} — exiting process.',
                    file=sys.stderr,
                )
                sys.exit(0)
        else:
            print(
                '[AUTO_STOP] RUNPOD_POD_ID or RUNPOD_API_KEY not set — exiting process.',
                file=sys.stderr,
            )
            sys.exit(0)

    # ── Frame sender (dedicated thread) ─────────────────────────────────────

    def _frame_sender_loop(self) -> None:
        """Drain _frame_queue and broadcast frames to clients.

        Runs on a dedicated thread so slow clients never stall the pipeline
        processing thread. Drops stale frames when a newer one is available.
        """
        while not self._stop_event.is_set():
            try:
                data = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # Drain to latest — skip intermediate frames to keep latency low
            while not self._frame_queue.empty():
                try:
                    data = self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._broadcast_binary(data)

    # ── Event handlers ───────────────────────────────────────────────────────

    def _on_frame_ready(self, frame: Any, seq: int, capture_ts: int = 0) -> None:
        """
        Handle FRAME_READY event — encode frame as JPEG and enqueue for broadcast.

        Prepends an 8-byte int64 capture_ts header before the JPEG payload so
        the desktop client can compute round-trip latency for A/V sync.

        Args:
            frame: numpy frame array
            seq: Sequence number
            capture_ts: Capture timestamp in nanoseconds (time.perf_counter_ns)
        """
        import cv2
        try:
            success, jpeg_data = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if success:
                header = struct.pack('<q', capture_ts)
                payload = header + jpeg_data.tobytes()
                try:
                    self._frame_queue.put_nowait(payload)
                except queue.Full:
                    # Drop oldest, enqueue latest — keeps display current
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(payload)
                    except queue.Full:
                        pass
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
