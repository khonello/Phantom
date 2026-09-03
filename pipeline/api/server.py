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

import hmac
import json
import os
import queue
import struct
import sys
import threading
import time
from typing import Any, Dict, Optional, Set

from pipeline.config import FaceSwapConfig, CONFIG
from pipeline.events import BUS, ERROR, FRAME_READY, DETECTION, PHOTO_RESULT, STATUS_CHANGED, PIPELINE_STARTED, PIPELINE_STOPPED, WARNING
from pipeline.api.handlers import (
    dispatch_command,
    handle_cleanup_session,
    HandlerContext,
)
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.logging import emit_status, emit_error


class WebSocketAPIServer:
    """
    WebSocket API server for Phantom pipeline.

    Accepts WebSocket connections at ws://host:9000/ws.
    Pushes JPEG frames (binary) and JSON events (text) to all connected clients.
    Receives commands as JSON text frames.

    Supports:
    - Frame streaming (FRAME_READY event -> binary JPEG push via dedicated sender thread)
    - Status updates (STATUS_CHANGED event -> JSON text push)
    - Command dispatch (JSON text received -> handler response)
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
        # Built locally then assigned: an annotation on another object's
        # attribute is not valid Python, and `frame_queue` is already typed on
        # ProcessingPipeline, so this needs no annotation of its own.
        # Two frames, not ten. This queue is pure latency: anything waiting in
        # it is a frame the operator has already moved past. Ten deep is half a
        # second at 20fps, which is felt as lag rather than seen as stutter.
        # Two is enough to absorb ordinary arrival jitter without becoming a
        # buffer, and the handler above now evicts the oldest rather than
        # refusing the newest, so depth costs freshness and nothing else.
        frame_queue: 'queue.Queue[Any]' = queue.Queue(maxsize=2)
        self.pipeline.frame_queue = frame_queue

        self._running = False
        self._stop_event = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._ws_server: Optional[Any] = None

        # Connected WebSocket clients (set of websocket objects)
        self._clients: Set[Any] = set()
        self._clients_lock = threading.Lock()

        # Erasing the session once the last client has gone for good.
        #
        # Not on the disconnect itself: `PipelineClient` reconnects
        # indefinitely by design, because a pod can be slow and a laptop can
        # sleep, so a dropped socket is usually a blip in a session that is
        # still live. Wiping on the drop would make a flaky link unusable —
        # the operator's face would be deleted mid-call and the swap would
        # stop. A grace period distinguishes the blip from the departure, and
        # a reconnect inside it cancels the sweep.
        self._session_grace = float(os.getenv('PHANTOM_SESSION_GRACE', '120'))
        self._session_sweep: Optional[threading.Timer] = None
        self._sweep_lock = threading.Lock()

        # Frame broadcast queue — decouples pipeline thread from network I/O.
        # Pipeline thread puts encoded frames here; a dedicated sender thread
        # drains and broadcasts them so slow clients never stall processing.
        # Holds encoded bytes, or (frame, capture_ts) when `async_encode`
        # moves the JPEG encode onto the sender thread.
        self._frame_queue: 'queue.Queue[Any]' = queue.Queue(maxsize=2)
        self._frame_sender_thread: Optional[threading.Thread] = None

        # Start time for uptime reporting
        self._start_time = time.time()

        # Auto-stop timer — stops the Vast instance after VAST_MAX_UPTIME
        # minutes to prevent billing overruns. Configurable via env vars;
        # disabled if VAST_MAX_UPTIME is 0 or unset.
        self._auto_stop_max = int(os.getenv('VAST_MAX_UPTIME', '0')) * 60  # seconds
        self._auto_stop_warning = int(os.getenv('VAST_STOP_WARNING', '5')) * 60  # seconds
        self._auto_stop_deadline = 0.0  # set on start()
        self._auto_stop_thread: Optional[threading.Thread] = None

        # Handler context (dependency injection — no globals)
        self._ctx = HandlerContext(
            pipeline=self.pipeline,
            shutdown_event=self.config.shutdown_event,
            reset_auto_stop=self._reset_auto_stop if self._auto_stop_max > 0 else None,
            server_stats=self._server_stats,
        )

        # Register event handlers
        BUS.on(FRAME_READY, self._on_frame_ready)
        BUS.on(STATUS_CHANGED, self._on_status_changed)
        BUS.on(DETECTION, self._on_detection)
        BUS.on(PIPELINE_STARTED, self._on_pipeline_started)
        BUS.on(PIPELINE_STOPPED, self._on_pipeline_stopped)
        BUS.on(PHOTO_RESULT, self._on_photo_result)
        BUS.on(WARNING, self._on_warning)
        BUS.on(ERROR, self._on_error)

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

        self._cancel_session_sweep()

        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
            self._server_thread = None

        if self._frame_sender_thread is not None:
            self._frame_sender_thread.join(timeout=3.0)
            self._frame_sender_thread = None

        emit_status('WebSocket API server stopped', scope='API_SERVER')

    # ── Server loop ──────────────────────────────────────────────────────────

    def _build_ssl_context(self) -> Optional[Any]:
        """
        TLS for the WebSocket, when the instance was given a certificate.

        RunPod terminated TLS at its proxy and handed out a wss:// hostname.
        Vast maps this port to a random external port on a shared public IP and
        terminates nothing, so on a rented machine the choice is between
        serving the operator's face in cleartext and doing this.

        Off when the paths are unset, which is how a local run stays plain
        ws:// — `desktop.py` against a pipeline on the same machine has no
        network to protect, and requiring a certificate there would only mean
        generating one nobody checks.

        A missing or unreadable certificate is fatal rather than a downgrade.
        Falling back to cleartext would be the silent-CPU-fallback mistake in a
        worse place: the session would work, look identical, and be readable by
        anyone on the path.
        """
        cert = os.getenv('PHANTOM_TLS_CERT', '').strip()
        key = os.getenv('PHANTOM_TLS_KEY', '').strip()
        if not cert or not key:
            return None

        import ssl
        if not os.path.isfile(cert) or not os.path.isfile(key):
            emit_error(
                f'PHANTOM_TLS_CERT/KEY set but not readable ({cert}, {key}). '
                'Refusing to start in cleartext.',
                scope='API_SERVER',
            )
            raise SystemExit(1)

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        return context

    def _authenticate(self, websocket: Any, client_addr: Any) -> bool:
        """
        Require the shared token in the first frame, before anything is sent.

        `docs/ACCEPTED_RISKS.md` accepted the unauthenticated WebSocket partly
        because the RunPod proxy URL was "pod-specific and unguessable in
        practice". On Vast the address is an IP and a port on a shared host,
        which is neither, so the move has to close this rather than inherit it.

        Unset means open, which keeps local development and the test suite
        working exactly as before. That is a real weakening of the default, and
        it is why `vast/startup.sh` always generates a token: the deployment
        that needs it never runs without one.
        """
        expected = os.getenv('PHANTOM_API_TOKEN', '').strip()
        if not expected:
            return True

        try:
            # Ten seconds: long enough for a saturated uplink to deliver one
            # small frame, short enough that a port scanner holding the
            # socket open does not occupy a slot indefinitely.
            message = websocket.recv(timeout=10)
        except Exception:
            self._reject(websocket, client_addr, 'no opening frame')
            return False

        token = ''
        if isinstance(message, str):
            try:
                token = str((json.loads(message) or {}).get('token') or '')
            except (ValueError, AttributeError):
                token = ''

        # compare_digest so a wrong token cannot be found a character at a time.
        if not token or not hmac.compare_digest(token, expected):
            self._reject(websocket, client_addr, 'bad or missing token')
            return False

        # The opening frame is a real command, so it is answered rather than
        # consumed — otherwise every client would have to send `health` twice.
        self._handle_text_message(websocket, message)
        return True

    def _reject(self, websocket: Any, client_addr: Any, why: str) -> None:
        """Close an unauthenticated connection, saying nothing useful to it."""
        emit_status(f'Rejected {client_addr}: {why}', scope='API_SERVER')
        try:
            websocket.close(code=1008, reason='unauthorized')
        except Exception:
            pass

    def _server_loop(self) -> None:
        """Main WebSocket server loop using websockets.sync.server."""
        try:
            from websockets.sync.server import serve as ws_serve

            def handler(websocket: Any) -> None:
                """Handle a new WebSocket connection."""
                # Capture address before the socket can close
                try:
                    client_addr = websocket.remote_address
                except Exception:
                    client_addr = 'unknown'

                # Authenticate BEFORE joining the broadcast set. The order is
                # the whole point: frames go to every client in `_clients`, so
                # a connection added first and checked second would receive the
                # operator's swapped video in the window between the two.
                if not self._authenticate(websocket, client_addr):
                    return

                with self._clients_lock:
                    self._clients.add(websocket)
                self._cancel_session_sweep()

                emit_status(f'Client connected: {client_addr}', scope='API_SERVER')

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
                        remaining = len(self._clients)
                    emit_status(
                        f'Client disconnected: {client_addr}',
                        scope='API_SERVER',
                    )
                    if remaining == 0:
                        self._arm_session_sweep()

            def process_request(connection: Any, request: Any) -> Any:
                """Respond to plain HTTP requests (proxy and port health probes).

                WebSocket upgrades pass through unchanged. Plain HTTP GETs get
                a 200 OK so anything probing the port considers it alive.

                Deliberately answered before authentication: this says only
                that a socket is listening, which is already observable from
                the outside, and a health probe that had to hold a credential
                would be one more place for the credential to live.
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

            ssl_context = self._build_ssl_context()
            serve_kwargs: Dict[str, Any] = {
                'max_size': 64 * 1024 * 1024,  # 64 MB max message (for file transfers)
                'ping_interval': 30,
                'ping_timeout': 120,  # generous for high-latency / saturated links
                'process_request': process_request,
            }
            if ssl_context is not None:
                serve_kwargs['ssl'] = ssl_context

            with ws_serve(
                handler,
                '0.0.0.0',
                self.port,
                **serve_kwargs,
            ) as server:
                self._ws_server = server
                scheme = 'wss' if ssl_context is not None else 'ws'
                emit_status(
                    f'WebSocket server listening on {scheme}://0.0.0.0:{self.port}/ws',
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

    def _arm_session_sweep(self) -> None:
        """
        Schedule the session erase, the last client having gone.

        Does nothing when the grace period is 0, which is how a long-running
        pipeline that several clients come and go from opts out.
        """
        if self._session_grace <= 0:
            return

        with self._sweep_lock:
            if self._session_sweep is not None:
                self._session_sweep.cancel()
            self._session_sweep = threading.Timer(
                self._session_grace, self._sweep_session,
            )
            self._session_sweep.daemon = True
            self._session_sweep.start()

    def _cancel_session_sweep(self) -> None:
        """Call off a scheduled erase — someone reconnected."""
        with self._sweep_lock:
            if self._session_sweep is not None:
                self._session_sweep.cancel()
                self._session_sweep = None

    def _sweep_session(self) -> None:
        """
        Erase the session, if nobody came back inside the grace period.

        The client set is re-checked here rather than trusted from when the
        timer was armed: a reconnect races the fire, and deleting the source
        out from under a client that is already streaming again is the one
        outcome this must not produce.
        """
        with self._clients_lock:
            if self._clients:
                return

        with self._sweep_lock:
            self._session_sweep = None

        try:
            handle_cleanup_session(self.config, self.pipeline)
            emit_status(
                f'No client for {self._session_grace:.0f}s — session erased',
                scope='API_SERVER',
            )
        except Exception as e:
            emit_error(
                f'Session sweep failed: {type(e).__name__}: {e}',
                exception=e,
                scope='API_SERVER',
            )

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
            # Echo the client's request id. Without it a client can only match
            # a reply to a request by action name, which means a reply to a
            # request that already timed out answers the next one of the same
            # name — see `PipelineClient._resolve_pending`.
            request_id = data.get('request_id')
            if request_id is not None:
                response.request_id = str(request_id)
            self._send_json(websocket, response.to_dict())
        except Exception as e:
            emit_error(f'Command dispatch error: {e}', exception=e, scope='API_SERVER')
            failure = {
                'type': 'response',
                'action': action,
                'success': False,
                'error': str(e),
            }
            # A failure has to carry the id too, or the caller waits out its
            # whole timeout for a reply that already came back.
            request_id = data.get('request_id')
            if request_id is not None:
                failure['request_id'] = str(request_id)
            self._send_json(websocket, failure)

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
        if fq is None:
            return

        try:
            fq.put_nowait((capture_ts, jpeg_bytes))
        except queue.Full:
            # Drop the **oldest** frame and take this one. The previous version
            # dropped the arriving frame instead, which is backwards for a live
            # call: under pressure it kept a backlog of stale frames and threw
            # away the only current one, so the operator's face lagged by the
            # whole depth of the queue and stayed there. What reaches the call
            # should always be the most recent frame that could be processed.
            try:
                fq.get_nowait()
            except queue.Empty:
                pass
            try:
                fq.put_nowait((capture_ts, jpeg_bytes))
            except queue.Full:
                pass

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

    def _server_stats(self) -> Dict[str, Any]:
        """
        The server's own runtime facts, for `get_stats`.

        Uptime and the auto-stop deadline live here rather than on the config
        or the pipeline, so this is the only place that can answer "how much of
        the paid hour is left" — the number someone wants before starting a
        measurement they cannot finish.

        Returns:
            uptime_seconds, connected clients, and the auto-stop picture:
            `auto_stop_minutes` as configured (0 = disabled) and
            `auto_stop_remaining_seconds`, None when disabled.
        """
        now = time.time()
        remaining = None
        if self._auto_stop_max > 0 and self._auto_stop_deadline > 0:
            remaining = max(0.0, self._auto_stop_deadline - now)

        return {
            'uptime_seconds': round(now - self._start_time, 1),
            'clients': len(getattr(self, '_clients', ()) or ()),
            'auto_stop_minutes': self._auto_stop_max // 60,
            'auto_stop_remaining_seconds': (
                None if remaining is None else round(remaining, 1)
            ),
        }

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
        """
        Stop the Vast instance to halt GPU billing.

        Stop rather than destroy, deliberately: a stopped instance keeps its
        disk, so the venv and the model weights survive and the next session
        resumes warm. Nothing is baked into an image, so destroying here would
        make every auto-stop cost a full cold start.

        It does not halt billing completely — storage keeps charging while the
        instance exists — which is the trade recorded in
        docs/VAST_MIGRATION.md and the reason VAST_DISK is sized rather than
        left generous.

        Falls back to exiting the process, which frees nothing but at least
        stops the pipeline pretending to serve a session that has ended.
        """
        instance_id = os.getenv('VAST_INSTANCE_ID', '')
        api_key = os.getenv('VAST_API_KEY', '')

        emit_status('Auto-stop deadline reached — stopping instance...', scope='AUTO_STOP')
        self._broadcast_text({
            'type': 'event',
            'event': 'auto_stop',
            'data': {'reason': 'uptime limit reached'},
        })

        # Give clients a moment to receive the final event
        time.sleep(1)

        if instance_id and api_key:
            try:
                import requests
                resp = requests.put(
                    f'https://console.vast.ai/api/v0/instances/{instance_id}/',
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={'state': 'stopped'},
                    timeout=30,
                )
                resp.raise_for_status()
                print(
                    f'[AUTO_STOP] Instance {instance_id} stopped via Vast API.',
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f'[AUTO_STOP] Vast API stop failed: {e} — exiting process.',
                    file=sys.stderr,
                )
                sys.exit(0)
        else:
            print(
                '[AUTO_STOP] VAST_INSTANCE_ID or VAST_API_KEY not set — exiting process.',
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

            # Encode here rather than at emit, when asked to. Two effects, and
            # the second is the one that was not obvious: the encode comes off
            # the pipeline thread, *and* it only happens to the frame actually
            # being sent. The drain above discards intermediate frames, so
            # encoding at emit time paid for pictures nobody ever saw.
            if isinstance(data, tuple):
                encoded = self._encode_frame(data[0], data[1])
                if encoded is None:
                    continue
                data = encoded

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
        payload: Any
        if getattr(self.config, 'async_encode', False):
            # Defer the encode to the sender thread. The frame is not copied:
            # the compositor returns a new array per frame, and the held frame
            # a guard re-sends is only ever read.
            payload = (frame, capture_ts)
        else:
            encoded = self._encode_frame(frame, capture_ts)
            if encoded is None:
                return
            payload = encoded

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

    def _encode_frame(self, frame: Any, capture_ts: int) -> Optional[bytes]:
        """
        JPEG-encode one frame, with the capture timestamp as an 8-byte header.

        The header lets the desktop compute round-trip latency for A/V sync.

        Args:
            frame: numpy frame array
            capture_ts: Capture timestamp in nanoseconds

        Returns:
            The payload, or None if encoding failed
        """
        import cv2
        try:
            # Preset-driven rather than fixed at 85: this is the return leg the
            # operator actually watches, and on `fast` a fixed 85 spent latency
            # and bandwidth re-encoding a 480x270 frame far above the quality it
            # was captured at.
            quality = int(getattr(self.config, 'jpeg_quality', 70) or 70)
            success, jpeg_data = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if not success:
                return None
            return struct.pack('<q', capture_ts) + jpeg_data.tobytes()
        except Exception as e:
            emit_error(f'Frame encoding error: {e}', exception=e, scope='API_SERVER')
            return None

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

    def _on_photo_result(self, result: Any, index: int, total: int) -> None:
        """
        Handle PHOTO_RESULT — push one photo's outcome to all clients.

        Sent as each photo finishes rather than only at the end, so a client
        can show four tiles resolving one at a time instead of nothing until
        the job stops. The swapped image itself is not attached here; clients
        fetch it with `get_photo_results` once the job is done, which keeps
        this event small enough to be a status update.

        Args:
            result: PhotoResult for the photo just processed
            index: Zero-based position in the job
            total: Photos in the job
        """
        self._broadcast_text({
            'type': 'event',
            'event': 'PHOTO_RESULT',
            'index': index,
            'total': total,
            'result': result.to_dict(),
        })

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

    def _on_error(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        scope: str = 'PHANTOM',
    ) -> None:
        """
        Handle ERROR event — push to all clients as STATUS_CHANGED with error level.

        Without this a failed batch reaches the client as nothing but
        PIPELINE_STOPPED, which is indistinguishable from success.

        Args:
            message: Error description
            exception: Originating exception, not forwarded — the client has no
                       use for a traceback and it does not serialise
            scope: Source scope
        """
        self._broadcast_text({
            'type': 'event',
            'event': 'STATUS_CHANGED',
            'message': message,
            'scope': scope,
            'level': 'error',
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
