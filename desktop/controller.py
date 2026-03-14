"""
WebSocket client for communicating with a running Phantom pipeline.

Replaces the HTTP-based PipelineClient with a WebSocket connection to
ws://host:9000/ws (or PHANTOM_API_URL env var for remote connections).

Protocol:
  - Send commands as JSON text: {"action": "<cmd>", "data": {...}}
  - Receive events as JSON text: {"type": "event", "event": "<name>", ...}
  - Receive frames as binary: raw JPEG bytes

Supports:
  - PHANTOM_API_URL env var for remote/RunPod connections
  - wss:// for secure connections
  - 30-second connection timeout
  - Exponential backoff retry (max 3 retries)
  - Connection status callbacks
"""

import json
import os
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

NAME = 'DESKTOP.CONTROLLER'
UDP_INGEST_PORT: int = 5000

# Default WebSocket URL (can be overridden by PHANTOM_API_URL env var)
_DEFAULT_WS_URL = f'ws://localhost:9000/ws'


def _get_ws_url() -> str:
    """
    Get WebSocket URL from environment or default.

    Supports PHANTOM_API_URL env var for remote connections.
    If URL is a plain host:port, constructs ws://host:port/ws.

    Returns:
        WebSocket URL string
    """
    url = os.environ.get('PHANTOM_API_URL', _DEFAULT_WS_URL)
    if not url.startswith(('ws://', 'wss://')):
        url = f'ws://{url}/ws'
    return url


class PipelineClient:
    """
    WebSocket client for communicating with a running Phantom pipeline.

    Single persistent connection to ws://host:9000/ws.
    Sends commands as JSON text frames.
    Receives events and frames over the same connection.
    Handles reconnection with exponential backoff (max 3 retries).

    Example:
        client = PipelineClient()
        client.set_source('/path/to/face.jpg')
        client.start_stream()
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 9000,
        on_frame: Optional[Callable[[bytes], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_connected: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """
        Initialize WebSocket pipeline client.

        Args:
            host: Pipeline server host (default localhost)
            port: Pipeline server port (default 9000)
            on_frame: Callback for received JPEG frame bytes
            on_event: Callback for received event dictionaries
            on_connected: Callback for connection status changes (True/False)
        """
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.on_event = on_event
        self.on_connected = on_connected

        # Use env var URL if provided, else build from host/port
        env_url = os.environ.get('PHANTOM_API_URL')
        if env_url:
            if not env_url.startswith(('ws://', 'wss://')):
                env_url = f'ws://{env_url}/ws'
            self._ws_url = env_url
        else:
            self._ws_url = f'ws://{host}:{port}/ws'

        self._ws: Optional[Any] = None
        self._ws_lock = threading.Lock()
        self._connected = False

        # Pending responses keyed by action name (simple matching)
        self._pending: Dict[str, Any] = {}
        self._pending_lock = threading.Lock()

        # Response events
        self._response_events: Dict[str, threading.Event] = {}
        self._response_data: Dict[str, Dict[str, Any]] = {}

        # Background receiver thread
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_receiver()

    # ── Connection management ────────────────────────────────────────────────

    def _start_receiver(self) -> None:
        """Start background WebSocket receiver thread."""
        self._stop_event.clear()
        self._recv_thread = threading.Thread(
            target=self._receiver_loop,
            daemon=True,
            name='PipelineClient.receiver',
        )
        self._recv_thread.start()

    def _receiver_loop(self) -> None:
        """Background thread: maintain WebSocket connection and receive messages."""
        from websockets.sync.client import connect as ws_connect

        retry_delay = 1.0
        max_retries = 3
        attempt = 0

        while not self._stop_event.is_set():
            try:
                with ws_connect(
                    self._ws_url,
                    open_timeout=30,
                    max_size=64 * 1024 * 1024,
                ) as ws:
                    with self._ws_lock:
                        self._ws = ws
                    self._set_connected(True)
                    attempt = 0
                    retry_delay = 1.0

                    for message in ws:
                        if self._stop_event.is_set():
                            break

                        if isinstance(message, bytes):
                            # Binary: JPEG frame
                            if self.on_frame:
                                try:
                                    self.on_frame(message)
                                except Exception:
                                    pass
                        elif isinstance(message, str):
                            # Text: JSON event or response
                            try:
                                data = json.loads(message)
                                self._dispatch_message(data)
                            except (json.JSONDecodeError, Exception):
                                pass

            except Exception:
                pass
            finally:
                with self._ws_lock:
                    self._ws = None
                self._set_connected(False)

            if self._stop_event.is_set():
                break

            # Exponential backoff
            attempt += 1
            if attempt > max_retries:
                attempt = max_retries  # cap
            self._stop_event.wait(timeout=min(retry_delay * (2 ** (attempt - 1)), 30.0))

    def _dispatch_message(self, data: Dict[str, Any]) -> None:
        """
        Route an inbound JSON message.

        Args:
            data: Parsed JSON dictionary
        """
        msg_type = data.get('type', '')
        action = data.get('action', data.get('type', ''))

        # Response to a command — unblock waiting caller
        if msg_type == 'response' or 'success' in data:
            key = action
            with self._pending_lock:
                if key in self._response_events:
                    self._response_data[key] = data
                    self._response_events[key].set()
                    return

        # Push event — call callback
        if self.on_event:
            try:
                self.on_event(data)
            except Exception:
                pass

    def _set_connected(self, value: bool) -> None:
        """Update connection status and fire callback."""
        if self._connected != value:
            self._connected = value
            if self.on_connected:
                try:
                    self.on_connected(value)
                except Exception:
                    pass

    def close(self) -> None:
        """Stop the receiver loop and close connection."""
        self._stop_event.set()
        with self._ws_lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=3)

    # ── Send / receive ────────────────────────────────────────────────────────

    def _send(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Send a command over WebSocket and wait for response.

        Args:
            action: Command action name
            **kwargs: Additional payload fields

        Returns:
            Response dictionary (or error dict on failure)
        """
        with self._ws_lock:
            ws = self._ws

        if ws is None:
            return {'error': 'not connected'}

        payload = json.dumps({'action': action, **kwargs})

        # Register response waiter
        event = threading.Event()
        with self._pending_lock:
            self._response_events[action] = event
            self._response_data.pop(action, None)

        try:
            ws.send(payload)
        except Exception as e:
            with self._pending_lock:
                self._response_events.pop(action, None)
            return {'error': str(e)}

        # Wait up to 5 seconds for response
        if event.wait(timeout=5.0):
            with self._pending_lock:
                result = self._response_data.pop(action, {})
                self._response_events.pop(action, None)
            return result
        else:
            with self._pending_lock:
                self._response_events.pop(action, None)
            return {'error': 'timeout waiting for response'}

    def status(self) -> Dict[str, Any]:
        """Get pipeline status (health check via WebSocket)."""
        return self._send('health')

    def get_frame(self) -> Optional[bytes]:
        """Frame delivery is push-based; this is a no-op."""
        return None

    # ── Source / target / output ──────────────────────────────────────────────

    def set_source(self, path: str) -> Dict[str, Any]:
        """Set source image path."""
        return self._send('set_source', path=path)

    def set_target(self, path: str) -> Dict[str, Any]:
        """Set target image/video path."""
        return self._send('set_target', path=path)

    def set_output(self, path: str) -> Dict[str, Any]:
        """Set output file path."""
        return self._send('set_output', path=path)

    # ── Processing settings ───────────────────────────────────────────────────

    def set_keep_fps(self, value: bool) -> Dict[str, Any]:
        """Set keep_fps flag."""
        return self._send('set_keep_fps', value=value)

    def set_keep_frames(self, value: bool) -> Dict[str, Any]:
        """Set keep_frames flag."""
        return self._send('set_keep_frames', value=value)

    def set_keep_audio(self, value: bool) -> Dict[str, Any]:
        """Set keep_audio flag."""
        return self._send('set_keep_audio', value=value)

    def set_many_faces(self, value: bool) -> Dict[str, Any]:
        """Set many_faces flag."""
        return self._send('set_many_faces', value=value)

    # ── Source embedding ──────────────────────────────────────────────────────

    def create_embedding(self, paths: List[str]) -> Dict[str, Any]:
        """Create face embedding from source paths."""
        return self._send('create_embedding', paths=paths)

    # ── Stream routing ────────────────────────────────────────────────────────

    def set_input_url(self, url: str) -> Dict[str, Any]:
        """Set network input stream URL."""
        return self._send('set_input_url', url=url)

    def set_stream_url(self, url: str) -> Dict[str, Any]:
        """Set stream URL (alias for set_input_url)."""
        return self._send('set_input_url', url=url)

    # ── Stream tuning ─────────────────────────────────────────────────────────

    def set_quality(self, preset: str) -> Dict[str, Any]:
        """Set quality preset."""
        return self._send('set_quality', preset=preset)

    def set_blend(self, value: float) -> Dict[str, Any]:
        """Set blend ratio."""
        return self._send('set_blend', value=value)

    def set_alpha(self, value: float) -> Dict[str, Any]:
        """Set alpha smoothing factor."""
        return self._send('set_alpha', value=value)

    # ── Pipeline control ──────────────────────────────────────────────────────

    def start(self) -> Dict[str, Any]:
        """Start batch processing."""
        return self._send('start')

    def start_stream(self) -> Dict[str, Any]:
        """Start stream processing."""
        return self._send('start_stream')

    def stop(self) -> Dict[str, Any]:
        """Stop pipeline."""
        return self._send('stop')

    def stop_stream(self) -> Dict[str, Any]:
        """Stop stream (alias for stop)."""
        return self._send('stop')

    def cleanup_session(self) -> Dict[str, Any]:
        """Clean up session."""
        return self._send('cleanup_session')

    def shutdown(self) -> Dict[str, Any]:
        """Request server shutdown."""
        return self._send('shutdown')


def _run_webcam_broadcast(
    webcam_index: int,
    server_host: str,
    udp_port: int,
    stop_event: threading.Event,
) -> None:
    cap = cv2.VideoCapture(webcam_index)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    cmd: List[str] = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'mpegts',
        f'udp://{server_host}:{udp_port}',
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f'[{NAME}] FFmpeg webcam broadcast failed to start: {e}')
        cap.release()
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                assert proc.stdin is not None
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                break
    finally:
        cap.release()
        try:
            assert proc.stdin is not None
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


def start_webcam_broadcast(
    webcam_index: int,
    server_host: str,
    udp_port: int = UDP_INGEST_PORT,
) -> Tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_run_webcam_broadcast,
        args=(webcam_index, server_host, udp_port, stop_event),
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def stop_webcam_broadcast(thread: threading.Thread, stop_event: threading.Event) -> None:
    stop_event.set()
    thread.join(timeout=5)
