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
  - Exponential backoff retry, indefinite; the delay is capped, not the count
  - expect_disconnect() to stop retrying when the pod was stopped on purpose
  - Connection status callbacks
"""

import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

NAME = 'DESKTOP.CONTROLLER'
UDP_INGEST_PORT: int = 5000

# Default WebSocket URL (can be overridden by PHANTOM_API_URL env var)
_DEFAULT_WS_URL = 'ws://localhost:9000/ws'


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


@dataclass
class _Pending:
    """One in-flight request, waiting for the reply that carries its id."""

    action: str
    event: threading.Event = field(default_factory=threading.Event)
    data: Dict[str, Any] = field(default_factory=dict)


class PipelineClient:
    """
    WebSocket client for communicating with a running Phantom pipeline.

    Single persistent connection to ws://host:9000/ws.
    Sends commands as JSON text frames.
    Receives events and frames over the same connection.
    Reconnects with exponential backoff, indefinitely — a pod can be slow, a
    laptop can sleep, and a live call is not a good moment to stop trying. The
    delay is what is capped (at 30s), not the number of attempts.

    The exception is a disconnect we were told to expect: `expect_disconnect()`
    marks the pod as having gone away on purpose, and the loop stops rather
    than hammering a stopped pod forever. Without it an expired session looks
    exactly like a network fault, which is the wrong thing to show someone
    whose paid time has simply run out.

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

        # Pending requests, keyed by request id rather than by action name.
        #
        # Keying by action is what let a reply the caller had already given up
        # on satisfy the *next* request of the same name. An upload that timed
        # out client-side does not stop the server working, so when the operator
        # retried, the first attempt's reply arrived and unblocked the retry —
        # which then reported a review of a request it was not waiting on, while
        # its own reply was dropped for having no waiter left.
        self._pending: "OrderedDict[str, _Pending]" = OrderedDict()
        self._pending_lock = threading.Lock()
        self._request_seq = 0

        # Background receiver thread
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Set when the pod has gone away on purpose — the paid hour ended, or
        # the pipeline announced an auto-stop. Distinguishes "we are done" from
        # "the network dropped", which are identical at the socket level and
        # must not look identical to the person watching.
        self._expected_disconnect = threading.Event()
        self._start_receiver()

    # ── Connection management ────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """
        Whether the socket is up right now.

        Exists because `on_connected` fires on *transitions* only, and the
        receiver thread starts inside `__init__` — before any caller has had a
        chance to attach a listener. A connection established in that window
        fires into `None` and is never re-announced, so a listener that arrives
        late has no way to learn the truth except by asking.
        """
        return self._connected

    def expect_disconnect(self) -> None:
        """
        Stop reconnecting: the pod is gone because it was meant to be.

        Called when the pipeline broadcasts `auto_stop`, or when the session's
        own clock runs out. Retrying past this point produces a UI that says
        "reconnecting…" forever about a pod that will never answer.
        """
        self._expected_disconnect.set()

    def expect_reconnect(self) -> None:
        """Undo `expect_disconnect` — a new session is starting."""
        self._expected_disconnect.clear()
        if self._recv_thread is None or not self._recv_thread.is_alive():
            self._start_receiver()

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
        # Caps the backoff exponent, not the number of attempts — the loop
        # below runs until stopped or until a disconnect we were told to
        # expect.
        max_backoff_steps = 3
        attempt = 0

        while not self._stop_event.is_set() and not self._expected_disconnect.is_set():
            try:
                with ws_connect(
                    self._ws_url,
                    # Short, because this is a *retry* loop and the timeout is
                    # dead air. A pod that is still booting does not refuse the
                    # connection — the RunPod proxy accepts it and holds — so
                    # every attempt before the pipeline is listening costs the
                    # full timeout, and the desktop can sit disconnected for
                    # that long after the pipeline actually comes up. Ten
                    # seconds is still ~28 round trips at the 350ms RTT this
                    # deployment actually runs at.
                    open_timeout=10,
                    max_size=64 * 1024 * 1024,
                    ping_interval=30,
                    ping_timeout=120,  # generous timeout for high-latency / saturated links
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
                                except Exception as e:
                                    print(f'[CONTROLLER] on_frame callback error: {type(e).__name__}: {e}', file=sys.stderr)
                        elif isinstance(message, str):
                            # Text: JSON event or response
                            try:
                                data = json.loads(message)
                                self._dispatch_message(data)
                            except json.JSONDecodeError as e:
                                print(f'[CONTROLLER] JSON decode error: {e} — raw: {message[:120]}', file=sys.stderr)
                            except Exception as e:
                                print(f'[CONTROLLER] message dispatch error: {type(e).__name__}: {e}', file=sys.stderr)

            except Exception as e:
                print(f'[CONTROLLER] Connection error ({self._ws_url}): {e}', file=sys.stderr)
            finally:
                with self._ws_lock:
                    self._ws = None
                self._set_connected(False)

            if self._stop_event.is_set():
                break

            if self._expected_disconnect.is_set():
                print('[CONTROLLER] Disconnect was expected — not reconnecting.',
                      file=sys.stderr)
                break

            # Exponential backoff
            attempt += 1
            if attempt > max_backoff_steps:
                attempt = max_backoff_steps  # cap the delay, keep retrying
            self._stop_event.wait(timeout=min(retry_delay * (2 ** (attempt - 1)), 30.0))

    def _resolve_pending(
        self,
        data: Dict[str, Any],
        action: str,
    ) -> Optional['_Pending']:
        """
        Find the waiter a reply belongs to, and retire it.

        Matched on `request_id` when the server echoed one. A reply carrying an
        id we no longer hold is stale — the caller timed out and gave up — and
        is dropped rather than handed to whoever is waiting now.

        The fall-back to matching by action exists for a pipeline old enough not
        to echo the id at all; it takes the oldest waiter for that action, which
        is the best guess available and the behaviour this had before ids.

        Args:
            data: The parsed reply
            action: Command name the reply claims to answer

        Returns:
            The retired waiter, or None if nothing is waiting for this reply.
        """
        request_id = str(data.get('request_id') or '')

        with self._pending_lock:
            if request_id:
                return self._pending.pop(request_id, None)

            for key, candidate in self._pending.items():
                if candidate.action == action:
                    self._pending.pop(key, None)
                    return candidate
        return None

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
            pending = self._resolve_pending(data, action)
            if pending is not None:
                pending.data = data
                pending.event.set()
                return

        # Push event — call callback
        if self.on_event:
            try:
                self.on_event(data)
            except Exception as e:
                print(f'[CONTROLLER] on_event callback error: {type(e).__name__}: {e}', file=sys.stderr)

    def _set_connected(self, value: bool) -> None:
        """Update connection status and fire callback."""
        if self._connected != value:
            self._connected = value
            if self.on_connected:
                try:
                    self.on_connected(value)
                except Exception as e:
                    print(f'[CONTROLLER] on_connected callback error: {type(e).__name__}: {e}', file=sys.stderr)

    def close(self) -> None:
        """Stop the receiver loop and close connection."""
        self._stop_event.set()
        with self._ws_lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception as e:
                    print(f'[CONTROLLER] WebSocket close error: {type(e).__name__}: {e}', file=sys.stderr)
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=3)

    # ── Send / receive ────────────────────────────────────────────────────────

    def _fire(self, action: str, **kwargs: Any) -> None:
        """Send a command without waiting for a response.

        Used for config-only commands (set_quality, set_enhance, etc.) where
        the caller doesn't need the server's reply before proceeding.
        """
        with self._ws_lock:
            ws = self._ws
        if ws is None:
            return
        try:
            ws.send(json.dumps({'action': action, **kwargs}))
        except Exception:
            pass  # best-effort — start_stream will catch real failures

    def _send(self, action: str, _timeout: float = 5.0, **kwargs: Any) -> Dict[str, Any]:
        """
        Send a command over WebSocket and wait for response.

        Args:
            action: Command action name
            _timeout: Seconds to wait for the response. Underscored so it
                      cannot collide with a payload field name. The default
                      suits control commands; bulk transfers pass their own,
                      since the response only arrives once the server has read
                      the whole message
            **kwargs: Additional payload fields

        Returns:
            Response dictionary (or error dict on failure)
        """
        with self._ws_lock:
            ws = self._ws

        if ws is None:
            return {'success': False, 'error': 'not connected'}

        # Register the waiter before sending, so a reply that arrives while
        # this thread is still in `ws.send` has somewhere to land.
        with self._pending_lock:
            self._request_seq += 1
            request_id = str(self._request_seq)
            pending = _Pending(action=action)
            self._pending[request_id] = pending

        payload = json.dumps({'action': action, 'request_id': request_id, **kwargs})

        try:
            ws.send(payload)
        except Exception as e:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            # `success` is set explicitly on every failure path. Callers read
            # `reply.get('success', True)` — defaulting to True, because a
            # handler that answers without the field has succeeded — so an
            # error dict without it was being read as a success with no data.
            return {'success': False, 'error': str(e)}

        # A request issued from the receive thread can never be answered: the
        # reply is delivered by `_dispatch_message`, which is what is currently
        # blocked here. It cost a whole render — the download of a finished
        # video ran inside the PIPELINE_STOPPED callback, timed out, and the
        # file was left on a pod that was later terminated. Callbacks must hand
        # this kind of work to a worker thread.
        if threading.current_thread() is self._recv_thread:
            import sys
            print(
                '[CONTROLLER] BUG: {} was sent from the receive thread and '
                'cannot be answered — move this call to a worker thread.'
                .format(action),
                file=sys.stderr,
            )
            with self._pending_lock:
                self._pending.pop(request_id, None)
            return {
                'success': False,
                'error': '{} called from the receive thread'.format(action),
            }

        # Wait for the response. `_resolve_pending` retires the entry when the
        # reply lands, so the only entry left to drop here is a timed-out one —
        # and dropping it is what makes its late reply stale rather than an
        # answer to whatever is sent next.
        if pending.event.wait(timeout=_timeout):
            return pending.data

        with self._pending_lock:
            self._pending.pop(request_id, None)
        return {
            'success': False,
            'error': 'timeout waiting for response to {}'.format(action),
        }

    def status(self) -> Dict[str, Any]:
        """Get pipeline status (health check via WebSocket)."""
        return self._send('health')

    def get_state(self) -> Dict[str, Any]:
        """Get full pipeline state for UI sync on reconnect."""
        return self._send('get_state')

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

    # ── Output format ─────────────────────────────────────────────────────────
    # `set_keep_frames` and `set_many_faces` used to sit here with no handler
    # behind them, so calling either would have returned `Unknown command`.
    # Both are deliberately CLI-only rather than missing: `many_faces` bypasses
    # every runtime guard and both temporal EMAs, and `keep_frames` is a
    # debugging flag that fills a pod's disk. See the note on COMMANDS in
    # pipeline/api/schema.py.
    #
    # The two below are real, and the desktop deliberately does *not* call them
    # on every render - see the note in bridge.startBatch.

    def set_keep_fps(self, value: bool) -> Dict[str, Any]:
        """Keep the target's frame rate, or retime the output to 30fps."""
        return self._send('set_keep_fps', value=value)

    def set_keep_audio(self, value: bool) -> Dict[str, Any]:
        """Keep the target's audio in the rendered output, or drop it."""
        return self._send('set_keep_audio', value=value)

    # ── Source embedding ──────────────────────────────────────────────────────

    def upload_source(self, images: List[Dict[str, str]]) -> Dict[str, Any]:
        """Upload source image(s) as base64 to the pipeline.

        Each entry: {'name': filename, 'data': base64_string}.
        Works for both single and multi-image (averaged embedding) cases.

        The generous timeout is not about the transfer. The reply only comes
        once the server has decoded and written every image *and* run the full
        source review over them — an InsightFace detection per photo, behind a
        first-upload model load that costs tens of seconds on its own. On the
        default 5s this timed out routinely, and the operator, seeing nothing,
        uploaded again.
        """
        return self._send('upload_source', _timeout=180.0, images=images)

    def list_templates(self) -> Dict[str, Any]:
        """Fetch the bundled template library, thumbnails included.

        Thumbnails travel inline because the library is on the pipeline's
        filesystem, which is not this machine when the pipeline is a pod.
        """
        return self._send('list_templates', _timeout=60.0)

    def set_template(self, template_id: str) -> Dict[str, Any]:
        """Choose a bundled template as the target."""
        return self._send('set_template', id=template_id)

    def upload_target(self, images: List[Dict[str, str]]) -> Dict[str, Any]:
        """Upload target photo(s) as base64 for a photo-mode job.

        Each entry: {'name': filename, 'data': base64_string}. At most four,
        enforced server-side as well. Unlike `set_source`/`set_target`, this
        needs no shared filesystem, which is what lets photo mode run against
        a remote pod.
        """
        return self._send('upload_target', _timeout=60.0, images=images)

    # ── Target video transfer ────────────────────────────────────────────────
    # A video cannot use `upload_target`'s shape. That carries a file base64 in
    # one message, and the server caps a message at 64 MB, which base64's 4/3
    # inflation turns into a 48 MB ceiling on the file. Chunking keeps the
    # message size out of the product limit.
    #
    # Each call is a normal request/response, so a chunk that is refused stops
    # the transfer at that chunk rather than at the end of it.

    def upload_video_begin(self, name: str, size: int) -> Dict[str, Any]:
        """Open a chunked target-video upload. Returns `upload_id`."""
        return self._send('upload_video_begin', name=name, size=size)

    def upload_video_chunk(
        self, upload_id: str, seq: int, data: str,
    ) -> Dict[str, Any]:
        """Send one base64 chunk. Sequence must be contiguous from zero."""
        return self._send(
            'upload_video_chunk', _timeout=120.0,
            upload_id=upload_id, seq=seq, data=data,
        )

    def upload_video_end(self, upload_id: str) -> Dict[str, Any]:
        """
        Close the upload and stage the clip as the target.

        The reply carries the server-side path, the probed duration and a
        first-frame thumbnail. Given a generous timeout because the server
        probes the file with ffprobe before answering.
        """
        return self._send('upload_video_end', _timeout=60.0, upload_id=upload_id)

    def upload_video_cancel(self, upload_id: str) -> Dict[str, Any]:
        """Abandon an upload and delete the partial file server-side."""
        return self._send('upload_video_cancel', upload_id=upload_id)

    def get_output_info(self) -> Dict[str, Any]:
        """Name and size of the finished render, before fetching it."""
        return self._send('get_output_info', _timeout=30.0)

    def get_output_chunk(self, offset: int, length: int) -> Dict[str, Any]:
        """Read one slice of the finished render."""
        return self._send(
            'get_output_chunk', _timeout=120.0, offset=offset, length=length)

    def get_render_thumbnails(self) -> Dict[str, Any]:
        """
        First frames of the configured render target and output.

        Both files live on the pipeline's filesystem, so the pictures travel
        rather than paths — the same reason `get_photo_results` returns images.
        """
        return self._send('get_render_thumbnails', _timeout=30.0)

    def set_target_faces(
        self, points: List[Optional[List[float]]],
    ) -> Dict[str, Any]:
        """Name the face to swap in each uploaded target photo.

        One normalised [x, y] per photo in `upload_target` order, or None for
        a photo the operator was not asked about. Detection order is not a
        stable contract, which is why this is a point rather than an index.
        """
        return self._send('set_target_faces', points=points)

    def get_photo_results(self, include_images: bool = True) -> Dict[str, Any]:
        """Fetch per-photo outcomes of the last photo job, images included."""
        return self._send('get_photo_results', _timeout=60.0, include_images=include_images)

    def create_embedding(self, paths: List[str]) -> Dict[str, Any]:
        """Create face embedding from source paths."""
        return self._send('create_embedding', paths=paths)

    # ── Stream routing ────────────────────────────────────────────────────────

    def set_restoration(self, preset: str) -> Dict[str, Any]:
        """Set restoration strength by name (auto/off/subtle/balanced/full)."""
        return self._send('set_restoration', preset=preset)

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

    def set_enhance(self, value: bool) -> Dict[str, Any]:
        """Enable or disable face restoration."""
        return self._send('set_enhance', value=value)

    def set_realism(self, **values: Any) -> Dict[str, Any]:
        """
        Set realism tuning parameters.

        Accepts any of: enhancer_model, enhancer_weight, enhance_strength,
        aligned_size, temporal_alpha, color_strength, grain, occluder.
        """
        return self._send('set_realism', values=values)

    def set_color_correction(self, value: bool) -> Dict[str, Any]:
        """Enable or disable color correction for cross-skin-tone swaps."""
        return self._send('set_color_correction', value=value)

    def set_preprocessing(self, value: bool) -> Dict[str, Any]:
        """Enable or disable frame preprocessing (CLAHE, white balance, denoise)."""
        return self._send('set_preprocessing', value=value)

    def keep_alive(self) -> Dict[str, Any]:
        """Reset the auto-stop timer, extending pod uptime."""
        return self._send('keep_alive')

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

    def send_frame(self, jpeg_bytes: bytes) -> None:
        """Send a raw JPEG frame to the pipeline (fire-and-forget, non-blocking).

        Drops the frame silently if the connection is busy or unavailable.
        """
        if not self._connected:
            return
        if not self._ws_lock.acquire(blocking=False):
            return  # drop frame — lock held by an in-flight command
        try:
            if self._ws is not None:
                self._ws.send(jpeg_bytes)
        except Exception as e:
            print(f'[CONTROLLER] send_frame error: {type(e).__name__}: {e}', file=sys.stderr)
        finally:
            self._ws_lock.release()

    def cleanup_session(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Erase the session on the pipeline: source, targets, outputs, embedding.

        Args:
            timeout: Seconds to wait. The default is short because the caller is
                     usually a window trying to close, and a pod that has already
                     gone away should not hold that up.
        """
        return self._send('cleanup_session', _timeout=timeout)

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
