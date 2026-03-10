import gc
import queue
import subprocess
import time
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Property, QTimer, QSize, Qt
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtQuick import QQuickPaintedItem

from pipeline.utilities import is_image
from desktop.controller import PipelineClient, UDP_INGEST_PORT

_PANEL_MAX_W = 800
_PANEL_MAX_H = 500

# Raise GC thresholds to avoid periodic freezes from frame allocations
gc.set_threshold(2800, 15, 15)


# ── Frame buffer (thread-safe storage) ────────────────────────────

class FrameBuffer:
    """Background threads write QImages, main thread promotes to QPixmap."""

    def __init__(self) -> None:
        self._pixmap: Optional[QPixmap] = None
        self._pending: Optional[QImage] = None
        self._lock = threading.Lock()
        self._dirty = False

    @property
    def pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    def update_from_numpy(self, frame: np.ndarray) -> None:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        if w > _PANEL_MAX_W or h > _PANEL_MAX_H:
            qimg = qimg.scaled(
                _PANEL_MAX_W, _PANEL_MAX_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        with self._lock:
            self._pending = qimg
            self._dirty = True

    def update_from_bytes(self, data: bytes) -> None:
        qimg = QImage()
        qimg.loadFromData(data)
        if not qimg.isNull() and (qimg.width() > _PANEL_MAX_W or qimg.height() > _PANEL_MAX_H):
            qimg = qimg.scaled(
                _PANEL_MAX_W, _PANEL_MAX_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        with self._lock:
            self._pending = qimg
            self._dirty = True

    def is_dirty(self) -> bool:
        return self._dirty

    def promote(self) -> None:
        """Main thread: convert pending QImage → QPixmap."""
        with self._lock:
            img = self._pending
            self._pending = None
            self._dirty = False
        if img is not None and not img.isNull():
            self._pixmap = QPixmap.fromImage(img)


# Global frame buffers
_buffers: Dict[str, FrameBuffer] = {
    'webcam': FrameBuffer(),
    'live': FrameBuffer(),
}

webcam_buffer = _buffers['webcam']
live_buffer = _buffers['live']


# ── QML painted item (replaces Image + ImageProvider) ─────────────

class FrameDisplay(QQuickPaintedItem):
    """Efficient video frame display. Reuses the same FBO — no texture churn."""

    sourceChanged = Signal()
    frameVersionChanged = Signal()

    def __init__(self, parent: Optional[QQuickPaintedItem] = None) -> None:
        super().__init__(parent)
        self._source = ''
        self._frame_version = 0
        self.setRenderTarget(QQuickPaintedItem.RenderTarget.FramebufferObject)

    def _get_source(self) -> str:
        return self._source

    def _set_source(self, val: str) -> None:
        if self._source != val:
            self._source = val
            self.sourceChanged.emit()
            self.update()

    source = Property(str, _get_source, _set_source, notify=sourceChanged)

    def _get_frame_version(self) -> int:
        return self._frame_version

    def _set_frame_version(self, val: int) -> None:
        if self._frame_version != val:
            self._frame_version = val
            self.frameVersionChanged.emit()
            self.update()

    frameVersion = Property(int, _get_frame_version, _set_frame_version,
                            notify=frameVersionChanged)

    def paint(self, painter: QPainter) -> None:
        buf = _buffers.get(self._source)
        if buf is None:
            return
        pm = buf.pixmap
        if pm is None or pm.isNull():
            return

        iw = self.width()
        ih = self.height()
        pw = pm.width()
        ph = pm.height()
        if pw <= 0 or ph <= 0 or iw <= 0 or ih <= 0:
            return

        # Aspect-crop: scale to fill, center the overflow
        scale = max(iw / pw, ih / ph)
        sw = int(pw * scale)
        sh = int(ph * scale)
        x = int((iw - sw) / 2)
        y = int((ih - sh) / 2)
        painter.drawPixmap(x, y, sw, sh, pm)


# ── Bridge ────────────────────────────────────────────────────────

class Bridge(QObject):
    webcamVersionChanged = Signal(int)
    liveVersionChanged = Signal(int)
    statusMessageChanged = Signal(str)
    connectedChanged = Signal(bool)
    connectionLabelChanged = Signal(str)
    embeddingPendingChanged = Signal(bool)
    pipelineRunningChanged = Signal(bool)
    virtualCamActiveChanged = Signal(bool)
    sourceSetChanged = Signal(bool)
    sourceThumbnailChanged = Signal(str)
    sourceLabelChanged = Signal(str)

    def __init__(self, client: PipelineClient) -> None:
        super().__init__()
        self._client = client
        self._source_set = False
        self._source_thumbnail: str = ''
        self._source_label: str = ''
        self._pipeline_running = False
        self._virtual_cam_active = False
        self._embedding_pending = False
        self._connected = False
        self._connection_label = 'connecting...'
        self._status_message = 'idle'
        self._webcam_version = 0
        self._live_version = 0
        self._quality = 'optimal'
        self._vcam_platform = 'obs'
        self._webcam_index = 0
        self._last_frame_time = 0.0

        # Single webcam thread — always running
        self._webcam_thread: Optional[threading.Thread] = None
        self._webcam_stop = threading.Event()
        self._broadcast_active = threading.Event()

        # WebSocket receiver
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_stop: Optional[threading.Event] = None

        # Virtual camera output
        self._vcam_thread: Optional[threading.Thread] = None
        self._vcam_stop: Optional[threading.Event] = None
        self._vcam_queue: queue.Queue = queue.Queue(maxsize=2)

        self._status_polling = False
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start(2000)

        # Single timer drives all frame updates on the main thread (~30fps)
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._poll_frames)
        self._frame_timer.start(33)

        self._start_webcam(0)

    # ── Properties ────────────────────────────────────────────────────

    @Property(int, notify=webcamVersionChanged)
    def webcamVersion(self) -> int:
        return self._webcam_version

    @Property(int, notify=liveVersionChanged)
    def liveVersion(self) -> int:
        return self._live_version

    @Property(str, notify=statusMessageChanged)
    def statusMessage(self) -> str:
        return self._status_message

    @Property(bool, notify=connectedChanged)
    def connected(self) -> bool:
        return self._connected

    @Property(str, notify=connectionLabelChanged)
    def connectionLabel(self) -> str:
        return self._connection_label

    @Property(bool, notify=embeddingPendingChanged)
    def embeddingPending(self) -> bool:
        return self._embedding_pending

    @Property(bool, notify=pipelineRunningChanged)
    def pipelineRunning(self) -> bool:
        return self._pipeline_running

    @Property(bool, notify=virtualCamActiveChanged)
    def virtualCamActive(self) -> bool:
        return self._virtual_cam_active

    @Property(bool, notify=sourceSetChanged)
    def sourceSet(self) -> bool:
        return self._source_set

    @Property(str, notify=sourceThumbnailChanged)
    def sourceThumbnail(self) -> str:
        return self._source_thumbnail

    @Property(str, notify=sourceLabelChanged)
    def sourceLabel(self) -> str:
        return self._source_label

    # ── Slots ─────────────────────────────────────────────────────────

    @Slot()
    def startPipeline(self) -> None:
        if self._pipeline_running or self._embedding_pending:
            return
        if not self._source_set:
            self._set_status('select a face image first')
            return
        result = self._client.status()
        if 'error' in result:
            self._set_status(f'cannot reach server — {result["error"]}')
            return
        self._client.set_quality(self._quality)
        self._client.set_input_url(
            f'tcp://0.0.0.0:{UDP_INGEST_PORT}?listen'
        )
        self._client.start_stream()
        self._broadcast_active.set()
        self._last_frame_time = time.time()
        self._start_ws_receiver()
        self._set_pipeline_running(True)
        self._set_status('pipeline connected · processing')

    @Slot()
    def stopPipeline(self) -> None:
        if self._virtual_cam_active:
            self._stop_vcam()
            self._set_virtual_cam_active(False)
        self._broadcast_active.clear()
        self._stop_ws_receiver()
        self._set_pipeline_running(False)
        self._client.stop_stream()
        self._set_status('stopped')
        self._live_version += 1
        self.liveVersionChanged.emit(self._live_version)

    @Slot()
    def toggleVirtualCam(self) -> None:
        if not self._virtual_cam_active:
            self._start_vcam()
        else:
            self._stop_vcam()
            self._set_virtual_cam_active(False)
            self._set_status('pipeline connected · processing')

    @Slot()
    def selectFaceImages(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            'Select face image(s)',
            '',
            'Images (*.jpg *.jpeg *.png *.webp)',
        )
        valid: List[str] = [p for p in paths if is_image(p)]
        if not valid:
            return
        self._source_thumbnail = valid[0].replace('\\', '/')
        if len(valid) == 1:
            self._client.set_source(valid[0])
            self._source_label = self._source_thumbnail.split('/')[-1]
            self._set_source_set(True)
            self._set_status(f'face set: {self._source_label}')
        else:
            self._source_label = f'{len(valid)} faces · averaged'
            self._set_source_set(True)
            self._set_embedding_pending(True)
            self._set_status(f'creating embedding from {len(valid)} images...')
            self._client.create_embedding(valid)

    @Slot()
    def resetSource(self) -> None:
        if self._pipeline_running:
            self.stopPipeline()
        self._source_thumbnail = ''
        self._source_label = ''
        self._set_source_set(False)
        self._client.cleanup_session()
        self._set_status('select a face source')

    @Slot(str)
    def setWebcamIndex(self, value: str) -> None:
        index = int(value) if value.strip().isdigit() else 0
        if index != self._webcam_index:
            self._webcam_index = index
            self._start_webcam(index)

    @Slot(str)
    def setQuality(self, preset: str) -> None:
        self._quality = preset

    @Slot(str)
    def setPlatform(self, platform: str) -> None:
        self._vcam_platform = platform

    @Slot()
    def cleanup(self) -> None:
        self._frame_timer.stop()
        self._status_timer.stop()
        self._stop_vcam()
        self._stop_ws_receiver()
        self._broadcast_active.clear()
        self._webcam_stop.set()
        if self._webcam_thread is not None:
            self._webcam_thread.join(timeout=3)
        self._client.stop_stream()
        self._client.shutdown()

    # ── Internal ──────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._status_message = msg
        self.statusMessageChanged.emit(msg)

    def _set_pipeline_running(self, value: bool) -> None:
        if self._pipeline_running != value:
            self._pipeline_running = value
            self.pipelineRunningChanged.emit(value)

    def _set_virtual_cam_active(self, value: bool) -> None:
        if self._virtual_cam_active != value:
            self._virtual_cam_active = value
            self.virtualCamActiveChanged.emit(value)

    def _set_source_set(self, value: bool) -> None:
        self._source_set = value
        self.sourceSetChanged.emit(value)
        self.sourceThumbnailChanged.emit(self._source_thumbnail)
        self.sourceLabelChanged.emit(self._source_label)

    def _set_embedding_pending(self, value: bool) -> None:
        if self._embedding_pending != value:
            self._embedding_pending = value
            self.embeddingPendingChanged.emit(value)

    def _poll_frames(self) -> None:
        if webcam_buffer.is_dirty():
            webcam_buffer.promote()
            self._webcam_version += 1
            self.webcamVersionChanged.emit(self._webcam_version)
        if live_buffer.is_dirty():
            live_buffer.promote()
            self._live_version += 1
            self.liveVersionChanged.emit(self._live_version)

    def _poll_status(self) -> None:
        if self._status_polling:
            return
        self._status_polling = True
        threading.Thread(target=self._fetch_status_bg, daemon=True).start()

    def _fetch_status_bg(self) -> None:
        result = self._client.status()
        label = f'{self._client.host}:{self._client.port}'
        if 'error' in result:
            print(f'[DESKTOP.BRIDGE] Status poll failed: {result["error"]}')
        self._apply_status(result, label)

    def _apply_status(self, result: Dict, label: str) -> None:
        self._status_polling = False
        if 'error' in result:
            if self._connected:
                self._connected = False
                self.connectedChanged.emit(False)
        else:
            if not self._connected:
                self._connected = True
                self.connectedChanged.emit(True)
            if self._embedding_pending:
                if result.get('embedding_ready'):
                    self._set_embedding_pending(False)
                    self._set_status('embedding ready')
                elif 'no face detected' in result.get('status_message', ''):
                    self._set_embedding_pending(False)
                    self._set_status('no face detected in selected images')
            elif not self._pipeline_running:
                msg = result.get('status_message', '')
                if msg:
                    self._set_status(msg)
        if self._connection_label != label:
            self._connection_label = label
            self.connectionLabelChanged.emit(label)

    # ── Webcam thread (preview + optional broadcast) ───────────────────

    def _start_webcam(self, webcam_index: int) -> None:
        self._webcam_stop.set()
        if self._webcam_thread is not None:
            self._webcam_thread.join(timeout=3)
        self._webcam_stop.clear()

        self._webcam_thread = threading.Thread(
            target=self._run_webcam,
            args=(webcam_index,),
            daemon=True,
        )
        self._webcam_thread.start()

    def _run_webcam(self, webcam_index: int) -> None:
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        proc: Optional[subprocess.Popen] = None

        while not self._webcam_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            webcam_buffer.update_from_numpy(frame)

            if self._broadcast_active.is_set():
                if proc is None:
                    proc = self._open_broadcast_ffmpeg(width, height, fps)
                if proc is not None and proc.stdin is not None:
                    try:
                        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                    except (BrokenPipeError, OSError):
                        proc = None
            else:
                if proc is not None:
                    try:
                        if proc.stdin:
                            proc.stdin.close()
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
                    proc = None

        cap.release()
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    def _open_broadcast_ffmpeg(self, width: int, height: int, fps: int) -> Optional[subprocess.Popen]:
        cmd = [
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
            f'tcp://{self._client.host}:{UDP_INGEST_PORT}',
        ]
        try:
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f'[DESKTOP.BRIDGE] ffmpeg broadcast failed to start: {e}')
            return None

    # ── WebSocket receiver ────────────────────────────────────────────

    def _start_ws_receiver(self) -> None:
        self._stop_ws_receiver()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_ws_receiver,
            args=(stop_event,),
            daemon=True,
        )
        self._ws_thread = thread
        self._ws_stop = stop_event
        thread.start()

    def _stop_ws_receiver(self) -> None:
        if self._ws_stop is not None:
            self._ws_stop.set()
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=3)
        self._ws_thread = None
        self._ws_stop = None

    def _run_ws_receiver(self, stop_event: threading.Event) -> None:
        from websockets.sync.client import connect
        ws_url = f'ws://{self._client.host}:{self._client.port + 1}'
        while not stop_event.is_set():
            try:
                with connect(ws_url) as ws:
                    self._last_frame_time = time.time()
                    while not stop_event.is_set():
                        try:
                            data = ws.recv(timeout=2.0)
                            if isinstance(data, bytes):
                                live_buffer.update_from_bytes(data)
                                self._last_frame_time = time.time()
                                if self._virtual_cam_active:
                                    self._push_to_vcam(data)
                        except TimeoutError:
                            if time.time() - self._last_frame_time > 3.0:
                                self._set_status('no signal from server')
            except Exception:
                if not stop_event.is_set():
                    time.sleep(1.0)

    # ── Virtual camera output ─────────────────────────────────────────

    def _start_vcam(self) -> None:
        self._stop_vcam()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_vcam,
            args=(stop_event,),
            daemon=True,
        )
        self._vcam_stop = stop_event
        self._vcam_thread = thread
        thread.start()

    def _stop_vcam(self) -> None:
        if self._vcam_stop is not None:
            self._vcam_stop.set()
        if self._vcam_thread is not None:
            self._vcam_thread.join(timeout=3)
        self._vcam_thread = None
        self._vcam_stop = None

    def _run_vcam(self, stop_event: threading.Event) -> None:
        try:
            import pyvirtualcam
        except ImportError:
            self._set_status('pyvirtualcam not installed — run: pip install pyvirtualcam')
            return

        import numpy as np

        kwargs: Dict[str, Any] = {'width': 960, 'height': 540, 'fps': 30, 'fmt': pyvirtualcam.PixelFormat.BGR}
        if self._vcam_platform:
            kwargs['backend'] = self._vcam_platform

        try:
            with pyvirtualcam.Camera(**kwargs) as cam:
                self._set_virtual_cam_active(True)
                self._set_status(f'virtual camera active · {cam.device}')
                while not stop_event.is_set():
                    try:
                        frame: np.ndarray = self._vcam_queue.get(timeout=0.1)
                        cam.send(frame)
                        cam.sleep_until_next_frame()
                    except queue.Empty:
                        continue
        except Exception as e:
            self._set_status(f'virtual camera error: {e}')
        finally:
            self._set_virtual_cam_active(False)

    def _push_to_vcam(self, jpeg_bytes: bytes) -> None:
        import cv2
        import numpy as np
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        if self._vcam_queue.full():
            try:
                self._vcam_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._vcam_queue.put_nowait(frame)
        except queue.Full:
            pass
