import gc
import os
import queue
import time
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Property, QTimer, Qt
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtQuick import QQuickPaintedItem

from pipeline.io.ffmpeg import is_image
from desktop.controller import PipelineClient

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
    detectionStatusChanged = Signal(str)

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
        self._detection_status = ''
        self._webcam_version = 0
        self._live_version = 0
        self._quality = 'optimal'
        self._vcam_platform = 'obs'
        self._webcam_index = 0
        self._last_frame_time = 0.0

        # Single webcam thread — always running
        self._webcam_thread: Optional[threading.Thread] = None
        self._webcam_stop = threading.Event()
        # Set when pipeline is running — webcam thread sends frames via WebSocket
        self._ws_push_active = threading.Event()

        # Virtual camera output
        self._vcam_thread: Optional[threading.Thread] = None
        self._vcam_stop: Optional[threading.Event] = None
        self._vcam_queue: queue.Queue = queue.Queue(maxsize=2)

        # Wire up WebSocket push callbacks from the client
        self._client.on_frame = self._on_ws_frame
        self._client.on_event = self._on_ws_event
        self._client.on_connected = self._on_ws_connected

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

    @Property(str, notify=detectionStatusChanged)
    def detectionStatus(self) -> str:
        return self._detection_status

    # ── Slots ─────────────────────────────────────────────────────────

    @Slot()
    def startPipeline(self) -> None:
        if self._pipeline_running or self._embedding_pending:
            return
        if not self._source_set:
            self._set_status('select a face image first')
            return
        if not self._connected:
            self._set_status('cannot reach server — not connected')
            return
        self._client.set_quality(self._quality)
        self._client.start_stream()
        self._ws_push_active.set()
        self._last_frame_time = time.time()
        self._set_pipeline_running(True)
        self._set_status('pipeline connected · processing')

    @Slot()
    def stopPipeline(self) -> None:
        if self._virtual_cam_active:
            self._stop_vcam()
            self._set_virtual_cam_active(False)
        self._ws_push_active.clear()
        self._client.stop_stream()
        self._set_status('stopping...')
        # _pipeline_running stays True until PIPELINE_STOPPED event arrives —
        # prevents the user from clicking Start before the pipeline thread has
        # fully exited, which would cause start_stream to be rejected silently.

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
        multi = len(valid) > 1
        self._source_label = (
            f'{len(valid)} faces · averaged' if multi
            else self._source_thumbnail.split('/')[-1]
        )
        self._set_source_set(True)
        if multi:
            self._set_embedding_pending(True)
            self._set_status(f'uploading {len(valid)} images...')
        else:
            self._set_status(f'uploading face...')

        def _do_upload(file_paths: List[str]) -> None:
            import base64
            images = []
            for fp in file_paths:
                try:
                    with open(fp, 'rb') as fh:
                        data = base64.b64encode(fh.read()).decode('ascii')
                    images.append({'name': os.path.basename(fp), 'data': data})
                except Exception as e:
                    self._set_embedding_pending(False)
                    self._set_status(f'upload error: {e}')
                    return

            result = self._client.upload_source(images)
            if result.get('success', False):
                self._set_embedding_pending(False)
                self._set_status(
                    'embedding ready' if multi else f'face set: {self._source_label}'
                )
            else:
                error = result.get('error', 'upload failed')
                self._set_embedding_pending(False)
                self._set_status(f'upload error: {error}')

        threading.Thread(target=_do_upload, args=(valid,), daemon=True).start()

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
        self._stop_vcam()
        self._ws_push_active.clear()
        self._webcam_stop.set()
        if self._webcam_thread is not None:
            self._webcam_thread.join(timeout=3)
        self._client.stop_stream()
        self._client.shutdown()
        self._client.close()

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

    def _set_detection_status(self, msg: str) -> None:
        if self._detection_status != msg:
            self._detection_status = msg
            self.detectionStatusChanged.emit(msg)

    def _poll_frames(self) -> None:
        if webcam_buffer.is_dirty():
            webcam_buffer.promote()
            self._webcam_version += 1
            self.webcamVersionChanged.emit(self._webcam_version)
        if live_buffer.is_dirty():
            live_buffer.promote()
            self._live_version += 1
            self.liveVersionChanged.emit(self._live_version)

    # ── WebSocket push callbacks (called from background thread) ──────────────

    def _on_ws_frame(self, jpeg_bytes: bytes) -> None:
        """Called by PipelineClient when a JPEG frame arrives."""
        live_buffer.update_from_bytes(jpeg_bytes)
        self._last_frame_time = time.time()
        if self._virtual_cam_active:
            self._push_to_vcam(jpeg_bytes)

    def _on_ws_event(self, data: Dict) -> None:
        """Called by PipelineClient when a JSON event arrives."""
        event = data.get('event', '')
        if event == 'STATUS_CHANGED':
            message = data.get('message', '')
            scope = data.get('scope', '')
            level = data.get('level', 'info')
            # Always update the detection badge regardless of pipeline state
            if scope == 'DETECTION':
                if level == 'warning':
                    self._set_detection_status('no face detected')
                else:
                    self._set_detection_status('')
            # Show general status messages only when the pipeline is not running
            # (avoids drowning the UI in per-frame debug messages during a run)
            if message and not self._pipeline_running:
                self._set_status(message)
        elif event == 'PIPELINE_STARTED':
            self._set_pipeline_running(True)
        elif event == 'PIPELINE_STOPPED':
            self._set_pipeline_running(False)
            self._set_detection_status('')
            self._set_status('stopped')

    def _on_ws_connected(self, connected: bool) -> None:
        """Called by PipelineClient when connection status changes."""
        if self._connected != connected:
            self._connected = connected
            self.connectedChanged.emit(connected)
            label = self._client._ws_url
            if self._connection_label != label:
                self._connection_label = label
                self.connectionLabelChanged.emit(label)
            if not connected:
                self._set_status('disconnected — reconnecting...')

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

        while not self._webcam_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            webcam_buffer.update_from_numpy(frame)

            if self._ws_push_active.is_set():
                # Encode as JPEG and send to pipeline via WebSocket binary.
                # Quality 75 balances upload bandwidth vs. swap quality.
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                self._client.send_frame(jpeg.tobytes())

        cap.release()

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
