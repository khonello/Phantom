import gc
import os
import queue
import struct
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Property, QTimer, Qt
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtQuick import QQuickPaintedItem

from pipeline.io.ffmpeg import is_image
from desktop.controller import PipelineClient
from desktop.audio import AudioCapture, AudioPlayback, JitterBuffer

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
        painter.save()
        painter.translate(iw, 0)
        painter.scale(-1, 1)
        painter.drawPixmap(x, y, sw, sh, pm)
        painter.restore()


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
    loadingMessageChanged = Signal(str)
    currentModeChanged = Signal(str)
    targetSetChanged = Signal(bool)
    targetLabelChanged = Signal(str)
    targetThumbnailChanged = Signal(str)
    outputPathChanged = Signal(str)
    batchRunningChanged = Signal(bool)
    batchCompleteChanged = Signal(bool)

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
        self._loading_message = ''
        self._current_mode: str = 'realtime'  # 'realtime' | 'video' | 'image'
        self._target_set: bool = False
        self._target_label: str = ''
        self._target_path: str = ''
        self._target_thumbnail: str = ''
        self._output_path: str = ''
        self._batch_running: bool = False
        self._batch_complete: bool = False
        self._webcam_version = 0
        self._live_version = 0
        self._quality = 'optimal'
        self._vcam_platform = 'obs'
        self._webcam_index = 0
        self._last_frame_time = 0.0
        self._last_capture_ts: int = 0  # perf_counter_ns from last received frame
        self._health_tick: int = 0  # counter for periodic health checks

        # Single webcam thread — always running
        self._webcam_thread: Optional[threading.Thread] = None
        self._webcam_stop = threading.Event()
        # Set when pipeline is running — webcam thread sends frames via WebSocket
        self._ws_push_active = threading.Event()

        # Audio capture (local mic, never sent to GPU)
        self._audio_capture = AudioCapture()

        # Jitter buffer: holds processed frames until their playout time
        self._jitter_buffer = JitterBuffer()

        # Audio playback: reads from capture ring buffer at the jitter
        # buffer's target_delay offset so audio stays in sync with video
        self._audio_playback = AudioPlayback(
            self._audio_capture.ring_buffer,
            self._jitter_buffer,
        )

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

    @Property(str, notify=loadingMessageChanged)
    def loadingMessage(self) -> str:
        return self._loading_message

    @Property(str, notify=currentModeChanged)
    def currentMode(self) -> str:
        return self._current_mode

    @Property(bool, notify=targetSetChanged)
    def targetSet(self) -> bool:
        return self._target_set

    @Property(str, notify=targetLabelChanged)
    def targetLabel(self) -> str:
        return self._target_label

    @Property(str, notify=targetThumbnailChanged)
    def targetThumbnail(self) -> str:
        return self._target_thumbnail

    @Property(str, notify=outputPathChanged)
    def outputPath(self) -> str:
        return self._output_path

    @Property(bool, notify=batchRunningChanged)
    def batchRunning(self) -> bool:
        return self._batch_running

    @Property(bool, notify=batchCompleteChanged)
    def batchComplete(self) -> bool:
        return self._batch_complete

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
        result = self._client.start_stream()
        if not result.get('success', True):
            self._set_status(f'start failed: {result.get("error", "unknown error")}')
            return
        self._ws_push_active.set()
        self._jitter_buffer.clear()
        self._audio_capture.start()
        self._audio_playback.start()
        self._last_frame_time = time.time()
        self._set_loading_message('Initializing...')
        self._set_pipeline_running(True)
        self._set_status('pipeline connected · processing')

    @Slot()
    def stopPipeline(self) -> None:
        if self._virtual_cam_active:
            self._stop_vcam()
            self._set_virtual_cam_active(False)
        self._ws_push_active.clear()
        self._audio_playback.stop()
        self._audio_capture.stop()
        self._jitter_buffer.clear()
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
        self._start_webcam(self._webcam_index)

    @Slot(str)
    def setPlatform(self, platform: str) -> None:
        self._vcam_platform = platform

    @Slot(str)
    def setMode(self, mode: str) -> None:
        """Switch between realtime, video, and image modes."""
        if mode not in ('realtime', 'video', 'image') or mode == self._current_mode:
            return
        if self._pipeline_running:
            self.stopPipeline()
        if self._batch_running:
            self._stop_batch_internal()
        self._current_mode = mode
        self.currentModeChanged.emit(mode)
        self._reset_batch_state()

    @Slot()
    def selectTargetFile(self) -> None:
        """Open a file dialog to select the target video or image."""
        from PySide6.QtWidgets import QFileDialog
        if self._current_mode == 'video':
            path, _ = QFileDialog.getOpenFileName(
                None, 'Select target video', '',
                'Videos (*.mp4 *.avi *.mov *.mkv *.webm)',
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, 'Select target image', '',
                'Images (*.jpg *.jpeg *.png *.webp *.bmp)',
            )
        if not path:
            return
        self._target_path = path.replace('\\', '/')
        self._target_label = self._target_path.split('/')[-1]
        self._target_thumbnail = (
            self._target_path if self._current_mode == 'image' else ''
        )
        self._target_set = True
        self._batch_complete = False
        self._output_path = ''
        self.targetSetChanged.emit(True)
        self.targetLabelChanged.emit(self._target_label)
        self.targetThumbnailChanged.emit(self._target_thumbnail)
        self.outputPathChanged.emit('')
        self.batchCompleteChanged.emit(False)

    @Slot()
    def selectOutputPath(self) -> None:
        """Open a save dialog to choose the output file path."""
        from PySide6.QtWidgets import QFileDialog
        if self._current_mode == 'video':
            path, _ = QFileDialog.getSaveFileName(
                None, 'Save output video', '', 'Videos (*.mp4)',
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                None, 'Save output image', '', 'Images (*.png *.jpg)',
            )
        if not path:
            return
        self._output_path = path.replace('\\', '/')
        self.outputPathChanged.emit(self._output_path)

    @Slot()
    def startBatch(self) -> None:
        """Start batch face swap processing on the selected target file."""
        if self._batch_running or not self._source_set or not self._target_set:
            return
        if not self._connected:
            self._set_status('cannot reach server — not connected')
            return

        # Auto-generate output path if none selected
        if not self._output_path:
            import os
            base, ext = os.path.splitext(self._target_path)
            self._output_path = base + '_swapped' + ext
            self.outputPathChanged.emit(self._output_path)

        self._batch_complete = False
        self.batchCompleteChanged.emit(False)
        self._set_status('processing...')
        self._client.set_target(self._target_path)
        self._client.set_output(self._output_path)
        result = self._client.start()
        if result.get('success', False) is False and 'error' in result:
            self._set_status(f'error: {result["error"]}')
            return
        self._batch_running = True
        self.batchRunningChanged.emit(True)

    @Slot()
    def stopBatch(self) -> None:
        """Cancel in-progress batch processing."""
        self._stop_batch_internal()

    @Slot()
    def openOutputFolder(self) -> None:
        """Open the folder containing the output file in the system file manager."""
        import os
        import sys as _sys
        import subprocess
        if not self._output_path:
            return
        folder = os.path.dirname(self._output_path)
        try:
            if _sys.platform == 'win32':
                os.startfile(folder)
            elif _sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
        except Exception as e:
            self._set_status(f'could not open folder: {e}')

    def _stop_batch_internal(self) -> None:
        """Internal: stop batch and reset running flag."""
        self._client.stop()
        self._batch_running = False
        self.batchRunningChanged.emit(False)
        self._set_status('stopped')

    def _reset_batch_state(self) -> None:
        """Clear all batch-related state."""
        self._target_set = False
        self._target_label = ''
        self._target_path = ''
        self._target_thumbnail = ''
        self._output_path = ''
        self._batch_running = False
        self._batch_complete = False
        self.targetSetChanged.emit(False)
        self.targetLabelChanged.emit('')
        self.targetThumbnailChanged.emit('')
        self.outputPathChanged.emit('')
        self.batchRunningChanged.emit(False)
        self.batchCompleteChanged.emit(False)

    @Slot()
    def cleanup(self) -> None:
        self._frame_timer.stop()
        self._stop_vcam()
        self._audio_playback.stop()
        self._audio_capture.stop()
        self._jitter_buffer.clear()
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

    def _set_loading_message(self, msg: str) -> None:
        if self._loading_message != msg:
            self._loading_message = msg
            self.loadingMessageChanged.emit(msg)

    def _poll_frames(self) -> None:
        if webcam_buffer.is_dirty():
            webcam_buffer.promote()
            self._webcam_version += 1
            self.webcamVersionChanged.emit(self._webcam_version)

        # Pop the most recent eligible frame from the jitter buffer.
        # If multiple frames are eligible the intermediate ones are dropped
        # so the display stays current.
        eligible = self._jitter_buffer.pop_eligible()
        if eligible is not None:
            _capture_ts, jpeg_bytes = eligible
            live_buffer.update_from_bytes(jpeg_bytes)
            if self._virtual_cam_active:
                self._push_to_vcam(jpeg_bytes)

        if live_buffer.is_dirty():
            live_buffer.promote()
            self._live_version += 1
            self.liveVersionChanged.emit(self._live_version)

        # Periodic health check (~every 2 seconds)
        if self._pipeline_running:
            self._health_tick += 1
            if self._health_tick >= 60:
                self._health_tick = 0
                self._check_av_health()

    def _check_av_health(self) -> None:
        """Periodic health check for audio streams and clock drift.

        Called every ~2 seconds from _poll_frames while the pipeline is running.
        Attempts automatic recovery of failed audio streams and logs sync stats.
        """
        import sys

        # 1. Audio capture health + clock drift
        if self._audio_capture.is_running:
            health = self._audio_capture.check_health()
            if not health['active']:
                print('[SYNC] Audio capture stream died — recovering', file=sys.stderr)
                self._audio_capture.try_recover()
            elif health['drift_warning']:
                print(
                    f'[SYNC] Clock drift warning: audio drifted '
                    f'{health["drift_ms"]:.1f}ms from wall clock',
                    file=sys.stderr,
                )

        # 2. Audio playback health
        if self._audio_playback.is_running:
            try:
                stream = self._audio_playback._stream
                if stream is not None and not stream.active:  # type: ignore[union-attr]
                    print('[SYNC] Audio playback stream died — recovering', file=sys.stderr)
                    self._audio_playback.try_recover()
            except Exception:
                pass

        # 3. Log sync stats
        stats = self._jitter_buffer.sync_stats()
        if stats['rtt_samples'] > 0:
            print(
                f'[SYNC] delay={stats["target_delay_ms"]}ms '
                f'rtt={stats["rtt_mean_ms"]}±{stats["rtt_stddev_ms"]}ms '
                f'buf={stats["buffer_depth"]}',
                file=sys.stderr,
            )

    # ── WebSocket push callbacks (called from background thread) ──────────────

    def _on_ws_frame(self, data: bytes) -> None:
        """Called by PipelineClient when a binary frame arrives.

        Expected format: [8 bytes int64 capture_ts_ns] [N bytes JPEG].
        Falls back gracefully if the header is missing (legacy server).

        Frames are pushed into the jitter buffer rather than displayed
        immediately — the Qt render timer (_poll_frames) pops them at the
        correct playout time.
        """
        if len(data) > self._TS_HEADER_SIZE:
            capture_ts = struct.unpack('<q', data[:self._TS_HEADER_SIZE])[0]
            jpeg_bytes = data[self._TS_HEADER_SIZE:]
        else:
            capture_ts = 0
            jpeg_bytes = data

        self._last_capture_ts = capture_ts
        self._last_frame_time = time.time()
        self._jitter_buffer.push(capture_ts, jpeg_bytes)

    def _on_ws_event(self, data: Dict) -> None:
        """Called by PipelineClient when a JSON event arrives."""
        event = data.get('event', '')
        if event == 'STATUS_CHANGED':
            message = data.get('message', '')
            scope = data.get('scope', '')
            level = data.get('level', 'info')
            if scope == 'MODEL_LOAD':
                # Update loading overlay; clear when models are ready
                self._set_loading_message('' if message == 'Models ready' else message)
            elif scope == 'DETECTION':
                if level == 'warning':
                    self._set_detection_status('no face detected')
                else:
                    self._set_detection_status('')
            # Show general status messages only when the pipeline is not running
            if message and not self._pipeline_running:
                self._set_status(message)
        elif event == 'PIPELINE_STARTED':
            self._set_loading_message('')
            if self._current_mode == 'realtime':
                self._set_pipeline_running(True)
        elif event == 'PIPELINE_STOPPED':
            self._set_loading_message('')
            self._set_detection_status('')
            if self._current_mode == 'realtime':
                self._set_pipeline_running(False)
                self._set_status('stopped')
            else:
                # Batch job finished — mark complete
                self._batch_running = False
                self.batchRunningChanged.emit(False)
                self._batch_complete = True
                self.batchCompleteChanged.emit(True)
                self._set_status('done')

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
            else:
                # GPU reconnected — reset jitter buffer so RTT stats
                # recalibrate from the new connection's latency profile.
                self._jitter_buffer.clear()

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

    # Size of the capture_ts header prepended to binary frames (int64 nanoseconds)
    _TS_HEADER_SIZE = 8

    # Capture settings per quality preset: (width, height, fps, jpeg_quality)
    _QUALITY_CAPTURE: Dict[str, tuple] = {
        'fast':       (480, 270,  15, 60),
        'optimal':    (640, 360,  20, 70),
        'production': (960, 540,  30, 85),
    }

    def _run_webcam(self, webcam_index: int) -> None:
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Apply capture settings for the current quality preset
        w, h, fps, _ = self._QUALITY_CAPTURE.get(self._quality, self._QUALITY_CAPTURE['optimal'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)

        while not self._webcam_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            capture_ts = time.perf_counter_ns()
            webcam_buffer.update_from_numpy(frame)

            if self._ws_push_active.is_set():
                _, _, _, jpeg_quality = self._QUALITY_CAPTURE.get(self._quality, self._QUALITY_CAPTURE['optimal'])
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                header = struct.pack('<q', capture_ts)
                self._client.send_frame(header + jpeg.tobytes())

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
        if frame.shape[0] != 540 or frame.shape[1] != 960:
            frame = cv2.resize(frame, (960, 540))
        if self._vcam_queue.full():
            try:
                self._vcam_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._vcam_queue.put_nowait(frame)
        except queue.Full:
            pass
