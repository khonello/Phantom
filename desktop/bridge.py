import base64
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
from pipeline.api.schema import MAX_PHOTO_BYTES, MAX_PHOTO_TARGETS, PRESETS
from desktop.controller import PipelineClient
from desktop.audio import AudioCapture, AudioPlayback, JitterBuffer
from desktop.voice import VoiceTransformer

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
    enhanceActiveChanged = Signal(bool)
    colorCorrectionActiveChanged = Signal(bool)
    preprocessingActiveChanged = Signal(bool)
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
    mediaTabChanged = Signal(str)
    templatesChanged = Signal()
    selectedTemplateChanged = Signal(str)
    photoTargetsChanged = Signal()
    photoResultsChanged = Signal()
    autoStopWarning = Signal(int)  # minutes remaining

    def __init__(self, client: PipelineClient) -> None:
        super().__init__()
        self._client = client
        self._source_set = False
        self._source_thumbnail: str = ''
        self._source_label: str = ''
        self._pipeline_running = False
        self._awaiting_first_frame = False
        self._virtual_cam_active = False
        self._enhance_active = True
        self._color_correction_active = True
        self._preprocessing_active = False
        self._embedding_pending = False
        self._connected = False
        self._connection_label = 'connecting...'
        self._status_message = 'idle'
        self._detection_status = ''
        self._loading_message = ''
        # Two levels of navigation. The media tab is what kind of thing is
        # being worked on; the mode is which job within it. LIVE and batch
        # video are one family because they share the video pipeline and the
        # compositor — a still is a different kind of job, not a third peer.
        self._media_tab: str = 'video'   # 'video' | 'image'
        self._current_mode: str = 'realtime'  # 'realtime' | 'video' | 'image'
        self._target_set: bool = False
        self._target_label: str = ''
        self._target_path: str = ''
        self._target_thumbnail: str = ''
        self._output_path: str = ''
        self._batch_running: bool = False
        self._batch_complete: bool = False
        self._batch_error: str = ''
        # Photo mode: up to MAX_PHOTO_TARGETS chosen images, each with its own
        # outcome. A single bool and a single output path cannot describe a job
        # where two of four succeeded, so photo mode keeps its own list.
        self._photo_targets: List[str] = []
        self._photo_results: List[Dict[str, Any]] = []
        # The subset of _photo_targets that reached the pipeline. Results come
        # back one per *uploaded* photo, so a photo dropped during encoding
        # would shift every result after it onto the wrong original.
        self._photo_uploaded: List[str] = []
        # Bundled templates, fetched from the pipeline on entering the tab.
        self._templates: List[Dict[str, Any]] = []
        self._selected_template: str = ''
        self._templates_loading = False
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

        # Voice transformer (CPU-based pitch/formant shifting)
        self._voice_transformer = VoiceTransformer()

        # Audio capture (local mic, never sent to GPU)
        self._audio_capture = AudioCapture()
        self._audio_capture.set_voice_transformer(self._voice_transformer)

        # Jitter buffer: holds processed frames until their playout time
        self._jitter_buffer = JitterBuffer()

        # Audio playback: reads from capture ring buffer at the jitter
        # buffer's target_delay offset so audio stays in sync with video
        self._audio_playback = AudioPlayback(
            self._audio_capture.ring_buffer,
            self._jitter_buffer,
            audio_capture=self._audio_capture,
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

    @Property(bool, notify=enhanceActiveChanged)
    def enhanceActive(self) -> bool:
        return self._enhance_active

    @Property(bool, notify=colorCorrectionActiveChanged)
    def colorCorrectionActive(self) -> bool:
        return self._color_correction_active

    @Property(bool, notify=preprocessingActiveChanged)
    def preprocessingActive(self) -> bool:
        return self._preprocessing_active

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
        # Show overlay immediately so the user sees feedback before the
        # round-trip WebSocket commands below (set_quality, set_enhance,
        # start_stream) which each block waiting for a server response.
        self._set_loading_message('Initializing...')
        self._awaiting_first_frame = True
        # Fire config commands without waiting for responses — the server
        # processes them in order before start_stream runs, and none of
        # these can fail in a way that should block startup.
        self._client._fire('set_quality', preset=self._quality)
        self._client._fire('set_enhance', value=self._enhance_active)
        self._client._fire('set_color_correction', value=self._color_correction_active)
        self._client._fire('set_preprocessing', value=self._preprocessing_active)
        result = self._client.start_stream()
        if not result.get('success', True):
            self._set_loading_message('')
            self._awaiting_first_frame = False
            self._set_status(f'start failed: {result.get("error", "unknown error")}')
            return
        # If we rejoined an existing pipeline, sync UI state from the server
        # so source thumbnail, quality, enhance, etc. reflect reality.
        rejoined = result.get('data', {}).get('rejoined', False)
        if rejoined:
            self._restore_state_from_server()

        self._ws_push_active.set()
        self._jitter_buffer.clear()
        self._audio_capture.start()
        self._audio_playback.start()
        self._last_frame_time = time.time()
        self._set_pipeline_running(True)
        self._set_status('pipeline connected · processing')

    @Slot()
    def stopPipeline(self) -> None:
        if self._virtual_cam_active:
            self._stop_vcam()
            self._set_virtual_cam_active(False)
        self._awaiting_first_frame = False
        self._ws_push_active.clear()
        self._audio_playback.stop()
        self._audio_capture.stop()
        self._jitter_buffer.clear()
        self._client.stop_stream()
        self._set_status('stopping...')
        # _pipeline_running stays True until PIPELINE_STOPPED event arrives —
        # prevents the user from clicking Start before the pipeline thread has
        # fully exited, which would cause start_stream to be rejected silently.

    def _restore_state_from_server(self) -> None:
        """Sync desktop UI with pipeline state after rejoining a running session."""
        state = self._client.get_state()
        if not state.get('success', False):
            return
        data = state.get('data', {})

        # Restore quality & enhance to match server
        self._quality = data.get('quality', self._quality)

        enhance = data.get('enhance', self._enhance_active)
        if enhance != self._enhance_active:
            self._enhance_active = enhance
            self.enhanceActiveChanged.emit(enhance)

        # Restore source indicator — we can't recover the local thumbnail
        # but we can show that a source is loaded on the server.
        source_loaded = data.get('source_loaded', False)
        if source_loaded and not self._source_set:
            source_path = data.get('source_path', '')
            paths = data.get('source_paths', [])
            if paths:
                self._source_label = f'{len(paths)} faces · averaged'
            elif source_path:
                self._source_label = os.path.basename(source_path)
            else:
                self._source_label = 'source (server)'
            # Thumbnail file lives on the server — use empty string so QML
            # shows the label only (no broken image path).
            self._source_thumbnail = ''
            self._set_source_set(True)

    @Slot()
    def toggleVirtualCam(self) -> None:
        if not self._virtual_cam_active:
            self._start_vcam()
        else:
            self._stop_vcam()
            self._set_virtual_cam_active(False)
            self._set_status('pipeline connected · processing')

    @Slot()
    def toggleEnhance(self) -> None:
        new_value = not self._enhance_active
        self._set_enhance_active(new_value)
        if self._connected:
            self._client.set_enhance(new_value)

    # Colour correction and preprocessing no longer have header toggles: the
    # first is correctness rather than preference (off produces a colour step
    # at the swap boundary), and the second defaults off and makes the frame
    # stop looking like the operator's real camera. These slots stay so the
    # capability survives for `set_realism`, the CLI and state sync.
    @Slot()
    def toggleColorCorrection(self) -> None:
        new_value = not self._color_correction_active
        self._set_color_correction_active(new_value)
        if self._connected:
            self._client.set_color_correction(new_value)

    @Slot()
    def togglePreprocessing(self) -> None:
        new_value = not self._preprocessing_active
        self._set_preprocessing_active(new_value)
        if self._connected:
            self._client.set_preprocessing(new_value)

    @Slot()
    def keepAlive(self) -> None:
        """Reset the auto-stop timer on the pipeline pod."""
        if self._connected:
            self._client.keep_alive()

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
            self._set_status('uploading face...')

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
            self._set_embedding_pending(False)

            if not result.get('success', False):
                # The error already names each refused image and why — the
                # handler builds it that way precisely so it can be shown.
                self._set_source_set(False)
                self._set_status(f'upload error: {result.get("error", "upload failed")}')
                return

            self._report_upload(result, len(images), multi)

        threading.Thread(target=_do_upload, args=(valid,), daemon=True).start()

    def _report_upload(self, result: Dict[str, Any], uploaded: int, multi: bool) -> None:
        """
        Report the outcome of a source upload, including partial rejection.

        A source built from 1 of 3 photos still succeeds, and saying only
        "embedding ready" hides the fact that two were thrown out — while the
        label would go on claiming all three were averaged. Someone choosing
        photos needs to know which one to replace, so the count is corrected and
        the first reason is named.

        Args:
            result: The upload_source response payload
            uploaded: How many images were sent
            multi: Whether this was a multi-image (averaged) upload
        """
        data = result.get('data') or {}
        rejected = data.get('rejected') or []
        accepted_paths = data.get('accepted') or data.get('paths') or []
        accepted = int(data.get('count', uploaded))

        if not rejected:
            self._set_status('embedding ready' if multi else f'face set: {self._source_label}')
            return

        # Correct the label. It was written before the server had an opinion, so
        # it still claims every uploaded photo went into the average.
        if accepted > 1:
            self._source_label = f'{accepted} faces · averaged'
        elif accepted_paths:
            self._source_label = os.path.basename(str(accepted_paths[0]))
        self.sourceLabelChanged.emit(self._source_label)

        first = rejected[0]
        summary = f'{first.get("name", "")}: {first.get("message", first.get("reason", ""))}'
        if len(rejected) > 1:
            summary += f' (+{len(rejected) - 1} more refused)'
        self._set_status(f'{accepted} of {uploaded} accepted — {summary}')

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
    def setVoiceTemplate(self, template: str) -> None:
        """Set the voice transformation preset (none/female/male/child/deep)."""
        self._voice_transformer.set_preset(template if template != 'none' else None)

    # Modes belonging to each media tab, first entry being that tab's default.
    _TAB_MODES = {
        'video': ('realtime', 'video'),
        'image': ('image', 'template'),
    }

    @Property(str, notify=mediaTabChanged)
    def mediaTab(self) -> str:
        """Which top-level tab is selected: 'video' or 'image'."""
        return self._media_tab

    @Slot(str)
    def setMediaTab(self, tab: str) -> None:
        """
        Switch the top-level tab, landing on that family's default mode.

        Switching tabs is switching what is being made, so it moves the mode
        with it rather than leaving a video job selected under the image tab.
        """
        if tab not in self._TAB_MODES or tab == self._media_tab:
            return
        self._media_tab = tab
        self.mediaTabChanged.emit(tab)
        self.setMode(self._TAB_MODES[tab][0])

    @Slot(str)
    def setMode(self, mode: str) -> None:
        """Switch between realtime, video, and image modes."""
        if mode not in ('realtime', 'video', 'image', 'template') or mode == self._current_mode:
            return
        if self._pipeline_running:
            self.stopPipeline()
        if self._batch_running:
            self._stop_batch_internal()
        self._current_mode = mode
        self.currentModeChanged.emit(mode)

        # A mode set directly — on reconnect, or from a shortcut — must not
        # leave the tab pointing at the other family.
        for tab, modes in self._TAB_MODES.items():
            if mode in modes and tab != self._media_tab:
                self._media_tab = tab
                self.mediaTabChanged.emit(tab)

        self._reset_batch_state()

        # The library lives on the pipeline, so it cannot be read until there
        # is one to ask. Fetched on entering the tab rather than at startup:
        # a user who never opens it should not pay for the transfer.
        if mode == 'template' and not self._templates:
            self.loadTemplates()

    # -- Photo mode ------------------------------------------------
    #
    # Several image targets, each swapped on its own, failures skipped. The
    # targets are uploaded rather than passed by path: `set_target` resolves
    # against the *pipeline's* filesystem, which on a pod is a different
    # machine, so a chosen file would not exist there. Photos are small enough
    # to carry inline; video is not, which is why this is image-only.

    # Long side below which downscaling stops and the photo is refused instead.
    # Losing detail defeats the point of a photo swap, so the transfer budget
    # gives way before the image does.
    _MIN_PHOTO_LONG_SIDE = 1600

    @Property(list, notify=photoTargetsChanged)
    def photoTargets(self) -> List[str]:
        """Chosen target photos, as local file paths."""
        return list(self._photo_targets)

    @Property(list, notify=photoResultsChanged)
    def photoResults(self) -> List[Dict[str, Any]]:
        """One entry per target: name, ok, reason, and where it was saved."""
        return list(self._photo_results)

    @Property(int, constant=True)
    def maxPhotoTargets(self) -> int:
        """How many photos one job accepts."""
        return MAX_PHOTO_TARGETS

    # -- Templates -------------------------------------------------
    #
    # A template is a target we ship: the same swap as an uploaded photo, but
    # against a scene chosen and verified in advance. The source face is
    # whatever was uploaded; only the picture it goes into differs.

    @Property(list, notify=templatesChanged)
    def templates(self) -> List[Dict[str, Any]]:
        """Available templates: id, name, and a local thumbnail path."""
        return list(self._templates)

    @Property(str, notify=selectedTemplateChanged)
    def selectedTemplate(self) -> str:
        """Id of the chosen template, or empty."""
        return self._selected_template

    def _cache_dir(self, *parts: str) -> str:
        """A local directory for files fetched from the pipeline."""
        from PySide6.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        ) or os.path.expanduser('~/.phantom')
        path = os.path.join(base, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _output_dir(self) -> str:
        """
        Where a template's result is saved locally.

        An uploaded photo has an original to sit beside; a template does not,
        so its output needs a home of its own that the operator can actually
        find. Pictures/Phantom is where someone would look.
        """
        from PySide6.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        ) or os.path.expanduser('~')
        path = os.path.join(base, 'Phantom')
        os.makedirs(path, exist_ok=True)
        return path

    @Slot()
    def loadTemplates(self) -> None:
        """Fetch the library. Runs on a worker — the request blocks."""
        if self._templates_loading:
            return
        if not self._connected:
            self._set_status('cannot load templates - not connected')
            return
        self._templates_loading = True
        self._set_status('loading templates...')
        threading.Thread(target=self._load_templates_worker, daemon=True).start()

    def _load_templates_worker(self) -> None:
        """Fetch, decode thumbnails to a local cache, publish to the gallery."""
        try:
            response = self._client.list_templates()
            if response.get('error'):
                self._set_status('templates unavailable: %s' % response['error'])
                return

            entries = (response.get('data') or {}).get('templates', [])
            cache = self._cache_dir('templates')
            loaded: List[Dict[str, Any]] = []

            for entry in entries:
                record: Dict[str, Any] = {
                    'id': entry.get('id', ''),
                    'name': entry.get('name', entry.get('id', '')),
                    'thumbnail': '',
                }
                data = entry.get('thumbnail')
                if data:
                    thumb = os.path.join(cache, '%s.jpg' % record['id'])
                    try:
                        with open(thumb, 'wb') as fh:
                            fh.write(base64.b64decode(data))
                        record['thumbnail'] = thumb.replace(chr(92), '/')
                    except (OSError, ValueError):
                        # A template without a picture is still selectable.
                        pass
                loaded.append(record)

            self._templates = loaded
            self.templatesChanged.emit()
            self._set_status(
                '%d template(s)' % len(loaded) if loaded else 'no templates available'
            )
        finally:
            self._templates_loading = False

    @Slot(str)
    def selectTemplate(self, template_id: str) -> None:
        """Choose a template as the target for the next job."""
        if self._batch_running or not template_id:
            return

        response = self._client.set_template(template_id)
        if response.get('error') or response.get('success') is False:
            self._set_status('error: %s' % (response.get('error') or 'template refused'))
            return

        self._selected_template = template_id
        self.selectedTemplateChanged.emit(template_id)

        name = next(
            (t['name'] for t in self._templates if t['id'] == template_id),
            template_id,
        )
        thumbnail = next(
            (t['thumbnail'] for t in self._templates if t['id'] == template_id),
            '',
        )

        # The header's "target set" gate is what enables PROCESS, so a template
        # has to satisfy it the same way a chosen file does.
        self._target_set = True
        self._target_label = name
        self._target_thumbnail = thumbnail
        self._photo_results = []
        self._batch_complete = False
        self.targetSetChanged.emit(True)
        self.targetLabelChanged.emit(name)
        self.targetThumbnailChanged.emit(thumbnail)
        self.photoResultsChanged.emit()
        self.batchCompleteChanged.emit(False)

    @Slot()
    def startTemplate(self) -> None:
        """Run the swap against the selected template."""
        if self._batch_running or not self._source_set or not self._selected_template:
            return
        if not self._connected:
            self._set_status('cannot reach server - not connected')
            return

        self._batch_complete = False
        self._batch_error = ''
        self._photo_results = []
        # The target is already on the pipeline, so nothing is uploaded and
        # there is no local original for the result to sit beside.
        self._photo_uploaded = []
        self.batchCompleteChanged.emit(False)
        self.photoResultsChanged.emit()

        result = self._client.start()
        if result.get('success', False) is False and 'error' in result:
            self._set_status('error: %s' % result['error'])
            return

        self._batch_running = True
        self.batchRunningChanged.emit(True)
        self._set_status('processing...')

    @Slot()
    def selectPhotoTargets(self) -> None:
        """Choose up to four target photos."""
        from PySide6.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(
            None, 'Select up to %d target photos' % MAX_PHOTO_TARGETS, '',
            'Images (*.jpg *.jpeg *.png *.webp *.bmp)',
        )
        if not paths:
            return

        chosen = [q.replace(chr(92), '/') for q in paths[:MAX_PHOTO_TARGETS]]
        if len(paths) > MAX_PHOTO_TARGETS:
            self._set_status(
                '%d chosen - using the first %d' % (len(paths), MAX_PHOTO_TARGETS)
            )

        self._photo_targets = chosen
        self._photo_results = []
        self._batch_complete = False
        self.photoTargetsChanged.emit()
        self.photoResultsChanged.emit()
        self.batchCompleteChanged.emit(False)

        # Keep the single-target properties meaningful so the existing header
        # and its "target set" gate keep working in photo mode.
        self._target_set = True
        self._target_path = chosen[0]
        self._target_label = (
            os.path.basename(chosen[0]) if len(chosen) == 1
            else '%d photos' % len(chosen)
        )
        self._target_thumbnail = chosen[0]
        self.targetSetChanged.emit(True)
        self.targetLabelChanged.emit(self._target_label)
        self.targetThumbnailChanged.emit(self._target_thumbnail)

    @Slot(int)
    def removePhotoTarget(self, index: int) -> None:
        """Drop one chosen photo before the job runs."""
        if self._batch_running or not (0 <= index < len(self._photo_targets)):
            return
        self._photo_targets.pop(index)
        self._photo_results = []
        self.photoTargetsChanged.emit()
        self.photoResultsChanged.emit()
        if not self._photo_targets:
            self._reset_batch_state()

    def _encode_photo(self, path: str) -> Tuple[str, str]:
        """
        Read one photo as base64, shrinking it only if it exceeds the cap.

        A photo that already fits is sent byte-for-byte - no decode, no
        re-encode, no generation loss. Only a camera original over the limit is
        touched, and then as gently as it can be: quality first, and dimensions
        only after quality has been spent, in 10% steps rather than one jump to
        a target size.

        Args:
            path: Local path to the photo

        Returns:
            (base64_data, error) - exactly one is non-empty
        """
        try:
            with open(path, 'rb') as fh:
                raw = fh.read()
        except OSError as e:
            return '', 'could not be read: %s' % e

        if len(raw) <= MAX_PHOTO_BYTES:
            return base64.b64encode(raw).decode('ascii'), ''

        image = cv2.imread(path)
        if image is None:
            return '', 'could not be read as an image'

        # Quality first - it costs the least visible detail per byte saved.
        for quality in (95, 92, 88):
            ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and buf.nbytes <= MAX_PHOTO_BYTES:
                return base64.b64encode(buf.tobytes()).decode('ascii'), ''

        # Then dimensions, in small steps, stopping well before the image
        # stops being worth swapping.
        scale = 1.0
        height, width = image.shape[:2]
        while max(height, width) * scale * 0.9 >= self._MIN_PHOTO_LONG_SIDE:
            scale *= 0.9
            resized = cv2.resize(
                image, (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            ok, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok and buf.nbytes <= MAX_PHOTO_BYTES:
                return base64.b64encode(buf.tobytes()).decode('ascii'), ''

        return '', (
            '%.1f MB and cannot be reduced under %d MB without losing too much detail'
            % (len(raw) / (1024 * 1024), MAX_PHOTO_BYTES // (1024 * 1024))
        )

    @Slot()
    def startPhotos(self) -> None:
        """Upload the chosen photos and run the swap over all of them."""
        if self._batch_running or not self._source_set or not self._photo_targets:
            return
        if not self._connected:
            self._set_status('cannot reach server - not connected')
            return

        self._batch_complete = False
        self._batch_error = ''
        self._photo_results = []
        self.batchCompleteChanged.emit(False)
        self.photoResultsChanged.emit()
        self._set_status('preparing photos...')

        images: List[Dict[str, str]] = []
        uploaded: List[str] = []
        skipped: List[str] = []
        for path in self._photo_targets:
            data, error = self._encode_photo(path)
            if error:
                skipped.append('%s: %s' % (os.path.basename(path), error))
                continue
            images.append({'name': os.path.basename(path), 'data': data})
            uploaded.append(path)

        if not images:
            self._set_status('no usable photo - %s' % '; '.join(skipped))
            return
        if skipped:
            self._set_status('skipping %d - %s' % (len(skipped), '; '.join(skipped)))

        self._set_status('uploading %d photo(s)...' % len(images))
        response = self._client.upload_target(images)
        if response.get('error') or response.get('success') is False:
            reason = response.get('error') or 'upload failed'
            self._set_status('error: %s' % reason)
            return

        result = self._client.start()
        if result.get('success', False) is False and 'error' in result:
            self._set_status('error: %s' % result['error'])
            return

        self._photo_uploaded = uploaded
        self._batch_running = True
        self.batchRunningChanged.emit(True)
        self._set_status('processing...')

    def _collect_photo_results(self) -> None:
        """
        Fetch the finished photos and write the swapped ones next to their
        originals.

        The outputs live on the pipeline's filesystem, which is not the
        operator's machine when the pipeline is a pod, so they come back inline
        and are written locally here - beside the photo the operator picked,
        with the `_swapped` suffix `startBatch` already uses for video.
        """
        response = self._client.get_photo_results()
        if response.get('error'):
            self._set_status('error reading results: %s' % response['error'])
            return

        entries = (response.get('data') or {}).get('results', [])
        collected: List[Dict[str, Any]] = []

        for index, entry in enumerate(entries):
            local_source = (
                self._photo_uploaded[index] if index < len(self._photo_uploaded) else ''
            )
            record: Dict[str, Any] = {
                'name': entry.get('target', os.path.basename(local_source)),
                'ok': bool(entry.get('ok')),
                'reason': entry.get('reason', ''),
                'source': local_source,
                'output': '',
            }

            data = entry.get('data')
            if record['ok'] and data:
                if local_source:
                    # An uploaded photo has an original to sit beside.
                    base, ext = os.path.splitext(local_source)
                    out_path = '%s_swapped%s' % (base, ext or '.png')
                else:
                    # A template has none, so its result needs a home the
                    # operator can actually find.
                    name = entry.get('target', 'template')
                    base, ext = os.path.splitext(name)
                    out_path = os.path.join(
                        self._output_dir(), '%s_swapped%s' % (base, ext or '.png')
                    )
                try:
                    with open(out_path, 'wb') as fh:
                        fh.write(base64.b64decode(data))
                    record['output'] = out_path
                except (OSError, ValueError) as e:
                    record['ok'] = False
                    record['reason'] = 'could not be saved locally: %s' % e
            collected.append(record)

        self._photo_results = collected
        self.photoResultsChanged.emit()

        swapped = sum(1 for r in collected if r['ok'])
        self._set_status(
            '%d swapped, %d skipped' % (swapped, len(collected) - swapped)
        )

    @Slot()
    def selectTargetFile(self) -> None:
        """Open a file dialog to select the target video or image."""
        from PySide6.QtWidgets import QFileDialog
        if self._current_mode == 'image':
            # Image mode takes several targets and uploads them, so it has its
            # own selection path. Routed here rather than duplicated in QML so
            # one button keeps working in every mode.
            self.selectPhotoTargets()
            return
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
        if self._current_mode == 'image':
            self.startPhotos()
            return
        if self._current_mode == 'template':
            self.startTemplate()
            return
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
        self._batch_error = ''
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
        self._photo_targets = []
        self._photo_results = []
        self._photo_uploaded = []
        self._selected_template = ''
        self.selectedTemplateChanged.emit('')
        self.photoTargetsChanged.emit()
        self.photoResultsChanged.emit()
        self._target_set = False
        self._target_label = ''
        self._target_path = ''
        self._target_thumbnail = ''
        self._output_path = ''
        self._batch_running = False
        self._batch_complete = False
        self._batch_error = ''
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

    def _set_enhance_active(self, value: bool) -> None:
        if self._enhance_active != value:
            self._enhance_active = value
            self.enhanceActiveChanged.emit(value)

    def _set_color_correction_active(self, value: bool) -> None:
        if self._color_correction_active != value:
            self._color_correction_active = value
            self.colorCorrectionActiveChanged.emit(value)

    def _set_preprocessing_active(self, value: bool) -> None:
        if self._preprocessing_active != value:
            self._preprocessing_active = value
            self.preprocessingActiveChanged.emit(value)

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
                    f'[SYNC] Clock drift: {health["drift_ms"]:.1f}ms '
                    f'(compensated in playback)',
                    file=sys.stderr,
                )
                # Reset drift counters periodically to prevent unbounded
                # accumulation — the playback callback reads drift_ns live
                if health['drift_ms'] > 200.0:
                    self._audio_capture.reset_drift()

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

        # Drop the loading overlay once the first processed frame arrives,
        # guaranteeing the user sees an active swap before the overlay clears.
        if self._awaiting_first_frame:
            self._awaiting_first_frame = False
            self._set_loading_message('')

    def _on_ws_event(self, data: Dict) -> None:
        """Called by PipelineClient when a JSON event arrives."""
        event = data.get('event', '')
        if event == 'STATUS_CHANGED':
            message = data.get('message', '')
            scope = data.get('scope', '')
            level = data.get('level', 'info')
            if scope == 'MODEL_LOAD':
                if message == 'Models ready':
                    # Models loaded; show a final status while we wait for
                    # the first processed frame to confirm the swap works.
                    if self._awaiting_first_frame:
                        self._set_loading_message('Starting stream...')
                    else:
                        self._set_loading_message('')
                else:
                    self._set_loading_message(message)
            elif scope == 'DETECTION':
                if level == 'warning':
                    self._set_detection_status('no face detected')
                else:
                    self._set_detection_status('')
            # Remember a failure so PIPELINE_STOPPED is not read as success.
            # A batch reports completion by stopping, so the stop event alone
            # cannot tell a finished job from a failed one.
            if level == 'error' and self._batch_running:
                self._batch_error = message
            if message and not self._pipeline_running:
                self._set_status(message)
        elif event == 'PHOTO_RESULT':
            # One photo finished. Recorded as it arrives so four tiles resolve
            # one at a time rather than all at the end; the swapped image
            # itself is fetched once the job stops.
            entry = data.get('result') or {}
            self._photo_results.append({
                'name': entry.get('target', ''),
                'ok': bool(entry.get('ok')),
                'reason': entry.get('reason', ''),
                'source': '',
                'output': '',
            })
            self.photoResultsChanged.emit()
        elif event == 'PIPELINE_STARTED':
            # Don't clear the overlay here — keep it up until the first
            # processed frame arrives so the user sees the swap is active.
            if not self._awaiting_first_frame:
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
                # Batch job finished. Complete only if nothing reported an
                # error — a failed job stops exactly like a successful one.
                self._batch_running = False
                self.batchRunningChanged.emit(False)
                if self._batch_error:
                    self._set_status(f'failed: {self._batch_error}')
                    self._batch_error = ''
                else:
                    self._batch_complete = True
                    self.batchCompleteChanged.emit(True)
                    if self._photo_targets:
                        # Fetching the images is a blocking request, and this
                        # runs on the socket's own receive thread — waiting
                        # here would block the very loop that has to deliver
                        # the response. Hence a worker.
                        threading.Thread(
                            target=self._collect_photo_results,
                            daemon=True,
                        ).start()
                    else:
                        self._set_status('done')
        elif event == 'auto_stop_warning':
            minutes = (data.get('data') or {}).get('minutes_remaining', 5)
            self.autoStopWarning.emit(minutes)

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

    def _capture_settings(self) -> Tuple[int, int, int, int]:
        """
        Capture settings for the current quality preset.

        Read from `pipeline.api.schema.PRESETS` rather than a table of our own,
        so the desktop's webcam and the pipeline's own VideoCapture loop cannot
        disagree about what a preset means.

        Returns:
            (width, height, fps, jpeg_quality)
        """
        preset = PRESETS.get(self._quality) or PRESETS['optimal']
        return (
            int(preset['capture_width']),
            int(preset['capture_height']),
            int(preset['capture_fps']),
            int(preset['jpeg_quality']),
        )

    def _run_webcam(self, webcam_index: int) -> None:
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Apply capture settings for the current quality preset
        w, h, fps, jpeg_quality = self._capture_settings()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        while not self._webcam_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            capture_ts = time.perf_counter_ns()
            webcam_buffer.update_from_numpy(frame)

            if self._ws_push_active.is_set():
                _, jpeg = cv2.imencode('.jpg', frame, encode_params)
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

        # The invariant this loop exists to hold:
        #
        #   The virtual camera shows the last augmented frame, or an augmented
        #   frame. Never the raw camera, and never nothing.
        #
        # It previously only called `cam.send()` when a frame arrived, so when
        # frames stopped the device stalled rather than froze — and a call
        # application can report a stalled device as a *disconnected camera*,
        # which is a louder and stranger signal to the other participants than a
        # frozen picture. Holding and re-sending keeps the stream alive at its
        # normal rate; it simply stops moving, which reads as a network hiccup.
        #
        # This covers every way frames can stop, not just guarded ones: the paid
        # hour expiring, the session ending, the worker dying, the pipeline
        # crashing. Expiry is the one most likely to be got wrong because it is
        # the only one that is *expected*, and it must behave exactly like the
        # failures — the operator's real face must not appear on the call the
        # moment their time runs out.
        held: Optional[np.ndarray] = None

        try:
            with pyvirtualcam.Camera(**kwargs) as cam:
                self._set_virtual_cam_active(True)
                self._set_status(f'virtual camera active · {cam.device}')
                while not stop_event.is_set():
                    try:
                        held = self._vcam_queue.get(timeout=0.1)
                    except queue.Empty:
                        # Nothing new. Re-send the previous frame rather than
                        # sending nothing. Until the first frame arrives there is
                        # genuinely nothing to send — and the raw camera is never
                        # the fallback, so the device simply waits.
                        if held is None:
                            continue

                    cam.send(held)
                    cam.sleep_until_next_frame()
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
