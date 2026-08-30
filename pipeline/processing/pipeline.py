"""
Main processing pipeline coordinator for Phantom.

Orchestrates frame processors into a complete pipeline.
Replaces monolithic stream.py:_pipeline_loop() and core.py:start().

Responsibilities:
- Build processor chains (batch vs stream)
- Coordinate enhancement (synchronous on GPU)
- Emit events (FRAME_READY, DETECTION, etc.)
- Listen to config changes and rebuild
- Manage I/O sources and sinks
"""

import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, FrameSwap, PhotoResult
from pipeline.events import (
    BATCH_PROGRESS,
    FRAME_READY,
    PHOTO_RESULT,
    DETECTION,
    PIPELINE_STARTED,
    PIPELINE_STOPPED,
)
from pipeline.logging import emit_status, emit_error, get_logger
from pipeline.io.ffmpeg import (
    clean_temp,
    create_video,
    detect_fps,
    extract_frames,
    get_temp_frame_paths,
    get_temp_output_path,
    has_audio,
    has_image_extension,
    is_image,
    is_video,
    move_temp,
    reset_temp,
    restore_audio,
)

from pipeline.services.face_detection import FaceDetector
from pipeline.services.face_swapping import FaceSwapper
from pipeline.services.enhancement import Enhancer
from pipeline.services.database import FaceDatabase
from pipeline.services.masking import FaceMasker
from pipeline.services.face_tracking import LandmarkStabilizer
from pipeline.services import guards
from pipeline.services import templates
from pipeline.services import execution
from pipeline.services.latency import LatencyBudget

from pipeline.processing.compositor import FaceCompositor
from pipeline.processing.frame_processor import (
    DetectionProcessor,
    SwappingProcessor,
    PreprocessingProcessor,
)


def _timecode(frame_index: int, fps: float) -> str:
    """
    Where in the clip a frame sits, as mm:ss.s.

    A frame number alone is not something an operator can act on — they have to
    open the video and find the offending face to fix it.
    """
    seconds = frame_index / fps if fps > 0 else 0.0
    return f'{int(seconds // 60):02d}:{seconds % 60:04.1f}'


class ProcessingPipeline:
    """
    Main face-swapping processing pipeline.

    Composes services and processors to handle both batch and realtime
    face swapping. Manages:
    - Processor chain construction
    - Frame routing and processing
    - Event emission
    - Config change handling

    Example:
        pipeline = ProcessingPipeline(CONFIG, BUS)
        BUS.on('frame_ready', my_handler)
        pipeline.run_stream()  # or run_batch()
        pipeline.stop()
    """

    # Frames between per-stage timing reports (debug log level only).
    _TIMING_INTERVAL = 30

    # Minimum seconds between batch progress reports.
    _PROGRESS_INTERVAL = 1.0

    def __init__(self, config: FaceSwapConfig, bus: Any) -> None:
        """
        Initialize the processing pipeline.

        Args:
            config: FaceSwapConfig object with all settings
            bus: EventBus for event emission
        """
        self.config = config
        self.bus = bus

        # Services (lazily created)
        self._detector: Optional[FaceDetector] = None
        self._swapper: Optional[FaceSwapper] = None
        self._enhancer: Optional[Enhancer] = None
        self._database: Optional[FaceDatabase] = None
        self._masker: Optional[FaceMasker] = None

        # Processors
        self._detection_proc: Optional[DetectionProcessor] = None
        self._swapping_proc: Optional[SwappingProcessor] = None
        self._preprocessing_proc: Optional[PreprocessingProcessor] = None

        # Compositing
        self._compositor: Optional[FaceCompositor] = None
        self._stabilizer: Optional[LandmarkStabilizer] = None

        # State
        self._running = False
        self._stop_event = threading.Event()
        # Per-photo outcomes of the last batch, read back by the API.
        self._photo_results: List[PhotoResult] = []
        # Why a render stopped itself, as opposed to being stopped. Both end a
        # job through the same `_stop_event`-shaped exit, and the difference
        # matters to the operator: one is a finished decision they made, the
        # other is a defect in the target they need told about.
        self._abort_reason: str = ''

        # Guard state. `_last_good_frame` is what a guarded frame emits — the
        # last frame that was actually swapped, held unchanged. `_guard_reason`
        # is the currently-reported reason, so a guard that holds for seconds is
        # reported once rather than every frame.
        self._last_good_frame: Optional[Frame] = None
        self._guard_reason: str = ''
        self._guard_starved = 0
        self._telemetry = guards.GuardTelemetry()
        self._latency = LatencyBudget()

        # Set by WebSocketAPIServer to enable push mode: desktop sends JPEG
        # frames via WebSocket instead of the pipeline capturing a local device.
        self.frame_queue: Optional['queue.Queue[Any]'] = None

        # Debug frame capture (see config.debug_frames_dir)
        self._debug_queue: Optional['queue.Queue[Any]'] = None
        self._debug_thread: Optional[threading.Thread] = None
        self._debug_written = 0
        self._debug_dropped = 0

        # Listen to config changes
        self.config.on_change(self._on_config_changed)

    def _get_detector(self) -> FaceDetector:
        """Get or create FaceDetector."""
        if self._detector is None:
            self._detector = FaceDetector(self.config)
        return self._detector

    def _get_swapper(self) -> FaceSwapper:
        """Get or create FaceSwapper."""
        if self._swapper is None:
            self._swapper = FaceSwapper(self.config)
        return self._swapper

    def _get_enhancer(self) -> Enhancer:
        """Get or create Enhancer."""
        if self._enhancer is None:
            self._enhancer = Enhancer(self.config)
        return self._enhancer

    def _get_database(self) -> FaceDatabase:
        """Get or create FaceDatabase."""
        if self._database is None:
            self._database = FaceDatabase(self._get_detector(), self.config)
        return self._database

    def _get_masker(self) -> FaceMasker:
        """Get or create FaceMasker."""
        if self._masker is None:
            self._masker = FaceMasker(self.config)
        return self._masker

    def face_boxes(self, path: str) -> List[Dict[str, float]]:
        """
        Every face in an image on disk, as normalised boxes.

        Exists for the upload path, which has to tell the operator how many
        faces a target photo holds before any job runs, so they can name the
        one they meant. It is the same detector the job itself will use — a
        different one could count differently, and a picker that disagrees with
        the guard it exists to answer is worse than no picker.

        Normalised for the same reason `target_face_point` is: these are drawn
        over a scaled preview, and a pixel box is wrong the moment the preview
        is not the photo's own size. Normalising here keeps it beside the
        detector that produced the pixels.

        Args:
            path: Path to an image

        Returns:
            One box per face, or an empty list if the file will not read
        """
        frame = cv2.imread(path)
        if frame is None:
            return []

        height, width = int(frame.shape[0]), int(frame.shape[1])
        if not width or not height:
            return []

        boxes: List[Dict[str, float]] = []
        for detection in self._get_detector().detect(frame):
            bbox = detection.bbox
            boxes.append({
                'x': bbox.x / width,
                'y': bbox.y / height,
                'w': bbox.w / width,
                'h': bbox.h / height,
                'score': float(detection.confidence),
            })
        return boxes

    def _build_processors(self) -> None:
        """Build processor and compositing instances."""
        detector = self._get_detector()
        swapper = self._get_swapper()
        enhancer = self._get_enhancer()
        database = self._get_database()
        masker = self._get_masker()

        # Create fresh processors
        self._detection_proc = DetectionProcessor(self.config, detector)
        self._swapping_proc = SwappingProcessor(self.config, swapper, database)
        self._preprocessing_proc = PreprocessingProcessor(self.config)

        self._compositor = FaceCompositor(self.config, enhancer, masker)
        self._stabilizer = LandmarkStabilizer(
            alpha=self.config.alpha,
            identity_sim=self.config.guard_identity_sim,
        )

    def _reset_temporal_state(self) -> None:
        """
        Drop everything that smooths across frames.

        Required whenever continuity is broken — a new source identity, a
        pipeline restart — otherwise the first frames afterwards blend
        against state belonging to the previous subject.
        """
        if self._stabilizer:
            self._stabilizer.reset()
        if self._compositor:
            self._compositor.reset()

    def _reset_guard_state(self) -> None:
        """
        Drop the held frame as well as the temporal state.

        Separate from `_reset_temporal_state` because a guard must *keep* the
        held frame — that frame is what it emits. This is for the cases where the
        held frame is no longer the right thing to show at all: a new source
        identity, or a new session on the same pipeline.
        """
        self._reset_temporal_state()
        self._last_good_frame = None
        self._guard_reason = ''
        self._guard_starved = 0

    def _on_config_changed(self, field: str, value: Any) -> None:
        """
        Handle configuration changes.

        Args:
            field: Config field name
            value: New value
        """
        # Source path changed -> drop temporal state and load new source. The
        # held frame goes too: it carries the previous identity, and holding it
        # after a source change would show the old face on the next guard.
        if field == 'source_path' or field == 'source_paths':
            self._reset_guard_state()
            if self._swapping_proc:
                sources = self.config.source_paths or (
                    [self.config.source_path] if self.config.source_path else []
                )
                self._swapping_proc.set_source(sources)

        # Landmark smoothing factor or identity floor changed -> rebuild the
        # stabilizer. Both are constructor arguments, and the identity floor is
        # one an operator sweeps live while calibrating it.
        elif field in ('alpha', 'guard_identity_sim'):
            self._stabilizer = LandmarkStabilizer(
                alpha=self.config.alpha,
                identity_sim=self.config.guard_identity_sim,
            )

        # Working resolution changed -> previously smoothed pixels are the
        # wrong shape. The compositor detects that and recovers on its own,
        # but dropping it here avoids a frame of unsmoothed output.
        elif field == 'aligned_size':
            if self._compositor:
                self._compositor.reset()

        # Restoration backend changed -> drop the loaded model so the new one
        # is picked up on the next frame.
        elif field in ('enhancer_model',):
            if self._enhancer:
                self._enhancer.clear()

        # Swap model changed -> drop the session, apply the model's realism
        # profile, and drop temporal state. The profile matters: the appearance
        # knobs are tuned per model, and carrying 128px tuning onto a 256px model
        # would make the better model look worse. Temporal state goes because the
        # smoothed buffer holds output from the previous model.
        elif field == 'swapper_model':
            if self._swapper:
                self._swapper.clear()
            self.config.apply_model_profile(value)
            self._reset_temporal_state()

        # A speed lever changed -> every ONNX session has to be rebuilt, because
        # all four are decided at session construction and none can be changed
        # on a live session. Dropping them here means they reload lazily on the
        # next frame with the new options.
        #
        # This exists so a measurement session does not need a pod restart per
        # lever. Model load is seconds; a cold start is minutes of a paid hour,
        # and six levers measured that way would be most of the hour.
        #
        # Temporal state goes with them: fp16 changes what the models output,
        # and a smoothed buffer holding fp32 pixels would blend the two.
        elif field in ('fp16', 'cuda_graphs', 'cuda_streams', 'trt', 'trt_gpus'):
            self._rebuild_sessions()

        # Restoration crop size changed -> the enhancer's session was built
        # declaring static shapes, and CUDA graph capture records fixed device
        # buffer addresses for the shape it saw. Replaying that against a
        # different crop would restore garbage, so drop the session and let it
        # reload. Temporal state goes too: the aligned buffer holds pixels
        # restored at the previous resolution.
        elif field == 'restore_size':
            if self._enhancer:
                self._enhancer.clear()
            self._reset_temporal_state()

    def _rebuild_sessions(self) -> None:
        """
        Drop every ONNX session so it reloads with the current speed levers.

        Sessions are rebuilt lazily on the next frame that needs one, so this
        returns immediately and the cost lands on that frame. On the live path
        that is a visible hitch of a second or two — acceptable for a
        deliberate A/B, and the reason these are not exposed to a consumer.
        """
        if self._swapper:
            self._swapper.clear()
        if self._enhancer:
            self._enhancer.clear()
        if self._masker:
            self._masker.reset()
        self._reset_temporal_state()
        emit_status(
            'ONNX sessions dropped — rebuilding with the current speed levers '
            'on the next frame.',
            scope='RUNTIME',
        )

    def run_stream(self) -> None:
        """
        Run realtime streaming pipeline (webcam or network stream).

        Main loop:
        1. Capture frame from source
        2. Detect faces
        3. Track across frames
        4. Swap if source available
        5. Enhance asynchronously
        6. Emit events
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        try:
            self._run_stream_impl()
        except Exception as e:
            emit_error(f"Stream pipeline error: {e}", exception=e, scope='PIPELINE')
        finally:
            self._running = False
            self._stop_event.set()
            # Before PIPELINE_STOPPED, so the summary is in the log above the
            # line a reader will scroll to when the session ends.
            self._report_telemetry()
            self.bus.emit(PIPELINE_STOPPED)

    def _warm_up_models(self) -> None:
        """
        Eagerly load ML models into GPU memory before the stream loop starts.

        All of these are lazily initialized by default, meaning the first
        frame that needs them blocks for 10-30s while 500MB+ of ONNX weights
        are loaded into CUDA — and, for the occluder, while it downloads.
        Pre-loading them here in parallel makes the first swap instant and
        cuts startup time roughly in half.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        emit_status('Loading models...', scope='MODEL_LOAD')

        def _load_detector() -> str:
            self._get_detector()._get_analyser()
            return 'detection'

        def _load_swapper() -> str:
            self._get_swapper()._get_swapper()
            return 'swap'

        def _load_masker() -> str:
            # Downloads on first use; failure is non-fatal by design.
            self._get_masker()._get_session()
            return 'occluder'

        def _load_enhancer() -> str:
            # Also downloads on first use when the backend is CodeFormer.
            self._get_enhancer().load()
            return 'restoration'

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_load_detector): 'detection',
                pool.submit(_load_swapper): 'swap',
                pool.submit(_load_masker): 'occluder',
                pool.submit(_load_enhancer): 'restoration',
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                except Exception as e:
                    emit_error(
                        f"{label.title()} model load failed: {e}",
                        exception=e, scope='PIPELINE',
                    )

        emit_status('Models ready', scope='MODEL_LOAD')

        # Everything is loaded, so this is the first moment the question can be
        # answered honestly: is each session actually on the device we asked
        # for? ONNX Runtime does not fail when a provider cannot initialise — it
        # quietly uses CPU, and every model that decides how the output looks is
        # ONNX. Checked here rather than trusted, because the last time it went
        # wrong it was found by reading a Dockerfile, not by anything failing.
        execution.verify(self.config, [
            ('detection', self._detector),
            ('swap', self._swapper),
            ('restoration', self._enhancer),
            ('occluder', self._masker),
        ])

    def _run_stream_impl(self) -> None:
        """Implementation of stream mode."""
        # A report has to describe the stream it is printed for. The budget
        # lives on the pipeline, which outlives any one stream, so without
        # this every measurement is diluted by every measurement before it.
        self._latency.reset()
        self._build_processors()
        self._warm_up_models()
        emit_status('Stream pipeline started', scope='PIPELINE')
        self.bus.emit(PIPELINE_STARTED)

        # Load source faces
        sources = self.config.source_paths or (
            [self.config.source_path] if self.config.source_path else []
        )
        if sources:
            if not self._swapping_proc.set_source(sources):
                emit_error(
                    'No face detected in source image(s) — stream will run '
                    'without face swapping until a valid source is set',
                    scope='PIPELINE',
                )

        # Push mode: desktop sends JPEG frames via WebSocket binary messages.
        # Used when no local VideoCapture source is available (e.g. RunPod).
        if self.frame_queue is not None and not self.config.input_url:
            emit_status('Stream mode: WebSocket push (receiving frames from desktop)', scope='PIPELINE')
            self._stream_loop_push()
        else:
            self._stream_loop_capture()

    def _swap_face(self, frame: Frame, face: Any) -> Optional[Frame]:
        """
        Swap and composite a single face.

        Prefers the aligned form so FaceCompositor can own masking, colour,
        detail and grain. Falls back to InsightFace's own compositing only if
        this build cannot hand back the affine.

        Args:
            frame: Frame to swap in
            face: Fresh (optionally stabilized) face for this frame

        Returns:
            Frame with the face composited in, or None if no swap could be
            produced — the occlusion guard refused it or compositing failed.
            Callers must decide what to show instead; returning `frame` here
            would put the operator's real face on the call.
        """
        result = self._swapping_proc.swap_aligned(frame, face)
        if result is None:
            return self._swapping_proc.swap_pasted(frame, face)

        crop, matrix = result
        return self._compositor.composite(frame, face, crop, matrix)

    def _guard_frame(self, reason: str, detail: str) -> None:
        """
        Drop everything that carries across frames, and report once.

        A guarded frame must not enter either EMA. If it did, whatever the guard
        objected to — a stranger's face, a half-occluded one — would be blended
        into the smoothed history and leak back out over the frames that follow,
        which is precisely the output the guard existed to prevent.

        Reported on transition only. A guard that holds for several seconds is
        one event, not one per frame, and at 20fps the per-frame version would
        push 20 messages a second at every client.

        Args:
            reason: Guard reason code
            detail: Measured value behind it
        """
        self._reset_temporal_state()

        if self._guard_reason != reason:
            self._guard_reason = reason
            emit_status(
                f'Frame guarded — {guards.describe(reason)}'
                + (f' ({detail})' if detail else ''),
                scope='GUARD', level='warning',
            )

    def _clear_guard(self) -> None:
        """Note that frames are being emitted normally again."""
        if self._guard_reason:
            self._guard_reason = ''
            emit_status('Guard cleared — swapping again', scope='GUARD')

    def _process_and_emit(self, frame: Frame, seq: int, capture_ts: int = 0) -> None:
        """Run preprocess -> detect -> swap -> composite -> emit for one frame.

        Detection runs on every frame, so the swap is always warped with
        current landmarks. Temporal continuity comes from EMA on those
        landmarks and on the composited pixels, not from a correlation
        tracker carrying a stale face forward.

        Args:
            frame: Input video frame
            seq: Sequence number
            capture_ts: Capture timestamp in nanoseconds (time.perf_counter_ns)
        """
        started = time.perf_counter()

        # Captured before anything touches it, so the pair is genuinely
        # "what the camera sent" against "what the far end sees".
        debug_input = frame.copy() if self.config.debug_frames_dir else None

        # Preprocessing: normalize lighting, white balance, denoise
        frame = self._preprocessing_proc.process(frame)

        frame = self._detection_proc.process(frame)
        detections = self._detection_proc.latest_detections
        detected = time.perf_counter()

        # Guards read what detection already produced, so this costs a handful
        # of comparisons. `all_detections` rather than `latest_detections`,
        # because outside many_faces the latter has already been trimmed to one
        # and the face *count* is what the multi-face guard is.
        verdict = guards.check_frame(self.config, self._detection_proc.all_detections)
        self._record_telemetry(verdict)

        # In observe mode every guard is still evaluated and recorded, but none
        # of them act. The thresholds were chosen without data, and a session
        # that enforces them cannot measure them: a guarded frame emits a held
        # frame, which is no longer a sample of what the camera was doing.
        if not verdict.ok and not self.config.guard_observe:
            self._guard_frame(verdict.reason, verdict.detail)
            self._emit_guarded(seq, capture_ts, debug_input)
            self._log_timing(seq, started, detected, time.perf_counter())
            return

        if not detections:
            self._stabilizer.mark_missing()
        elif self._swapping_proc.source_face is not None:
            for detection in detections:
                face = detection.face
                # Landmark smoothing needs a stable subject identity; with
                # several faces the per-frame detection order is not stable.
                if not self.config.many_faces:
                    face = self._stabilizer.stabilize(face)

                swapped_frame = self._swap_face(frame, face)
                if swapped_frame is None:
                    # Occlusion guard, or a compositing failure. Either way there
                    # is no swapped frame for this face, and the partially
                    # composited result must not be emitted.
                    self._guard_frame(guards.OCCLUDED, '')
                    self._emit_guarded(seq, capture_ts, debug_input)
                    self._log_timing(seq, started, detected, time.perf_counter())
                    return

                frame = swapped_frame

            self._clear_guard()
            self._last_good_frame = frame
            self.bus.emit(DETECTION, detection=detections[0].to_dict(), seq=seq)

        swapped = time.perf_counter()

        if debug_input is not None:
            self._queue_debug_pair(seq, debug_input, frame)

        self.bus.emit(FRAME_READY, frame=frame, seq=seq, capture_ts=capture_ts)

        self._log_timing(seq, started, detected, swapped)

    def _record_telemetry(self, verdict: 'guards.GuardResult') -> None:
        """
        Record what this frame measured, for threshold calibration.

        Coverage and identity similarity come from the previous frame's work —
        the masker and stabilizer each record what they last computed, so this
        adds no inference. Coverage therefore lags by one frame, which does not
        matter for a distribution.

        Args:
            verdict: What `check_frame` decided for this frame
        """
        coverage = self._masker.last_coverage if self._masker else None
        similarity = self._stabilizer.last_similarity if self._stabilizer else None

        self._telemetry.observing = self.config.guard_observe
        self._telemetry.record(
            self._detection_proc.all_detections,
            verdict,
            coverage=coverage,
            identity_sim=similarity,
        )

    def _report_telemetry(self) -> None:
        """
        Emit the guard calibration summary, and write it if asked.

        Called once when the stream stops. The text form goes to the log so a
        session is self-documenting; `guard_report` additionally writes JSON,
        which is what a calibration run should keep.
        """
        if not self._telemetry.frames:
            return

        if self._detector is not None:
            self._telemetry.capabilities = dict(self._detector.capabilities)

        emit_status(self._telemetry.format_report(self.config), scope='GUARD')
        emit_status(self._latency.format_report(self.config), scope='PERF')

        path = self.config.guard_report
        if path and self._telemetry.write(path, self.config):
            emit_status(f'Guard telemetry written to {path}', scope='GUARD')
        elif path:
            emit_error(f'Could not write guard telemetry to {path}', scope='GUARD')

    def _emit_guarded(
        self,
        seq: int,
        capture_ts: int,
        debug_input: Optional[Frame],
    ) -> None:
        """
        Emit the last good swapped frame, unchanged.

        Nothing is drawn on it — no banner, border, text or tint. This frame
        goes to the virtual camera and therefore to everyone on the call, so
        anything added would be visible to every participant. A held frame reads
        as a network hiccup, which is the most innocuous way this can fail in
        front of other people.

        With no good frame yet — guarded from the very first frame — nothing is
        emitted at all. That is deliberate: the alternative is the raw camera,
        and the operator is on the call precisely because they do not want their
        own face transmitted.

        Args:
            seq: Frame sequence number
            capture_ts: Capture timestamp in nanoseconds
            debug_input: The unprocessed frame, when debug capture is on
        """
        held = self._last_good_frame
        if held is None:
            self._guard_starved += 1
            if self._guard_starved == 1:
                emit_status(
                    'Guarded before any frame was swapped — nothing to hold, so '
                    'nothing is sent. The raw camera is never a fallback.',
                    scope='GUARD', level='warning',
                )
            return

        if debug_input is not None:
            self._queue_debug_pair(seq, debug_input, held)

        self.bus.emit(FRAME_READY, frame=held, seq=seq, capture_ts=capture_ts)

    # ------------------------------------------------------------------
    # Debug frame capture
    # ------------------------------------------------------------------

    def _queue_debug_pair(self, seq: int, source: Frame, output: Frame) -> None:
        """
        Hand an (input, output) pair to the writer thread.

        Never blocks and never writes on this thread: encoding PNGs inline
        would add tens of milliseconds per frame and change the latency of the
        very thing the capture exists to measure. Pairs are dropped rather than
        queued when the writer falls behind.

        Args:
            seq: Frame sequence number, used as the filename stem
            source: Frame as received, before any processing
            output: Final composited frame, as emitted
        """
        stride = max(1, int(self.config.debug_frames_stride or 1))
        if seq % stride:
            return

        limit = int(self.config.debug_frames_limit or 0)
        if limit and self._debug_written >= limit:
            return

        if self._debug_queue is None:
            self._start_debug_writer()
        assert self._debug_queue is not None

        try:
            self._debug_queue.put_nowait((seq, source, output.copy()))
        except queue.Full:
            self._debug_dropped += 1

    def _start_debug_writer(self) -> None:
        """Create the capture directory and start the background writer."""
        import os

        directory = self.config.debug_frames_dir or ''
        os.makedirs(directory, exist_ok=True)

        # 32 pairs of headroom, ~45 MB at 640x360. The writer sustains about
        # 39 pairs/second at that size, comfortably ahead of a 20fps stream,
        # so the depth is there to absorb a disk hiccup rather than a deficit.
        self._debug_queue = queue.Queue(maxsize=32)
        self._debug_thread = threading.Thread(
            target=self._debug_writer_loop,
            name='debug-frame-writer',
            daemon=True,
        )
        self._debug_thread.start()
        emit_status(f'Debug frame capture writing to: {directory}', scope='DEBUG_FRAMES')

    def _debug_writer_loop(self) -> None:
        """
        Write queued pairs as PNG until the pipeline stops.

        PNG rather than JPEG because these frames are measured, and a lossy
        encode would add its own artefacts to exactly the statistics — noise,
        high-frequency energy, blocking — the capture exists to compare.
        """
        import os

        assert self._debug_queue is not None
        directory = self.config.debug_frames_dir or ''

        while not self._stop_event.is_set():
            try:
                seq, source, output = self._debug_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                cv2.imwrite(os.path.join(directory, f'{seq:06d}_in.png'), source)
                cv2.imwrite(os.path.join(directory, f'{seq:06d}_out.png'), output)
                self._debug_written += 1
            except Exception as e:
                emit_error(
                    f'Debug frame write failed: {type(e).__name__}: {e}',
                    exception=e, scope='DEBUG_FRAMES',
                )
                return

        if self._debug_written:
            emit_status(
                f'Debug frame capture: {self._debug_written} pairs written, '
                f'{self._debug_dropped} dropped',
                scope='DEBUG_FRAMES',
            )

    def _log_timing(
        self,
        seq: int,
        started: float,
        detected: float,
        swapped: float,
    ) -> None:
        """
        Report per-stage timing periodically, at debug level only.

        Args:
            seq: Frame sequence number
            started: perf_counter at frame start
            detected: perf_counter after detection
            swapped: perf_counter after swap and compositing
        """
        detect_ms = (detected - started) * 1000.0
        swap_ms = (swapped - detected) * 1000.0
        total_ms = (swapped - started) * 1000.0

        # Recorded for every frame regardless of log level. The per-line debug
        # output below answers "how long did frame 240 take"; whether the preset
        # holds is a question about the distribution, and a 1-in-30 sample taken
        # only at debug level cannot answer it.
        # The compositor's own breakdown of the swap bucket, if it composited
        # this frame. A guarded frame leaves it empty, which is correct — the
        # stages it names did not run.
        stages = self._compositor.last_stage_ms if self._compositor else None
        self._latency.record(detect_ms, swap_ms, total_ms, dict(stages) if stages else None)

        if self.config.log_level != 'debug' or seq % self._TIMING_INTERVAL:
            return

        get_logger('PERF').debug(
            'seq=%d detect=%.0fms swap+composite=%.0fms total=%.0fms',
            seq, detect_ms, swap_ms, total_ms,
        )

    @staticmethod
    def _unpack_timestamped_frame(
        data: Union[bytes, 'Tuple[int, bytes]'],
    ) -> Tuple[int, bytes]:
        """Extract capture_ts and JPEG bytes from a frame queue item.

        Supports two formats:
        - Tuple (capture_ts, jpeg_bytes): set by server when 8-byte header present
        - Raw bytes: legacy/fallback, capture_ts = 0

        Returns:
            (capture_ts_ns, jpeg_bytes)
        """
        if isinstance(data, tuple):
            return data
        return (0, data)

    def _stream_loop_push(self) -> None:
        """Stream loop for WebSocket push mode — reads JPEG frames from frame_queue."""
        assert self.frame_queue is not None

        # Drain stale frames queued while the pipeline was stopped — prevents
        # latency buildup across multiple stop/start cycles.
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        seq = 0
        while not self._stop_event.is_set():
            try:
                raw = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            capture_ts, jpeg_bytes = self._unpack_timestamped_frame(raw)

            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            seq += 1
            self._process_and_emit(frame, seq, capture_ts)

    def _stream_loop_capture(self) -> None:
        """Stream loop for VideoCapture mode — local webcam or network URL."""
        input_source = self.config.input_url or 0
        cap = cv2.VideoCapture(input_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.config.input_url:
            # Driven by the quality preset, matching what the desktop requests
            # of its own webcam in push mode. Previously hardcoded to 960x540
            # at 30fps, which meant a local run always paid production capture
            # cost no matter which preset was selected.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.capture_fps)

        frame_count = 0
        seq = 0
        drop_count = 0
        drop_window_start = time.time()
        warmup_frames = getattr(self.config, 'warmup_frames', 0)

        try:
            # A file input loops rather than ending. Measuring a lever means
            # streaming for a fixed wall-clock time, and frames are read as
            # fast as they decode rather than paced to real time — so a clip is
            # consumed far quicker than its running length and would otherwise
            # end the run early, leaving later configurations with no data.
            #
            # Only for a file. On a webcam or a network stream a failed read
            # means the device or the connection is gone, and looping would
            # turn that into a silent stall instead of a stop.
            loops = bool(
                self.config.input_url and os.path.isfile(str(self.config.input_url))
            )

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if not loops:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

                frame_count += 1
                seq += 1

                if frame_count <= warmup_frames:
                    continue

                capture_ts = time.perf_counter_ns()

                self._process_and_emit(frame, seq, capture_ts)

                if frame_count % 30 == 0:
                    now = time.time()
                    window = now - drop_window_start
                    if window > 1.0:
                        drop_rate = drop_count / frame_count
                        self.bus.emit('drop_rate', dropped=drop_count, total=frame_count, rate=drop_rate)
                        drop_window_start = now
                        drop_count = 0
        finally:
            cap.release()

    def run_batch(self) -> None:
        """
        Run batch processing mode (single image or video file).

        Processes target image/video with source face swapping.
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._abort_reason = ''

        try:
            self._run_batch_impl()
        except Exception as e:
            emit_error(f"Batch pipeline error: {e}", exception=e, scope='PIPELINE')
        finally:
            self._running = False
            self._stop_event.set()
            self.bus.emit(PIPELINE_STOPPED)

    def _run_batch_impl(self) -> None:
        """Implementation of batch mode."""
        self._build_processors()
        emit_status('Batch pipeline started', scope='PIPELINE')
        self.bus.emit(PIPELINE_STARTED)

        # Photo mode is several image targets, each swapped on its own; the
        # single-file path stays exactly as it was. Which one this is comes
        # from which field was set, not from a mode flag, so the two cannot be
        # in disagreement.
        photos = list(self.config.target_paths)

        # Validate inputs
        if not photos and not self.config.target_path:
            emit_error('No target path specified', scope='PIPELINE')
            return

        # Load source
        sources = self.config.source_paths or (
            [self.config.source_path] if self.config.source_path else []
        )
        if not sources or not self._swapping_proc.set_source(sources):
            emit_error('No valid source face', scope='PIPELINE')
            return

        # Process
        try:
            if photos:
                self._process_photos_batch(photos)
            else:
                self._process_target_batch(self.config.target_path, self.config.output_path)
        except Exception as e:
            emit_error(f"Batch processing failed: {e}", exception=e, scope='PIPELINE')

    def _process_target_batch(self, target_path: str, output_path: Optional[str]) -> None:
        """
        Process a single target image or video file.

        Args:
            target_path: Path to target image or video
            output_path: Where to save output (optional)
        """
        if not os.path.isfile(target_path):
            emit_error(f"Target file not found: {target_path}", scope='PIPELINE')
            return

        if is_image(target_path) or has_image_extension(target_path):
            result = self._process_image_batch(target_path, output_path)
            self._photo_results = [result]
            if not result.ok:
                # A single-image job reports through the error channel, which
                # is what the desktop reads to tell a failed batch from a
                # finished one. Photo mode reports per item instead.
                emit_error(
                    f"{os.path.basename(target_path)}: {result.reason}",
                    scope='PIPELINE',
                )
        elif is_video(target_path):
            self._process_video_batch(target_path, output_path)
        else:
            emit_error(
                f"Unsupported target, neither image nor video: {target_path}",
                scope='PIPELINE',
            )

    def _swap_frame_detail(
        self,
        frame: Frame,
        stabilize: bool,
        face_point: Optional[Tuple[float, float]] = None,
    ) -> FrameSwap:
        """
        Run preprocess -> detect -> swap -> composite for one batch frame, and
        say whether a swap actually happened.

        Shared by the image and video paths so the two cannot drift apart, and
        the same compositor as stream mode so batch output matches live.

        The frame is returned unswapped on every failure path, which is what
        the video path wants for most of them — one unswapped frame mid-clip
        beats a hole. A still image has no such context: an unswapped photo is
        simply a copy of the input wearing the output's name, so the photo path
        needs to know the difference. Hence the reason, rather than a bare
        frame — and the reason *code* alongside it, since only video's caller
        cares which guard it was.

        Args:
            frame: Input frame
            stabilize: Smooth landmarks across frames. True for video, where
                       frames are consecutive; False for a lone image
            face_point: The face named for this target, if the operator or a
                        template named one

        Returns:
            FrameSwap. On failure the frame is the input, unmodified apart
            from preprocessing.
        """
        frame = self._preprocessing_proc.process(frame)
        self._detection_proc.face_point = face_point
        frame = self._detection_proc.process(frame)
        detections = self._detection_proc.latest_detections

        if not detections:
            if stabilize:
                self._stabilizer.mark_missing()
            return FrameSwap(frame, 'no face detected')

        # Batch guards pass the original frame through rather than holding the
        # last good one. The privacy argument that forces a held frame on the
        # live path does not apply: the target here is a file the operator
        # supplied, not their camera, and freezing a frame mid-clip would be a
        # more visible defect than one unswapped frame.
        verdict = guards.check_frame(
            self.config, self._detection_proc.all_detections, face_point,
        )
        if not verdict.ok:
            self._reset_temporal_state()
            return FrameSwap(frame, verdict.message or verdict.reason, verdict.reason)

        swapped_count = 0
        for detection in detections:
            face = detection.face
            # Landmark smoothing needs a stable subject identity; with several
            # faces the per-frame detection order is not stable.
            if stabilize and not self.config.many_faces:
                face = self._stabilizer.stabilize(face)

            swapped = self._swap_face(frame, face)
            if swapped is None:
                self._reset_temporal_state()
                return FrameSwap(
                    frame, 'the compositor produced no swap', faces=swapped_count,
                )
            frame = swapped
            swapped_count += 1

        return FrameSwap(frame, faces=swapped_count)

    def _process_image_batch(
        self,
        target_path: str,
        output_path: Optional[str],
        face_point: Optional[Tuple[float, float]] = None,
    ) -> PhotoResult:
        """
        Swap faces in a single image.

        Writes an output file only when a swap actually happened. Previously
        this wrote unconditionally, so a guarded or faceless target produced a
        file that was byte-for-byte the input but named like a result — the
        exact "confidently wrong output" the guards exist to prevent, and
        indistinguishable from success to whoever opens the folder.

        Args:
            target_path: Path to target image
            output_path: Where to save output, or None to derive one
            face_point: The face the operator picked in this photo, if it holds
                        more than one and they were asked

        Returns:
            PhotoResult describing what happened to this image
        """
        frame = cv2.imread(target_path)
        if frame is None:
            reason = 'could not be read as an image'
            emit_error(f"{os.path.basename(target_path)}: {reason}", scope='PIPELINE')
            return PhotoResult.skipped(target_path, reason)

        # There is no previous frame to smooth against — but the compositor's
        # pixel EMA would otherwise still hold whatever the last job left in it,
        # so the image has to be given a clean slate rather than assumed one.
        self._reset_temporal_state()
        result = self._swap_frame_detail(frame, stabilize=False, face_point=face_point)

        if result.reason:
            emit_status(
                f"Skipped {os.path.basename(target_path)}: {result.reason}",
                scope='PIPELINE',
            )
            return PhotoResult.skipped(target_path, result.reason)

        frame = result.frame

        # A template can carry an authored layer that belongs in front of the
        # face. Applied after the swap and before writing, so it occludes the
        # result exactly as it occluded the original.
        if self.config.target_foreground:
            frame = templates.composite_foreground(frame, self.config.target_foreground)

        out_path = output_path or self._photo_output_path(target_path)
        try:
            written = cv2.imwrite(out_path, frame)
        except Exception as e:
            written = False
            emit_error(f"Failed writing {out_path}: {e}", exception=e, scope='PIPELINE')
        if not written:
            return PhotoResult.skipped(target_path, f'could not write output to {out_path}')

        emit_status(f"Batch output saved to: {out_path}", scope='PIPELINE')
        return PhotoResult.swapped(target_path, out_path, result.faces)

    def _photo_output_path(self, target_path: str) -> str:
        """
        Where a photo's swap is written when no explicit output was given.

        Beside the target with a `_swapped` suffix. In a remote job the target
        already lives in that job's own upload directory, so the outputs land
        there too and are removed with it by `cleanup_session`.

        `output_dir` overrides that, and a template job sets it: a template's
        target lives in the shared library, so writing beside it would leave
        one user's face in an asset directory for the next job to find.

        Args:
            target_path: Path to the target image

        Returns:
            Output path for this target
        """
        base, ext = os.path.splitext(target_path)
        if self.config.output_dir:
            base = os.path.join(
                self.config.output_dir, os.path.basename(base)
            )
        return f'{base}_swapped{ext or ".png"}'

    def _process_photos_batch(self, target_paths: List[str]) -> List[PhotoResult]:
        """
        Swap each target photo independently, skipping the ones that fail.

        Independence is the contract: one unusable photo must not cost the
        operator the other three, so every failure — unreadable file, no face,
        a guard, or an exception from the swap itself — is recorded against
        that photo and the loop continues.

        Args:
            target_paths: Target images, in the order they were given

        Returns:
            One PhotoResult per target, in the same order
        """
        results: List[PhotoResult] = []
        total = len(target_paths)
        # Aligned with `target_paths` by index, and allowed to be shorter or
        # absent: a photo nobody was asked about simply has no point, and is
        # refused by the multi-face guard exactly as before.
        points = self.config.target_face_points

        for index, target in enumerate(target_paths):
            if self._stop_event.is_set():
                emit_status('Photo batch cancelled', scope='PIPELINE')
                break

            emit_status(
                f"Processing photo {index + 1}/{total}: {os.path.basename(target)}",
                scope='PIPELINE',
            )

            if not os.path.isfile(target):
                result = PhotoResult.skipped(target, 'file not found')
            else:
                point = points[index] if index < len(points) else None
                try:
                    result = self._process_image_batch(target, None, point)
                except Exception as e:
                    # A failure here is this photo's failure, not the job's.
                    emit_error(
                        f"{os.path.basename(target)}: {type(e).__name__}: {e}",
                        exception=e,
                        scope='PIPELINE',
                    )
                    result = PhotoResult.skipped(target, f'{type(e).__name__}: {e}')

            results.append(result)
            self.bus.emit(PHOTO_RESULT, result=result, index=index, total=total)
            self.bus.emit(
                BATCH_PROGRESS,
                done=index + 1,
                total=total,
                percent=((index + 1) / total) * 100.0 if total else 100.0,
            )

        self._photo_results = results
        swapped = sum(1 for r in results if r.ok)
        emit_status(
            f"Photos complete: {swapped} swapped, {len(results) - swapped} skipped",
            scope='PIPELINE',
        )
        return results

    def _process_video_batch(self, target_path: str, output_path: Optional[str]) -> None:
        """
        Swap faces through a video, then reassemble it with its audio.

        Frames are extracted to lossless PNG, swapped in place and re-encoded.
        Working through files rather than streaming costs disk — roughly 4 MB per
        1080p frame — but it keeps each frame lossless between decode and encode,
        so the only generational loss is the final encode, and it makes
        `keep_frames` and a resumable job possible.

        Args:
            target_path: Path to target video
            output_path: Where to save output
        """
        if not output_path:
            emit_error('No output path specified for video batch', scope='PIPELINE')
            return

        # The source rate is needed either way: it is what the extracted frames
        # are read back at, and reading them at anything else would rescale the
        # clip against the audio restored afterwards. `keep_fps` decides only
        # whether the result is then re-timed.
        source_fps = detect_fps(target_path)
        output_fps = None if self.config.keep_fps else 30.0

        reset_temp(target_path)

        try:
            emit_status(
                f"Extracting frames from {target_path} at {source_fps:.3f}fps",
                scope='PIPELINE',
            )
            extract_frames(self.config, target_path)

            frame_paths = get_temp_frame_paths(target_path)
            if not frame_paths:
                emit_error(
                    f"No frames extracted from {target_path} — is FFmpeg installed "
                    f"and the file readable?",
                    scope='PIPELINE',
                )
                return

            if not self._process_frame_files(frame_paths, source_fps):
                if self._abort_reason:
                    # The error channel, not a status: the desktop reads a
                    # batch's success from whether an error arrived, and a
                    # warning here would let this render as "complete".
                    emit_error(self._abort_reason, scope='PIPELINE')
                else:
                    emit_status('Batch cancelled', scope='PIPELINE')
                return

            emit_status(f"Encoding {len(frame_paths)} frames", scope='PIPELINE')
            create_video(self.config, target_path, source_fps, output_fps)

            if not os.path.isfile(get_temp_output_path(target_path)):
                emit_error('Video encoding produced no output', scope='PIPELINE')
                return

            if self.config.keep_audio and has_audio(target_path):
                restore_audio(self.config, target_path, output_path)
            else:
                move_temp(target_path, output_path)

            if os.path.isfile(output_path):
                emit_status(f"Batch output saved to: {output_path}", scope='PIPELINE')
            else:
                emit_error(f"Output file was not written: {output_path}", scope='PIPELINE')
        finally:
            clean_temp(self.config, target_path)

    def _process_frame_files(self, frame_paths: List[str], fps: float) -> bool:
        """
        Swap every extracted frame in place.

        Stops the whole job at the first frame holding more than one face.
        That is deliberately the only guard that aborts: low confidence, pose
        and occlusion describe one frame and pass through, but a second face
        describes the *target*, will almost certainly persist, and every frame
        it appears in would otherwise be written unswapped — a video that
        silently stops being a swap partway through, which is the confidently
        wrong output the guards exist to prevent.

        Args:
            frame_paths: Extracted frame paths, in playback order
            fps: Source frame rate, for naming where in the clip it stopped

        Returns:
            True if all frames were processed, False if the job was stopped —
            by the operator, or by itself with `_abort_reason` set
        """
        total = len(frame_paths)

        # Consecutive frames of one clip, so temporal smoothing applies — but it
        # must not begin against state left by a previous run.
        self._reset_temporal_state()

        # Progress is rate-limited two ways, because either alone fails at one
        # end of the range: a 1% step is every frame on a 90-frame clip, and a
        # time gate alone is thousands of messages across a feature-length job.
        report_every = max(1, total // 100)
        started = time.perf_counter()
        last_report = 0.0

        for index, frame_path in enumerate(frame_paths):
            if self._stop_event.is_set():
                return False

            frame = cv2.imread(frame_path)
            if frame is None:
                # A frame that will not decode is passed through rather than
                # dropped: dropping one shifts every later frame against the
                # audio, and a single unswapped frame is the smaller fault.
                emit_status(
                    f"Skipping unreadable frame: {frame_path}",
                    scope='PIPELINE', level='warning',
                )
                continue

            swap = self._swap_frame_detail(frame, stabilize=True)
            if swap.code == guards.MULTIPLE_FACES:
                self._abort_reason = (
                    f'Stopped at frame {index + 1} of {total} '
                    f'({_timecode(index, fps)}) — {swap.reason}. A video target '
                    f'must show one face throughout. No output was written.'
                )
                return False
            cv2.imwrite(frame_path, swap.frame)

            now = time.perf_counter()
            final = index == total - 1
            if final or (index % report_every == 0 and now - last_report >= self._PROGRESS_INTERVAL):
                last_report = now
                self._emit_batch_progress(index + 1, total, started)

        return True

    def _emit_batch_progress(self, done: int, total: int, started: float) -> None:
        """
        Report batch progress, with an estimate of the time remaining.

        Goes out as a status message, which the desktop already displays in
        batch modes, rather than as a new event needing its own plumbing.

        Args:
            done: Frames completed
            total: Frames in the job
            started: perf_counter when frame processing began
        """
        percent = (done / total) * 100.0 if total else 100.0
        elapsed = time.perf_counter() - started

        remaining = ''
        if done and elapsed > 1.0:
            eta = (elapsed / done) * (total - done)
            remaining = f", {int(eta // 60)}m{int(eta % 60):02d}s left"

        emit_status(
            f"Processing frame {done}/{total} ({percent:.0f}%{remaining})",
            scope='PIPELINE',
        )
        self.bus.emit(BATCH_PROGRESS, done=done, total=total, percent=percent)

    def stop(self) -> None:
        """Stop the pipeline."""
        self._stop_event.set()

    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self._running

    @property
    def photo_results(self) -> List[PhotoResult]:
        """Per-photo outcomes of the most recent batch, in target order."""
        return list(self._photo_results)
