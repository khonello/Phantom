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

import queue
import struct
import threading
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig, CONFIG
from pipeline.types import Frame
from pipeline.events import BUS, FRAME_READY, DETECTION, PIPELINE_STARTED, PIPELINE_STOPPED
from pipeline.logging import emit_status, emit_error

from pipeline.services.face_detection import FaceDetector
from pipeline.services.face_swapping import FaceSwapper
from pipeline.services.enhancement import Enhancer
from pipeline.services.database import FaceDatabase

from pipeline.processing.frame_processor import (
    FrameProcessor,
    DetectionProcessor,
    TrackingProcessor,
    SwappingProcessor,
    EnhancementProcessor,
    BlendingProcessor,
    ColorCorrectionProcessor,
    PreprocessingProcessor,
)


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

        # Processors
        self._detection_proc: Optional[DetectionProcessor] = None
        self._tracking_proc: Optional[TrackingProcessor] = None
        self._swapping_proc: Optional[SwappingProcessor] = None
        self._enhancement_proc: Optional[EnhancementProcessor] = None
        self._blending_proc: Optional[BlendingProcessor] = None
        self._color_correction_proc: Optional[ColorCorrectionProcessor] = None
        self._preprocessing_proc: Optional[PreprocessingProcessor] = None

        # State
        self._running = False
        self._stop_event = threading.Event()

        # Set by WebSocketAPIServer to enable push mode: desktop sends JPEG
        # frames via WebSocket instead of the pipeline capturing a local device.
        self.frame_queue: Optional[queue.Queue] = None

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
            self._enhancer = Enhancer()
        return self._enhancer

    def _get_database(self) -> FaceDatabase:
        """Get or create FaceDatabase."""
        if self._database is None:
            self._database = FaceDatabase(self._get_detector())
        return self._database

    def _build_processors(self) -> None:
        """Build processor instances."""
        detector = self._get_detector()
        swapper = self._get_swapper()
        enhancer = self._get_enhancer()
        database = self._get_database()

        # Create fresh processors
        self._detection_proc = DetectionProcessor(self.config, detector)
        self._tracking_proc = TrackingProcessor(self.config, detector)
        self._swapping_proc = SwappingProcessor(self.config, swapper, database)
        self._enhancement_proc = EnhancementProcessor(self.config, enhancer)
        self._blending_proc = BlendingProcessor(self.config)
        self._color_correction_proc = ColorCorrectionProcessor(self.config)
        self._preprocessing_proc = PreprocessingProcessor(self.config)

    def _on_config_changed(self, field: str, value: Any) -> None:
        """
        Handle configuration changes.

        Some changes require rebuilding processor chain.

        Args:
            field: Config field name
            value: New value
        """
        # Source path changed → reset tracker and load new source
        if field == 'source_path' or field == 'source_paths':
            if self._tracking_proc:
                self._tracking_proc.reset()
            if self._swapping_proc:
                sources = self.config.source_paths or (
                    [self.config.source_path] if self.config.source_path else []
                )
                self._swapping_proc.set_source(sources)

        # Tracker type or alpha changed → rebuild processors
        elif field in ('tracker', 'alpha', 'blend', 'luminance_blend'):
            self._build_processors()

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
            self.bus.emit(PIPELINE_STOPPED)

    def _warm_up_models(self) -> None:
        """
        Eagerly load ML models into GPU memory before the stream loop starts.

        Both models are lazily initialized by default, meaning the first frame
        that needs them blocks for 10-30s while 500MB+ of ONNX weights are
        loaded into CUDA. Pre-loading them here makes the first swap instant.
        """
        emit_status('Loading detection model...', scope='MODEL_LOAD')
        try:
            self._get_detector()._get_analyser()
        except Exception as e:
            emit_error(f"Detection model load failed: {e}", exception=e, scope='PIPELINE')

        emit_status('Loading swap model...', scope='MODEL_LOAD')
        try:
            self._get_swapper()._get_swapper()
        except Exception as e:
            emit_error(f"Swap model load failed: {e}", exception=e, scope='PIPELINE')

        emit_status('Models ready', scope='MODEL_LOAD')

    def _run_stream_impl(self) -> None:
        """Implementation of stream mode."""
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

    def _process_and_emit(self, frame: Frame, seq: int, capture_ts: int = 0) -> None:
        """Run detection → tracking → swap → enhance → emit for one frame.

        Args:
            frame: Input video frame
            seq: Sequence number
            capture_ts: Capture timestamp in nanoseconds (time.perf_counter_ns)
        """
        # TEMPORARY: per-stage timing to diagnose GPU performance
        import sys
        _t0 = time.perf_counter()

        # Preprocessing: normalize lighting, white balance, denoise
        frame = self._preprocessing_proc.process(frame)

        frame = self._detection_proc.process(frame)
        detections = self._detection_proc.latest_detections
        _t1 = time.perf_counter()

        if detections:
            for detection in detections:
                if self._tracking_proc.get_tracked_detection() is None:
                    self._tracking_proc.set_tracked_face(detection, frame)

        frame = self._tracking_proc.process(frame)
        tracked = self._tracking_proc.get_tracked_detection()
        _t2 = time.perf_counter()

        # If the CV2 tracker is unavailable (e.g. opencv-python without contrib),
        # fall back to the raw detection so swapping still works.
        if tracked is None and detections:
            tracked = detections[0]

        if tracked and self._swapping_proc.source_face:
            # Save original before swap — needed for color correction reference.
            # Only copy when color correction is enabled to avoid allocation overhead.
            original_frame = frame.copy() if self.config.color_correction else None
            frame = self._swapping_proc.swap_detection(frame, tracked)

            # Color correction: match swapped face color to original target skin
            if original_frame is not None:
                bbox = tracked.bbox
                frame = self._color_correction_proc.correct(
                    frame, original_frame,
                    (bbox.x, bbox.y, bbox.w, bbox.h),
                )

            self.bus.emit(DETECTION, detection=tracked.to_dict(), seq=seq)
        _t3 = time.perf_counter()

        if self.config.enhance:
            frame = self._enhancement_proc.process(frame)
        _t4 = time.perf_counter()

        self.bus.emit(FRAME_READY, frame=frame, seq=seq, capture_ts=capture_ts)

        # TEMPORARY: print per-stage timing every frame
        print(
            f'[PERF] seq={seq} '
            f'detect={(_t1-_t0)*1000:.0f}ms '
            f'track={(_t2-_t1)*1000:.0f}ms '
            f'swap={(_t3-_t2)*1000:.0f}ms '
            f'enhance={(_t4-_t3)*1000:.0f}ms '
            f'total={(_t4-_t0)*1000:.0f}ms',
            file=sys.stderr,
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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        seq = 0
        drop_count = 0
        drop_window_start = time.time()
        warmup_frames = getattr(self.config, 'warmup_frames', 0)

        try:
            while not self._stop_event.is_set():
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

        # Validate inputs
        if not self.config.target_path:
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
        # Simple implementation for images
        if target_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            frame = cv2.imread(target_path)
            if frame is None:
                emit_error(f"Failed to load image: {target_path}", scope='PIPELINE')
                return

            # Process
            frame = self._preprocessing_proc.process(frame)
            frame = self._detection_proc.process(frame)
            detections = self._detection_proc.latest_detections

            original_frame = frame.copy() if self.config.color_correction else None
            for detection in detections:
                frame = self._swapping_proc.swap_detection(frame, detection)

                if original_frame is not None:
                    bbox = detection.bbox
                    frame = self._color_correction_proc.correct(
                        frame, original_frame,
                        (bbox.x, bbox.y, bbox.w, bbox.h),
                    )

            if self.config.enhance:
                frame = self._enhancement_proc.process(frame)

            # Save
            if output_path:
                cv2.imwrite(output_path, frame)
                emit_status(f"Batch output saved to: {output_path}", scope='PIPELINE')
        else:
            emit_error('Video batch processing not yet implemented', scope='PIPELINE')

    def stop(self) -> None:
        """Stop the pipeline."""
        self._stop_event.set()

    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self._running
