"""
Main processing pipeline coordinator for Phantom.

Orchestrates frame processors into a complete pipeline.
Replaces monolithic stream.py:_pipeline_loop() and core.py:start().

Responsibilities:
- Build processor chains (batch vs stream)
- Coordinate async enhancement
- Emit events (FRAME_READY, DETECTION, etc.)
- Listen to config changes and rebuild
- Manage I/O sources and sinks
"""

import threading
import time
from typing import Any, Callable, List, Optional

import cv2

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
)
from pipeline.processing.async_processor import AsyncProcessor


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
        self._async_enhancement: Optional[AsyncProcessor] = None

        # State
        self._running = False
        self._stop_event = threading.Event()

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

        # Async enhancement wrapper
        if self._async_enhancement is not None:
            self._async_enhancement.stop()
            self._async_enhancement.join()

        self._async_enhancement = AsyncProcessor(self._enhancement_proc, self._stop_event)

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
        elif field in ('tracker', 'alpha', 'enhance_interval', 'blend', 'luminance_blend'):
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

    def _run_stream_impl(self) -> None:
        """Implementation of stream mode."""
        self._build_processors()
        emit_status('Stream pipeline started', scope='PIPELINE')
        self.bus.emit(PIPELINE_STARTED)

        # Set up source
        input_source = self.config.input_url or 0
        cap = cv2.VideoCapture(input_source)
        if not self.config.input_url:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # Load source faces
        sources = self.config.source_paths or (
            [self.config.source_path] if self.config.source_path else []
        )
        if sources:
            self._swapping_proc.set_source(sources)

        # Start async enhancement
        self._async_enhancement.start()

        frame_count = 0
        seq = 0
        drop_count = 0
        drop_window_start = time.time()

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                seq += 1

                # Run detection
                frame = self._detection_proc.process(frame)
                detections = self._detection_proc.latest_detections

                # Run tracking (and swap if we have detections)
                if detections:
                    for detection in detections:
                        # Initialize or update tracker
                        if not self._tracking_proc._tracker or not self._tracking_proc._tracker.is_valid:
                            self._tracking_proc.set_tracked_face(detection)

                frame = self._tracking_proc.process(frame)
                tracked = self._tracking_proc.get_tracked_detection()

                # Swap
                if tracked and self._swapping_proc.source_face:
                    frame = self._swapping_proc.swap_detection(frame, tracked)
                    self.bus.emit(DETECTION, detection=tracked.to_dict(), seq=seq)

                # Async enhancement
                self._async_enhancement.submit(seq, frame)
                result = self._async_enhancement.get_latest()
                if result:
                    result_seq, enhanced_frame = result
                    frame = enhanced_frame

                # Emit frame ready
                self.bus.emit(FRAME_READY, frame=frame, seq=seq)

                # Track drops
                if frame_count % 30 == 0:
                    now = time.time()
                    window = now - drop_window_start
                    if window > 1.0:
                        drop_rate = drop_count / frame_count
                        self.bus.emit('drop_rate', dropped=drop_count, total=frame_count, rate=drop_rate)
                        drop_window_start = now
                        drop_count = 0

        finally:
            self._async_enhancement.stop()
            self._async_enhancement.join()
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
            frame = self._detection_proc.process(frame)
            detections = self._detection_proc.latest_detections

            for detection in detections:
                frame = self._swapping_proc.swap_detection(frame, detection)

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
