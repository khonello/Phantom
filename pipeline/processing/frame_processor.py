"""
Composable frame processors for the Phantom pipeline.

Each processor encapsulates a single processing step (detection, swap, enhance, etc.)
and can be chained together. Processors are stateless (or maintain only internal state)
and don't depend on global variables.

Abstract base:
    FrameProcessor - process(frame: Frame) -> Frame

Implementations:
    DetectionProcessor - Detect faces in frame
    TrackingProcessor - Track faces across frames
    SwappingProcessor - Swap detected faces
    EnhancementProcessor - Enhance faces (async-aware wrapper)
    BlendingProcessor - Blend swapped/original
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import cv2

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Detection
from pipeline.services.face_detection import FaceDetector
from pipeline.services.face_swapping import FaceSwapper
from pipeline.services.enhancement import Enhancer
from pipeline.services.face_tracking import FaceTrackerState
from pipeline.services.database import FaceDatabase
from pipeline.logging import emit_status, emit_warning


class FrameProcessor(ABC):
    """
    Abstract base for frame processors.

    Each processor transforms a frame (and optionally maintains state).
    Processors are chained together to form a pipeline.
    """

    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        """
        Process a frame.

        Args:
            frame: Input frame (numpy array)

        Returns:
            Processed frame
        """
        pass


class DetectionProcessor(FrameProcessor):
    """
    Detect faces in a frame.

    Returns frame unchanged but stores detections in state for downstream
    processors. Uses FaceDetector service.
    """

    def __init__(self, config: FaceSwapConfig, detector: FaceDetector) -> None:
        """
        Initialize detection processor.

        Args:
            config: Configuration object
            detector: FaceDetector service instance
        """
        self.config = config
        self.detector = detector
        self.latest_detections: List[Detection] = []

    def process(self, frame: Frame) -> Frame:
        """
        Detect faces in frame.

        Args:
            frame: Input frame

        Returns:
            Frame unchanged; detections stored in self.latest_detections
        """
        try:
            if self.config.many_faces:
                self.latest_detections = self.detector.detect(frame)
            else:
                det = self.detector.detect_one(frame)
                self.latest_detections = [det] if det else []
        except Exception as e:
            emit_warning(f"Detection failed: {e}", scope='DETECTION')
            self.latest_detections = []

        return frame


class TrackingProcessor(FrameProcessor):
    """
    Track faces across frames using OpenCV trackers.

    Maintains FaceTrackerState internally. Returns frame unchanged
    but updates tracking state for downstream processors.
    """

    def __init__(
        self,
        config: FaceSwapConfig,
        detector: FaceDetector,
        redetect_interval: int = 30,
    ) -> None:
        """
        Initialize tracking processor.

        Args:
            config: Configuration object
            detector: FaceDetector for handling detection refresh
            redetect_interval: How often to re-detect faces (in frames)
        """
        self.config = config
        self.detector = detector
        self.redetect_interval = redetect_interval

        self._tracker: Optional[FaceTrackerState] = None
        self._frame_count = 0
        self.latest_detection: Optional[Detection] = None

    def set_tracked_face(self, detection: Detection, frame: Optional[Frame] = None) -> None:
        """
        Initialize tracking with a detected face.

        Called by upstream processor or pipeline when face is detected.
        Pass `frame` to immediately initialize the CV2 tracker; without it
        the tracker is created but stays uninitialized until the next process() call.

        Args:
            detection: Detection to track
            frame: Current frame (required to initialize the CV2 tracker)
        """
        self._tracker = FaceTrackerState(
            tracker_type=self.config.tracker,
            ema_alpha=self.config.alpha,
        )
        self.latest_detection = detection
        if frame is not None:
            self._tracker.initialize(frame, detection)

    def process(self, frame: Frame) -> Frame:
        """
        Update tracker state.

        Args:
            frame: Current frame

        Returns:
            Frame unchanged; tracking state updated in self
        """
        self._frame_count += 1

        # If no valid tracker, nothing to do this frame
        if self._tracker is None or not self._tracker.is_valid:
            return frame

        # Try to update tracker
        if self._tracker.update(frame):
            pass
        else:
            # Tracker lost face
            self.latest_detection = None
            self._tracker.reset()

        # Re-detect periodically to handle drift / re-entry
        if self._frame_count % self.redetect_interval == 0:
            det = self.detector.detect_one(frame)
            if det:
                self.set_tracked_face(det, frame)

        return frame

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker = None
        self.latest_detection = None
        self._frame_count = 0

    def get_tracked_detection(self) -> Optional[Detection]:
        """Get the current tracked detection (if valid)."""
        if self._tracker and self._tracker.is_valid:
            # Return cached detection with updated bbox from tracker
            if self.latest_detection:
                return Detection(
                    face=self.latest_detection.face,
                    bbox=self._tracker.get_bbox(),
                    kps=self._tracker.get_kps(),
                    confidence=self.latest_detection.confidence,
                )
        return None


class SwappingProcessor(FrameProcessor):
    """
    Swap detected faces using FaceSwapper service.

    Takes input from DetectionProcessor or TrackingProcessor.
    """

    def __init__(
        self,
        config: FaceSwapConfig,
        swapper: FaceSwapper,
        database: FaceDatabase,
    ) -> None:
        """
        Initialize swapping processor.

        Args:
            config: Configuration object
            swapper: FaceSwapper service instance
            database: FaceDatabase for source face lookup
        """
        self.config = config
        self.swapper = swapper
        self.database = database
        self.source_face = None

    def set_source(self, paths: List[str]) -> bool:
        """
        Load source face from paths.

        Args:
            paths: List of image or .npy paths

        Returns:
            True if source loaded successfully
        """
        try:
            self.source_face = self.database.get_source_face(paths)
            if self.source_face:
                emit_status(f"Source face loaded from {len(paths)} path(s)", scope='SWAPPER')
                return True
            else:
                emit_warning("No face found in source paths", scope='SWAPPER')
                return False
        except Exception as e:
            emit_warning(f"Failed to load source: {e}", scope='SWAPPER')
            return False

    def process(self, frame: Frame) -> Frame:
        """
        Process frame (no-op without source or detections).

        Actual swapping is done via swap_detection().

        Args:
            frame: Input frame

        Returns:
            Frame unchanged
        """
        return frame

    def swap_detection(self, frame: Frame, detection: Detection) -> Frame:
        """
        Swap a detected face in the frame.

        Args:
            frame: Frame containing face
            detection: Detection to swap

        Returns:
            Frame with swapped face
        """
        if self.source_face is None:
            return frame

        try:
            return self.swapper.swap(self.source_face, detection, frame)
        except Exception as e:
            emit_warning(f"Swap failed: {e}", scope='SWAPPER')
            return frame

    def reset(self) -> None:
        """Clear source face."""
        self.source_face = None
        self.database.clear()


class EnhancementProcessor(FrameProcessor):
    """
    Enhance faces using GFPGAN (if available).

    Gracefully falls back to unchanged frame if enhancement unavailable.
    """

    def __init__(self, config: FaceSwapConfig, enhancer: Enhancer) -> None:
        """
        Initialize enhancement processor.

        Args:
            config: Configuration object
            enhancer: Enhancer service instance
        """
        self.config = config
        self.enhancer = enhancer
        self.frame_count = 0

    def process(self, frame: Frame) -> Frame:
        """
        Enhance frame (every N frames based on config).

        Args:
            frame: Input frame

        Returns:
            Enhanced frame (or original if enhancement disabled/unavailable)
        """
        self.frame_count += 1

        # Skip enhancement if disabled
        if self.config.enhance_interval <= 0:
            return frame

        # Skip based on interval
        if self.frame_count % self.config.enhance_interval != 0:
            return frame

        # Enhance
        if self.enhancer.available:
            try:
                return self.enhancer.enhance(frame)
            except Exception as e:
                emit_warning(f"Enhancement failed: {e}", scope='ENHANCER')
                return frame

        return frame

    def reset(self) -> None:
        """Reset frame counter."""
        self.frame_count = 0


class BlendingProcessor(FrameProcessor):
    """
    Blend swapped and original frames based on alpha/luminance settings.
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize blending processor.

        Args:
            config: Configuration object (blend, alpha, luminance_blend)
        """
        self.config = config

    def process(self, frame: Frame) -> Frame:
        """
        Blend frame (no-op; actual blending done via blend()).

        Args:
            frame: Input frame

        Returns:
            Frame unchanged
        """
        return frame

    def blend(
        self,
        swapped: Frame,
        original: Frame,
        face_bbox: tuple,
    ) -> Frame:
        """
        Blend swapped face with original frame.

        Args:
            swapped: Frame with swapped face
            original: Original frame
            face_bbox: Tuple (x, y, w, h) of face location

        Returns:
            Blended frame
        """
        if self.config.luminance_blend:
            return self._luminance_adaptive_blend(swapped, original, face_bbox)
        else:
            return cv2.addWeighted(
                swapped,
                self.config.blend,
                original,
                1.0 - self.config.blend,
                0,
            )

    @staticmethod
    def _luminance_adaptive_blend(
        swapped: Frame,
        original: Frame,
        face_bbox: tuple,
    ) -> Frame:
        """
        Blend with luminance-adaptive alpha.

        Reduces blending amount when luminance difference is high.

        Args:
            swapped: Frame with swapped face
            original: Original frame
            face_bbox: Tuple (x, y, w, h)

        Returns:
            Blended frame
        """
        x, y, w, h = face_bbox
        fh, fw = original.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        if x2 <= x1 or y2 <= y1:
            # Invalid region, use fixed blend
            return cv2.addWeighted(swapped, 0.65, original, 0.35, 0)

        try:
            orig_lum = float(
                cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[:, :, 0].mean()
            )
            swap_lum = float(
                cv2.cvtColor(swapped[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[:, :, 0].mean()
            )
            lum_delta = abs(swap_lum - orig_lum)

            if lum_delta < 10.0:
                # Similar luminance, use normal blend
                return cv2.addWeighted(swapped, 0.65, original, 0.35, 0)

            # Adaptive blend: reduce when delta is high
            adaptive_blend = 0.65 * max(0.5, 1.0 - lum_delta / 255.0)
            return cv2.addWeighted(swapped, adaptive_blend, original, 1.0 - adaptive_blend, 0)
        except Exception:
            # Fallback to normal blend if LAB conversion fails
            return cv2.addWeighted(swapped, 0.65, original, 0.35, 0)


class OutputProcessor(FrameProcessor):
    """
    Output frame to sink (file, HTTP, WebSocket, etc.).

    Placeholder; actual output handled by pipeline coordinator.
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        """Initialize output processor."""
        self.config = config

    def process(self, frame: Frame) -> Frame:
        """No-op; output handled separately."""
        return frame
