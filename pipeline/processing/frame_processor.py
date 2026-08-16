"""
Composable frame processors for the Phantom pipeline.

Each processor encapsulates a single processing step and can be chained
together. Processors are stateless (or maintain only internal state) and don't
depend on global variables.

Abstract base:
    FrameProcessor - process(frame: Frame) -> Frame

Implementations:
    PreprocessingProcessor - Normalize lighting/colour before detection
    DetectionProcessor     - Detect faces in frame
    SwappingProcessor      - Swap detected faces
    OutputProcessor        - Hand off to a sink

Compositing (masking, colour, detail, grain) lives in
`pipeline/processing/compositor.py`, not here — it operates in aligned face
space rather than on whole frames, so it does not fit the FrameProcessor
contract.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Face, Detection, Matrix
from pipeline.services.face_detection import FaceDetector
from pipeline.services.face_swapping import FaceSwapper
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
        self._frame_count = 0
        # State-change tracking: None = unknown (first frame), True/False = last known state
        self._face_present: Optional[bool] = None
        # Consecutive frames with no detection — emit warning after threshold to avoid flicker
        self._no_face_streak = 0
        self._NO_FACE_THRESHOLD = 3

    def process(self, frame: Frame) -> Frame:
        """
        Detect faces in frame.

        Args:
            frame: Input frame

        Returns:
            Frame unchanged; detections stored in self.latest_detections
        """
        self._frame_count += 1
        try:
            if self.config.many_faces:
                self.latest_detections = self.detector.detect(frame)
            else:
                det = self.detector.detect_one(frame)
                self.latest_detections = [det] if det else []
        except Exception as e:
            emit_warning(f"Detection failed: {e}", scope='DETECTION')
            self.latest_detections = []

        if self.latest_detections:
            self._no_face_streak = 0
            if self._face_present is not True:
                self._face_present = True
                emit_status('Face detected — swap active', scope='DETECTION')
        else:
            self._no_face_streak += 1
            # Emit only when streak crosses the threshold (avoids badge flicker)
            if self._no_face_streak == self._NO_FACE_THRESHOLD:
                self._face_present = False
                emit_warning(
                    'No face detected — ensure face is clearly visible and well-lit',
                    scope='DETECTION',
                )

        return frame


class SwappingProcessor(FrameProcessor):
    """
    Swap detected faces using FaceSwapper service.

    Owns the source face. Prefers the aligned form of the swap so the
    compositor can do the pasting; falls back to InsightFace's own
    compositing when the aligned form is unavailable.
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
        self.source_face: Optional[Face] = None

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

        Actual swapping is done via swap_aligned() / swap_pasted().

        Args:
            frame: Input frame

        Returns:
            Frame unchanged
        """
        return frame

    def swap_aligned(
        self,
        frame: Frame,
        face: Face,
    ) -> Optional[Tuple[Frame, Matrix]]:
        """
        Generate the swapped face as an aligned crop plus its affine.

        Args:
            frame: Frame containing the face
            face: Fresh detection for this frame

        Returns:
            (aligned_crop, matrix), or None if unavailable
        """
        if self.source_face is None:
            return None
        return self.swapper.swap_aligned(self.source_face, face, frame)

    def swap_pasted(self, frame: Frame, face: Face) -> Frame:
        """
        Fallback swap using InsightFace's own compositing.

        Args:
            frame: Frame containing the face
            face: Fresh detection for this frame

        Returns:
            Frame with swapped face
        """
        if self.source_face is None:
            return frame

        try:
            return self.swapper.swap(self.source_face, face, frame)
        except Exception as e:
            emit_warning(f"Swap failed: {e}", scope='SWAPPER')
            return frame

    def reset(self) -> None:
        """Clear source face."""
        self.source_face = None
        self.database.clear()


class PreprocessingProcessor(FrameProcessor):
    """
    Preprocess frames to handle poor lighting and low camera quality.

    Applies lightweight corrections before detection/swapping:
    - CLAHE: Adaptive histogram equalization to normalize uneven lighting
    - White balance: Gray-world algorithm to remove color casts
    - Denoise: Bilateral filter for edge-preserving noise reduction

    All operations run on the full frame and are fast enough for realtime.
    Controlled by config.preprocessing (bool toggle), which defaults off:
    these corrections change the whole image, so the output stops looking
    like the operator's real camera.
    """

    # CLAHE parameters
    _CLAHE_CLIP = 2.0
    _CLAHE_GRID = (8, 8)

    # Bilateral filter parameters (edge-preserving denoise)
    _BILATERAL_D = 5
    _BILATERAL_SIGMA_COLOR = 50
    _BILATERAL_SIGMA_SPACE = 50

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize preprocessing processor.

        Args:
            config: Configuration object (preprocessing toggle)
        """
        self.config = config
        self._clahe = cv2.createCLAHE(
            clipLimit=self._CLAHE_CLIP,
            tileGridSize=self._CLAHE_GRID,
        )

    def process(self, frame: Frame) -> Frame:
        """
        Apply preprocessing corrections to input frame.

        Args:
            frame: Raw camera frame

        Returns:
            Corrected frame with normalized lighting, white balance, and reduced noise
        """
        if not self.config.preprocessing:
            return frame

        try:
            frame = self._apply_clahe(frame)
            frame = self._apply_white_balance(frame)
            frame = self._apply_denoise(frame)
            return frame
        except Exception as e:
            emit_warning(f'Preprocessing error: {type(e).__name__}: {e}', scope='PREPROCESS')
            return frame

    def _apply_clahe(self, frame: Frame) -> Frame:
        """
        Apply CLAHE to the L channel in LAB space.

        Normalizes brightness adaptively across the frame — handles
        shadows on one side of the face, overexposed highlights, etc.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_white_balance(frame: Frame) -> Frame:
        """
        Gray-world white balance correction.

        Assumes the average color of a scene should be neutral gray.
        Removes color casts from artificial lighting (fluorescent green,
        tungsten orange, LED blue, etc.).
        """
        avg_b = frame[:, :, 0].mean()
        avg_g = frame[:, :, 1].mean()
        avg_r = frame[:, :, 2].mean()
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        if avg_b < 1 or avg_g < 1 or avg_r < 1:
            return frame

        result = frame.astype(np.float32)
        result[:, :, 0] *= avg_all / avg_b
        result[:, :, 1] *= avg_all / avg_g
        result[:, :, 2] *= avg_all / avg_r
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_denoise(self, frame: Frame) -> Frame:
        """
        Bilateral filter for edge-preserving noise reduction.

        Smooths noise/grain while keeping face edges sharp — important
        for detection accuracy and swap quality on cheap webcams.
        """
        return cv2.bilateralFilter(
            frame,
            self._BILATERAL_D,
            self._BILATERAL_SIGMA_COLOR,
            self._BILATERAL_SIGMA_SPACE,
        )


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
