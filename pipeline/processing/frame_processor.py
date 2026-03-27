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
import numpy as np

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

    def process(self, frame: Frame) -> Frame:
        """
        Enhance frame using GFPGAN.

        Args:
            frame: Input frame

        Returns:
            Enhanced frame (or original if enhancement unavailable)
        """
        if self.enhancer.available:
            try:
                return self.enhancer.enhance(frame)
            except Exception as e:
                emit_warning(f"Enhancement failed: {e}", scope='ENHANCER')
                return frame

        return frame


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
        except Exception as e:
            emit_warning(f'Luminance blend error: {type(e).__name__}: {e}', scope='BLENDER')
            return cv2.addWeighted(swapped, 0.65, original, 0.35, 0)


class PreprocessingProcessor(FrameProcessor):
    """
    Preprocess frames to handle poor lighting and low camera quality.

    Applies lightweight corrections before detection/swapping:
    - CLAHE: Adaptive histogram equalization to normalize uneven lighting
    - White balance: Gray-world algorithm to remove color casts
    - Denoise: Bilateral filter for edge-preserving noise reduction

    All operations run on the full frame and are fast enough for realtime.
    Controlled by config.preprocessing (bool toggle).
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


class ColorCorrectionProcessor(FrameProcessor):
    """
    Correct color mismatch between swapped face and target skin tone.

    Uses LAB color transfer to match the swapped face's color distribution
    to the original target face, then applies seamless cloning for
    boundary-free compositing. Essential for cross-skin-tone swaps
    (e.g. fair source onto dark target) that would otherwise look cartoonish.
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize color correction processor.

        Args:
            config: Configuration object (color_correction toggle)
        """
        self.config = config

    def process(self, frame: Frame) -> Frame:
        """No-op; actual correction done via correct()."""
        return frame

    def correct(
        self,
        swapped: Frame,
        original: Frame,
        face_bbox: tuple,
    ) -> Frame:
        """
        Apply color correction to the swapped face region.

        Pipeline:
        1. Extract face ROI from both frames
        2. LAB color transfer: match mean+std of each channel to original
        3. Feathered mask to avoid hard edges
        4. Seamless clone for final compositing

        Args:
            swapped: Frame with swapped face (from inswapper)
            original: Original unmodified frame
            face_bbox: Tuple (x, y, w, h) of face location

        Returns:
            Color-corrected frame
        """
        x, y, w, h = face_bbox
        fh, fw = original.shape[:2]

        # Clamp ROI to frame bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return swapped

        try:
            # --- Step 0: Fast bail-out when skin tones already match ---
            orig_roi = original[y1:y2, x1:x2].astype(np.float32)
            swap_roi = swapped[y1:y2, x1:x2].astype(np.float32)

            orig_lab = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2LAB)
            swap_lab = cv2.cvtColor(swap_roi, cv2.COLOR_BGR2LAB)

            # Euclidean distance between LAB means — below threshold means
            # skin tones are close enough that correction is unnecessary.
            lab_delta = float(np.sqrt(sum(
                (orig_lab[:, :, ch].mean() - swap_lab[:, :, ch].mean()) ** 2
                for ch in range(3)
            )))
            if lab_delta < 12.0:
                return swapped

            # --- Step 1: LAB color transfer on face region ---
            # Per-channel mean/std transfer
            for ch in range(3):
                o_mean, o_std = orig_lab[:, :, ch].mean(), orig_lab[:, :, ch].std()
                s_mean, s_std = swap_lab[:, :, ch].mean(), swap_lab[:, :, ch].std()

                if s_std < 1e-6:
                    continue

                # Shift and scale to match original distribution
                swap_lab[:, :, ch] = (swap_lab[:, :, ch] - s_mean) * (o_std / s_std) + o_mean

            swap_lab = np.clip(swap_lab, 0, 255)
            corrected_roi = cv2.cvtColor(swap_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

            # --- Step 2: Build elliptical feathered mask ---
            roi_h, roi_w = corrected_roi.shape[:2]
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            center = (roi_w // 2, roi_h // 2)
            axes = (int(roi_w * 0.4), int(roi_h * 0.4))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            # Feather the edges with Gaussian blur
            ksize = max(roi_w, roi_h) // 4
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            ksize = max(ksize, 3)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

            # --- Step 3: Alpha-blend corrected ROI using feathered mask ---
            mask_f = mask.astype(np.float32) / 255.0
            mask_3ch = np.stack([mask_f] * 3, axis=-1)

            blended_roi = (
                corrected_roi.astype(np.float32) * mask_3ch
                + swap_roi * (1.0 - mask_3ch)
            ).astype(np.uint8)

            # --- Step 4: Seamless clone into the frame ---
            # Build a full-frame mask for seamlessClone
            result = swapped.copy()
            full_mask = np.zeros((fh, fw), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask

            # seamlessClone center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Place the corrected ROI into a full-frame src image
            src_frame = swapped.copy()
            src_frame[y1:y2, x1:x2] = blended_roi

            result = cv2.seamlessClone(
                src_frame, original, full_mask, (cx, cy), cv2.NORMAL_CLONE
            )

            return result

        except Exception as e:
            emit_warning(f'Color correction error: {type(e).__name__}: {e}', scope='COLOR')
            return swapped


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
