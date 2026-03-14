"""
Face tracking service for the Phantom pipeline.

Extracted from pipeline/stream.py. Provides stateful face tracking
across video frames using OpenCV trackers.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from pipeline.types import Frame, Detection, Bbox


def make_tracker(tracker_name: str) -> Optional[cv2.Tracker]:
    """
    Create a tracker by name.

    Supported: 'kcf', 'mosse', 'csrt' (default)

    Args:
        tracker_name: Name of tracker to create

    Returns:
        Tracker instance or None if creation failed
    """
    name = tracker_name.lower()
    try:
        if name == 'kcf':
            creator = getattr(cv2, 'TrackerKCF_create', None) or getattr(cv2.legacy, 'TrackerKCF_create', None)
        elif name == 'mosse':
            creator = getattr(cv2.legacy, 'TrackerMOSSE_create', None)
        else:
            creator = getattr(cv2, 'TrackerCSRT_create', None) or getattr(cv2.legacy, 'TrackerCSRT_create', None)

        return creator() if creator else None
    except Exception as e:
        import sys
        print(f'[FaceTracking] make_tracker({tracker_name!r}) error: {type(e).__name__}: {e}', file=sys.stderr)
        return None


def _ema(current: np.ndarray, previous: Optional[np.ndarray], alpha: float) -> np.ndarray:
    """
    Exponential Moving Average smoothing.

    Args:
        current: Current frame keypoints
        previous: Previous frame keypoints (or None for first frame)
        alpha: Blend factor (0-1)

    Returns:
        Smoothed keypoints
    """
    if previous is None:
        return current.copy()
    return alpha * current + (1.0 - alpha) * previous


class FaceTrackerState:
    """
    Stateful face tracking across video frames.

    Tracks a single face using an OpenCV tracker and maintains:
    - Tracker state (bounding box, initialized flag)
    - Cached face detection
    - Smoothed keypoints (via EMA)

    Example:
        state = FaceTrackerState(tracker_type='csrt', ema_alpha=0.6)
        # On first detection:
        state.initialize(frame, detection)
        # On subsequent frames:
        ok = state.update(frame)
        if ok:
            bbox = state.get_bbox()
            kps = state.get_kps()
    """

    def __init__(self, tracker_type: str = 'csrt', ema_alpha: float = 0.6) -> None:
        """
        Initialize face tracker state.

        Args:
            tracker_type: Name of CV2 tracker to use ('kcf', 'mosse', 'csrt')
            ema_alpha: EMA smoothing factor for keypoints (0-1)
        """
        self.tracker_type = tracker_type
        self.ema_alpha = ema_alpha

        self._tracker: Optional[cv2.Tracker] = None
        self._initialized: bool = False
        self._bbox: Optional[Bbox] = None
        self._cached_face: Optional[Detection] = None
        self._prev_kps: Optional[np.ndarray] = None

    def initialize(self, frame: Frame, detection: Detection) -> bool:
        """
        Initialize tracker with a detected face.

        Should be called when a face is first detected.

        Args:
            frame: Current frame
            detection: Detection object with face location

        Returns:
            True if tracker initialized successfully
        """
        self._cached_face = detection
        self._bbox = detection.bbox

        # Smooth keypoints with EMA
        if detection.kps is not None:
            self._prev_kps = _ema(detection.kps, self._prev_kps, self.ema_alpha)
        else:
            self._prev_kps = None

        # Initialize CV2 tracker
        self._tracker = make_tracker(self.tracker_type)
        if self._tracker is None:
            self._initialized = False
            return False

        # CV2 tracker expects (x, y, w, h) format
        bbox_tuple = (self._bbox.x, self._bbox.y, self._bbox.w, self._bbox.h)
        try:
            self._tracker.init(frame, bbox_tuple)
            self._initialized = True
            return True
        except Exception as e:
            import sys
            print(f'[FaceTracking] tracker init error: {type(e).__name__}: {e}', file=sys.stderr)
            self._initialized = False
            return False

    def update(self, frame: Frame) -> bool:
        """
        Update tracker with new frame.

        Args:
            frame: Current frame

        Returns:
            True if track is valid, False if tracking failed
        """
        if not self._initialized or self._tracker is None:
            return False

        try:
            ok, bbox_raw = self._tracker.update(frame)
            if not ok:
                return False

            # Update bbox from tracker
            x, y, w, h = (
                int(bbox_raw[0]),
                int(bbox_raw[1]),
                int(bbox_raw[2]),
                int(bbox_raw[3]),
            )
            self._bbox = Bbox(x=x, y=y, w=w, h=h)
            return True
        except Exception as e:
            import sys
            print(f'[FaceTracking] tracker update error: {type(e).__name__}: {e}', file=sys.stderr)
            return False

    def get_bbox(self) -> Optional[Bbox]:
        """
        Get current tracked bounding box.

        Returns:
            Bbox object or None if not initialized
        """
        return self._bbox

    def get_kps(self) -> Optional[np.ndarray]:
        """
        Get current smoothed keypoints.

        Returns:
            Keypoints array or None
        """
        return self._prev_kps

    def get_cached_face(self) -> Optional[Detection]:
        """
        Get cached detection (face model from initialization).

        Returns:
            Detection object from last initialize() call
        """
        return self._cached_face

    def reset(self) -> None:
        """
        Reset tracker state (e.g., when face is lost).

        Clears all cached data but keeps configuration.
        """
        self._tracker = None
        self._initialized = False
        self._bbox = None
        self._cached_face = None
        self._prev_kps = None

    @property
    def is_initialized(self) -> bool:
        """Check if tracker is currently initialized."""
        return self._initialized

    @property
    def is_valid(self) -> bool:
        """Check if tracker is initialized and has valid bbox."""
        return self._initialized and self._bbox is not None
