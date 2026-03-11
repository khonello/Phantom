"""
Face detection service for the Phantom pipeline.

Extracted from pipeline/face_analyser.py. Provides a clean interface
for detecting and analyzing faces in frames without global state.
"""

import threading
from typing import Any, List, Optional

import insightface
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Detection, Bbox


class FaceDetector:
    """
    Face detection using InsightFace's FaceAnalysis model.

    This service is thread-safe and maintains an internal cache of the
    face analysis model. Configuration is passed in the constructor.

    Example:
        detector = FaceDetector(CONFIG)
        detections = detector.detect(frame)
        for det in detections:
            print(f"Face at {det.bbox}")
    """

    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize the face detector.

        Args:
            config: FaceSwapConfig with execution_providers configured
        """
        self.config = config
        self._analyser: Optional[Any] = None
        self._lock = threading.Lock()

    def _get_analyser(self) -> Any:
        """
        Get or create the FaceAnalysis model (lazy initialization).

        Thread-safe. Model is cached after first access.
        """
        if self._analyser is None:
            with self._lock:
                if self._analyser is None:
                    self._analyser = insightface.app.FaceAnalysis(
                        name='buffalo_l',
                        providers=self.config.execution_providers,
                    )
                    self._analyser.prepare(ctx_id=0, det_size=(640, 640))
        return self._analyser

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect all faces in a frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of Detection objects (may be empty if no faces found)
        """
        analyser = self._get_analyser()
        try:
            raw_faces = analyser.get(frame)
            if not raw_faces:
                return []

            detections = []
            for face in raw_faces:
                bbox = Bbox.from_insightface(face.bbox)
                det = Detection(
                    face=face,
                    bbox=bbox,
                    kps=face.kps if hasattr(face, 'kps') else np.array([]),
                    confidence=float(face.score) if hasattr(face, 'score') else 0.0,
                )
                detections.append(det)
            return detections
        except IndexError:
            return []
        except Exception:
            return []

    def detect_one(self, frame: Frame) -> Optional[Detection]:
        """
        Detect a single face in a frame (leftmost face).

        Args:
            frame: Input frame as numpy array

        Returns:
            Detection of the leftmost face, or None if no face found
        """
        detections = self.detect(frame)
        if not detections:
            return None

        # Return leftmost face (smallest x coordinate)
        return min(detections, key=lambda d: d.bbox.x)

    def clear(self) -> None:
        """Clear the cached model (useful for memory cleanup)."""
        with self._lock:
            self._analyser = None
