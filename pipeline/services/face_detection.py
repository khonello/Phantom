"""
Face detection service for the Phantom pipeline.

Extracted from pipeline/face_analyser.py. Provides a clean interface
for detecting and analyzing faces in frames without global state.

Model cache priority:
1. /workspace/models/insightface/ (RunPod Network Volume)
2. ~/.insightface/models/ (default InsightFace path)
"""

import os
import threading
from typing import Any, List, Optional

import insightface

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Detection
from pipeline.logging import emit_status

# RunPod Network Volume model cache path
_RUNPOD_CACHE = '/workspace/models/insightface'
# Default InsightFace model cache path
_DEFAULT_CACHE = os.path.expanduser('~/.insightface')


def _get_insightface_root() -> str:
    """
    Resolve InsightFace model root directory.

    Checks RunPod Network Volume first, falls back to default.

    Returns:
        Absolute path to InsightFace model root
    """
    if os.path.isdir(_RUNPOD_CACHE):
        emit_status(f'Using RunPod model cache: {_RUNPOD_CACHE}', scope='FACE_DETECTOR')
        # insightface uses INSIGHTFACE_HOME env var to override root
        return os.path.dirname(_RUNPOD_CACHE)  # parent of 'insightface/'
    emit_status(f'Using default model cache: {_DEFAULT_CACHE}', scope='FACE_DETECTOR')
    return _DEFAULT_CACHE


class FaceDetector:
    """
    Face detection using InsightFace's FaceAnalysis model.

    This service is thread-safe and maintains an internal cache of the
    face analysis model. Configuration is passed in the constructor.

    Checks RunPod Network Volume (/workspace/models/insightface/) before
    downloading to the default InsightFace cache (~/.insightface/models/).

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
        Resolves model root to RunPod volume if available.
        """
        if self._analyser is None:
            with self._lock:
                if self._analyser is None:
                    root = _get_insightface_root()
                    self._analyser = insightface.app.FaceAnalysis(
                        name='buffalo_l',
                        root=root,
                        providers=self.config.execution_providers,
                    )
                    # det_thresh=0.35: lower than the default 0.5 to handle
                    # JPEG-compressed webcam frames and varied lighting.
                    self._analyser.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)
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
                det = Detection.from_insightface(face)
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
