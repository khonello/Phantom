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
        self._prepared_size: int = 0
        self._lock = threading.Lock()

    # Detector input is square and must be a multiple of 32 for the retinaface
    # backbone's stride. Values outside this range are not useful: below 256 the
    # detector starts missing faces at ordinary call framing, and above 640 it
    # costs more than it finds.
    _DET_MIN = 256
    _DET_MAX = 640
    _DET_STRIDE = 32

    def _resolve_det_size(self) -> int:
        """Detector input size from config, snapped to a valid value."""
        requested = int(getattr(self.config, 'det_size', 448) or 448)
        clamped = max(self._DET_MIN, min(self._DET_MAX, requested))
        return clamped - (clamped % self._DET_STRIDE)

    def _get_analyser(self) -> Any:
        """
        Get or create the FaceAnalysis model (lazy initialization).

        Thread-safe. Model is cached after first access, and re-prepared if
        `det_size` changed — switching quality preset changes it, and the
        prepared size is baked in at `prepare()` time.

        Resolves model root to RunPod volume if available.
        """
        size = self._resolve_det_size()

        if self._analyser is not None and self._prepared_size == size:
            return self._analyser

        with self._lock:
            if self._analyser is None:
                root = _get_insightface_root()
                self._analyser = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    root=root,
                    providers=self.config.execution_providers,
                )

            if self._prepared_size != size:
                # det_thresh=0.35: lower than the default 0.5 to handle
                # JPEG-compressed webcam frames and varied lighting.
                self._analyser.prepare(ctx_id=0, det_size=(size, size), det_thresh=0.35)
                self._prepared_size = size

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
        except IndexError as e:
            emit_status(f'Detection IndexError: {e}', scope='FACE_DETECTOR', level='error')
            return []
        except Exception as e:
            emit_status(f'Detection error: {type(e).__name__}: {e}', scope='FACE_DETECTOR', level='error')
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
