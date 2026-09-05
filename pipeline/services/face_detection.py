"""
Face detection service for the Phantom pipeline.

Extracted from pipeline/face_analyser.py. Provides a clean interface
for detecting and analyzing faces in frames without global state.

Model cache priority:
1. /workspace/models/insightface/ (the instance disk)
2. ~/.insightface/models/ (default InsightFace path)
"""

import os
import threading
from typing import Any, Dict, List, Optional

import insightface

from pipeline.config import FaceSwapConfig
from pipeline.types import Frame, Detection
from pipeline.services import guards
from pipeline.logging import emit_status

# Model cache on the instance disk
_VOLUME_CACHE = '/workspace/models/insightface'
# Default InsightFace model cache path
_DEFAULT_CACHE = os.path.expanduser('~/.insightface')


def _get_insightface_root() -> str:
    """
    Resolve InsightFace model root directory.

    Checks /workspace/models first, falls back to default.

    Returns:
        Absolute path to InsightFace model root
    """
    if os.path.isdir(_VOLUME_CACHE):
        emit_status(f'Using workspace model cache: {_VOLUME_CACHE}', scope='FACE_DETECTOR')
        # insightface uses INSIGHTFACE_HOME env var to override root
        return os.path.dirname(_VOLUME_CACHE)  # parent of 'insightface/'
    emit_status(f'Using default model cache: {_DEFAULT_CACHE}', scope='FACE_DETECTOR')
    return _DEFAULT_CACHE


class FaceDetector:
    """
    Face detection using InsightFace's FaceAnalysis model.

    This service is thread-safe and maintains an internal cache of the
    face analysis model. Configuration is passed in the constructor.

    Checks /workspace/models/insightface/ before
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
        # Set for the duration of `detect_source`, which pins the detector size
        # rather than taking it from the capture preset. Guarded by its own lock
        # because `_lock` is held inside `_get_analyser` and is not reentrant.
        self._det_override: Optional[int] = None
        self._override_lock = threading.Lock()
        # What this model pack actually provides, probed on the first real
        # detection. Several guards silently become no-ops when their input is
        # absent, and a guard that never fires because its data is missing looks
        # exactly like one that never had cause to fire.
        self.capabilities: Dict[str, bool] = {}

    # Detector input is square and must be a multiple of 32 for the retinaface
    # backbone's stride. Values outside this range are not useful: below 256 the
    # detector starts missing faces at ordinary call framing, and above 640 it
    # costs more than it finds.
    _DET_MIN = 256
    _DET_MAX = 640
    _DET_STRIDE = 32

    # Detector input used when reviewing an uploaded source photo. Fixed, and
    # deliberately *not* the capture preset's `det_size`.
    #
    # A source is a still of arbitrary resolution and has nothing to do with the
    # webcam's, but `det_size` moves 320 / 448 / 640 across the presets and the
    # source guards ran at whichever happened to be loaded. That made the verdict
    # on a photo depend on when it was uploaded relative to `set_quality`: a face
    # in the background that 320 misses and 640 finds turns a clean photo into a
    # MULTIPLE_FACES rejection, and size, pose and landmarks all shift with it.
    # One flipped verdict then changes the set the identity outlier pass sees,
    # and leave-one-out is set-dependent — so a single difference cascades into
    # several rejections, on the same photos that were accepted a moment before.
    #
    # `_DET_MAX` rather than a middle value: a review runs once per upload, so
    # its cost does not matter, and failing to see a second face is the
    # expensive error here — it is the one that lets a stranger into the
    # identity every frame is swapped to.
    _SOURCE_DET_SIZE = _DET_MAX

    def _resolve_det_size(self) -> int:
        """Detector input size from config, snapped to a valid value."""
        override = self._det_override
        requested = (
            override if override is not None
            else int(getattr(self.config, 'det_size', 448) or 448)
        )
        clamped = max(self._DET_MIN, min(self._DET_MAX, requested))
        return clamped - (clamped % self._DET_STRIDE)

    def _get_analyser(self) -> Any:
        """
        Get or create the FaceAnalysis model (lazy initialization).

        Thread-safe. Model is cached after first access, and re-prepared if
        `det_size` changed — switching quality preset changes it, and the
        prepared size is baked in at `prepare()` time.

        Resolves model root to /workspace/models if available.
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

    def detect_source(self, frame: Frame) -> List[Detection]:
        """
        Detect faces in an uploaded source photo, at a fixed detector size.

        Same detection as `detect`, pinned to `_SOURCE_DET_SIZE` so a source
        review returns the same verdict whatever capture preset happens to be
        loaded. See the note on that constant for what varied before.

        The analyser is left prepared at the review size; the next `detect` sees
        the mismatch against `config.det_size` and re-prepares itself, so no
        restore step is needed here.

        Args:
            frame: Decoded source image

        Returns:
            List of Detection objects (may be empty if no faces found)
        """
        with self._override_lock:
            self._det_override = self._SOURCE_DET_SIZE
            try:
                return self.detect(frame)
            finally:
                self._det_override = None

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

            self._probe_once(raw_faces[0])

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
        Detect the primary face in a frame — the largest one.

        Previously the *leftmost* (`min(detections, key=lambda d: d.bbox.x)`),
        which is not a heuristic but an arbitrary tie-break: nothing about how
        images are composed makes the smallest x coordinate the subject, so it
        picked wrong whenever there was a second candidate — someone beside you,
        someone walking behind, a face on a television. It also meant the choice
        could change between frames as people moved, feeding the stabilizer
        alternating identities.

        Largest-face is a better rule but still fails silently when two faces
        are of comparable size, which is what the guards are for: see
        `pipeline.services.guards.check_frame`.

        Args:
            frame: Input frame as numpy array

        Returns:
            Detection of the largest face, or None if no face found
        """
        return self.select_primary(self.detect(frame))

    @staticmethod
    def select_primary(detections: List[Detection]) -> Optional[Detection]:
        """
        Pick the primary face from an existing detection list.

        Separate from `detect_one` so a caller that needs the full list — to
        count faces for a guard — does not have to run detection twice.

        Ties broken by bounding-box area, then by distance from the frame's left
        edge, purely so the result is deterministic for identical boxes.

        Args:
            detections: Faces found in one frame

        Returns:
            The largest detection, or None if the list is empty
        """
        if not detections:
            return None

        return max(detections, key=lambda d: (d.bbox.w * d.bbox.h, -d.bbox.x))

    def _probe_once(self, face: Any) -> None:
        """
        Record and report what this model pack provides, on first detection.

        Cheap enough to do unconditionally: it is a handful of getattr calls,
        once per process. Reported at info level because the answer changes how
        several guards behave and is otherwise invisible — `buffalo_l` bundles
        `1k3d68.onnx`, whose 3D-landmark model sets `face.pose`, but a pack
        trimmed with `allowed_modules` would not, and yaw would silently fall
        back to a keypoint approximation on a different scale.

        Args:
            face: A raw InsightFace Face from a successful detection
        """
        if self.capabilities:
            return

        self.capabilities = guards.probe_capabilities(face)
        emit_status(
            guards.describe_capabilities(self.capabilities),
            scope='FACE_DETECTOR',
        )

    def clear(self) -> None:
        """Clear the cached model (useful for memory cleanup)."""
        with self._lock:
            self._analyser = None
