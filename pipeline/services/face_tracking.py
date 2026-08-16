"""
Landmark stabilization for the Phantom pipeline.

Replaces the previous OpenCV-tracker approach (FaceTrackerState/make_tracker).
Detection runs on every frame, so a correlation tracker added latency without
contributing anything — and worse, it caused the swap to be warped with a
cached, stale Face object.

What actually needs smoothing is `face.kps`, since InsightFace's INSwapper
derives its affine warp from those five points. Jitter there is what makes a
swapped face shimmer and drift relative to the head.
"""

import copy
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pipeline.types import Face


def _ema(
    current: npt.NDArray[Any],
    previous: Optional[npt.NDArray[Any]],
    alpha: float,
) -> npt.NDArray[Any]:
    """
    Exponential Moving Average smoothing.

    Args:
        current: Current frame keypoints
        previous: Previous frame keypoints (or None for first frame)
        alpha: Blend factor (0.0 = maximum smoothing, 1.0 = no smoothing)

    Returns:
        Smoothed keypoints
    """
    if previous is None:
        return current.copy()
    if previous.shape != current.shape:
        # Landmark model changed shape between frames — cannot blend safely
        return current.copy()
    return alpha * current + (1.0 - alpha) * previous


def _copy_face(face: Face) -> Face:
    """
    Shallow-copy an InsightFace Face so smoothed landmarks can be attached
    without mutating the detector's object.

    `Face` subclasses dict, so `Face(dict(face))` produces an independent
    mapping. The numpy arrays inside are shared by reference, which is safe
    here because the stabilizer only ever *replaces* those attributes with
    new arrays — it never writes into them in place.
    """
    try:
        return Face(dict(face))
    except Exception:
        return copy.copy(face)


class LandmarkStabilizer:
    """
    Temporally smooths a face's landmarks across frames.

    Smooths both `kps` (the 5 points that drive the swap warp) and
    `landmark_2d_106` (which drives the compositing mask) with the same
    factor, so the warp and the mask never disagree with each other.

    Only meaningful for a single tracked subject: with multiple faces the
    per-frame detection order is unstable, so the caller must bypass this
    (see `ProcessingPipeline._process_and_emit`).

    Example:
        stabilizer = LandmarkStabilizer(alpha=0.6)
        smoothed = stabilizer.stabilize(detection.face)
    """

    # Consecutive missing frames after which smoothing state is dropped.
    _MISSING_LIMIT = 3
    # Centroid jump beyond this fraction of face width is treated as a
    # different face (or a re-acquisition) rather than motion to smooth.
    _JUMP_RATIO = 0.5

    def __init__(self, alpha: float = 0.6) -> None:
        """
        Initialize the stabilizer.

        Args:
            alpha: EMA factor (0.0 = maximum smoothing, 1.0 = disabled)
        """
        self.alpha = alpha
        self._prev_kps: Optional[npt.NDArray[Any]] = None
        self._prev_landmarks: Optional[npt.NDArray[Any]] = None
        self._missing = 0

    def stabilize(self, face: Face) -> Face:
        """
        Return a copy of `face` with temporally smoothed landmarks.

        Args:
            face: Fresh detection for the current frame

        Returns:
            A Face copy carrying smoothed `kps` and `landmark_2d_106`.
            Returns `face` unchanged when smoothing is disabled or the
            landmarks are unusable.
        """
        kps = getattr(face, 'kps', None)
        if self.alpha >= 1.0 or kps is None or len(kps) == 0:
            self._missing = 0
            return face

        kps = np.asarray(kps, dtype=np.float32)

        if self._should_reset(face, kps):
            self.reset()

        self._missing = 0

        smoothed_kps = _ema(kps, self._prev_kps, self.alpha)
        self._prev_kps = smoothed_kps

        stabilized = _copy_face(face)
        stabilized.kps = smoothed_kps

        landmarks = getattr(face, 'landmark_2d_106', None)
        if landmarks is not None and len(landmarks) > 0:
            landmarks = np.asarray(landmarks, dtype=np.float32)
            smoothed_landmarks = _ema(landmarks, self._prev_landmarks, self.alpha)
            self._prev_landmarks = smoothed_landmarks
            stabilized.landmark_2d_106 = smoothed_landmarks

        return stabilized

    def _should_reset(self, face: Face, kps: npt.NDArray[Any]) -> bool:
        """
        Decide whether the incoming face is a continuation of the tracked one.

        A large centroid jump means either a re-acquisition after a dropout or
        a different person. Blending across that would drag the alignment from
        the old position for several frames.
        """
        if self._prev_kps is None:
            return False

        bbox = getattr(face, 'bbox', None)
        if bbox is None or len(bbox) < 4:
            return False

        face_width = float(bbox[2] - bbox[0])
        if face_width <= 0:
            return False

        jump = float(np.linalg.norm(kps.mean(axis=0) - self._prev_kps.mean(axis=0)))
        return jump > self._JUMP_RATIO * face_width

    def mark_missing(self) -> None:
        """
        Record a frame in which no face was detected.

        After `_MISSING_LIMIT` consecutive misses the smoothing state is
        dropped, so re-acquisition starts clean instead of interpolating
        from a stale position.
        """
        self._missing += 1
        if self._missing >= self._MISSING_LIMIT:
            self.reset()

    def reset(self) -> None:
        """Drop all smoothing state (face lost, source changed, etc.)."""
        self._prev_kps = None
        self._prev_landmarks = None
        self._missing = 0
