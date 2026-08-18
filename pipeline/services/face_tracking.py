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
from collections import deque
from typing import Any, Deque, Optional

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
    # Default cosine below which consecutive frames are treated as different
    # people. The same face across two frames sits well above this even through
    # expression and moderate pose change; two different people sit well below.
    #
    # Set to fail toward resetting, because the two errors are not symmetric: a
    # needless reset costs one frame of unsmoothed output, while a missed one
    # blends two identities into the smoothing history and leaks a stranger back
    # out over the following frames.
    #
    # It is a guess, and the one most able to cost realism if wrong in the other
    # direction — see `config.guard_identity_sim`, which overrides it, and the
    # `identity_sim` row of the guard telemetry, which measures where the same
    # person actually lands.
    _IDENTITY_SIM = 0.35
    # Consecutive frames below the floor before the identity is believed to have
    # changed. One frame is not evidence: an embedding is computed from a crop
    # that can be motion-blurred, half-turned or badly lit for a single frame and
    # recover on the next, and resetting on that drops the landmark EMA *during
    # movement* — which is precisely when shimmer is most visible. A real change
    # of person persists; a bad crop does not.
    #
    # The cost of waiting is bounded and small. Any frame containing two faces is
    # already refused by the multi-face guard before it reaches the stabilizer,
    # so this is a backstop against detection flicker rather than the primary
    # defence. And `reset()` discards the smoothing history wholesale, so the few
    # frames blended before confirmation are thrown away rather than lingering.
    # Evaluated over a sliding window rather than a run of consecutive frames.
    # Consecutive counting is defeated by alternation: a detector flickering
    # between two people gives good, bad, good, bad, and a consecutive counter
    # is reset by every good frame, so it never fires. Counting within a window
    # catches both shapes — a sustained change (3 in a row) and a flicker (3 of
    # the last 6) — while still ignoring an isolated bad crop.
    _IDENTITY_CONFIRM = 3
    _IDENTITY_WINDOW = 6

    def __init__(self, alpha: float = 0.6, identity_sim: Optional[float] = None) -> None:
        """
        Initialize the stabilizer.

        Args:
            alpha: EMA factor (0.0 = maximum smoothing, 1.0 = disabled)
            identity_sim: Cosine floor for treating a face as the same person.
                          None uses the class default
        """
        self.alpha = alpha
        self.identity_sim = self._IDENTITY_SIM if identity_sim is None else identity_sim
        self._prev_kps: Optional[npt.NDArray[Any]] = None
        self._prev_landmarks: Optional[npt.NDArray[Any]] = None
        self._prev_embedding: Optional[npt.NDArray[Any]] = None
        self._missing = 0
        self._identity_recent: Deque[bool] = deque(maxlen=self._IDENTITY_WINDOW)
        # Most recent frame-to-frame identity cosine, or None when there was
        # nothing to compare. Read by the guard telemetry: `_IDENTITY_SIM` is a
        # guess, and if it sits above where the same person actually lands under
        # motion blur, this resets every frame and the shimmer comes back.
        self.last_similarity: Optional[float] = None

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

        self._remember_identity(face)
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

        Two independent tests, because each misses what the other catches:

        - **Centroid jump** — a re-acquisition after a dropout, or a face that
          moved impossibly far for one frame. Blending across that would drag
          the alignment from the old position for several frames.
        - **Identity change** — the selected face is a *different person*. This
          is what the jump test cannot see: two people standing still beside
          each other are only a few pixels apart, so whichever one is selected
          flipping between frames produces no jump at all, and the two get
          blended into one smoothed face.
        """
        if self._prev_kps is None:
            return False

        if self._identity_changed(face):
            return True

        bbox = getattr(face, 'bbox', None)
        if bbox is None or len(bbox) < 4:
            return False

        face_width = float(bbox[2] - bbox[0])
        if face_width <= 0:
            return False

        jump = float(np.linalg.norm(kps.mean(axis=0) - self._prev_kps.mean(axis=0)))
        return jump > self._JUMP_RATIO * face_width

    def _identity_changed(self, face: Face) -> bool:
        """
        Whether this face is a different person from the one being smoothed.

        Uses the recognition embedding InsightFace already computed as part of
        detection, so this costs a dot product. Pure: the remembered identity is
        updated by `_remember_identity` after the reset decision, not here, so a
        reset cannot swallow the update and leave the next frame with nothing to
        compare against — which would make an alternating pair of faces reset on
        only every second frame.

        Faces without an embedding are treated as continuous. The geometric test
        still applies, and inferring "different person" from missing data would
        reset constantly.
        """
        self.last_similarity = None

        previous = self._prev_embedding
        if previous is None:
            return False

        current = self._embedding_of(face)
        if current is None or current.shape != previous.shape:
            return False

        norms = float(np.linalg.norm(current) * np.linalg.norm(previous))
        if norms < 1e-9:
            return False

        similarity = float(np.dot(current, previous) / norms)
        self.last_similarity = similarity

        self._identity_recent.append(similarity < self.identity_sim)
        return sum(self._identity_recent) >= self._IDENTITY_CONFIRM

    @staticmethod
    def _embedding_of(face: Face) -> Optional[npt.NDArray[Any]]:
        """Recognition embedding as a flat float32 array, or None if absent."""
        embedding = getattr(face, 'normed_embedding', None)
        if embedding is None:
            return None
        array = np.asarray(embedding, dtype=np.float32).ravel()
        return array if array.size else None

    def _remember_identity(self, face: Face) -> None:
        """
        Record who is currently being smoothed.

        Skipped while a suspected change is unconfirmed, which is what makes the
        confirmation count work at all: comparing each frame against the one
        before it would adopt the new face immediately, similarity would return
        to ~1.0 on the next frame, and the counter could never reach its limit.
        Holding the last *confirmed* identity means the comparison keeps asking
        the right question — "is this still the person I was following?"

        Kept across `reset()` on purpose — see the note there.
        """
        if self._identity_recent and self._identity_recent[-1]:
            return

        current = self._embedding_of(face)
        if current is not None:
            self._prev_embedding = current

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
        """
        Drop all smoothing state (face lost, source changed, etc.).

        The remembered identity deliberately survives. It is not smoothing state
        — it is the answer to "who was I following", and that question still has
        the same answer after a dropout. Clearing it would mean the frame after a
        reset has nothing to compare against, so a switch between two people
        would be caught on only every second frame; and after a genuine dropout,
        keeping it is what lets the same person returning resume smoothly while a
        different person still triggers a reset.
        """
        self._prev_kps = None
        self._prev_landmarks = None
        self._missing = 0
        # Cleared so the frame after a reset adopts whoever is now in shot,
        # rather than still being measured against the person who just left.
        self._identity_recent.clear()
