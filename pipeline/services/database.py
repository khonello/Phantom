"""
Face database service for the Phantom pipeline.

Handles caching of face embeddings, averaging multiple faces,
and loading pre-saved embeddings. Extracted from face_analyser.py.
"""

import hashlib
import os
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Bbox, Detection, Face, Frame
from pipeline.services import guards
from pipeline.services.face_detection import FaceDetector


# Weights for `_texture_score`. **Starting points chosen on the design target,
# not measured** — the same footing the restoration presets are on. Sharpness
# leads because a soft photograph has no high-frequency band to extract at all,
# and size follows it because a small face has one that is real but thin.
# Frontality matters less than either: the canonical warp is a similarity
# transform and does not correct yaw, so an angled source yields a foreshortened
# map — but selection can only choose among what was uploaded, and refusing an
# angled source is the guards' job, not this one's.
_TEXTURE_WEIGHTS = {
    'sharpness': 0.40,
    'size': 0.30,
    'frontality': 0.20,
    'exposure': 0.10,
}

# Laplacian variance at which the sharpness term is worth half its weight. Ten
# times `guard_min_sharpness`'s default floor of 40 — the guard asks "is this
# photo usable at all", and this asks "which of these usable photos is best",
# which is a question about the top of the range rather than the bottom.
_SHARPNESS_HALF = 400.0

# Face extent, shorter side, at which the size term saturates. Beyond this the
# canonical crop is downsampling real detail rather than gaining any.
_SIZE_FULL = 400.0


def _texture_score(frame: Frame, detection: Detection) -> float:
    """
    How suitable one source image is as the texture donor, in [0, 1].

    Called from `_review_image`, where the frame has just been read and the
    detection has just been made — see `select_texture_source` for why it lives
    there rather than in a selector of its own.

    Occlusion is **not** scored, and its absence is deliberate rather than
    pending: the source guards do not run XSeg, so there is no occlusion signal
    on this path to reuse, and inventing one here would mean a second masking
    definition that could disagree with the one the extractor uses.

    Args:
        frame: The source image, BGR
        detection: Its primary detection

    Returns:
        Weighted score in [0, 1]; higher is a better texture donor
    """
    bbox = detection.bbox

    # The face, not the frame — `guards.sharpness` already takes the bbox for
    # exactly this reason, since a portrait's blurred background is the point of
    # the photograph rather than a defect in it.
    variance = guards.sharpness(frame, bbox)
    sharpness = variance / (variance + _SHARPNESS_HALF)

    size = min(1.0, min(bbox.w, bbox.h) / _SIZE_FULL)

    yaw = guards.estimate_yaw(detection)
    # An unreadable pose scores as neutral rather than as frontal. Scoring it
    # frontal would let a model pack without `pose` promote every image to the
    # top of this term, which is a silent change of behaviour with the pack.
    frontality = 0.5 if yaw is None else max(0.0, 1.0 - abs(yaw) / 90.0)

    return float(
        _TEXTURE_WEIGHTS['sharpness'] * sharpness
        + _TEXTURE_WEIGHTS['size'] * size
        + _TEXTURE_WEIGHTS['frontality'] * frontality
        + _TEXTURE_WEIGHTS['exposure'] * _exposure_score(frame, bbox)
    )


def _exposure_score(frame: Frame, bbox: Bbox) -> float:
    """
    How much of the face is neither crushed nor blown out, in [0, 1].

    Clipped pixels carry no texture in either direction — a blown-out cheek is
    flat white and a crushed shadow is flat black, and no amount of high-pass
    recovers detail from either. This is the cheapest possible proxy for
    "well lit" and is not a substitute for looking at the photograph.
    """
    height, width = frame.shape[:2]
    box = bbox.clip_to_frame((height, width))
    if box.w < 8 or box.h < 8:
        return 0.5

    gray = cv2.cvtColor(
        frame[box.y:box.y + box.h, box.x:box.x + box.w], cv2.COLOR_BGR2GRAY,
    )
    clipped = int(np.count_nonzero((gray <= 4) | (gray >= 251)))
    return float(1.0 - clipped / gray.size)


@dataclass
class SourceReview:
    """
    Result of validating a batch of source images.

    Carries per-image outcomes rather than a single verdict, because this is an
    upload flow with a person present: they need to know *which* photo to
    replace, not that something was wrong.
    """

    accepted: List[str] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)  # (path, reason code)
    messages: Dict[str, str] = field(default_factory=dict)         # path -> explanation

    @property
    def ok(self) -> bool:
        """True if at least one image survived and none were rejected."""
        return bool(self.accepted) and not self.rejected

    @property
    def usable(self) -> bool:
        """True if anything at all can be built from this batch."""
        return bool(self.accepted)

    def reject(self, path: str, result: 'guards.GuardResult') -> None:
        """Record a rejection and its explanation."""
        self.rejected.append((path, result.reason))
        self.messages[path] = result.message

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the API, one entry per image."""
        return {
            'ok': self.ok,
            'accepted': list(self.accepted),
            'rejected': [
                {
                    'path': path,
                    'name': os.path.basename(path),
                    'reason': reason,
                    'message': self.messages.get(path, ''),
                }
                for path, reason in self.rejected
            ],
        }


class FaceDatabase:
    """
    In-memory cache of face embeddings.

    Handles:
    - Loading and caching embeddings from image files
    - Loading pre-computed .npy embeddings
    - Averaging multiple faces into a single embedding
    - Validating source images against the input guards
    - Clearing cache on demand

    Example:
        db = FaceDatabase(detector, CONFIG)
        review = db.review_sources(['face1.jpg', 'face2.jpg'])
        source_face = db.get_source_face(review.accepted)
        db.clear()  # cleanup
    """

    def __init__(
        self,
        detector: FaceDetector,
        config: Optional[FaceSwapConfig] = None,
    ) -> None:
        """
        Initialize face database.

        Args:
            detector: FaceDetector instance for face extraction
            config: Guard thresholds. Optional so existing callers that only
                    want embeddings keep working; guards are skipped without it
        """
        self.detector = detector
        self.config = config
        self._cache: Dict[str, Face] = {}

        # Texture suitability per accepted source, recorded during the review
        # that already read and detected every image. Scoring here rather than
        # in a selector of its own is the whole reason this is thirty lines:
        # the frame is in hand, the detection is in hand, and re-reading four
        # photos to ask a question the review could have answered is work for
        # nothing. Keyed by path, cleared with the cache.
        self._texture_scores: Dict[str, float] = {}

    @staticmethod
    def _cache_key(image_path: str) -> str:
        """
        Cache key for a source image: path, plus what is at that path now.

        Keying on the path alone is wrong here specifically because uploads are
        written to `uploads/<original filename>`. Two different photos named
        `IMG_0001.jpg` — which is what phones and cameras actually produce —
        land on the same path, and the second upload returned the **first
        person's embedding**. That is the confidently-wrong output the source
        guards exist to prevent, arriving underneath them: the guards check the
        image that was uploaded while the swap uses the face that was cached.

        **Hashed, not stat'd.** Modification time and size look sufficient and
        are not: filesystem timestamp granularity is coarser than the writes it
        has to separate — on Windows the clock behind `st_mtime_ns` advances in
        milliseconds — so two uploads close together can produce byte-identical
        keys for different photos. That failed under the test suite the moment
        the two writes landed in the same tick, which is the same thing that
        would happen to a person clicking twice.

        The cost is reading the file, which is bounded: source images are
        capped at 6 MB and this runs when a source is set, not per frame.

        A file that cannot be read falls back to the bare path, which is no
        worse than the behaviour this replaced.

        Args:
            image_path: Path to the source image

        Returns:
            A key that changes when the file's contents change
        """
        try:
            digest = hashlib.blake2b(digest_size=16)
            with open(image_path, 'rb') as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b''):
                    digest.update(block)
            return '{}:{}'.format(image_path, digest.hexdigest())
        except OSError:
            return image_path

    def get_source_face(self, paths: List[str]) -> Optional[Face]:
        """
        Get a source face from one or more paths.

        If multiple paths, returns averaged face embedding.
        Handles both image files (.jpg, .png) and embeddings (.npy).

        Args:
            paths: List of image or .npy file paths

        Returns:
            Face object with averaged embedding, or None if no valid faces
        """
        if not paths:
            return None

        faces = []

        for path in paths:
            if path.lower().endswith('.npy'):
                face = self._load_embedding(path)
            else:
                face = self._extract_from_image(path)

            if face is not None:
                faces.append(face)

        if not faces:
            return None

        # Return single face or average of multiple
        if len(faces) == 1:
            return faces[0]

        return self._average_faces(faces)

    def select_texture_source(
        self,
        paths: List[str],
    ) -> Optional[Tuple[str, Face]]:
        """
        Pick the single image to lift skin texture from.

        **Deliberately not the averaged identity.** Every accepted image feeds
        the embedding, because identity is a distributed representation and
        averaging it is sound. High-frequency texture is not — it is spatially
        localised, so blending detail maps taken at different angles, focal
        lengths and expressions makes misaligned pores cancel rather than
        reinforce. One image, chosen on its own merits.

        The merits are already measured. `_review_image` had the frame and the
        detection in hand and recorded a score there; this is the `max()` over
        them. Nothing is re-read and nothing is re-detected.

        Args:
            paths: Accepted source paths, from `SourceReview.accepted`

        Returns:
            (path, face) for the best image, or None if none of them is an
            image this can score — an all-`.npy` source set is the ordinary
            case, and it has no pixels to extract from.
        """
        best: Optional[Tuple[str, Face]] = None
        best_score = -1.0

        for path in paths:
            score = self._texture_scores.get(path)
            if score is None:
                continue
            face = self._cache.get(self._cache_key(path))
            if face is None:
                continue
            if score > best_score:
                best_score = score
                best = (path, face)

        return best

    def review_sources(self, paths: List[str]) -> SourceReview:
        """
        Validate source images against every source guard.

        Runs before any embedding is built, so a rejected photo never
        contributes to the identity. Two passes, because they need different
        information: the per-image guards look at one photo at a time, and the
        outlier check can only run once the surviving embeddings exist.

        `.npy` embeddings are accepted without per-image checks — there is no
        image to measure — but they do take part in the outlier comparison.

        Args:
            paths: Source image or .npy paths, in upload order

        Returns:
            SourceReview with per-image outcomes and explanations
        """
        review = SourceReview()

        if self.config is None:
            review.accepted = list(paths)
            return review

        embeddings: Dict[str, Any] = {}

        for path in paths:
            if path.lower().endswith('.npy'):
                face = self._load_embedding(path)
                if face is None:
                    review.reject(path, guards.GuardResult.failed(guards.UNREADABLE))
                    continue
                review.accepted.append(path)
                embeddings[path] = face.normed_embedding
                continue

            result, face = self._review_image(path)
            if not result.ok:
                review.reject(path, result)
                continue

            review.accepted.append(path)
            if face is not None and hasattr(face, 'normed_embedding'):
                embeddings[path] = face.normed_embedding

        self._review_identity(review, embeddings)
        return review

    def _review_image(
        self,
        image_path: str,
    ) -> Tuple['guards.GuardResult', Optional[Face]]:
        """
        Apply the per-image source guards to one file.

        Returns:
            (result, face) — the face is only present when the guards passed
        """
        assert self.config is not None

        if not os.path.exists(image_path):
            return guards.GuardResult.failed(guards.UNREADABLE, 'file not found'), None

        try:
            frame = cv2.imread(image_path)
        except Exception as e:
            return guards.GuardResult.failed(
                guards.UNREADABLE, f'{type(e).__name__}: {e}',
            ), None

        if frame is None:
            return guards.GuardResult.failed(guards.UNREADABLE), None

        # The full list, not `detect_one`: counting faces is the multi-face guard.
        # `detect_source` rather than `detect`, so the verdict on a photo does
        # not depend on which capture preset was loaded when it was uploaded.
        detections = self.detector.detect_source(frame)
        result = guards.check_source(self.config, frame, detections)
        if not result.ok:
            return result, None

        primary = self.detector.select_primary(detections)
        if primary is None:
            return guards.GuardResult.failed(guards.NO_FACE), None

        # Cache it so `get_source_face` does not detect the same file twice.
        self._cache[self._cache_key(image_path)] = primary.face
        self._texture_scores[image_path] = _texture_score(frame, primary)
        return guards.GuardResult.passed(), primary.face

    def _review_identity(
        self,
        review: SourceReview,
        embeddings: Dict[str, Any],
    ) -> None:
        """
        Reject accepted images whose identity disagrees with the rest.

        This is the failure `_average_faces` cannot see: it takes the mean of
        whatever it is given, so one photo of a different person pulls the
        identity toward a blend of two people — a face that resembles nobody,
        with nothing reported.

        Mutates `review` in place.
        """
        assert self.config is not None

        paths = [p for p in review.accepted if p in embeddings]
        if len(paths) < 2:
            return

        vectors = [embeddings[p] for p in paths]

        outliers = guards.find_identity_outliers(self.config, vectors)
        if outliers:
            for index in outliers:
                path = paths[index]
                review.accepted.remove(path)
                review.reject(path, guards.GuardResult.failed(
                    guards.IDENTITY_OUTLIER,
                    f'cosine below {self.config.guard_outlier_sim:.2f} '
                    f'against the other {len(paths) - 1}',
                ))
            return

        # Two images that disagree have no majority, so there is no way to tell
        # which one is the intruder. Report the disagreement rather than guessing
        # — rejecting the wrong one of the two would be worse than either.
        if len(paths) == 2:
            agreement = guards.group_agreement(vectors)
            if agreement < self.config.guard_outlier_sim:
                for path in list(paths):
                    review.accepted.remove(path)
                    review.reject(path, guards.GuardResult.failed(
                        guards.IDENTITY_OUTLIER,
                        f'the two images disagree (cosine {agreement:.2f}) and with '
                        f'only two there is no way to tell which is wrong — add a '
                        f'third, or upload them separately',
                    ))

    def _load_embedding(self, npy_path: str) -> Optional[Face]:
        """
        Load a pre-computed face embedding from .npy file.

        Args:
            npy_path: Path to .npy file containing embedding vector

        Returns:
            Face object with embedding, or None if file not found
        """
        if not os.path.exists(npy_path):
            return None

        try:
            embedding = np.load(npy_path)
            # Create a Face-like object with just the embedding
            return types.SimpleNamespace(normed_embedding=embedding)
        except Exception as e:
            import sys
            print(f'[FaceDatabase] _load_embedding error ({npy_path}): {type(e).__name__}: {e}', file=sys.stderr)
            return None

    def _extract_from_image(self, image_path: str) -> Optional[Face]:
        """
        Extract face from an image file.

        Uses the FaceDetector to find and extract a face.
        Caches result for repeated access.

        Args:
            image_path: Path to image file

        Returns:
            Face object, or None if file not found or no face detected
        """
        # Check cache first. Keyed by file identity rather than path, so an
        # upload that reuses a filename is treated as the new photo it is.
        key = self._cache_key(image_path)
        if key in self._cache:
            return self._cache[key]

        if not os.path.exists(image_path):
            return None

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return None

            detection = self.detector.detect_one(frame)
            if detection is None:
                return None

            # Cache it
            face = detection.face
            self._cache[self._cache_key(image_path)] = face
            return face
        except Exception as e:
            import sys
            print(f'[FaceDatabase] _extract_from_image error: {type(e).__name__}: {e}', file=sys.stderr)
            return None

    def _average_faces(self, faces: List[Face]) -> Optional[Face]:
        """
        Average embeddings from multiple faces.

        Args:
            faces: List of Face objects with normed_embedding attribute

        Returns:
            New Face-like object with averaged embedding, or None if empty
        """
        if not faces:
            return None

        # Extract embeddings
        embeddings = []
        for face in faces:
            if hasattr(face, 'normed_embedding'):
                embeddings.append(face.normed_embedding)

        if not embeddings:
            return None

        # Average
        embeddings_array = np.array(embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)

        # Normalize to unit vector
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        # Return Face-like object
        return types.SimpleNamespace(normed_embedding=avg_embedding)

    def save_embedding(self, face: Face, path: str) -> None:
        """
        Save a face embedding to a .npy file.

        Args:
            face: Face object with normed_embedding
            path: Output path for .npy file
        """
        if not hasattr(face, 'normed_embedding'):
            return

        # Create directory if needed (dirname is '' for bare filenames)
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            np.save(path, face.normed_embedding)
        except Exception as e:
            import sys
            print(f'[FaceDatabase] save_embedding error ({path}): {type(e).__name__}: {e}', file=sys.stderr)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._texture_scores.clear()
