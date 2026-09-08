"""
Does the output still look like the source?

Every other realism reading in this pipeline is about *texture* — is the face as
detailed as the frame, is there a seam, does the grain match. None of them can
tell you whether the person on screen is the right person, and that is the one
question the whole product turns on.

The instrument is the recognition model the detector already loads. ArcFace maps
a face to a 512-d vector where cosine distance is identity distance; it is what
`review_sources` already uses to refuse a photograph of the wrong person, and
what `LandmarkStabilizer` already uses to notice the subject changed. Pointing it
at the *output* instead of the input costs one inference and answers the
question directly.

**Read differences, not absolutes.** A swap is not a photograph of the source: it
is the source's identity rendered in the target's pose, lighting and camera, and
it will not score like two photos of the same person. Useful bands, for
`buffalo_l`'s `w600k_r50`:

    > 0.6    the same person, comfortably
    0.4-0.6  recognisably related — where a good swap lands
    0.28     InsightFace's own verification threshold
    < 0.2    not this person

What the readings are actually for is the *gap between two of them*: between
stages of one composite, which attributes a loss to a stage; and between two
configurations on the same clip, which is how a lever gets judged.

Measuring in aligned space is the fiddly part, and the reason this is a service
rather than three lines at the call site. The recognition model wants an
**arcface_112** crop — a specific framing, not merely a 112px face — and the
compositor's working crops are in the *swapper's* alignment, which is
`arcface_128` for some models and `mtcnn_512` for others. Both are similarity
transforms of the same five points, so the map between them is closed-form and
exact: fit the crop's own template to the 112 one and warp. No detection is
needed, and no crop is re-derived from the frame.
"""

from typing import Any, Optional

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.processing.geometry import estimate_similarity
from pipeline.types import Frame

Embedding = npt.NDArray[Any]

# InsightFace's `arcface_dst`: the five destination points its recognition
# models were trained against, in a 112x112 crop.
#
# Hard-coded rather than imported for the same reason `_ARCFACE_TEMPLATE` in
# face_swapping.py is. It is a constant of the trained model, not of whichever
# InsightFace version happens to be installed, and a silent move upstream would
# be a quiet accuracy loss rather than an ImportError. Verified equal to
# `insightface.utils.face_align.arcface_dst`.
ARCFACE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)

_CROP = 112


def cosine(a: Optional[Embedding], b: Optional[Embedding]) -> Optional[float]:
    """
    Cosine similarity between two embeddings.

    Args:
        a: An embedding, or None
        b: An embedding, or None

    Returns:
        Similarity in [-1, 1], or None if either side is missing or degenerate
    """
    if a is None or b is None:
        return None

    first = np.asarray(a, dtype=np.float64).ravel()
    second = np.asarray(b, dtype=np.float64).ravel()
    if first.size != second.size or first.size == 0:
        return None

    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    if scale < 1e-8:
        return None

    return float(np.dot(first, second) / scale)


class IdentityProbe:
    """
    Embeds faces with the detector's own recognition model.

    Shares the detector rather than loading a second copy: `buffalo_l` already
    has `w600k_r50` in memory and on the GPU, and a private session would double
    both for a diagnostic.

    Example:
        probe = IdentityProbe(detector)
        source = probe.embed_frame(photo, face.kps)
        after = probe.embed_aligned(fake, template)
        print(cosine(source, after))
    """

    def __init__(self, detector: Any) -> None:
        """
        Args:
            detector: A `FaceDetector`, consulted lazily for its recognition
                model so constructing a probe never loads anything
        """
        self._detector = detector
        self._model: Optional[Any] = None
        self._checked = False

    @property
    def available(self) -> bool:
        """Whether a recognition model could be resolved."""
        return self._recognition() is not None

    def _recognition(self) -> Optional[Any]:
        """
        The recognition model, or None on a pack that does not carry one.

        Resolved once. A trimmed model pack is a capability gap, not a fault —
        the same reasoning `_probe_once` applies to `face.pose` — so the probe
        goes quiet rather than raising, and the readings simply do not appear.
        """
        if self._checked:
            return self._model

        self._checked = True
        try:
            self._model = self._detector.recognition_model()
        except Exception:
            self._model = None

        return self._model

    def _feature(self, crop112: Frame) -> Optional[Embedding]:
        """
        Run recognition on an arcface_112 crop.

        Args:
            crop112: 112x112 BGR crop in the recognition model's own framing

        Returns:
            L2-normalised 512-d embedding, or None
        """
        model = self._recognition()
        if model is None:
            return None

        try:
            feature = np.asarray(model.get_feat(crop112), dtype=np.float32).ravel()
        except Exception:
            return None

        norm = float(np.linalg.norm(feature))
        if norm < 1e-8:
            return None

        return feature / norm

    def embed_aligned(
        self,
        crop: Frame,
        template: npt.NDArray[Any],
    ) -> Optional[Embedding]:
        """
        Embed a crop that is already in a swapper's aligned space.

        The crop's alignment is fully described by the template it was built
        with, so re-framing it for recognition is a closed-form similarity
        between two sets of five points — no detection, and no going back to the
        frame for pixels that are already here.

        Args:
            crop: Square aligned crop, BGR
            template: The normalised 5-point template that defines its space

        Returns:
            L2-normalised embedding, or None if the warp or the model failed
        """
        if crop is None or crop.ndim != 3 or crop.shape[0] < 8:
            return None

        source = np.asarray(template, dtype=np.float64) * float(crop.shape[0])
        matrix = estimate_similarity(source, ARCFACE_112)
        if matrix is None:
            return None

        warped = cv2.warpAffine(crop, matrix, (_CROP, _CROP))
        return self._feature(warped)

    def embed_frame(
        self,
        frame: Frame,
        kps: Optional[npt.NDArray[Any]],
    ) -> Optional[Embedding]:
        """
        Embed a face in a full frame, from its five keypoints.

        This is the measurement that counts: it sees the face at the size and
        through the mask a viewer sees it, so everything the compositor did —
        the paste, the silhouette, the colour match, the grain — is in the
        number.

        Args:
            frame: Full frame, BGR
            kps: The face's five keypoints in frame coordinates

        Returns:
            L2-normalised embedding, or None
        """
        if frame is None or kps is None:
            return None

        points = np.asarray(kps, dtype=np.float64)
        if points.shape != ARCFACE_112.shape:
            return None

        matrix = estimate_similarity(points, ARCFACE_112)
        if matrix is None:
            return None

        warped = cv2.warpAffine(frame, matrix, (_CROP, _CROP))
        return self._feature(warped)
