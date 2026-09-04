"""
Input guards for the Phantom pipeline.

Refusing inputs that would produce a wrong swap, instead of swapping them badly.

The pipeline otherwise has no notion of an input it should refuse: it finds a
face or it does not, and everything in between — the wrong person, an unusable
photo, two people in shot — is processed as though it were fine. That produces
output which is *confidently wrong*, and confidently wrong is worse than absent.
A frame with no face is obviously broken and the operator fixes it; a frame with
a stranger's face swapped in looks like it worked.

Two call sites, and the difference between them is whether a human is available
to be told:

- **Source images**, at upload. Someone is right there, so reject and say which
  image and why.
- **Runtime**, per frame. Nobody can be asked, so guard the frame and hold the
  last good one.

Everything here is a pure predicate over data the pipeline already computed, so
the runtime guards cost effectively nothing. See docs/INPUT_GUARDS.md.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.types import Bbox, Detection, Frame

# Guard reason codes. Strings rather than an enum because they travel to the
# desktop over JSON and appear in logs, where a stable readable token is worth
# more than type safety at the boundary.
NO_FACE = 'no_face'
MULTIPLE_FACES = 'multiple_faces'
TOO_SMALL = 'too_small'
BLURRED = 'blurred'
EXTREME_POSE = 'extreme_pose'
LOW_CONFIDENCE = 'low_confidence'
OCCLUDED = 'occluded'
NO_SOURCE = 'no_source'
IDENTITY_OUTLIER = 'identity_outlier'
UNREADABLE = 'unreadable'
NOT_EVALUABLE = 'not_evaluable'

# Human-readable explanation per reason. The source-guard flow shows these to a
# person deciding which photo to replace, so they name the fix, not the rule.
_MESSAGES = {
    NO_FACE: 'no face found',
    NO_SOURCE: 'no source face loaded - upload a photo of the face to wear',
    MULTIPLE_FACES: 'more than one face — use a photo of one person alone',
    TOO_SMALL: 'face too small — use a closer or higher-resolution photo',
    BLURRED: 'too blurred — a soft source gives a soft swap on every frame',
    EXTREME_POSE: 'turned too far from the camera — use a more frontal photo',
    LOW_CONFIDENCE: 'face detected but not confidently',
    OCCLUDED: 'face mostly hidden',
    IDENTITY_OUTLIER: 'looks like a different person from the others',
    UNREADABLE: 'file could not be read as an image',
    NOT_EVALUABLE: 'could not be checked',
}


def describe(reason: str) -> str:
    """
    Human-readable explanation for a guard reason code.

    Args:
        reason: One of the reason constants in this module

    Returns:
        A short phrase naming the fix, or the raw code if unrecognised
    """
    return _MESSAGES.get(reason, reason)


@dataclass
class GuardResult:
    """
    Outcome of evaluating guards against one input.

    `ok` is the only thing callers must branch on. `reason` and `detail` exist
    to be reported: at upload to the person who can fix it, at runtime to the
    log.
    """

    ok: bool
    reason: str = ''
    detail: str = ''

    @property
    def message(self) -> str:
        """Full explanation, including the measured value where there is one."""
        if self.ok:
            return ''
        base = describe(self.reason)
        return f'{base} ({self.detail})' if self.detail else base

    @classmethod
    def passed(cls) -> 'GuardResult':
        """An input that cleared every guard."""
        return cls(ok=True)

    @classmethod
    def failed(cls, reason: str, detail: str = '') -> 'GuardResult':
        """A guarded input, with its reason."""
        return cls(ok=False, reason=reason, detail=detail)


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------

def face_size(detection: Detection) -> int:
    """
    Face size in pixels, on the **shorter** side of its bounding box.

    The shorter side rather than the longer one, or the area: a box that is tall
    and narrow has too few pixels across the features regardless of its height,
    and the shorter side is what actually limits the embedding.

    Args:
        detection: Detection to measure

    Returns:
        Shorter side of the bounding box, in pixels
    """
    return int(min(detection.bbox.w, detection.bbox.h))


# Edge length the face crop is normalised to before measuring focus. Laplacian
# variance is scale-dependent — the same face photographed larger scores higher —
# so without a canonical size the reading says as much about the camera's
# megapixels as about whether the photo is sharp.
_SHARPNESS_SIZE = 256


def sharpness(frame: Frame, bbox: Optional[Bbox] = None) -> float:
    """
    Laplacian variance of the **face**, at a canonical size.

    Two things were wrong with measuring the whole frame, which is what this
    did. It answered a question nobody asked — a portrait shot at f/1.8 has a
    deliberately blurred background, and averaged over the frame that reads as
    an out-of-focus photo even when the face is perfectly sharp, so good photos
    were refused for the composition that makes them good ones. And it missed
    the case the guard exists for, since a sharp busy background carries a soft
    face over the floor.

    Normalising the crop removes the other half of the problem, which the
    previous docstring admitted rather than fixed: the reading no longer moves
    with how many pixels the camera happened to spend on the face, so one
    threshold means the same thing for a phone portrait and a webcam grab.

    Args:
        frame: Image to measure
        bbox: Face to measure within it. None measures the whole frame, which
              is only right when there is no detection to speak of

    Returns:
        Variance of the Laplacian; higher is sharper
    """
    region = frame
    if bbox is not None:
        # The box exactly, with no padding for context. Padding was the first
        # attempt and it quietly reintroduced the bug: a portrait's blurred
        # background bleeds back in around the edges and drags a sharp face
        # down, which is the thing this crop exists to stop.
        height, width = frame.shape[:2]
        x0 = max(0, bbox.x)
        y0 = max(0, bbox.y)
        x1 = min(width, bbox.x + bbox.w)
        y1 = min(height, bbox.y + bbox.h)
        # A degenerate box measures the frame rather than an empty array. It
        # cannot happen behind the size guard, which runs first and needs a real
        # box to pass, but this is also called from telemetry.
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            region = frame[y0:y1, x0:x1]

    gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # INTER_AREA downsamples without the aliasing that would read as detail;
    # a face smaller than this is left alone rather than upsampled, since
    # interpolating cannot add the focus the guard is looking for.
    if gray.shape[0] > _SHARPNESS_SIZE or gray.shape[1] > _SHARPNESS_SIZE:
        scale = _SHARPNESS_SIZE / float(max(gray.shape[:2]))
        gray = cv2.resize(
            gray, (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Where a yaw reading came from, for telemetry.
YAW_POSE = 'pose'
YAW_KEYPOINTS = 'keypoints'
YAW_NONE = 'none'


def estimate_yaw(detection: Detection) -> Optional[float]:
    """
    Yaw in degrees, preferring the detector's own estimate.

    Args:
        detection: Detection to measure

    Returns:
        Signed yaw in degrees, or None if it cannot be determined
    """
    return measure_yaw(detection)[0]


def measure_yaw(detection: Detection) -> Tuple[Optional[float], str]:
    """
    Yaw in degrees plus where the number came from.

    `buffalo_l` bundles `1k3d68.onnx`, and InsightFace's 3D-landmark model sets
    `face.pose` as a side effect of detection — so on a stock pack this is
    already computed and free. It is preferred when present because it is a
    real pose estimate rather than a proxy.

    The keypoint approximation is the fallback for packs that do not carry it
    (a build using `allowed_modules` to trim the pack, for instance). It reads
    the nose's horizontal position between the eyes: dead centre is 0, and as
    the head turns the nose moves toward the eye on the near side. Projected
    onto the inter-eye axis and normalised by its length, so it is scale and
    roll invariant.

    The two are *not* the same scale. The approximation is a small-angle fit
    that grows increasingly pessimistic toward profile — the safe direction for
    a guard, but it means a threshold calibrated against one source should not
    be assumed correct for the other. Which source is in use is recorded in the
    guard telemetry for exactly that reason.

    Args:
        detection: Detection carrying `kps` in InsightFace's five-point order
                   (left eye, right eye, nose, left mouth, right mouth)

    Returns:
        (yaw in degrees or None, source constant)
    """
    pose = getattr(detection.face, 'pose', None)
    if pose is not None:
        try:
            values = np.asarray(pose, dtype=np.float64).ravel()
            if values.size >= 2 and np.isfinite(values[1]):
                # InsightFace orders this (pitch, yaw, roll).
                return float(values[1]), YAW_POSE
        except (TypeError, ValueError):
            pass

    kps = detection.kps
    if kps is None or len(kps) < 3:
        return None, YAW_NONE

    points = np.asarray(kps, dtype=np.float64)
    left_eye, right_eye, nose = points[0], points[1], points[2]

    eye_span = float(np.linalg.norm(right_eye - left_eye))
    if eye_span < 1e-6:
        return None, YAW_NONE

    axis = (right_eye - left_eye) / eye_span
    midpoint = (left_eye + right_eye) / 2.0
    offset = float(np.dot(nose - midpoint, axis)) / eye_span

    return float(np.clip(offset * 90.0, -90.0, 90.0)), YAW_KEYPOINTS


# Face attributes the pipeline depends on, and what breaks without each.
_CAPABILITIES = {
    'pose': 'real yaw (falls back to a keypoint approximation)',
    'kps': 'the swap warp itself',
    'landmark_2d_106': 'the compositing mask (falls back to a template ellipse)',
    'landmark_3d_68': 'nothing directly; its presence is why `pose` exists',
    'normed_embedding': 'the stabilizer identity reset and source outlier check',
    'det_score': 'the confidence guard',
}


def probe_capabilities(face: Any) -> Dict[str, bool]:
    """
    Which of the attributes the pipeline relies on this model pack provides.

    Called once on the first real detection. Several thresholds silently become
    no-ops if their input is absent — the identity reset never fires without
    `normed_embedding`, the confidence guard never fires without `det_score` —
    and a silent no-op is indistinguishable from a guard that simply never had
    cause to fire.

    Args:
        face: A raw InsightFace Face from a successful detection

    Returns:
        Attribute name -> present and non-empty
    """
    present: Dict[str, bool] = {}
    for name in _CAPABILITIES:
        value = getattr(face, name, None)
        if value is None:
            present[name] = False
            continue
        try:
            present[name] = bool(np.asarray(value).size)
        except (TypeError, ValueError):
            present[name] = True
    return present


def describe_capabilities(present: Dict[str, bool]) -> str:
    """
    One-line summary of a capability probe, naming what is lost.

    Args:
        present: Result of `probe_capabilities`

    Returns:
        Human-readable summary
    """
    have = sorted(n for n, ok in present.items() if ok)
    missing = sorted(n for n, ok in present.items() if not ok)

    summary = f'Model provides: {", ".join(have) or "nothing recognised"}'
    if missing:
        losses = '; '.join(f'{n} — affects {_CAPABILITIES[n]}' for n in missing)
        summary += f'. Missing: {losses}'
    return summary


def cosine_similarity(a: Any, b: Any) -> float:
    """
    Cosine similarity between two embedding vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity in [-1, 1]; 0.0 if either vector is degenerate
    """
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.shape != vb.shape or va.size == 0:
        return 0.0

    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / norm)


# ----------------------------------------------------------------------------
# Source guards — applied at upload, before any embedding is built
# ----------------------------------------------------------------------------

def check_source(
    config: FaceSwapConfig,
    frame: Optional[Frame],
    detections: Sequence[Detection],
) -> GuardResult:
    """
    Evaluate every source guard against one uploaded image.

    Order matters only for which reason gets reported first, and it is chosen so
    the most actionable complaint wins: "there are two people in this photo" is
    more useful than "this photo is slightly soft".

    Args:
        config: Thresholds
        frame: The decoded image, or None if it could not be read
        detections: Every face found in it

    Returns:
        GuardResult naming the first guard that failed
    """
    if frame is None:
        return GuardResult.failed(UNREADABLE)

    if not detections:
        return GuardResult.failed(NO_FACE)

    if config.guard_multi_face and len(detections) > 1:
        return GuardResult.failed(
            MULTIPLE_FACES, f'{len(detections)} faces',
        )

    detection = detections[0]

    size = face_size(detection)
    if size < config.guard_min_source_px:
        return GuardResult.failed(
            TOO_SMALL, f'{size}px, need {config.guard_min_source_px}px',
        )

    # The face, not the frame: a portrait's blurred background is the point of
    # the photograph, and averaging it in refuses good pictures for it.
    variance = sharpness(frame, detection.bbox)
    if variance < config.guard_min_sharpness:
        return GuardResult.failed(
            # One decimal, because :.0f rounded 39.6 up and produced
            # 'sharpness 40, need 40' -- a refusal that reads as a
            # contradiction, and sends the reader looking for an off-by-one
            # that is not there. The comparison is <, so equality passes.
            BLURRED,
            f'sharpness {variance:.1f}, need {config.guard_min_sharpness:.1f}',
        )

    yaw = estimate_yaw(detection)
    if yaw is None:
        # Fail closed: an un-evaluable guard is a failed guard.
        return GuardResult.failed(NOT_EVALUABLE, 'pose could not be estimated')
    if abs(yaw) > config.guard_max_yaw:
        return GuardResult.failed(
            EXTREME_POSE,
            f'{abs(yaw):.1f} degrees, limit {config.guard_max_yaw:.1f}',
        )

    return GuardResult.passed()


def find_identity_outliers(
    config: FaceSwapConfig,
    embeddings: Sequence[Any],
) -> List[int]:
    """
    Indices of embeddings that disagree with the rest of the group.

    Leave-one-out cosine similarity: each embedding is compared against the mean
    of *the others*, not against the mean of everything. Including a candidate
    in the mean it is tested against is what lets a single wrong photo drag the
    reference toward itself and pass.

    Needs three images to mean anything. With two that disagree there is no
    majority and no way to tell which is the intruder, so nothing is flagged —
    the caller reports the disagreement instead of guessing.

    Args:
        config: Provides `guard_outlier_sim`, the cosine floor
        embeddings: One embedding vector per source image, in upload order

    Returns:
        Indices into `embeddings` that sit below the floor, ascending
    """
    if len(embeddings) < 3:
        return []

    vectors = [np.asarray(e, dtype=np.float64).ravel() for e in embeddings]
    if len({v.shape for v in vectors}) != 1:
        return []

    outliers = []
    for index, vector in enumerate(vectors):
        others = [v for i, v in enumerate(vectors) if i != index]
        mean = np.mean(others, axis=0)
        if cosine_similarity(vector, mean) < config.guard_outlier_sim:
            outliers.append(index)

    return outliers


def group_agreement(embeddings: Sequence[Any]) -> float:
    """
    Lowest pairwise similarity in a group, for reporting a two-image conflict.

    Args:
        embeddings: Embedding vectors

    Returns:
        Minimum pairwise cosine similarity, or 1.0 for fewer than two vectors
    """
    if len(embeddings) < 2:
        return 1.0

    worst = 1.0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            worst = min(worst, cosine_similarity(embeddings[i], embeddings[j]))
    return worst


# ----------------------------------------------------------------------------
# Runtime guards — applied per frame, before swapping
# ----------------------------------------------------------------------------

def check_frame(
    config: FaceSwapConfig,
    detections: Sequence[Detection],
    face_point: Optional[Tuple[float, float]] = None,
) -> GuardResult:
    """
    Evaluate the runtime guards that can be answered from detection alone.

    Occlusion is not here: it needs the aligned crop and the segmentation mask,
    which only exist once alignment has been computed. `coverage_ok` handles it
    at that point.

    Zero faces is deliberately *not* a guarded frame **here**, because this
    function serves batch as well, and a rendered video of an empty room should
    come back as a video of an empty room.

    The live path decides differently and does so itself, in
    `ProcessingPipeline._process_and_emit`: while a stream is running it holds
    the last swapped frame rather than emitting an unswapped one. Nothing
    distinguishes "stepped out of shot" from "still sitting there, but the light
    dropped and detection failed" — both arrive as zero detections, and the
    second puts the operator's real face on the call. See
    `tests/test_live_exposure.py`.

    Args:
        config: Thresholds
        detections: Every face found in this frame
        face_point: A face named for *this* target specifically, overriding
                    `config.target_face_point`. Photo mode carries one per
                    uploaded photo, so the config's single value cannot say it

    Returns:
        GuardResult naming the first guard that failed
    """
    if not config.guards:
        return GuardResult.passed()

    # With `many_faces` on, a second face is the point rather than a problem,
    # and per-face guarding would need per-face held frames to fall back to.
    if config.many_faces:
        return GuardResult.passed()

    # A named face answers the question this guard exists to ask. It refuses
    # a crowd because "which face did you mean?" has no safe default — once
    # someone has answered it, whether a template's author offline or the
    # operator clicking a face in their own photo, there is nothing left to
    # protect against. Refusing anyway would reject a scene we shipped on
    # purpose, or a choice the operator just made.
    named = face_point or config.target_face_point
    if config.guard_multi_face and len(detections) > 1 and named is None:
        return GuardResult.failed(MULTIPLE_FACES, f'{len(detections)} faces')

    if not detections:
        return GuardResult.passed()

    detection = detections[0]

    if detection.confidence < config.guard_min_confidence:
        return GuardResult.failed(
            LOW_CONFIDENCE,
            f'{detection.confidence:.2f} < {config.guard_min_confidence:.2f}',
        )

    size = face_size(detection)
    if size < config.guard_min_frame_px:
        return GuardResult.failed(
            TOO_SMALL, f'{size}px < {config.guard_min_frame_px}px',
        )

    yaw = estimate_yaw(detection)
    if yaw is None:
        return GuardResult.failed(NOT_EVALUABLE, 'pose')
    if abs(yaw) > config.guard_max_yaw:
        return GuardResult.failed(
            EXTREME_POSE, f'{abs(yaw):.0f} > {config.guard_max_yaw:.0f} degrees',
        )

    return GuardResult.passed()


def coverage_ok(config: FaceSwapConfig, coverage: Optional[float]) -> bool:
    """
    Whether enough of the face is unoccluded to swap it.

    Args:
        config: Provides `guard_min_coverage`
        coverage: Fraction of the landmark hull the occluder left in, or None
                  when occlusion could not be evaluated

    Returns:
        True if the frame may be swapped
    """
    if not config.guards or config.guard_observe:
        return True

    if coverage is None:
        # Nothing to measure against. This is the case where the occluder is
        # switched off or unavailable, which is a supported configuration rather
        # than a failure, so it does not guard — otherwise turning the occluder
        # off would guard every frame.
        return True

    return coverage >= config.guard_min_coverage


def hull_coverage(hull: Frame, occlusion: Optional[Frame]) -> Optional[float]:
    """
    Fraction of the landmark hull that the occlusion mask leaves in.

    Args:
        hull: Hull mask in [0, 1]
        occlusion: XSeg inclusion mask in [0, 1], or None if unavailable

    Returns:
        Coverage in [0, 1], or None if it cannot be computed
    """
    if occlusion is None:
        return None

    hull_area = float(hull.sum())
    if hull_area < 1e-6:
        return None

    if occlusion.shape != hull.shape:
        occlusion = cv2.resize(occlusion, (hull.shape[1], hull.shape[0]))

    return float(np.clip((hull * occlusion).sum() / hull_area, 0.0, 1.0))


# ----------------------------------------------------------------------------
# Calibration telemetry
# ----------------------------------------------------------------------------

# Each metric, paired with the config field whose threshold it is compared
# against and which side of that threshold is the failing one. This is what lets
# the report say "3.2% of frames would fail" rather than just printing a median.
_METRICS = {
    'confidence': ('guard_min_confidence', 'below'),
    'face_px': ('guard_min_frame_px', 'below'),
    'yaw_abs': ('guard_max_yaw', 'above'),
    'coverage': ('guard_min_coverage', 'below'),
    'identity_sim': ('guard_identity_sim', 'below'),
    'faces': (None, None),
}


@dataclass
class GuardTelemetry:
    """
    Records what every guard measured, not just what it decided.

    The point is calibration. Eight thresholds were chosen without data behind
    them, and watching the output only ever reveals *that* something is wrong —
    "it guards constantly", "the shimmer is back" — never *which number*. A
    distribution per metric turns one session into eight answers.

    Three of the defaults can make things actively worse if mis-set, and all
    three are invisible without this:

    - `guard_min_coverage` is compared against XSeg coverage of an *expanded*
      hull, and what that reads on a completely clear face is unknown
    - the stabilizer's identity floor, if it sits above the same-person
      similarity under motion blur, resets every frame and brings the shimmer
      back — a realism regression caused by a guard
    - `guard_min_confidence` is 0.5 against a detector threshold of 0.35, so
      everything scoring in between is guarded

    Samples are capped rather than reservoir-sampled: the cap is large enough
    for many minutes of video, and keeping the first N is easier to reason
    about than a weighted sample when the report is read by a person.
    """

    # ~28 minutes at 30fps. Past this the distribution has long since converged.
    limit: int = 50000

    samples: Dict[str, List[float]] = field(default_factory=dict)
    reasons: Dict[str, int] = field(default_factory=dict)
    yaw_sources: Dict[str, int] = field(default_factory=dict)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    frames: int = 0
    would_guard: int = 0
    observing: bool = False

    def record(
        self,
        detections: Sequence[Detection],
        verdict: 'GuardResult',
        coverage: Optional[float] = None,
        identity_sim: Optional[float] = None,
    ) -> None:
        """
        Record one frame's measurements and verdict.

        Args:
            detections: Every face found in the frame
            verdict: What `check_frame` decided
            coverage: Occlusion coverage, if the mask was built
            identity_sim: Frame-to-frame identity cosine, if computed
        """
        self.frames += 1
        self._add('faces', float(len(detections)))

        if not verdict.ok:
            self.would_guard += 1
            self.reasons[verdict.reason] = self.reasons.get(verdict.reason, 0) + 1

        if coverage is not None:
            self._add('coverage', coverage)
        if identity_sim is not None:
            self._add('identity_sim', identity_sim)

        if detections:
            primary = max(detections, key=lambda d: d.bbox.w * d.bbox.h)
            self._add('confidence', float(primary.confidence))
            self._add('face_px', float(face_size(primary)))

            yaw, source = measure_yaw(primary)
            self.yaw_sources[source] = self.yaw_sources.get(source, 0) + 1
            if yaw is not None:
                self._add('yaw_abs', abs(yaw))

    def _add(self, metric: str, value: float) -> None:
        """Append a sample, respecting the cap."""
        bucket = self.samples.setdefault(metric, [])
        if len(bucket) < self.limit:
            bucket.append(value)

    def report(self, config: FaceSwapConfig) -> Dict[str, Any]:
        """
        Summarise the session against the thresholds currently configured.

        Args:
            config: The thresholds these samples should be judged against

        Returns:
            A JSON-serialisable report
        """
        metrics: Dict[str, Any] = {}

        for metric, values in sorted(self.samples.items()):
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            entry: Dict[str, Any] = {
                'count': int(array.size),
                'min': round(float(array.min()), 4),
                'p1': round(float(np.percentile(array, 1)), 4),
                'p5': round(float(np.percentile(array, 5)), 4),
                'p50': round(float(np.percentile(array, 50)), 4),
                'p95': round(float(np.percentile(array, 95)), 4),
                'p99': round(float(np.percentile(array, 99)), 4),
                'max': round(float(array.max()), 4),
            }

            field_name, direction = _METRICS.get(metric, (None, None))
            if field_name:
                threshold = float(getattr(config, field_name))
                failing = (array > threshold) if direction == 'above' else (array < threshold)
                entry['threshold'] = threshold
                entry['threshold_field'] = field_name
                entry['fail_pct'] = round(float(failing.mean() * 100.0), 2)
                # Distance from the threshold to the bulk of the distribution.
                # A small or negative margin means the threshold sits inside
                # normal operating range and will fire on ordinary frames.
                edge = float(np.percentile(array, 99 if direction == 'above' else 1))
                entry['margin'] = round(
                    (threshold - edge) if direction == 'above' else (edge - threshold), 4,
                )
            metrics[metric] = entry

        return {
            'frames': self.frames,
            'observing': self.observing,
            'would_guard': self.would_guard,
            'would_guard_pct': round(
                (self.would_guard / self.frames * 100.0) if self.frames else 0.0, 2,
            ),
            'reasons': dict(sorted(self.reasons.items(), key=lambda kv: -kv[1])),
            'yaw_source': dict(self.yaw_sources),
            'capabilities': dict(self.capabilities),
            'metrics': metrics,
        }

    def format_report(self, config: FaceSwapConfig) -> str:
        """
        The report as text, for the log.

        Args:
            config: Thresholds to judge against

        Returns:
            A multi-line summary
        """
        data = self.report(config)
        if not data['frames']:
            return 'Guard telemetry: no frames recorded'

        mode = 'observing' if data['observing'] else 'enforcing'
        lines = [
            f'Guard telemetry ({mode}) — {data["frames"]} frames, '
            f'{data["would_guard"]} would guard ({data["would_guard_pct"]}%)',
        ]

        if data['reasons']:
            lines.append('  reasons: ' + ', '.join(
                f'{k}={v}' for k, v in data['reasons'].items()
            ))
        if data['yaw_source']:
            lines.append('  yaw from: ' + ', '.join(
                f'{k}={v}' for k, v in data['yaw_source'].items()
            ))

        for metric, entry in data['metrics'].items():
            line = (
                f'  {metric:<13} p1={entry["p1"]:<9} p50={entry["p50"]:<9} '
                f'p99={entry["p99"]:<9}'
            )
            if 'threshold' in entry:
                verdict = 'OK' if entry['margin'] > 0 else 'TIGHT'
                line += (
                    f' vs {entry["threshold_field"]}={entry["threshold"]} '
                    f'-> {entry["fail_pct"]}% fail, margin {entry["margin"]} [{verdict}]'
                )
            lines.append(line)

        return '\n'.join(lines)

    def write(self, path: str, config: FaceSwapConfig) -> bool:
        """
        Write the report as JSON.

        Args:
            path: Destination file
            config: Thresholds to judge against

        Returns:
            True if written
        """
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(self.report(config), handle, indent=2)
            return True
        except OSError:
            return False


# ----------------------------------------------------------------------------
# Validation for the set_realism API surface
# ----------------------------------------------------------------------------

# field -> (type, minimum, maximum). Bounds are clamps, not rejections: an
# operator A/B-ing thresholds live should get the nearest legal value rather
# than an error.
GUARD_FIELDS = {
    'guards': (bool, None, None),
    'guard_multi_face': (bool, None, None),
    'guard_min_source_px': (int, 0, 2048),
    'guard_min_frame_px': (int, 0, 2048),
    'guard_max_yaw': (float, 0.0, 90.0),
    'guard_min_confidence': (float, 0.0, 1.0),
    'guard_min_coverage': (float, 0.0, 1.0),
    'guard_identity_sim': (float, -1.0, 1.0),
    'guard_min_sharpness': (float, 0.0, 10000.0),
    'guard_outlier_sim': (float, -1.0, 1.0),
    'guard_observe': (bool, None, None),
}


def validate_guard_value(field: str, value: Any) -> Tuple[bool, Any, str]:
    """
    Coerce and clamp one guard threshold.

    Args:
        field: Config field name, which must be in GUARD_FIELDS
        value: Proposed value

    Returns:
        (accepted, coerced_value, error_message)
    """
    if field not in GUARD_FIELDS:
        return False, None, f'unknown guard field: {field}'

    kind, low, high = GUARD_FIELDS[field]

    if kind is bool:
        if not isinstance(value, bool):
            return False, None, f'{field} must be a boolean'
        return True, value, ''

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False, None, f'{field} must be a number'

    coerced = kind(value)
    if low is not None:
        coerced = max(kind(low), coerced)
    if high is not None:
        coerced = min(kind(high), coerced)

    return True, coerced, ''
