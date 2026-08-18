"""
Exercise the input guards.

The ML layer is stubbed: these are pure predicates over detection data, which is
exactly what makes them testable without a model. Detections are synthesised with
the geometry each guard is supposed to react to.
"""

import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from unittest.mock import MagicMock


class StubModule(MagicMock):
    """MagicMock that also satisfies `from x.y import z` for nested paths."""

    __path__: list = []


for name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'onnxruntime', 'torch', 'torchvision', 'psutil',
    'tensorflow', 'opennsfw2', 'gfpgan', 'onnx',
):
    sys.modules.setdefault(name, StubModule())

import logging
import os

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.services import guards
from pipeline.types import Bbox, Detection

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def make_detection(
    size: int = 200,
    x: int = 100,
    y: int = 100,
    confidence: float = 0.9,
    yaw_offset: float = 0.0,
    embedding=None,
) -> Detection:
    """
    Synthesise a Detection with controllable geometry.

    `yaw_offset` shifts the nose along the inter-eye axis as a fraction of the
    eye span, which is exactly what estimate_yaw measures.
    """
    eye_y = y + size * 0.4
    left_x = x + size * 0.3
    right_x = x + size * 0.7
    span = right_x - left_x
    nose_x = (left_x + right_x) / 2.0 + yaw_offset * span

    kps = np.array([
        [left_x, eye_y],
        [right_x, eye_y],
        [nose_x, y + size * 0.55],
        [x + size * 0.35, y + size * 0.75],
        [x + size * 0.65, y + size * 0.75],
    ], dtype=np.float32)

    face = MagicMock()
    face.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
    face.kps = kps
    if embedding is not None:
        face.normed_embedding = np.asarray(embedding, dtype=np.float32)

    return Detection(face=face, bbox=Bbox(x, y, size, size), kps=kps,
                     confidence=confidence)


def sharp_image(size: int = 400) -> np.ndarray:
    """An image with plenty of high-frequency content."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def blurred_image(size: int = 400) -> np.ndarray:
    """The same content, heavily blurred."""
    return cv2.GaussianBlur(sharp_image(size), (0, 0), 8.0)


print('=' * 70)
print('Input guards')
print('=' * 70)

# -- Yaw estimation -----------------------------------------------------
print('\nYaw estimation from keypoints')
frontal = guards.estimate_yaw(make_detection(yaw_offset=0.0))
check('frontal face reads near zero', frontal is not None and abs(frontal) < 5.0,
      f'{frontal:.1f} degrees' if frontal is not None else 'None')
turned = guards.estimate_yaw(make_detection(yaw_offset=0.3))
check('turned face reads high', turned is not None and abs(turned) > 20.0,
      f'{turned:.1f} degrees' if turned is not None else 'None')
left = guards.estimate_yaw(make_detection(yaw_offset=-0.3))
check('sign distinguishes direction', left is not None and turned is not None
      and left < 0 < turned, f'{left:.1f} vs {turned:.1f}')
bare = make_detection()
bare.kps = np.array([])
check('unusable keypoints report None', guards.estimate_yaw(bare) is None)

# -- Size measurement ---------------------------------------------------
print('\nFace size')
wide = Detection(face=MagicMock(), bbox=Bbox(0, 0, 300, 90),
                 kps=np.array([]), confidence=0.9)
check('measures the shorter side', guards.face_size(wide) == 90,
      f'{guards.face_size(wide)} from 300x90')

# -- Sharpness ----------------------------------------------------------
print('\nSharpness')
sharp_v = guards.sharpness(sharp_image())
blur_v = guards.sharpness(blurred_image())
check('sharp scores above blurred', sharp_v > blur_v * 10,
      f'{sharp_v:.0f} vs {blur_v:.0f}')

# -- Source guards ------------------------------------------------------
print('\nSource guards')
config = FaceSwapConfig()
image = sharp_image()

ok = guards.check_source(config, image, [make_detection(size=200)])
check('a clean source passes', ok.ok, ok.message)

r = guards.check_source(config, None, [])
check('unreadable file rejected', r.reason == guards.UNREADABLE, r.reason)

r = guards.check_source(config, image, [])
check('no face rejected', r.reason == guards.NO_FACE, r.reason)

r = guards.check_source(config, image,
                        [make_detection(x=50), make_detection(x=250)])
check('two faces rejected', r.reason == guards.MULTIPLE_FACES, r.message)

r = guards.check_source(config, image, [make_detection(size=80)])
check('small face rejected', r.reason == guards.TOO_SMALL, r.message)

r = guards.check_source(config, blurred_image(), [make_detection(size=200)])
check('blurred source rejected', r.reason == guards.BLURRED, r.message)

r = guards.check_source(config, image, [make_detection(size=200, yaw_offset=0.5)])
check('extreme pose rejected', r.reason == guards.EXTREME_POSE, r.message)

nokps = make_detection(size=200)
nokps.kps = np.array([])
r = guards.check_source(config, image, [nokps])
check('un-evaluable pose fails closed', r.reason == guards.NOT_EVALUABLE, r.message)

relaxed = FaceSwapConfig()
relaxed.guard_multi_face = False
r = guards.check_source(relaxed, image, [make_detection(x=50), make_detection(x=250)])
check('multi-face guard is switchable off', r.ok, r.message)

check('rejection messages name the fix',
      'one person' in guards.describe(guards.MULTIPLE_FACES),
      guards.describe(guards.MULTIPLE_FACES))

# -- Identity outliers --------------------------------------------------
print('\nIdentity outlier check')
rng = np.random.default_rng(7)
base = rng.normal(size=512)
base /= np.linalg.norm(base)


def near(vector, jitter: float):
    """
    A unit vector `jitter` away from `vector`, in relative terms.

    The perturbation is a *unit-norm* random direction scaled by `jitter`, not
    per-component noise of that magnitude: the components of a 512-dim unit
    vector are only ~1/sqrt(512) each, so per-component noise of 0.15 would
    swamp the signal by more than 3x and produce a group with no shared
    identity at all.
    """
    direction = rng.normal(size=vector.shape)
    direction /= np.linalg.norm(direction)
    out = vector + direction * jitter
    return out / np.linalg.norm(out)


stranger = rng.normal(size=512)
stranger /= np.linalg.norm(stranger)

consistent = [near(base, 0.15) for _ in range(4)]
check('a consistent group flags nothing',
      guards.find_identity_outliers(config, consistent) == [],
      str(guards.find_identity_outliers(config, consistent)))

with_intruder = [near(base, 0.15), near(base, 0.15), near(base, 0.15), stranger]
found = guards.find_identity_outliers(config, with_intruder)
check('the intruder is found', found == [3], str(found))

check('two images flag nothing - no majority',
      guards.find_identity_outliers(config, [near(base, 0.1), stranger]) == [])

check('leave-one-out excludes the candidate from its own reference',
      guards.find_identity_outliers(config, [stranger, stranger, near(base, 0.1)]) == [2],
      'a pair of matching strangers makes the odd one out the outlier')

check('group_agreement spots a two-image conflict',
      guards.group_agreement([near(base, 0.1), stranger]) < config.guard_outlier_sim,
      f'{guards.group_agreement([near(base, 0.1), stranger]):.2f}')

# -- Runtime guards -----------------------------------------------------
print('\nRuntime guards')
r = guards.check_frame(config, [make_detection(size=200)])
check('a clean frame passes', r.ok, r.message)

r = guards.check_frame(config, [])
check('zero faces is not a guard', r.ok,
      'stepping out of shot must not hold a stale face over an empty chair')

r = guards.check_frame(config, [make_detection(x=50), make_detection(x=250)])
check('two faces guarded', r.reason == guards.MULTIPLE_FACES, r.message)

r = guards.check_frame(config, [make_detection(size=200, confidence=0.3)])
check('low confidence guarded', r.reason == guards.LOW_CONFIDENCE, r.message)

r = guards.check_frame(config, [make_detection(size=50)])
check('small face guarded', r.reason == guards.TOO_SMALL, r.message)

r = guards.check_frame(config, [make_detection(size=200, yaw_offset=0.5)])
check('extreme pose guarded', r.reason == guards.EXTREME_POSE, r.message)

many = FaceSwapConfig()
many.many_faces = True
r = guards.check_frame(many, [make_detection(x=50), make_detection(x=250)])
check('many_faces bypasses guards', r.ok, r.message)

off = FaceSwapConfig()
off.guards = False
r = guards.check_frame(off, [make_detection(size=10, confidence=0.01)])
check('guards can be disabled wholesale', r.ok, r.message)

# -- Occlusion coverage -------------------------------------------------
print('\nOcclusion coverage')
hull = np.zeros((64, 64), np.float32)
cv2.circle(hull, (32, 32), 20, (1.0,), -1)

clear = np.ones((64, 64), np.float32)
check('unoccluded hull is full coverage',
      abs((guards.hull_coverage(hull, clear) or 0) - 1.0) < 1e-6,
      f'{guards.hull_coverage(hull, clear)}')

half = np.ones((64, 64), np.float32)
half[:, 32:] = 0.0
coverage = guards.hull_coverage(hull, half) or 0.0
check('half-covered hull reads about half', 0.4 < coverage < 0.6, f'{coverage:.2f}')

check('missing occlusion mask reports None',
      guards.hull_coverage(hull, None) is None)

check('heavy occlusion fails the guard',
      not guards.coverage_ok(config, 0.2), 'coverage 0.2 under floor 0.4')
check('light occlusion passes', guards.coverage_ok(config, 0.8))
check('unevaluated occlusion does not guard', guards.coverage_ok(config, None),
      'occluder off is a supported configuration, not a failure')

# -- Threshold validation -----------------------------------------------
print('\nset_realism validation')
accepted, value, err = guards.validate_guard_value('guard_max_yaw', 200.0)
check('out-of-range value is clamped, not refused',
      accepted and value == 90.0, f'200 -> {value}')

accepted, value, err = guards.validate_guard_value('guard_min_confidence', -5)
check('below-range value is clamped', accepted and value == 0.0, f'-5 -> {value}')

accepted, _, err = guards.validate_guard_value('guard_multi_face', 'yes')
check('a string for a boolean is refused', not accepted, err)

accepted, value, _ = guards.validate_guard_value('guards', False)
check('False is accepted, not read as absent', accepted and value is False)

accepted, _, err = guards.validate_guard_value('guard_nonsense', 1)
check('unknown guard field is refused', not accepted, err)

accepted, value, _ = guards.validate_guard_value('guard_min_frame_px', 96.7)
check('int field coerces from float', accepted and value == 96, f'96.7 -> {value}')

# -- Pose preference ----------------------------------------------------
print('\nYaw source preference')
posed = make_detection(yaw_offset=0.0)
posed.face.pose = np.array([3.0, 42.0, 1.0], dtype=np.float32)  # pitch, yaw, roll
value, source = guards.measure_yaw(posed)
check('face.pose is preferred over the approximation',
      source == guards.YAW_POSE and abs(value - 42.0) < 1e-3,
      f'{value} from {source}')

no_pose = make_detection(yaw_offset=0.3)
no_pose.face.pose = None
value, source = guards.measure_yaw(no_pose)
check('falls back to keypoints when pose is absent',
      source == guards.YAW_KEYPOINTS and value is not None,
      f'{value:.1f} from {source}')

junk = make_detection()
junk.face.pose = 'not an array'
value, source = guards.measure_yaw(junk)
check('unusable pose falls back rather than crashing',
      source == guards.YAW_KEYPOINTS, f'from {source}')

nan_pose = make_detection(yaw_offset=0.3)
nan_pose.face.pose = np.array([0.0, np.nan, 0.0], dtype=np.float32)
check('NaN pose falls back', guards.measure_yaw(nan_pose)[1] == guards.YAW_KEYPOINTS)

blind = make_detection()
blind.face.pose = None
blind.kps = np.array([])
check('no pose and no keypoints reports none',
      guards.measure_yaw(blind) == (None, guards.YAW_NONE))

posed_guard = make_detection()
posed_guard.face.pose = np.array([0.0, 60.0, 0.0], dtype=np.float32)
r = guards.check_source(config, image, [posed_guard])
check('the pose guard uses the real value',
      r.reason == guards.EXTREME_POSE and '60' in r.detail, r.message)

# -- Capability probe ---------------------------------------------------
print('\nCapability probe')
full = make_detection().face
full.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
full.landmark_2d_106 = np.zeros((106, 2), dtype=np.float32)
full.landmark_3d_68 = np.zeros((68, 3), dtype=np.float32)
full.normed_embedding = np.zeros(512, dtype=np.float32)
full.det_score = 0.9
present = guards.probe_capabilities(full)
check('a complete pack reports everything present', all(present.values()),
      str(sorted(k for k, v in present.items() if not v)) or 'none missing')

trimmed = make_detection().face
trimmed.pose = None
trimmed.normed_embedding = None
trimmed.landmark_2d_106 = np.zeros((106, 2), dtype=np.float32)
trimmed.landmark_3d_68 = None
trimmed.det_score = 0.9
present = guards.probe_capabilities(trimmed)
check('a trimmed pack reports what is missing',
      not present['pose'] and not present['normed_embedding'] and present['kps'],
      str(sorted(k for k, v in present.items() if not v)))

described = guards.describe_capabilities(present)
check('the description names the consequence, not just the field',
      'identity reset' in described, described[described.find('Missing'):][:90])

empty = make_detection().face
empty.normed_embedding = np.array([])
check('an empty array counts as absent',
      not guards.probe_capabilities(empty)['normed_embedding'])

# -- Telemetry ----------------------------------------------------------
print('\nGuard telemetry')
tele = guards.GuardTelemetry()
strict = FaceSwapConfig()

# 100 frames: 90 clean, 10 with low confidence.
for index in range(100):
    conf = 0.9 if index >= 10 else 0.4
    dets = [make_detection(size=200, confidence=conf)]
    tele.record(dets, guards.check_frame(strict, dets),
                coverage=0.75, identity_sim=0.95)

data = tele.report(strict)
check('every frame counted', data['frames'] == 100, str(data['frames']))
check('the would-guard rate is measured',
      data['would_guard'] == 10 and data['would_guard_pct'] == 10.0,
      f'{data["would_guard"]} ({data["would_guard_pct"]}%)')
check('guards are attributed by reason',
      data['reasons'].get(guards.LOW_CONFIDENCE) == 10, str(data['reasons']))
check('the failing percentage matches the threshold',
      data['metrics']['confidence']['fail_pct'] == 10.0,
      f'{data["metrics"]["confidence"]["fail_pct"]}% below '
      f'{data["metrics"]["confidence"]["threshold"]}')
check('percentiles are recorded',
      data['metrics']['confidence']['p50'] == 0.9,
      f'p50={data["metrics"]["confidence"]["p50"]}')
check('the yaw source is tracked',
      data['yaw_source'].get(guards.YAW_KEYPOINTS) == 100, str(data['yaw_source']))
check('coverage is recorded for calibration',
      data['metrics']['coverage']['p50'] == 0.75)
check('identity similarity is recorded',
      data['metrics']['identity_sim']['p50'] == 0.95,
      'the floor that could reintroduce shimmer is now measurable')

# Margin is the number that says whether a threshold sits inside normal range.
tight = guards.GuardTelemetry()
for _ in range(100):
    dets = [make_detection(size=200, confidence=0.52)]
    tight.record(dets, guards.check_frame(strict, dets), coverage=0.41)
report = tight.report(strict)
check('a threshold just under the distribution shows a small margin',
      0 < report['metrics']['confidence']['margin'] < 0.05,
      f'margin {report["metrics"]["confidence"]["margin"]}')
check('a coverage floor close to normal readings is flagged tight',
      report['metrics']['coverage']['margin'] < 0.02,
      f'margin {report["metrics"]["coverage"]["margin"]} - 0.41 measured vs 0.40 floor')

text = tele.format_report(strict)
check('the text report names the threshold field',
      'guard_min_confidence' in text)
check('the text report flags tight margins',
      'TIGHT' in tight.format_report(strict) or 'OK' in text)

import json as _json
import tempfile as _tempfile
out = os.path.join(_tempfile.mkdtemp(), 'nested', 'report.json')
check('the report writes as JSON, creating directories', tele.write(out, strict))
check('the written JSON round-trips',
      _json.load(open(out))['frames'] == 100)

check('an empty session reports nothing rather than dividing by zero',
      'no frames' in guards.GuardTelemetry().format_report(strict))

observe = FaceSwapConfig()
observe.guard_observe = True
check('observe mode stops the coverage guard from acting',
      guards.coverage_ok(observe, 0.01),
      'so the frame is still composited and its coverage still measured')

print('\n' + '=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
for f in FAIL:
    print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """
    Surface the checks above to pytest as one assertion.

    The bodies run at import: these are scripts first, so they stay runnable
    directly (`python tests/test_x.py`) when a failure needs poking at, and the
    per-check output is the diagnostic. This function is what makes the same
    file a pytest test without duplicating any of it.
    """
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
