"""
Exercise the stateful half of the guards.

Face selection, the stabilizer's identity reset, the source review flow, and the
held-frame behaviour on the live path. These are where the bugs live: the pure
predicates are covered by test_guards.py.
"""

import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from unittest.mock import MagicMock


class StubModule(MagicMock):
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
import tempfile

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.events import EventBus, FRAME_READY
from pipeline.services import guards
from pipeline.services.database import FaceDatabase
from pipeline.services.face_detection import FaceDetector
from pipeline.services.face_tracking import LandmarkStabilizer
from pipeline.types import Bbox, Detection

logging.disable(logging.INFO)

WORK = tempfile.mkdtemp(prefix='phantom-guards-')
PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    print(f'  [{"PASS" if condition else "FAIL"}] {label}' + (f' - {detail}' if detail else ''))


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def make_detection(size=200, x=100, y=100, confidence=0.9, yaw_offset=0.0,
                   embedding=None) -> Detection:
    eye_y = y + size * 0.4
    left_x, right_x = x + size * 0.3, x + size * 0.7
    span = right_x - left_x
    nose_x = (left_x + right_x) / 2.0 + yaw_offset * span
    kps = np.array([
        [left_x, eye_y], [right_x, eye_y], [nose_x, y + size * 0.55],
        [x + size * 0.35, y + size * 0.75], [x + size * 0.65, y + size * 0.75],
    ], dtype=np.float32)

    face = MagicMock()
    face.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
    face.kps = kps
    face.landmark_2d_106 = np.tile(kps, (22, 1))[:106].astype(np.float32)
    face.normed_embedding = embedding if embedding is not None else unit(1)
    return Detection(face=face, bbox=Bbox(x, y, size, size), kps=kps,
                     confidence=confidence)


print('=' * 70)
print('Guards - stateful behaviour')
print('=' * 70)

# -- Face selection -----------------------------------------------------
print('\nPrimary face selection')
small_left = make_detection(size=100, x=10)
big_right = make_detection(size=300, x=400)
picked = FaceDetector.select_primary([small_left, big_right])
check('largest wins, not leftmost', picked is big_right,
      f'picked {picked.bbox.w}px box at x={picked.bbox.x}')

check('order does not matter',
      FaceDetector.select_primary([big_right, small_left]) is big_right)
check('empty list gives None', FaceDetector.select_primary([]) is None)

identical_a = make_detection(size=200, x=10)
identical_b = make_detection(size=200, x=500)
first = FaceDetector.select_primary([identical_a, identical_b])
second = FaceDetector.select_primary([identical_b, identical_a])
check('ties are deterministic', first is second,
      'same box size resolves to the same face regardless of input order')

# -- Stabilizer identity reset ------------------------------------------
print('\nStabilizer identity reset')
person_a, person_b = unit(1), unit(99)
check('the two test identities are dissimilar',
      guards.cosine_similarity(person_a, person_b) < 0.5,
      f'cosine {guards.cosine_similarity(person_a, person_b):.2f}')

stab = LandmarkStabilizer(alpha=0.5)
# Two frames of A, standing still in the same position.
stab.stabilize(make_detection(x=100, embedding=person_a).face)
stab.stabilize(make_detection(x=100, embedding=person_a).face)
smoothed_a = stab._prev_kps.copy()

# B appears at the same position - no centroid jump at all.
b_detection = make_detection(x=100, embedding=person_b).face
out = stab.stabilize(b_detection)
check('a different person at the same position resets smoothing',
      np.allclose(out.kps, b_detection.kps),
      'output is B unsmoothed, not a blend of A and B')

# The position has to move a little between frames or smoothing is
# unobservable: an EMA of two identical keypoint sets is those keypoints. 10px
# is well inside the centroid-jump threshold (0.5 x 200px face width), so this
# isolates the identity test from the geometric one.
stab2 = LandmarkStabilizer(alpha=0.5)
stab2.stabilize(make_detection(x=100, embedding=person_a).face)
stab2.stabilize(make_detection(x=110, embedding=person_a).face)
same = make_detection(x=120, embedding=person_a).face
out = stab2.stabilize(same)
check('the same person keeps smoothing',
      not np.allclose(out.kps, same.kps),
      'output is a blend, so continuity was preserved')

# Alternating identities must reset on *every* switch, not every other one.
stab3 = LandmarkStabilizer(alpha=0.5)
resets = []
for index in range(6):
    who = person_a if index % 2 == 0 else person_b
    det = make_detection(x=100 + index * 10, embedding=who).face
    out = stab3.stabilize(det)
    resets.append(bool(np.allclose(out.kps, det.kps)))
# Alternation used to reset on every frame. It now takes a few frames, because
# a single low reading is no longer treated as proof - see the transient-dip
# block below, which is the realism-critical half of this trade. A detector
# flickering between two people still gets caught, just not instantly.
check('alternating identities are still caught, within the window',
      any(resets[1:]), 'unsmoothed frames: {}'.format(resets))

check('identity survives reset()',
      stab3._prev_embedding is not None,
      'reset drops smoothing state but keeps who was being followed')

# ── Transient dips must not reset ──────────────────────────────────────
# This is the realism-critical case. An embedding is computed from a crop that
# can be motion-blurred for a single frame and recover on the next; resetting on
# that drops the landmark EMA *during movement*, which is exactly when shimmer
# is most visible. A guard reinstating the shimmer it exists to remove would be
# a realism regression caused by a safety feature.
print('\nTransient identity dips (realism protection)')


def run_sequence(similarities, alpha=0.5):
    """Feed embeddings producing the given similarities to person A in turn."""
    stab = LandmarkStabilizer(alpha=alpha)
    unsmoothed = []
    for index, sim in enumerate(similarities):
        # Build a vector at the requested cosine to person_a.
        if sim >= 0.999:
            embedding = person_a
        else:
            perp = person_b - np.dot(person_b, person_a) * person_a
            perp = perp / np.linalg.norm(perp)
            embedding = sim * person_a + np.sqrt(max(0.0, 1 - sim ** 2)) * perp
        det = make_detection(x=100 + index * 8, embedding=embedding).face
        out = stab.stabilize(det)
        unsmoothed.append(bool(np.allclose(out.kps, det.kps)))
    return stab, unsmoothed


# One bad frame in the middle of a steady sequence.
_, flags = run_sequence([1.0, 1.0, 0.1, 1.0, 1.0])
check('a single low frame does not reset smoothing',
      not any(flags[1:]),
      'unsmoothed frames: {} - one blurred embedding must not drop the EMA'
      .format(flags))

# Two bad frames, still under the confirmation limit of 3.
_, flags = run_sequence([1.0, 1.0, 0.1, 0.1, 1.0])
check('two low frames still do not reset',
      not any(flags[1:]), 'unsmoothed frames: {}'.format(flags))

# Three consecutive: a real change, and it must be caught.
_, flags = run_sequence([1.0, 1.0, 0.1, 0.1, 0.1])
check('three consecutive low frames do reset',
      flags[4], 'unsmoothed frames: {}'.format(flags))

check('confirmation is counted over a window, not a consecutive run',
      LandmarkStabilizer._IDENTITY_CONFIRM == 3
      and LandmarkStabilizer._IDENTITY_WINDOW == 6,
      '{} of the last {}'.format(LandmarkStabilizer._IDENTITY_CONFIRM,
                                 LandmarkStabilizer._IDENTITY_WINDOW))

# The window is what closes the hole a consecutive counter leaves:
# good/bad/good/bad zeroes a consecutive counter on every good frame and so
# never fires, but reaches the limit within six frames here.
_, flags = run_sequence([1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0])
check('alternating readings reach the limit within the window',
      any(flags[1:]), 'unsmoothed frames: {}'.format(flags))

# An isolated dip every few frames must never accumulate into a reset.
_, flags = run_sequence([1.0] * 5 + [0.1] + [1.0] * 5 + [0.1] + [1.0] * 5)
check('sparse dips never accumulate into a reset',
      not any(flags[1:]),
      'a blurred frame here and there is normal on a real call')

# The reference must be held while suspicious, or the counter can never climb:
# comparing each frame against the one before it would adopt the new face and
# similarity would snap back to 1.0 on the very next frame.
stab_hold = LandmarkStabilizer(alpha=0.5)
stab_hold.stabilize(make_detection(x=100, embedding=person_a).face)
stab_hold.stabilize(make_detection(x=108, embedding=person_b).face)
held = stab_hold._prev_embedding.copy()
check('the remembered identity is held while a change is unconfirmed',
      float(np.dot(held, person_a) / (np.linalg.norm(held) * np.linalg.norm(person_a))) > 0.99,
      'still comparing against the person being followed, not the intruder')

stab_adopt = LandmarkStabilizer(alpha=0.5)
for index in range(4):
    stab_adopt.stabilize(make_detection(x=100 + index * 8, embedding=person_b).face)
adopted = stab_adopt._prev_embedding.copy()
check('a confirmed change adopts the new identity',
      float(np.dot(adopted, person_b) / (np.linalg.norm(adopted) * np.linalg.norm(person_b))) > 0.99)

check('the default floor sits between the two populations',
      FaceSwapConfig().guard_identity_sim == 0.35,
      'different people ~0.0-0.2, same person >0.9 normally')

disabled = LandmarkStabilizer(alpha=0.5, identity_sim=-1.0)
disabled.stabilize(make_detection(x=100, embedding=person_a).face)
out = disabled.stabilize(make_detection(x=108, embedding=person_b).face)
check('identity resetting can be disabled outright',
      not np.allclose(out.kps, make_detection(x=108).face.kps),
      'guard_identity_sim = -1.0 leaves only the centroid test')

stab4 = LandmarkStabilizer(alpha=0.5)
no_embed = make_detection(x=100).face
no_embed.normed_embedding = None
stab4.stabilize(no_embed)
check('a face with no embedding does not crash the reset check', True)

# -- Source review ------------------------------------------------------
print('\nSource review')


class StubDetector:
    """Returns whatever detections were registered for a given image path."""

    def __init__(self) -> None:
        self.by_shape: dict = {}

    def detect(self, frame):
        return self.by_shape.get(frame.shape[0], [])

    def detect_source(self, frame):
        # Source review goes through this rather than `detect`, so that a
        # photo's verdict does not move with the capture preset's det_size.
        return self.detect(frame)

    @staticmethod
    def select_primary(detections):
        return FaceDetector.select_primary(detections)


def write_image(name: str, height: int, blur: bool = False) -> str:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (height, 400, 3), dtype=np.uint8)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), 8.0)
    path = os.path.join(WORK, name)
    cv2.imwrite(path, image)
    return path


config = FaceSwapConfig()
detector = StubDetector()

good = write_image('good.png', 400)
two_faces = write_image('two.png', 401)
blurry = write_image('blurry.png', 402, blur=True)

detector.by_shape[400] = [make_detection(size=200, embedding=person_a)]
detector.by_shape[401] = [make_detection(size=200, x=10), make_detection(size=200, x=300)]
detector.by_shape[402] = [make_detection(size=200, embedding=person_a)]

db = FaceDatabase(detector, config)
review = db.review_sources([good, two_faces, blurry])

check('good image accepted', good in review.accepted)
check('multi-face image rejected',
      any(p == two_faces and r == guards.MULTIPLE_FACES for p, r in review.rejected))
check('blurred image rejected',
      any(p == blurry and r == guards.BLURRED for p, r in review.rejected))
check('review is usable but not ok', review.usable and not review.ok,
      'one survived, so a source can be built, but something was refused')

payload = review.to_dict()
check('report names each rejected file', len(payload['rejected']) == 2,
      str([r['name'] for r in payload['rejected']]))
check('report explains each rejection',
      all(r['message'] for r in payload['rejected']),
      str([r['message'][:34] for r in payload['rejected']]))
check('report carries a machine-readable reason too',
      all(r['reason'] for r in payload['rejected']))

# An intruder inside an otherwise consistent batch.
paths = []
for index in range(3):
    height = 500 + index
    paths.append(write_image(f'p{index}.png', height))
    detector.by_shape[height] = [make_detection(size=200, embedding=unit(1))]
intruder_height = 510
intruder = write_image('intruder.png', intruder_height)
detector.by_shape[intruder_height] = [make_detection(size=200, embedding=unit(99))]

db2 = FaceDatabase(StubDetector(), config)
db2.detector = detector
review2 = db2.review_sources(paths + [intruder])
check('the odd one out of four is rejected',
      [p for p, r in review2.rejected] == [intruder],
      f'rejected {[os.path.basename(p) for p, _ in review2.rejected]}')
check('the other three survive', len(review2.accepted) == 3)

# Two images that disagree: report the conflict, do not guess.
pair = [write_image('x.png', 600), write_image('y.png', 601)]
detector.by_shape[600] = [make_detection(size=200, embedding=unit(1))]
detector.by_shape[601] = [make_detection(size=200, embedding=unit(99))]
review3 = FaceDatabase(detector, config).review_sources(pair)
check('two conflicting images are both refused', len(review3.rejected) == 2,
      'with no majority there is no way to tell which is the intruder')
check('the conflict message asks for a third image',
      all('third' in review3.messages[p] for p, _ in review3.rejected))

check('guards are skipped without a config',
      FaceDatabase(detector).review_sources(pair).accepted == pair,
      'a database built without thresholds must not silently reject')

# -- Held frame on the live path ----------------------------------------
print('\nHeld frame on the live path')
from pipeline.processing.pipeline import ProcessingPipeline


class StubPipeline(ProcessingPipeline):
    """Pipeline whose detection and swap stages are scripted."""

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.script: list = []
        self.swap_returns_none = False
        self._build_processors()

    def _build_processors(self):
        self._preprocessing_proc = MagicMock()
        self._preprocessing_proc.process = lambda f: f
        self._detection_proc = MagicMock()
        self._detection_proc.process = lambda f: f
        self._swapping_proc = MagicMock()
        self._swapping_proc.source_face = object()
        self._compositor = MagicMock()
        self._stabilizer = MagicMock()
        self._stabilizer.stabilize = lambda face: face

    def _swap_face(self, frame, face):
        # A frame filled with 99 stands for one the compositor refuses - the
        # occlusion guard, or a compositing failure. Keyed off the frame rather
        # than a flag so a single run can cover both outcomes without a second
        # EventBus subscription.
        if self.swap_returns_none or int(frame[1, 1, 0]) == 99:
            return None
        # A "swapped" frame is marked so it can be told from the input.
        out = frame.copy()
        out[0, 0] = 255
        return out


def run_frames(pipe, frames_and_detections):
    """Feed frames through _process_and_emit, returning what was emitted."""
    emitted = []
    pipe.bus.on(FRAME_READY, lambda **kw: emitted.append(kw['frame']))
    for seq, (frame, dets) in enumerate(frames_and_detections):
        pipe._detection_proc.all_detections = dets
        pipe._detection_proc.latest_detections = dets[:1]
        pipe._process_and_emit(frame, seq)
    import time
    time.sleep(0.3)
    return emitted


def blank(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, np.uint8)


config = FaceSwapConfig()
one_face = [make_detection(size=200)]
two = [make_detection(size=200, x=10), make_detection(size=200, x=400)]

pipe = StubPipeline(config, EventBus())
emitted = run_frames(pipe, [
    (blank(10), one_face),   # good - becomes the held frame
    (blank(20), two),        # guarded: two faces
    (blank(30), one_face),   # good again
])
check('three frames in, three out', len(emitted) == 3, f'{len(emitted)} emitted')
check('the guarded frame emits the previous swapped frame',
      emitted[1][0, 0, 0] == 255 and emitted[1][1, 1, 0] == 10,
      'held frame is frame 1 swapped, not frame 2')
check('nothing is drawn on the held frame',
      np.array_equal(emitted[1], emitted[0]),
      'byte-for-byte identical - no banner, border or tint')
check('the raw frame is never emitted',
      all(f[0, 0, 0] == 255 for f in emitted),
      'every emitted frame carries the swap marker')

pipe2 = StubPipeline(config, EventBus())
emitted2 = run_frames(pipe2, [(blank(10), two), (blank(20), two)])
check('guarded from the first frame emits nothing',
      len(emitted2) == 0,
      'there is no augmented frame to hold, and raw is never a fallback')

pipe3 = StubPipeline(config, EventBus())
emitted3 = run_frames(pipe3, [
    (blank(10), one_face),   # good - becomes the held frame
    (blank(99), one_face),   # composite refused: single face, guards all pass
])
check('a refused composite holds the last frame too',
      len(emitted3) == 2 and np.array_equal(emitted3[1], emitted3[0]),
      f'{len(emitted3)} emitted; occlusion guard and compositing failure both hold')

pipe4 = StubPipeline(config, EventBus())
run_frames(pipe4, [(blank(10), one_face), (blank(20), two)])
check('a guarded frame resets temporal state',
      pipe4._compositor.reset.called and pipe4._stabilizer.reset.called,
      'neither EMA may absorb a guarded frame')

pipe5 = StubPipeline(config, EventBus())
run_frames(pipe5, [(blank(10), one_face)])
check('a good frame does not reset temporal state',
      not pipe5._compositor.reset.called)

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
