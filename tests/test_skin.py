"""
The skin segmentation service — RESEMBLANCE.md §3.2, seeded backend.

What would fail silently: a body mask that quietly includes the face (every
neck reading becomes a face reading), one that fires on the shirt or the
wall (the neck reading becomes a wall reading), one that misses a neck in
shadow (the seam reading is absent and the seam is real), and a mask that
does not clear on reset (a new person's neck graded with the last one's
colour model). Each is a check below, on a synthetic frame with each region
painted where the test can find it.

No detector: the face is supplied with keypoints on the FFHQ template.
"""

import os
import sys
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

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.services import skin
from pipeline.services import complexion
from pipeline.processing.geometry import FFHQ_TEMPLATE

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def lab_to_bgr(lab: tuple) -> tuple:
    pixel = np.array([[lab]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0])


W, H = 640, 360
FX, FY, FS = 240.0, 60.0, 160.0
SKIN = (150, 146, 150)


def make_face() -> MagicMock:
    face = MagicMock()
    face.kps = (FFHQ_TEMPLATE * FS + np.array([FX, FY])).astype(np.float32)
    face.bbox = np.array([FX, FY, FX + FS, FY + FS], dtype=np.float32)
    face.landmark_2d_106 = None
    face.pose = None
    return face


def scene(skin_lab: tuple = SKIN, neck_shade: float = 0.75,
          wall_patch: bool = True) -> np.ndarray:
    """
    A person against a blue wall: face, a neck in shadow, a hand catching
    more light, a dark shirt — and, optionally, a skin-coloured patch of wall
    in the top-right corner, which is the seeded model's known blind spot.
    """
    frame = np.full((H, W, 3), (200, 120, 40), dtype=np.uint8)
    if wall_patch:
        frame[0:100, 520:640] = lab_to_bgr((160, 144, 148))
    face = make_face()
    cv2.ellipse(frame, (int(FX + FS / 2), int(FY + FS / 2)),
                (int(FS * 0.42), int(FS * 0.5)), 0, 0, 360, lab_to_bgr(skin_lab), -1)
    for index, colour in ((0, (20, 20, 20)), (1, (20, 20, 20)),
                          (3, (40, 40, 200)), (4, (40, 40, 200))):
        cv2.circle(frame, (int(face.kps[index][0]), int(face.kps[index][1])),
                   8, colour, -1)
    shaded = (int(skin_lab[0] * neck_shade), skin_lab[1], skin_lab[2])
    cv2.rectangle(frame, (int(FX + FS * 0.3), int(FY + FS * 0.95)),
                  (int(FX + FS * 0.7), int(FY + FS * 1.4)), lab_to_bgr(shaded), -1)
    cv2.ellipse(frame, (90, 300), (40, 55), 0, 0, 360,
                lab_to_bgr((min(255, skin_lab[0] + 20), skin_lab[1], skin_lab[2] - 1)), -1)
    cv2.rectangle(frame, (150, int(FY + FS * 1.4)), (500, H), (60, 60, 60), -1)
    return frame


NECK = (slice(int(FY + FS * 1.0), int(FY + FS * 1.35)),
        slice(int(FX + FS * 0.35), int(FX + FS * 0.65)))
HAND = (slice(260, 340), slice(60, 120))
WALL = (slice(200, 300), slice(540, 640))
PATCH = (slice(10, 90), slice(530, 630))
SHIRT = (slice(300, 350), slice(200, 450))
FACE_BOX = (slice(int(FY + FS * 0.3), int(FY + FS * 0.7)),
            slice(int(FX + FS * 0.3), int(FX + FS * 0.7)))


def coverage(mask: np.ndarray, region: tuple) -> float:
    return float(mask[region].mean())


print('=' * 70)
print('Skin segmentation')
print('=' * 70)

# ── Registry ───────────────────────────────────────────────────────────
print('\nRegistry')

check('the default backend is built and needs no file',
      skin.resolve(None).key == 'seeded' and not skin.resolve(None).needs_file)
check('an unknown name resolves to the default rather than raising',
      skin.resolve('nonsense').key == 'seeded')
check('a registered but unbuilt backend resolves to the default',
      skin.resolve('mediapipe_multiclass').key == 'seeded')
check('the config default is a built backend',
      skin.resolve(FaceSwapConfig().skin_model).built)

# ── The seeded backend on a synthetic scene ────────────────────────────
print('\nSeeded backend')

face = make_face()
frame = scene()
segmenter = skin.SkinSegmenter()
masks = segmenter.segment(frame, face)
check('a frame with a face yields masks', masks is not None)
assert masks is not None

check('both masks are frame-sized float32 in [0, 1]',
      masks.face.shape == (H, W) and masks.body.shape == (H, W)
      and masks.face.dtype == np.float32 and masks.body.dtype == np.float32
      and float(masks.body.max()) <= 1.0 and float(masks.body.min()) >= 0.0)

check('the neck in shadow is found', coverage(masks.body, NECK) > 0.6,
      '{:.2f}'.format(coverage(masks.body, NECK)))
check('a hand catching more light is found', coverage(masks.body, HAND) > 0.8,
      '{:.2f}'.format(coverage(masks.body, HAND)))
check('the face is NOT in the body mask', coverage(masks.body, FACE_BOX) < 0.02,
      '{:.3f}'.format(coverage(masks.body, FACE_BOX)))
check('the face IS in the face mask', coverage(masks.face, FACE_BOX) > 0.3,
      '{:.2f}'.format(coverage(masks.face, FACE_BOX)))
check('a blue wall is not skin', coverage(masks.body, WALL) < 0.02,
      '{:.3f}'.format(coverage(masks.body, WALL)))
check('a dark shirt is not skin', coverage(masks.body, SHIRT) < 0.02,
      '{:.3f}'.format(coverage(masks.body, SHIRT)))
check('the combined mask is the union',
      np.allclose(masks.skin, np.maximum(masks.face, masks.body)))
check('the cost is measured', segmenter.last_ms > 0.0,
      '{:.1f}ms at 640x360 on this CPU'.format(segmenter.last_ms))

# The known blind spot, pinned so it is a stated limitation and not a
# surprise: a wall the colour of the person's skin is classified as skin.
check('a skin-coloured wall patch IS picked up (known limitation of seeded)',
      coverage(masks.body, PATCH) > 0.8,
      '{:.2f} — the parsing backend is the answer to this'.format(
          coverage(masks.body, PATCH)))

# ── A darker complexion works the same way ─────────────────────────────
print('\nAcross complexions')

dark = scene(skin_lab=(95, 140, 148))
segmenter.reset()
dark_masks = segmenter.segment(dark, face)
assert dark_masks is not None
check('a darker skin finds its own neck', coverage(dark_masks.body, NECK) > 0.6,
      '{:.2f}'.format(coverage(dark_masks.body, NECK)))
check('and its own hand', coverage(dark_masks.body, HAND) > 0.8,
      '{:.2f}'.format(coverage(dark_masks.body, HAND)))
check('and does not take the lighter wall patch for skin',
      coverage(dark_masks.body, PATCH) < 0.2,
      '{:.2f}'.format(coverage(dark_masks.body, PATCH)))

# ── State ──────────────────────────────────────────────────────────────
print('\nState')

segmenter.reset()
check('reset drops the smoothed model',
      segmenter._centre is None and segmenter._body is None)

# A model fitted to one person must not grade the next: after a light face,
# a dark person's neck would fail a stale model. Segment the light scene,
# reset, then the dark one; the dark neck must be found as well as fresh.
segmenter.segment(frame, face)
segmenter.reset()
fresh = segmenter.segment(dark, face)
assert fresh is not None
check('after reset a new person is segmented from scratch',
      coverage(fresh.body, NECK) > 0.6, '{:.2f}'.format(coverage(fresh.body, NECK)))

# Without a reset the parameters are smoothed, so the first frame of a new
# person reads through the previous model — that is what reset is for.
check('a face too small to sample declines rather than raising',
      skin.SkinSegmenter().segment(
          np.zeros((40, 40, 3), np.uint8),
          MagicMock(kps=(FFHQ_TEMPLATE * 8 + 16).astype(np.float32),
                    bbox=np.array([16, 16, 24, 24], np.float32),
                    landmark_2d_106=None, pose=None)) is None)
check('a face without keypoints declines rather than raising',
      skin.SkinSegmenter().segment(frame, MagicMock(kps=None)) is None)

# ── Feeding the complexion reading ─────────────────────────────────────
print('\nFeeding the complexion reading')

source = complexion.SourceComplexion(
    lab=np.array([130.0, 152.0, 156.0]), photographs=3, spread=1.0)

# First, the blind spot's consequence, pinned: with the skin-coloured wall in
# shot, the body median is the WALL (15k pixels against the neck's 2k), so
# `complexion_neck` reads the wall's colour. Not a fixture artefact — this is
# what the seeded backend does in front of a pine door, and why a parsing
# backend is the answer for the neck reading as well as for Route A.
walled = complexion.measure_output(
    frame, frame.copy(), face, source, body=masks.body)
assert walled is not None and walled.seam is not None
check('a skin-coloured wall corrupts the neck reading under seeded (known)',
      walled.seam > 2.0, 'seam {:.1f} on an untouched frame'.format(walled.seam))

frame = scene(wall_patch=False)
segmenter.reset()
masks = segmenter.segment(frame, face)
assert masks is not None
reading = complexion.measure_output(frame, frame.copy(), face, source, body=masks.body)
assert reading is not None
check('with a body mask the neck and seam readings exist',
      reading.neck is not None and reading.seam is not None)
assert reading.neck is not None and reading.seam is not None
check('an untouched frame has no seam: face and neck are the same skin',
      reading.seam < 2.0, '{:.1f}'.format(reading.seam))
check('the neck reads the same distance from the source as the face',
      abs(reading.neck - reading.face) < 2.5,
      'neck {:.1f} face {:.1f}'.format(reading.neck, reading.face))

# Now recolour only the face toward the source — the face-only transfer
# that Route A exists to avoid — and the seam must appear.
recoloured = frame.copy()
cv2.ellipse(recoloured, (int(FX + FS / 2), int(FY + FS / 2)),
            (int(FS * 0.42), int(FS * 0.5)), 0, 0, 360,
            lab_to_bgr((130, 152, 156)), -1)
seamed = complexion.measure_output(frame, recoloured, face, source, body=masks.body)
assert seamed is not None and seamed.seam is not None and seamed.neck is not None
check('a face-only complexion transfer shows up as a seam',
      seamed.seam > 5.0 and seamed.face < 2.5,
      'seam {:.1f}, face {:.1f} from source'.format(seamed.seam, seamed.face))
check('while the neck still reads the target\'s distance',
      abs(seamed.neck - reading.neck) < 1.0)
check('without a body mask the two readings are simply absent',
      'complexion_neck' not in complexion.measure_output(
          frame, frame.copy(), face, source).as_readings())

# ── Summary ────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('{} passed, {} failed'.format(len(PASS), len(FAIL)))
if FAIL:
    for f in FAIL:
        print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
