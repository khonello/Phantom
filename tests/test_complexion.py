"""
The complexion reading — RESEMBLANCE.md §3.1, face half.

What is pinned here is what would otherwise fail silently. A reading that
sampled the whole crop rather than the skin would still return a plausible
LAB triple; a source reference that averaged rather than took the median would
still print three numbers; and a compositor that never called the measurement
would leave `last_complexion` empty in a way indistinguishable from "nothing
to read". Each of those is a confident wrong answer, and each is a check below.

No detector, no model: faces are supplied with keypoints on the FFHQ template
so the canonical warp is a known scale, and "photographs" are flat colours
with features painted where the mask exclusions will land.
"""

import os
import sys
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import tempfile
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

from pipeline.services import complexion
from pipeline.services.readings import Readings
from pipeline.processing.geometry import FFHQ_TEMPLATE

logging.disable(logging.INFO)

WORK = tempfile.mkdtemp(prefix='phantom-complexion-test-')

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def make_face(x: float, y: float, size: float) -> MagicMock:
    """A face whose five keypoints sit where the FFHQ template expects them."""
    face = MagicMock()
    face.kps = (FFHQ_TEMPLATE * size + np.array([x, y])).astype(np.float32)
    face.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
    face.landmark_2d_106 = None
    face.pose = None
    return face


def lab_to_bgr(lab: tuple) -> tuple:
    """One LAB triple (OpenCV 8-bit) as a BGR triple."""
    pixel = np.array([[lab]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0])


def photo(skin_lab: tuple, face: MagicMock, canvas: int = 700) -> np.ndarray:
    """
    A flat 'face' of one skin colour, with red lips and dark eyes painted
    exactly where the FFHQ template puts them — so the test can tell a
    reading that sampled skin from one that sampled everything.
    """
    image = np.full((canvas, canvas, 3), lab_to_bgr(skin_lab), dtype=np.uint8)
    kps = face.kps
    size = float(face.bbox[2] - face.bbox[0])
    for index, colour, radius in (
            (0, (20, 20, 20), 0.06), (1, (20, 20, 20), 0.06),
            (3, (40, 40, 200), 0.08), (4, (40, 40, 200), 0.08)):
        centre = (int(kps[index][0]), int(kps[index][1]))
        cv2.circle(image, centre, int(radius * size), colour, -1)
    # And a background nothing like skin, outside the hull.
    image[: int(face.bbox[1]) - 20, :] = (255, 255, 0)
    return image


def write(name: str, image: np.ndarray) -> str:
    path = os.path.join(WORK, name)
    cv2.imwrite(path, image)
    return path


FACE = make_face(x=150.0, y=150.0, size=400.0)

# Two complexions well apart in chroma — a warm, darker skin and a cool,
# lighter one — and a target between them for the transfer checks.
SOURCE = (120, 148, 152)
TARGET = (170, 136, 138)

print('=' * 70)
print('Complexion reading')
print('=' * 70)

# ── Sampling skin, not the face ────────────────────────────────────────
print('\nSampling')

sample = complexion.skin_lab(photo(SOURCE, FACE), FACE)
check('a flat face yields its own colour', sample is not None)
assert sample is not None
check('the reading is the skin colour, not a mean over lips and eyes',
      complexion.chroma_distance(sample, SOURCE) < 1.5
      and abs(float(sample[0]) - SOURCE[0]) < 2.0,
      'read L{:.0f} a{:.0f} b{:.0f} for L{} a{} b{}'.format(
          sample[0], sample[1], sample[2], *SOURCE))

# Paint the lips and eyes far larger, so a mean over the hull would drift.
loud = photo(SOURCE, FACE)
for index in (0, 1, 3, 4):
    cv2.circle(loud, (int(FACE.kps[index][0]), int(FACE.kps[index][1])),
               int(0.16 * 400), (40, 40, 220), -1)
sample_loud = complexion.skin_lab(loud, FACE)
assert sample_loud is not None
check('exaggerated features barely move the median',
      complexion.chroma_distance(sample_loud, SOURCE) < 3.0,
      '{:.1f} units'.format(complexion.chroma_distance(sample_loud, SOURCE)))

check('a face without keypoints reads nothing',
      complexion.skin_lab(photo(SOURCE, FACE), MagicMock(kps=None)) is None)

# ── The source reference ───────────────────────────────────────────────
print('\nSource reference')

near = [(SOURCE[0], SOURCE[1] + d, SOURCE[2] - d) for d in (-1, 0, 1)]
off = (SOURCE[0] + 30, SOURCE[1] - 20, SOURCE[2] + 25)   # one bad white balance
paths = [write('src{}.png'.format(i), photo(c, FACE)) for i, c in enumerate(near)]
paths.append(write('src_off.png', photo(off, FACE)))

reference = complexion.measure_source([(p, FACE) for p in paths])
check('a source set yields a reference', reference is not None)
assert reference is not None
check('every readable photograph contributes', reference.photographs == 4)
check('one badly balanced photograph does not move the consensus',
      complexion.chroma_distance(reference.lab, SOURCE) < 2.0,
      '{:.1f} units off with an outlier {:.1f} away'.format(
          complexion.chroma_distance(reference.lab, SOURCE),
          complexion.chroma_distance(off, SOURCE)))
check('the spread reports the disagreement rather than hiding it',
      0.5 < reference.spread < 10.0, '{:.1f}'.format(reference.spread))
check('an unreadable path is skipped, not fatal',
      complexion.measure_source(
          [(os.path.join(WORK, 'nope.png'), FACE)] + [(paths[0], FACE)])
      is not None)
check('no readable photograph means no reference',
      complexion.measure_source([(os.path.join(WORK, 'nope.png'), FACE)]) is None)
check('frames are accepted in place of paths',
      complexion.measure_source([(photo(SOURCE, FACE), FACE)]) is not None)

# ── The output reading ─────────────────────────────────────────────────
print('\nOutput reading')

target_frame = photo(TARGET, FACE)
kept_target = complexion.measure_output(target_frame, target_frame.copy(), FACE, reference)
assert kept_target is not None
check('the gap is the pairing: source against target',
      abs(kept_target.gap - complexion.chroma_distance(SOURCE, TARGET)) < 2.5,
      '{:.1f} vs {:.1f}'.format(kept_target.gap, complexion.chroma_distance(SOURCE, TARGET)))
check('an output that kept the target\'s skin reads the whole gap from the source',
      abs(kept_target.face - kept_target.gap) < 2.5,
      'face {:.1f} gap {:.1f}'.format(kept_target.face, kept_target.gap))
check('and reads zero from the target', kept_target.target < 1.5,
      '{:.1f}'.format(kept_target.target))

took_source = complexion.measure_output(target_frame, photo(SOURCE, FACE), FACE, reference)
assert took_source is not None
check('an output that took the source\'s skin reads near zero from the source',
      took_source.face < 2.5, '{:.1f}'.format(took_source.face))
check('and reads the whole gap from the target',
      abs(took_source.target - took_source.gap) < 2.5,
      'target {:.1f} gap {:.1f}'.format(took_source.target, took_source.gap))
check('lightness is signed and separate from the chroma distances',
      abs(kept_target.lightness - (TARGET[0] - SOURCE[0])) < 3.0
      and abs(took_source.lightness) < 3.0,
      'kept {:+.0f} took {:+.0f}'.format(kept_target.lightness, took_source.lightness))

names = set(took_source.as_readings())
check('the four readings carry the documented names',
      names == {'complexion_gap', 'complexion_face',
                'complexion_target', 'complexion_lum'}, str(sorted(names)))

# ── The report ─────────────────────────────────────────────────────────
print('\nReport')

readings = Readings()
for _ in range(5):
    for name, value in kept_target.as_readings().items():
        readings.record(name, value)
text = readings.format_report()
check('the REALISM block names whose complexion the face has',
      "TARGET's complexion" in text, text.splitlines()[-3][:60])
check('and says how far apart the pair was',
      'differ by {:.1f} LAB units'.format(kept_target.gap) in text)

readings = Readings()
for _ in range(5):
    for name, value in took_source.as_readings().items():
        readings.record(name, value)
text = readings.format_report()
check('a transferred complexion sends the reader to the neck',
      "SOURCE's complexion" in text and 'neck' in text)

close = Readings()
for _ in range(5):
    close.record('complexion_gap', 1.0)
    close.record('complexion_face', 0.8)
text = close.format_report()
check('a pair that already agrees is said to, and nothing is recommended',
      'already agree' in text and 'Route A' not in text)

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
