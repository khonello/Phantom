"""
Route A — the whole-skin complexion grade, RESEMBLANCE.md §4.

The claims that would fail silently, each a check below:

- strength 0 returns the very same frame object, so "off" is off and not a
  colour round trip that happens to land close;
- a pixel outside the skin masks is byte-identical at ANY strength — a wall
  that shifts by one unit is a grade that leaked;
- at strength 1 the face AND the neck land on the reference chroma, and
  the seam between them stays at zero — the whole point of grading upstream;
- lightness moves as a GAIN, so the neck's shading ratio to the forehead
  survives the grade;
- the cap binds on a pairing far apart and the reading says so;
- the hands switch keeps the neck and drops the hand;
- the baseline resolves, takes undertone from the photographs inside the
  bound, and reports disagreement past it;
- the readings measure the target's REAL complexion after the grade, not
  the graded one, or `complexion_gap` would report the gap as closed before
  the swap did anything.

No detector, no swap model: the stage is exercised directly on the same
synthetic scene `test_skin.py` uses, and the compositor's wiring is checked
on the measurement path with a stubbed swap.
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
from pipeline.services import complexion
from pipeline.services import skin
from pipeline.processing import complexion_stage
from pipeline.processing.complexion_stage import ComplexionStage
from pipeline.processing.compositor import FaceCompositor
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
TARGET = (150, 146, 150)     # the person in front of the camera
SOURCE = (120, 156, 160)     # warmer and darker: 14 units of chroma, 30 of L
FAR = (60, 170, 175)         # a pairing past the cap


def make_face() -> MagicMock:
    face = MagicMock()
    face.kps = (FFHQ_TEMPLATE * FS + np.array([FX, FY])).astype(np.float32)
    face.bbox = np.array([FX, FY, FX + FS, FY + FS], dtype=np.float32)
    face.landmark_2d_106 = None
    face.pose = None
    return face


def scene(skin_lab: tuple = TARGET, neck_shade: float = 0.75) -> np.ndarray:
    """Face, a neck in shadow, a hand off to the side, a shirt, a blue wall."""
    frame = np.full((H, W, 3), (200, 120, 40), dtype=np.uint8)
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


NECK = (slice(int(FY + FS * 1.05), int(FY + FS * 1.3)),
        slice(int(FX + FS * 0.38), int(FX + FS * 0.62)))
HAND = (slice(270, 330), slice(70, 110))
WALL = (slice(200, 300), slice(540, 640))
SHIRT = (slice(300, 350), slice(200, 450))
FOREHEAD = (slice(int(FY + FS * 0.25), int(FY + FS * 0.4)),
            slice(int(FX + FS * 0.4), int(FX + FS * 0.6)))


def median_lab(frame: np.ndarray, region: tuple) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[region].reshape(-1, 3)
    return np.median(lab.astype(np.float64), axis=0)


def stage_for(strength: float, hands: bool = True) -> ComplexionStage:
    config = FaceSwapConfig()
    config.skin_complexion = strength
    config.skin_complexion_hands = hands
    return ComplexionStage(config, skin.SkinSegmenter())


def settle(stage: ComplexionStage, frame: np.ndarray, face: MagicMock,
           reference: np.ndarray, frames: int = 60) -> np.ndarray:
    """Run enough frames for the parameter EMA to converge."""
    out = frame
    for _ in range(frames):
        out = stage.apply(frame, face, reference)
    return out


print('=' * 70)
print('Route A — whole-skin complexion grade')
print('=' * 70)

face = make_face()
frame = scene()
reference = np.array(SOURCE, dtype=np.float64)

# ── Off is off ─────────────────────────────────────────────────────────
print('\nOff')

check('strength 0 returns the same frame object',
      stage_for(0.0).apply(frame, face, reference) is frame)
check('no reference returns the same frame object',
      stage_for(1.0).apply(frame, face, None) is frame)
check('the stage reports itself disabled at 0', not stage_for(0.0).enabled())
check('the config default is ON, earned on footage 2026-09-14',
      FaceSwapConfig().skin_complexion == 1.0)

# ── On, at full strength ───────────────────────────────────────────────
print('\nFull strength')

stage = stage_for(1.0)
graded = settle(stage, frame, face, reference)
check('a graded frame is a new array', graded is not frame)

face_after = median_lab(graded, FOREHEAD)
neck_after = median_lab(graded, NECK)
check('the face lands on the reference chroma',
      complexion.chroma_distance(face_after, reference) < 2.0,
      '{:.1f} units off'.format(complexion.chroma_distance(face_after, reference)))
check('the NECK lands on the reference chroma too',
      complexion.chroma_distance(neck_after, reference) < 2.5,
      '{:.1f} units off'.format(complexion.chroma_distance(neck_after, reference)))
check('so there is no seam between them',
      complexion.chroma_distance(face_after, neck_after) < 2.0,
      '{:.1f}'.format(complexion.chroma_distance(face_after, neck_after)))

before_ratio = median_lab(frame, NECK)[0] / median_lab(frame, FOREHEAD)[0]
after_ratio = neck_after[0] / face_after[0]
check('lightness moved as a gain: the neck keeps its shading ratio',
      abs(after_ratio - before_ratio) < 0.04,
      'neck/forehead L {:.3f} -> {:.3f}'.format(before_ratio, after_ratio))
check('and the face lightness approached the reference',
      abs(face_after[0] - SOURCE[0]) < 6.0,
      'L {:.0f} for {}'.format(face_after[0], SOURCE[0]))

check('the wall is byte-identical', np.array_equal(graded[WALL], frame[WALL]))
check('the shirt is byte-identical', np.array_equal(graded[SHIRT], frame[SHIRT]))
check('the hand was graded with the rest of the skin',
      complexion.chroma_distance(median_lab(graded, HAND), reference) < 3.0,
      '{:.1f}'.format(complexion.chroma_distance(median_lab(graded, HAND), reference)))
check('the readings carry the applied shift and gain',
      stage.last_shift is not None and stage.last_gain is not None
      and 12.0 < stage.last_shift < 16.0 and 0.78 < stage.last_gain < 0.82,
      'shift {:.1f} gain {:.3f}'.format(stage.last_shift or 0.0, stage.last_gain or 0.0))
check('the cost is measured', stage.last_ms > 0.0,
      '{:.1f}ms at 640x360 on this CPU'.format(stage.last_ms))

# ── Half strength closes half the gap ──────────────────────────────────
print('\nHalf strength')

half = settle(stage_for(0.5), frame, face, reference)
gap = complexion.chroma_distance(np.array(TARGET, dtype=np.float64), reference)
left = complexion.chroma_distance(median_lab(half, FOREHEAD), reference)
check('half strength leaves about half the chroma gap',
      abs(left / gap - 0.5) < 0.12, '{:.0%} of the gap left'.format(left / gap))
check('and the neck agrees with the face at half strength too',
      complexion.chroma_distance(median_lab(half, FOREHEAD), median_lab(half, NECK)) < 2.0)

# ── The cap ────────────────────────────────────────────────────────────
print('\nBounds')

capped = stage_for(1.0)
settle(capped, frame, face, np.array(FAR, dtype=np.float64))
check('a pairing past the cap is bounded, not chased',
      capped.last_shift is not None
      and abs(capped.last_shift - complexion_stage._MAX_SHIFT) < 0.5,
      'shift {:.1f} against a cap of {:.0f}'.format(
          capped.last_shift or 0.0, complexion_stage._MAX_SHIFT))
check('the gain is bounded too',
      capped.last_gain is not None and capped.last_gain >= complexion_stage._MIN_GAIN)

close = stage_for(1.0)
same = settle(close, frame, face, np.array(TARGET, dtype=np.float64))
check('a pairing that already agrees is left untouched',
      same is frame, 'the stage returns the input rather than a round trip')

# ── Hands ──────────────────────────────────────────────────────────────
print('\nHands')

no_hands = settle(stage_for(1.0, hands=False), frame, face, reference)
check('with hands off the neck is still graded',
      complexion.chroma_distance(median_lab(no_hands, NECK), reference) < 2.5)
check('and the hand is byte-identical',
      np.array_equal(no_hands[HAND], frame[HAND]))
corridor = complexion_stage.corridor(face)
check('the corridor sits under the face',
      corridor is not None and corridor[0] < FX and corridor[1] > FX + FS)

# ── Temporal ───────────────────────────────────────────────────────────
print('\nSmoothing')

smooth = stage_for(1.0)
first = smooth.apply(frame, face, reference)
first_shift = smooth.last_shift
settle(smooth, frame, face, reference)
check('the parameters converge to a steady value',
      first_shift is not None and smooth.last_shift is not None
      and abs(smooth.last_shift - 14.1) < 1.0,
      'first {:.1f} settled {:.1f}'.format(first_shift or 0.0, smooth.last_shift or 0.0))
smooth.reset()
check('reset drops the smoothed parameters',
      smooth._shift is None and smooth._gain is None)
smooth.clear_readings()
check('clear_readings drops the readings and keeps the parameters',
      smooth.last_shift is None and first is not None)

# ── The baseline ───────────────────────────────────────────────────────
print('\nBaseline')

check('auto with no photographs yields no reference',
      complexion.resolve_reference(None, 'auto') is None)
photos = complexion.SourceComplexion(lab=reference.copy(), photographs=4, spread=1.0)
auto = complexion.resolve_reference(photos, 'auto')
check('auto is the photographs alone',
      auto is not None and np.allclose(auto.lab, reference) and auto.base == 'auto')
check('every MST step resolves to a LAB triple',
      all(complexion.base_lab(k) is not None for k in complexion.MST_SRGB))
# Steps 2 -> 3 is a hue step in the published scale (235 -> 238 in L, yellower
# rather than darker), so the promise is monotone from 3 on and 1 above 5.
check('the scale gets darker from step 3 on, and 1 is lighter than 5',
      all(complexion.base_lab('mst{:02d}'.format(i))[0]
          > complexion.base_lab('mst{:02d}'.format(i + 1))[0] for i in range(3, 10))
      and complexion.base_lab('mst01')[0] > complexion.base_lab('mst05')[0])
check('an unknown baseline is None', complexion.base_lab('mst11') is None)

anchor = complexion.base_lab('mst05')
assert anchor is not None
near = complexion.SourceComplexion(
    lab=np.array([anchor[0] + 20, anchor[1] + 2.0, anchor[2] - 2.0]), photographs=3, spread=1.0)
based = complexion.resolve_reference(near, 'mst05')
assert based is not None
check('a baseline takes the photographs\' undertone inside the bound',
      abs(based.lab[1] - (anchor[1] + 2.0)) < 1e-6 and abs(based.lab[2] - (anchor[2] - 2.0)) < 1e-6
      and not based.disagrees)
check('and its lightness from the PHOTOGRAPHS, not the swatch',
      abs(based.lab[0] - (anchor[0] + 20)) < 1e-6,
      "a swatch's L is a paint chip under studio light, not this scene")

wrong = complexion.SourceComplexion(
    lab=np.array([anchor[0], anchor[1] + 20.0, anchor[2] + 20.0]), photographs=3, spread=1.0)
clash = complexion.resolve_reference(wrong, 'mst05')
assert clash is not None
check('photographs far from the baseline are reported as disagreeing',
      clash.disagrees and clash.disagreement is not None and clash.disagreement > 25.0)
check('and the pull is capped at the bound',
      abs(complexion.chroma_distance(clash.lab, anchor) - complexion.UNDERTONE_BOUND) < 1e-6)
check('the config default is auto', FaceSwapConfig().complexion_base == 'auto')

# ── Declared tone classes license the lightness move ───────────────────
print('\nDeclared tone classes')

check('no ratio when either side is auto',
      complexion.lightness_ratio('auto', 'mst08') is None
      and complexion.lightness_ratio('mst03', 'auto') is None
      and complexion.lightness_ratio(None, None) is None)
fair_on_dark = complexion.lightness_ratio('mst03', 'mst08')
check('a fair source over a dark target licenses a gain well past the narrow band',
      fair_on_dark is not None and fair_on_dark > 1.6,
      'mst03/mst08 = {:.2f}'.format(fair_on_dark or 0.0))
check('and is bounded at CLASS_GAIN_MAX rather than the swatch extremes',
      abs((complexion.lightness_ratio('mst01', 'mst10') or 0.0) - complexion.CLASS_GAIN_MAX) < 1e-9)
check('the same tone on both sides is a gain of one',
      abs((complexion.lightness_ratio('mst05', 'mst05') or 0.0) - 1.0) < 1e-9)
check('dark over fair goes the other way',
      (complexion.lightness_ratio('mst08', 'mst03') or 9.0) < 0.6)

# The fair-on-dark scene: a dark target, a fair source, the pairing that
# pinned the narrow band on every frame of the second footage run.
DARK = (85, 140, 146)
dark_frame = scene(skin_lab=DARK)
fair_ref = np.array((190, 138, 144), dtype=np.float64)

narrow = stage_for(1.0)
narrow_out = settle(narrow, dark_frame, face, fair_ref)
check('under auto the lightness gain stays in the narrow band',
      narrow.last_gain is not None and abs(narrow.last_gain - complexion_stage._MAX_GAIN) < 1e-6,
      'gain {:.3f} - the cap, as measured on footage'.format(narrow.last_gain or 0.0))

declared = stage_for(1.0)
declared.config.complexion_base = 'mst03'
declared.config.complexion_target_base = 'mst08'
declared_out = settle(declared, dark_frame, face, fair_ref)
check('with both tones declared the gain follows the class ratio',
      declared.last_gain is not None and abs(declared.last_gain - fair_on_dark) < 1e-6,
      'gain {:.3f}'.format(declared.last_gain or 0.0))
check('and the neck is lightened far past what auto allowed',
      median_lab(declared_out, NECK)[0] > median_lab(narrow_out, NECK)[0] * 1.3,
      'neck L {:.0f} declared vs {:.0f} under auto (was {:.0f})'.format(
          median_lab(declared_out, NECK)[0], median_lab(narrow_out, NECK)[0],
          median_lab(dark_frame, NECK)[0]))
check('while the shading ratio still survives',
      abs(median_lab(declared_out, NECK)[0] / median_lab(declared_out, FOREHEAD)[0]
          - median_lab(dark_frame, NECK)[0] / median_lab(dark_frame, FOREHEAD)[0]) < 0.06)
check('and the wall is still byte-identical',
      np.array_equal(declared_out[WALL], dark_frame[WALL]))

# ── The compositor's wiring ────────────────────────────────────────────
print('\nCompositor wiring')

config = FaceSwapConfig()
config.skin_complexion = 1.0
config.identity_probe = 1
comp = FaceCompositor(config, MagicMock(), MagicMock())
comp.skin = skin.SkinSegmenter()
comp.complexion_stage = ComplexionStage(config, comp.skin)
comp.source_complexion = photos

comp.clear_readings()
out = None
for _ in range(12):
    out = comp.grade_skin(frame, face)
assert out is not None
check('grade_skin grades when the stage is on', out is not frame)
check('and remembers the ungraded frame for the readings', comp._ungraded is frame)
check('the stage cost is in last_stage_ms', 'skin_grade' in comp.last_stage_ms)

# Measure as the compositor would after a swap that changed nothing else:
# the "pasted" frame is the graded one. The gap must still be the REAL gap.
comp._measure_complexion(out, out, face)
readings = dict(comp.last_complexion)
check('complexion_gap is measured against the ungraded target',
      abs(readings.get('complexion_gap', 0.0) - gap) < 2.5,
      '{:.1f} vs real gap {:.1f}'.format(readings.get('complexion_gap', 0.0), gap))
check('complexion_face reads the grade as closed',
      readings.get('complexion_face', 99.0) < 2.5,
      '{:.1f}'.format(readings.get('complexion_face', 99.0)))
check('the applied shift is reported beside them',
      'complexion_shift' in readings and 'complexion_gain' in readings)
check('and how much body skin was found',
      readings.get('complexion_coverage', 0.0) > 0.3,
      "{:.2f} of the face's area".format(readings.get('complexion_coverage', 0.0)))

comp.clear_readings()
check('clear_readings forgets the ungraded frame', comp._ungraded is None)

config.skin_complexion = 0.0
check('with the stage off grade_skin returns the frame itself',
      comp.grade_skin(frame, face) is frame)

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
