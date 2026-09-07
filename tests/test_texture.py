"""
Exercise the source-texture layer: selection, extraction, and the frame-space add.

The layer exists because the swapped face carries 58% of the frame's
high-frequency energy against an ideal of 1.00, and restoration was measured to
close 7% of that gap. What is tested here is not whether it looks better — that
is a footage question and no assertion can answer it — but the handful of
properties that decide whether it can:

1. **The band is chosen at the size it will be displayed at.** A map built at 512
   and warped onto a 101px face is decimated; this is the single mistake that
   would make the whole layer a no-op while appearing to work.
2. **Detail lands on skin, not on features.** Reprojected eyelashes over the
   swap's own eyes is worse than no texture at all.
3. **The map is normalised**, so `texture_strength` means the same thing for
   every source photograph rather than tracking whichever one had most contrast.
4. **Selection picks one image, not an average**, and prefers the sharp one.
5. **Off means off** — at strength 0 the composited frame is bit-identical to
   what it was before this existed.
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

from pipeline.config import FaceSwapConfig
from pipeline.processing import texture
from pipeline.services.masking import FaceMasker
from pipeline.services.readings import Readings
from pipeline.processing.geometry import (
    ALIGNED_STEPS,
    DETAIL_SIGMA,
    DETAIL_SIGMA_REFERENCE,
    FFHQ_TEMPLATE,
    canonical_from_frame,
)

logging.disable(logging.INFO)

WORK = tempfile.mkdtemp(prefix='phantom-texture-test-')

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def make_face(x: float, y: float, size: float) -> MagicMock:
    """
    A face whose five keypoints sit where the FFHQ template expects them.

    Built from the template itself rather than by hand, so the canonical warp is
    exactly the identity-plus-scale and a feature's canonical position is
    predictable — which is what lets the mask test assert where the eyes landed.
    """
    face = MagicMock()
    face.kps = (FFHQ_TEMPLATE * size + np.array([x, y])).astype(np.float32)
    face.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
    face.landmark_2d_106 = None
    face.pose = None
    return face


def write_source(name: str, sigma: float, size: int = 700) -> str:
    """
    A synthetic 'photograph': noise blurred by `sigma`.

    Sigma is the knob under test for selection — a larger blur is a softer photo
    and must score lower. Noise rather than a real face because nothing here
    detects; the detection is supplied.
    """
    rng = np.random.default_rng(11)
    image = rng.integers(60, 200, (size, size, 3), dtype=np.uint8)
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    path = os.path.join(WORK, name)
    cv2.imwrite(path, image)
    return path


print('=' * 70)
print('Source texture')
print('=' * 70)

# ── The band follows the face's size in frame ──────────────────────────
print('\nWorking size')

check('a webcam-sized face maps to a small band, not 512',
      texture.map_size(101.0) <= 128,
      'a 101px face gets {}'.format(texture.map_size(101.0)))
check('a large face maps to a large band',
      texture.map_size(500.0) == 512,
      'a 500px face gets {}'.format(texture.map_size(500.0)))
check('every size comes off the compositing ladder',
      all(texture.map_size(e) in ALIGNED_STEPS for e in (60, 101, 200, 333, 900)))
check('the ladder is shared with the compositor, not copied',
      texture._MAP_STEPS is ALIGNED_STEPS)

# ── Extraction ─────────────────────────────────────────────────────────
print('\nExtraction')

SHARP = write_source('sharp.png', sigma=0.0)
SOFT = write_source('soft.png', sigma=4.0)
FACE = make_face(x=150.0, y=150.0, size=400.0)

extracted = texture.extract(SHARP, FACE)
check('a readable source yields a texture', extracted is not None)
assert extracted is not None

check('the crop is cached at the canonical size, not the working size',
      extracted.crop.shape[:2] == (texture.CANONICAL_SIZE, texture.CANONICAL_SIZE),
      str(extracted.crop.shape))
check('the source face extent is recorded', extracted.native_px == 400,
      '{}px'.format(extracted.native_px))
check('an unreadable file is not fatal',
      texture.extract(os.path.join(WORK, 'nope.png'), FACE) is None)
check('a face without keypoints is not fatal',
      texture.extract(SHARP, MagicMock(kps=None)) is None)

# ── The map itself ─────────────────────────────────────────────────────
print('\nDetail map')

small = extracted.detail_for(128)
check('a map is produced at a webcam working size', small is not None)
assert small is not None

check('the map is single-channel', small.ndim == 2, str(small.shape))
check('the map is built at the size asked for, not the canonical size',
      small.shape == (128, 128), str(small.shape))

skin = cv2.resize(extracted.skin, (128, 128))
inside = small[skin > 0.5]
check('the map has near-zero mean inside the mask',
      abs(float(inside.mean())) < 0.05, 'mean {:.4f}'.format(float(inside.mean())))
check('the map is normalised to unit deviation inside the mask',
      abs(float(inside.std()) - 1.0) < 0.15, 'std {:.3f}'.format(float(inside.std())))

# Normalisation is what makes the strength knob portable. Without it the same
# `texture_strength` would add four times as much detail from a contrasty
# photograph as from a flat one, and the number would mean nothing.
soft_texture = texture.extract(SOFT, FACE)
assert soft_texture is not None
soft_map = soft_texture.detail_for(128)
assert soft_map is not None
soft_inside = soft_map[cv2.resize(soft_texture.skin, (128, 128)) > 0.5]
check('a soft source normalises to the same deviation as a sharp one',
      abs(float(soft_inside.std()) - float(inside.std())) < 0.15,
      'sharp {:.3f} vs soft {:.3f}'.format(float(inside.std()), float(soft_inside.std())))

check('maps are memoised per size',
      extracted.detail_for(128) is small)
check('a different size is a different map',
      extracted.detail_for(256) is not small)

# ── Features are excluded ──────────────────────────────────────────────
print('\nSkin mask')

mask = extracted.skin
edge = texture.CANONICAL_SIZE


def at(index: int) -> float:
    point = (FFHQ_TEMPLATE[index] * edge).astype(int)
    return float(mask[point[1], point[0]])


check('the left eye carries no detail', at(0) < 0.01, '{:.3f}'.format(at(0)))
check('the right eye carries no detail', at(1) < 0.01, '{:.3f}'.format(at(1)))
check('the mouth corners carry no detail',
      at(3) < 0.01 and at(4) < 0.01,
      '{:.3f} / {:.3f}'.format(at(3), at(4)))
check('the cheek does carry detail',
      float(mask[int(edge * 0.60), int(edge * 0.28)]) > 0.5,
      '{:.3f}'.format(float(mask[int(edge * 0.60), int(edge * 0.28)])))
check('the mask is not simply everything',
      0.05 < float(mask.mean()) < 0.75, 'coverage {:.2f}'.format(float(mask.mean())))

# ── Geometry round-trips ───────────────────────────────────────────────
print('\nGeometry')

matrix = canonical_from_frame(FACE, 256)
assert matrix is not None
projected = cv2.transform(
    np.asarray(FACE.kps, dtype=np.float32).reshape(-1, 1, 2), matrix,
).reshape(-1, 2)
expected = FFHQ_TEMPLATE * 256
check('frame keypoints land on the canonical template',
      float(np.abs(projected - expected).max()) < 0.5,
      'max error {:.3f}px'.format(float(np.abs(projected - expected).max())))

check('canonical space scales linearly with size',
      np.allclose(
          canonical_from_frame(FACE, 512), canonical_from_frame(FACE, 256) * 2.0,
          atol=1e-3),
      'so a map at one size is the same framing as at another')

# ── The band matches the compositor's ──────────────────────────────────
print('\nBand agreement')

check('the texture layer and _match_detail split at the same sigma',
      DETAIL_SIGMA == 1.5 and DETAIL_SIGMA_REFERENCE == 256.0,
      'additive and multiplicative stages must describe one band')

from pipeline.processing.compositor import FaceCompositor  # noqa: E402

check('the compositor reads that same constant',
      FaceCompositor._DETAIL_SIGMA is DETAIL_SIGMA
      and FaceCompositor._DETAIL_SIGMA_REFERENCE is DETAIL_SIGMA_REFERENCE)

# ── Applying it ────────────────────────────────────────────────────────
print('\nCompositing')

config = FaceSwapConfig()
config.grain = False          # isolated: grain also consumes headroom, tested below
compositor = FaceCompositor(config, MagicMock(available=False), MagicMock())
compositor.source_texture = extracted

# The face being composited *into* is not the face texture was taken *from*.
# A webcam target of ~100px, which is the case the whole layer is aimed at.
TARGET = make_face(x=150.0, y=150.0, size=100.0)
# The region of interest has to actually contain the face: the canonical square
# maps back to frame [150, 250] for this target, and a ROI at the origin would
# put every warped pixel outside the array.
ROI = (150, 150, 120, 120)
alpha = np.ones((120, 120), dtype=np.float32)
aligned_matrix = canonical_from_frame(TARGET, 256).astype(np.float32)

# `swap` is smooth and `real` carries texture — the actual situation: a face
# generated at 128 native sitting in a frame whose skin has detail. The gap
# between them is the headroom the layer is allowed to fill.
rng = np.random.default_rng(5)
swap = cv2.GaussianBlur(
    rng.normal(128, 6, (120, 120, 3)).astype(np.float32), (0, 0), 2.0)
real = rng.normal(128, 6, (120, 120, 3)).astype(np.float32)


def add(strength, blended=None, target=None, mask=None):
    """Run the layer at a strength and hand back what it added."""
    config.texture_strength = strength
    base = swap if blended is None else blended
    return compositor._add_texture(
        base.copy(), real if target is None else target,
        alpha if mask is None else mask, TARGET, 100.0, ROI,
    ) - base


check('texture defaults to off', FaceSwapConfig().texture_strength == 0.0)
check('at strength 0 the frame is untouched',
      not np.any(add(0.0)),
      'the layer must cost nothing and change nothing when off')

delta = add(0.5)
check('at strength 0.5 detail is added', float(np.abs(delta).max()) > 0.5,
      'max {:.2f} units'.format(float(np.abs(delta).max())))
check('the working size follows the target face, not the source',
      texture.map_size(100.0) == 128,
      'a 100px target gets a 128 map, not the 512 crop it was cached at')
# Tolerance rather than exact equality because `delta` is recovered by
# subtracting a ~128 base from a ~128 sum in float32, and that cancellation
# costs about 1e-5 per channel. The added term is monochrome by construction;
# this checks the construction, not the arithmetic used to read it back.
check('the added detail is monochrome',
      np.allclose(delta[:, :, 0], delta[:, :, 1], atol=1e-3)
      and np.allclose(delta[:, :, 1], delta[:, :, 2], atol=1e-3),
      'per-channel high frequency reads as coloured speckle, not skin')
check('the added detail is roughly zero-mean',
      abs(float(delta.mean())) < 0.5,
      'mean {:.3f} — texture must not shift exposure'.format(float(delta.mean())))
check('the stage reports its own cost', 'texture' in compositor.last_stage_ms)

# ── The headroom bound ─────────────────────────────────────────────────
print('\nHeadroom')

headroom = compositor.last_texture_headroom
check('headroom is measured and published', headroom is not None and headroom > 0,
      '{:.2f} units against a real face'.format(headroom or 0.0))
check('added deviation is a fraction of the headroom, not an open gain',
      float(delta[:, :, 0].std()) <= headroom + 0.5,
      '{:.2f} added against {:.2f} available'.format(
          float(delta[:, :, 0].std()), headroom))
# `strength` is spent once, in `_texture_reserve`. It used to be spent again
# here, so the delivered amplitude went as its square — 0.4 asked detail
# matching to stand back by 40%, then filled 40% of what that freed, delivering
# 16%. With the base fixed (no `_match_detail` in front of these calls) the
# headroom is the same at every strength, so below parity every strength must
# now spend the same amount: all of it.
below_parity = [float(add(s)[:, :, 0].std()) for s in (0.3, 0.5, 1.0)]
check('below parity the whole measured headroom is spent, whatever the strength',
      max(below_parity) - min(below_parity) < 1e-3,
      '{} — the reservation is the control, not a second multiplier'.format(
          ', '.join('{:.3f}'.format(v) for v in below_parity)))
check('past parity the knob still multiplies',
      abs(float(add(2.0)[:, :, 0].std()) - 2.0 * below_parity[-1]) < 0.05,
      'the diagnostic overshoot has no reservation left to act through, so '
      'above 1.0 it has to scale the amount directly')

# The case the bound exists for: a swap that already carries as much texture as
# the frame does. Nothing may be added, because past parity the face is noisier
# than the camera that shot it.
nothing = add(1.0, blended=real.copy())
check('a swap already at the frame texture level gets nothing',
      not np.any(nothing),
      'headroom {:.3f} — overshoot is what B1 exists to prevent'.format(
          compositor.last_texture_headroom or 0.0))

smooth_target = cv2.GaussianBlur(real, (0, 0), 3.0)
add(1.0, target=smooth_target)
check('a smoother reference offers less headroom',
      (compositor.last_texture_headroom or 0.0) < headroom,
      'the reference is the real face in these pixels, so it sets the ceiling')

config.grain = True
add(1.0)
grain_headroom = compositor.last_texture_headroom or 0.0
config.grain = False
check('grain is counted against the headroom',
      grain_headroom < headroom,
      '{:.2f} with grain vs {:.2f} without — the two layers must not each '
      'reach the target and overshoot together'.format(grain_headroom, headroom))

check('the compositing mask gates the layer',
      not np.any(add(0.8, mask=np.zeros((120, 120), dtype=np.float32))),
      'texture outside the swap would be a seam made of pores')

compositor.source_texture = None
check('no source texture is not fatal', not np.any(add(0.8)),
      'the layer is decorative; its absence must never break a swap')
compositor.source_texture = extracted

# ── The budget the two stages share ────────────────────────────────────
#
# `_match_detail` and `_add_texture` both aim at the target face's
# high-frequency energy, and the first one to run used to take all of it: the
# clamp was measured binding on 0% of 2267 frames, so parity was reached every
# frame and the texture layer measured only what the warp had lost — p50 0.78 of
# an 8-bit unit. What is pinned here is that the reservation exists, that it is
# conditional on the layer actually running, and that the two stages describe
# the same band. Getting any of the three wrong restores the silent no-op.
print('\nBudget reservation')

config.grain = False
config.texture_strength = 0.0
compositor.source_texture = extracted
extracted.yaw = 0.0

check('no reserve when the layer is off',
      compositor._texture_reserve(TARGET, 100.0) == 0.0,
      'reserving is a deliberate undershoot; with nothing to fill it the '
      'face just comes out softer than before any of this existed')

config.texture_strength = 0.5
check('the reserve follows the strength',
      abs(compositor._texture_reserve(TARGET, 100.0) - 0.5) < 1e-6,
      '{:.2f}'.format(compositor._texture_reserve(TARGET, 100.0)))

config.texture_strength = 1.0
check('the reserve is bounded well short of the whole band',
      compositor._texture_reserve(TARGET, 100.0) <= FaceCompositor._RESERVE_MAX
      < 1.0,
      'a face whose surface is entirely reprojected cannot follow an '
      'expression — that is a different failure, not a better one')

config.texture_strength = 2.0
check('a strength past parity does not reserve past the ceiling',
      compositor._texture_reserve(TARGET, 100.0)
      == FaceCompositor._RESERVE_MAX,
      'the overshoot is spent on top of a full reservation, not through it')

config.texture_strength = 0.5
def _posed(yaw):
    """A target face at a given yaw. Defined here because the pose section
    below runs later and this check must not depend on ordering."""
    f = make_face(x=150.0, y=150.0, size=100.0)
    f.pose = np.array([0.0, yaw, 0.0], dtype=np.float32)
    return f


check('an off-pose frame reserves nothing',
      compositor._texture_reserve(_posed(70.0), 100.0) == 0.0,
      'every gate _add_texture applies before it commits must be applied '
      'here first, or the gap it left goes unfilled')

compositor.source_texture = None
check('no source texture reserves nothing',
      compositor._texture_reserve(TARGET, 100.0) == 0.0)
compositor.source_texture = extracted

# The reservation has to reach the stage and lower what it aims at.
detail_sharp = rng.normal(128, 20, (128, 128, 3)).astype(np.uint8)
detail_soft = cv2.GaussianBlur(detail_sharp, (0, 0), 2.0)
detail_mask = np.ones((128, 128), dtype=np.float32)

compositor._match_detail(detail_soft, detail_sharp, detail_mask, reserve=0.0)
unreserved = compositor.last_detail_ratio or 0.0
compositor._match_detail(detail_soft, detail_sharp, detail_mask, reserve=0.6)
reserved = compositor.last_detail_ratio or 0.0

check('reserving lowers what detail matching aims at',
      reserved < unreserved,
      'wanted {:.2f} reserved vs {:.2f} not'.format(reserved, unreserved))

# The quadrature share is checked *within* the reserve path, against a
# vanishing reserve rather than against no reserve at all. Reserving switches
# the measurement to the basis `_texture_headroom` uses — grayscale, skin only,
# the same centred window — because the two stages reading the same face three
# different ways is what left the reservation short. So the two paths are no
# longer numerically comparable, deliberately, and a test that compared them
# would be pinning the bug rather than the property.
compositor._match_detail(detail_soft, detail_sharp, detail_mask, reserve=1e-4)
barely = compositor.last_detail_ratio or 0.0
check('it lowers it by exactly the quadrature share',
      abs(reserved - barely * float(np.sqrt(1.0 - 0.36))) < 1e-3,
      'independent fields add in quadrature; anything else and the two '
      'stages do not sum to parity')
compositor._match_detail(detail_soft, detail_sharp, detail_mask, reserve=0.6)
check('the reserve is published for the readings',
      compositor.last_detail_reserve == 0.6,
      'a reserve of zero while texture_strength is set is the layer '
      'declining, and looks exactly like a strength set too low')

# The band has to travel with the reservation. Reserving one octave of a field
# measured over three barely moves the headroom — measured at 0.30 -> 0.31,
# which is the bug this check exists to catch.
compositor._match_detail(detail_soft, detail_sharp, detail_mask,
                         reserve=0.5, band=1.0)
narrow = compositor.last_detail_ratio or 0.0
compositor._match_detail(detail_soft, detail_sharp, detail_mask,
                         reserve=0.5, band=3.0)
wide = compositor.last_detail_ratio or 0.0
check('the band travels with the reserve into detail matching',
      abs(wide - narrow) > 1e-3,
      'reserving over a narrower span than the map fills leaves the '
      'headroom where it was: {:.3f} vs {:.3f}'.format(narrow, wide))

# ── The reserve must survive the grain term ────────────────────────────
# The defect this section exists for: the target's band is skin *and* sensor
# noise, and `_add_grain` supplies the noise separately, so aiming detail
# matching at the total spent that budget twice. `_texture_headroom` then
# subtracted `grain^2` from a difference the double count had already closed,
# and returned **exactly zero** across the whole 0.3-0.5 range the layer
# documents as its working range — while detail matching had already undershot
# to honour a reservation nothing then filled. Softer than the layer switched
# off, silently, which is the one outcome `_texture_reserve` exists to prevent.
print('\nThe reserve against grain')

config.grain = True
compositor._match_detail(detail_soft, detail_sharp, detail_mask, reserve=0.5)
discounted = compositor.last_detail_ratio or 0.0
config.grain = False
compositor._match_detail(detail_soft, detail_sharp, detail_mask, reserve=0.5)
undiscounted = compositor.last_detail_ratio or 0.0

check('with grain on, detail matching aims lower still',
      discounted < undiscounted,
      'wanted {:.3f} with grain vs {:.3f} without — matching the swap to '
      'skin plus noise and then adding the noise again spends it '
      'twice'.format(discounted, undiscounted))
compositor._match_detail(detail_soft, detail_sharp, detail_mask, reserve=1e-4)
barely_off = compositor.last_detail_ratio or 0.0
check('with grain off it is exactly the quadrature share and no more',
      abs(undiscounted - barely_off * float(np.sqrt(1.0 - 0.25))) < 1e-3,
      'the discount is gated on grain, so with grain off the reserve is the '
      'only thing lowering the aim')

# The path nothing may disturb: no reserve means no texture layer waiting, and
# every measurement this project has taken was taken there.
before = compositor._match_detail(detail_soft, detail_sharp, detail_mask)
after = compositor._match_detail(detail_soft, detail_sharp, detail_mask,
                                 reserve=0.0, band=3.0)
check('without a reserve the stage is untouched, band included',
      np.array_equal(before, after),
      'the reserve path changes the measurement basis; the unreserved one '
      'must stay bit-identical to what every existing measurement was taken '
      'against')

# End to end, both stages, on a target that carries noise as a real photograph
# does. The property is the one that failed: a reservation made must be a
# reservation filled.
config.grain = True

# **Channel-correlated, and with the noise a realistic share of the band.**
# Both matter. Per-channel independent content puts the pooled deviation
# `_match_detail` measures a factor of sqrt(3) above the grayscale one
# `_texture_headroom` measures, and no arithmetic here survives that; a
# demosaiced photograph is nothing like it. And the noise has to be a
# *plausible* share of the band — measured on a real freckled photograph at
# 275px, the band is 5.50 and the sensor noise 2.32, a ratio of 0.42. These
# numbers reproduce that ratio; drift far above it and the fixture is a
# photograph with no skin detail in it, where finding no headroom is the
# correct answer rather than the defect.
# Its own generator: drawing from the shared one here would shift every
# fixture built after this block and make an unrelated test fail.
_rng = np.random.default_rng(11)
_structure = cv2.GaussianBlur(_rng.normal(0, 45.0, (120, 120)), (0, 0), 1.4)
_sensor = _rng.normal(0, 0.8, (120, 120))
noisy_real = np.clip(
    128.0 + np.repeat((_structure + _sensor)[:, :, None], 3, axis=2),
    0, 255).astype(np.float32)
# A swap that kept 70% of the band: soft, but inside `_DETAIL_RATIO`, which is
# the regime the pipeline actually runs in. A heavier blur puts the wanted gain
# past the clamp, and then the clamp rather than the reserve decides everything.
_low = cv2.GaussianBlur(noisy_real, (0, 0), FaceCompositor._DETAIL_SIGMA * 2.0
                        * 120.0 / FaceCompositor._DETAIL_SIGMA_REFERENCE)
flat_swap = _low + (noisy_real - _low) * 0.7


def two_stage(strength):
    """`_texture_reserve` -> `_match_detail` -> `_add_texture`, as composite does."""
    config.texture_strength = strength
    compositor._warned_no_headroom = True          # measuring, not reporting
    reserve = compositor._texture_reserve(TARGET, 100.0)
    matched = compositor._match_detail(
        flat_swap.copy(), noisy_real, alpha,
        reserve=reserve, band=compositor._texture_shaping()[2],
    ).astype(np.float32)
    out = compositor._add_texture(
        matched.copy(), noisy_real, alpha, TARGET, 100.0, ROI,
    )
    return float((out - matched)[:, :, 0].std()), compositor.last_texture_headroom


working = {s: two_stage(s) for s in (0.3, 0.4, 0.5)}
check('a reservation in the working range is actually filled',
      all(added > 0.0 for added, _ in working.values()),
      'added ' + ', '.join('{:.2f}@{:.1f}'.format(v[0], k)
                           for k, v in working.items())
      + ' — every one of these was 0.00 before the grain term was counted once')
check('and more of it is filled as the strength rises',
      working[0.5][0] > working[0.4][0] > working[0.3][0],
      'the reservation is what the knob moves, so the delivered amount has '
      'to follow it')

# The state that has to be said out loud rather than merely recorded: a reserve
# made, and no room found to fill it. The readings carry the zero, but only a
# stream reports them and a still render is exactly where this lands.
config.texture_strength = 0.5
compositor._warned_no_headroom = False
compositor._add_texture(
    noisy_real.copy(), noisy_real, alpha, TARGET, 100.0, ROI,
)
check('an unfilled reservation says so',
      compositor._warned_no_headroom,
      'a face softer than with the layer off, reported once rather than '
      'per frame')
config.grain = False
config.texture_strength = 0.0

compositor._match_detail(detail_soft, detail_sharp, detail_mask,
                         reserve=0.0, band=3.0)
check('band is ignored when nothing is reserved',
      abs((compositor.last_detail_ratio or 0.0) - unreserved) < 1e-6,
      'a run without the texture layer must split where it always did')

# ── Octaves: one pass per kind of skin feature ─────────────────────────
#
# What separates a pore from a spot from a mole from a crease is *scale*, so the
# map is cut into octaves and each is normalised and weighted on its own. That
# is a general statement about localised skin features and deliberately not a
# detector for any one of them.
print('\nOctaves')

pores_only = extracted.detail_for(128, contrast=1.0, band=1.0, relief=0.0)
with_marks = extracted.detail_for(128, contrast=1.0, band=2.0, relief=0.65)
shaped = extracted.detail_for(128, contrast=2.0, band=2.0, relief=0.65)
assert pores_only is not None and with_marks is not None and shaped is not None

check('the shaping is part of the cache key, not just the size',
      with_marks is not pores_only and shaped is not with_marks,
      'keyed on size alone, a live A/B would compare a setting against a '
      'map built under the previous one')
check('relief 0.0 reproduces the pores-only map exactly',
      extracted.detail_for(128, 1.0, 2.0, 0.0) is not None
      and np.allclose(extracted.detail_for(128, 1.0, 2.0, 0.0), pores_only,
                      atol=1e-5),
      'the new path must contain the old one, or nothing can be A/B\'d '
      'against what shipped')
check('a wider band changes what the map carries',
      not np.allclose(with_marks, pores_only, atol=1e-3),
      'a 4px mark has most of its energy below a 1px high-pass')
check('every variant still normalises to unit deviation',
      all(abs(float(m[cv2.resize(extracted.skin, (128, 128)) > 0.5].std())
              - 1.0) < 0.2
          for m in (pores_only, with_marks, shaped)),
      'the compositor multiplies by a measured budget and depends on this')

# Shaping moves energy from the filler into the marks, at constant total energy.
# That is the whole argument: RMS is the wrong statistic for a sparse field.
def peak_ratio(m):
    """Peak amplitude against RMS, inside the skin mask."""
    inside = m[cv2.resize(extracted.skin, (128, 128)) > 0.5]
    return float(np.percentile(np.abs(inside), 99.5) / max(1e-6, inside.std()))


check('shaping raises peak contrast at the same total energy',
      peak_ratio(shaped) > peak_ratio(with_marks),
      'p99.5/RMS {:.2f} shaped vs {:.2f} flat — a sparse mark is what a '
      'second moment cannot see'.format(
          peak_ratio(shaped), peak_ratio(with_marks)))
check('contrast 1.0 is exactly no shaping',
      extracted.detail_for(128, 1.0, 2.0, 0.65) is with_marks)

check('the cache does not grow without bound',
      (([extracted.detail_for(s, 1.0 + i * 0.1, 2.0, 0.5)
         for i, s in enumerate([128] * 24)]),
       len(extracted._maps) <= texture._MAP_CACHE_MAX)[1],
      '{} entries after a 24-value sweep'.format(len(extracted._maps)))

check('the source photograph reports what it contains',
      len(extracted.octaves) == 2 and extracted.octaves[0] > 0.0,
      'pores {:.2f}, marks {:.2f} — a soft donor and a starved budget look '
      'identical in the output'.format(*extracted.octaves))

check('the strength ceiling reaches past parity for diagnosis',
      texture.STRENGTH_MAX > 1.0,
      'separating "the map is weak" from "the budget is small" must not '
      'need a code change mid-session')

config.texture_strength = 0.0
config.texture_band = FaceSwapConfig().texture_band
config.texture_relief = FaceSwapConfig().texture_relief
config.texture_contrast = FaceSwapConfig().texture_contrast

# ── The seam ───────────────────────────────────────────────────────────
print('\nSeam')

# White swap onto a black frame makes the composite equal to the alpha itself,
# so the transition can be measured directly rather than inferred.
config.grain = False
config.texture_strength = 0.0
FRAME = np.zeros((400, 400, 3), dtype=np.uint8)
WHITE = np.full((256, 256, 3), 255, dtype=np.uint8)
disc = np.zeros((256, 256), dtype=np.float32)
cv2.circle(disc, (128, 128), 90, (1.0,), -1)
seam_matrix = canonical_from_frame(TARGET, 256).astype(np.float32)


def transition_px(feather):
    """Width in frame pixels over which the composite fades in."""
    config.mask_feather = feather
    out = compositor._paste(FRAME, WHITE, disc, seam_matrix, TARGET, 100.0)
    # One edge only. A whole scanline crosses the disc twice, and the span
    # between the two crossings is its diameter rather than its softness.
    row = out[200, :200, 0].astype(np.float32)
    ramp = np.flatnonzero((row > 25) & (row < 230))
    return int(ramp.max() - ramp.min() + 1) if ramp.size else 0


narrow = transition_px(0.01)
wide = transition_px(0.04)
wider = transition_px(0.10)
check('the feather widens with the knob',
      narrow < wide < wider,
      '{}px / {}px / {}px at 1%, 4%, 10% of a 100px face'.format(
          narrow, wide, wider))
check('the default is a real transition, not an edge',
      wide >= 6, '{}px on a 100px face'.format(wide))
check('a floor keeps a small face from having no transition at all',
      transition_px(0.0) > 0,
      'below the floor the frame-space blur would be identity')
check('the feather is bounded, not unlimited',
      transition_px(1.0) == transition_px(0.25),
      'set_realism clamps at 0.25 — a quarter-face feather is a dissolve')

config.mask_feather = 0.04
check('padding grows with the feather, so the blur is not reflected back',
      compositor._region_of_interest(
          cv2.invertAffineTransform(seam_matrix), 256, 400, 400, 32)[0]
      < compositor._region_of_interest(
          cv2.invertAffineTransform(seam_matrix), 256, 400, 400, 4)[0],
      'a mask blurred wider than its ROI never reaches zero at the border')

check('the colour deadband no longer swallows a visible difference',
      FaceCompositor._COLOR_FLOOR < 2.0,
      'floor {:.1f} — a 3-unit LAB step at a boundary used to get no '
      'correction at all'.format(FaceCompositor._COLOR_FLOOR))

# ── Selection ──────────────────────────────────────────────────────────
print('\nSource selection')

from pipeline.services.database import _texture_score  # noqa: E402
from pipeline.types import Bbox, Detection  # noqa: E402


def make_detection(path: str) -> tuple:
    image = cv2.imread(path)
    face = make_face(x=150.0, y=150.0, size=400.0)
    return image, Detection(
        face=face,
        bbox=Bbox(x=150, y=150, w=400, h=400),
        kps=face.kps,
        confidence=0.9,
    )


sharp_score = _texture_score(*make_detection(SHARP))
soft_score = _texture_score(*make_detection(SOFT))
check('the sharper photograph scores higher',
      sharp_score > soft_score,
      'sharp {:.3f} vs soft {:.3f}'.format(sharp_score, soft_score))
check('scores stay in range', 0.0 <= soft_score <= 1.0 and 0.0 <= sharp_score <= 1.0)

big_image, big = make_detection(SHARP)
small_det = Detection(
    face=big.face, bbox=Bbox(x=150, y=150, w=120, h=120),
    kps=big.kps, confidence=0.9,
)
check('a larger face scores higher than a small one at equal sharpness',
      _texture_score(big_image, big) > _texture_score(big_image, small_det),
      'a small face carries a real but thin high-frequency band')

blown = np.full((700, 700, 3), 255, dtype=np.uint8)
check('a blown-out face is penalised',
      _texture_score(blown, big) < _texture_score(big_image, big),
      'clipped pixels carry no texture in either direction')

check('the weights sum to one',
      abs(sum(__import__('pipeline.services.database', fromlist=['x'])
              ._TEXTURE_WEIGHTS.values()) - 1.0) < 1e-9)


# ── Pose confidence ────────────────────────────────────────────────────
print('\nPose confidence')


def posed(yaw):
    """A target face at a given yaw, everything else identical."""
    f = make_face(x=150.0, y=150.0, size=100.0)
    f.pose = np.array([0.0, yaw, 0.0], dtype=np.float32)
    return f


compositor.source_texture = extracted
extracted.yaw = 0.0

check('a matched pose is full confidence',
      compositor._pose_confidence(posed(0.0)) == 1.0)
check('a small disagreement is still full confidence',
      compositor._pose_confidence(posed(10.0)) == 1.0,
      'a head does not sit still; attenuating at every twitch would flicker')
check('confidence falls as the poses diverge',
      1.0 > compositor._pose_confidence(posed(25.0))
      > compositor._pose_confidence(posed(38.0)) > 0.0,
      '{:.2f} at 25 deg, {:.2f} at 38'.format(
          compositor._pose_confidence(posed(25.0)),
          compositor._pose_confidence(posed(38.0))))
check('past the limit the map is not used at all',
      compositor._pose_confidence(posed(60.0)) == 0.0,
      'a source shot frontally says nothing about a profile')
check('the sign of the disagreement does not matter',
      compositor._pose_confidence(posed(30.0))
      == compositor._pose_confidence(posed(-30.0)),
      'only the magnitude is used — the directional term needs footage first')

extracted.yaw = 20.0
check('confidence is measured against the source, not against frontal',
      compositor._pose_confidence(posed(20.0)) == 1.0,
      'an angled source is fine for an equally angled frame')

# Capability gaps must not silently become behaviour changes.
extracted.yaw = None
check('an unmeasurable source pose means full confidence, not none',
      compositor._pose_confidence(posed(80.0)) == 1.0)
extracted.yaw = 0.0
check('an unmeasurable frame pose means full confidence, not none',
      compositor._pose_confidence(make_face(150.0, 150.0, 100.0)) == 1.0,
      'a pack without `pose` would otherwise disable the layer in silence')

# And it actually gates the composite.
config.grain = False
config.texture_strength = 0.8
off_pose = compositor._add_texture(
    swap.copy(), real, alpha, posed(70.0), 100.0, ROI) - swap
check('an off-pose frame gets no texture',
      not np.any(off_pose),
      'confidence {:.2f}'.format(compositor.last_texture_confidence or 0.0))
check('confidence is published for the readings',
      compositor.last_texture_confidence == 0.0)

on_pose = compositor._add_texture(
    swap.copy(), real, alpha, posed(0.0), 100.0, ROI) - swap
check('a matched frame gets the full amount',
      float(np.abs(on_pose).max()) > 0.5)
half = compositor._add_texture(
    swap.copy(), real, alpha, posed(28.5), 100.0, ROI) - swap
check('a partly-disagreeing pose gets a partial amount',
      0.0 < float(half[:, :, 0].std()) < float(on_pose[:, :, 0].std()),
      'confidence scales the amount rather than switching it')

# ── Readings ───────────────────────────────────────────────────────────
print('\nReadings')

readings = Readings()
for value in (1.2, 1.4, 1.6, 1.6, 1.6):
    readings.record('detail_ratio', value, limit=1.6)
report = readings.report()['detail_ratio']

check('a clamped reading reports what share reached the limit',
      report['share_at_limit'] == 0.6, str(report['share_at_limit']))
check('the unclamped value is what is recorded',
      report['max'] == 1.6 and report['n'] == 5)
check('the verdict says to raise the clamp when it binds',
      any('CLAMPED' in n for n in readings.format_report().split('\n')),
      'the whole point of the reading is that it names the next action')

loose = Readings()
for value in (0.9, 1.0, 1.1):
    loose.record('detail_ratio', value, limit=1.6)
check('a clamp that never binds says so instead',
      'not clamp-bound' in loose.format_report(),
      'which is the case for adding real detail rather than amplifying')

empty = Readings()
check('nothing recorded is not an error', 'none recorded' in empty.format_report())
check('reset drops everything', (readings.reset() or readings.report()) == {})

pinned = Readings()
pinned.record('texture_headroom', 0.0)
check('no headroom is called out, not left as a number',
      'nothing to spend' in pinned.format_report(),
      'a layer with no headroom looks exactly like one set too low')

# ── The detail-ratio reading is real, not synthetic ────────────────────
print('\nDetail ratio')

compositor.last_detail_ratio = None
sharp = rng.normal(128, 20, (128, 128, 3)).astype(np.float32)
flat = cv2.GaussianBlur(sharp, (0, 0), 3.0)
full = np.ones((128, 128), dtype=np.float32)
compositor._match_detail(flat.astype(np.uint8), sharp.astype(np.uint8), full)
check('the pre-clamp ratio is published, not the clamped one',
      (compositor.last_detail_ratio or 0.0) > FaceCompositor._DETAIL_RATIO[1],
      'wanted {:.2f} against a clamp of {:.2f} — a percentile of the clamped '
      'value could never show this'.format(
          compositor.last_detail_ratio or 0.0, FaceCompositor._DETAIL_RATIO[1]))

compositor.last_detail_ratio = None
compositor._match_detail(sharp.astype(np.uint8), sharp.astype(np.uint8),
                         np.zeros((128, 128), dtype=np.float32))
check('a stage that did not run records nothing',
      compositor.last_detail_ratio is None,
      'a zero would be a claim about a frame that has no reading')

# -- Scatter --------------------------------------------------------------
print('\nScatter')

# A face-sized aligned crop with shading (a gradient) and texture (noise) on it,
# so the two bands can be told apart in the result.
ALIGNED = 256
scatter_face = make_face(x=0.0, y=0.0, size=float(ALIGNED))
scatter_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
shade = np.tile(
    np.linspace(90, 170, ALIGNED, dtype=np.float32), (ALIGNED, 1))[:, :, None]
grain = rng.normal(0, 8, (ALIGNED, ALIGNED, 3)).astype(np.float32)
crop = np.clip(shade + grain, 0, 255).astype(np.uint8)
full_mask = np.ones((ALIGNED, ALIGNED), dtype=np.float32)


def to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)


def from_lab(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def scatter(strength, mask=None):
    """Run the pass the way the compositor does: LAB in, LAB out.

    The colour conversion is the caller's since `_match_color` needs the same
    space straight afterwards, so the test owns it too.
    """
    config.diffuse_strength = strength
    return from_lab(compositor._scatter(
        to_lab(crop), full_mask if mask is None else mask,
        scatter_face, scatter_matrix,
    ))


def band(image, sigma):
    """Deviation of the high band, as a stand-in for 'texture'."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((grey - cv2.GaussianBlur(grey, (0, 0), sigma)).std())


check('scatter defaults to off', FaceSwapConfig().diffuse_strength == 0.0)
off_lab = to_lab(crop)
config.diffuse_strength = 0.0
check('at strength 0 the crop is untouched',
      np.array_equal(
          compositor._scatter(off_lab.copy(), full_mask, scatter_face,
                              scatter_matrix),
          off_lab),
      'checked in LAB, the space the stage is handed: the BGR round trip in '
      'this helper belongs to the test and would mask an exact no-op')

softened = scatter(0.6)
check('scatter changes the crop when on', not np.array_equal(softened, crop))
check('it softens rather than sharpens',
      band(softened, 4.0) < band(crop, 4.0),
      '{:.2f} against {:.2f} in the shading band'.format(
          band(softened, 4.0), band(crop, 4.0)))
check('more strength softens more',
      band(scatter(0.9), 4.0) < band(scatter(0.3), 4.0))

# The scoping that keeps this from being the failure it imitates.
lab_before = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.int16)
lab_after = cv2.cvtColor(scatter(0.9), cv2.COLOR_BGR2LAB).astype(np.int16)
chroma = np.abs(lab_after[:, :, 1:] - lab_before[:, :, 1:]).max()
luma = np.abs(lab_after[:, :, 0] - lab_before[:, :, 0]).max()
check('only the luminance channel moves',
      chroma <= 2 < luma,
      'chroma max {} vs luma max {} — on full RGB this would drift skin '
      'tone as well as shading'.format(int(chroma), int(luma)))

# Features are where softening reads as identity loss, so they are cut out.
eye = (np.asarray(scatter_face.kps, dtype=np.float32)[0]).astype(int)
cheek = (int(ALIGNED * 0.22), int(ALIGNED * 0.62))
hard = scatter(0.9)
eye_delta = float(abs(int(hard[eye[1], eye[0], 0]) - int(crop[eye[1], eye[0], 0])))
cheek_delta = float(
    abs(int(hard[cheek[1], cheek[0], 0]) - int(crop[cheek[1], cheek[0], 0])))
check('the eyes are excluded from the softening',
      eye_delta < cheek_delta,
      'eye moved {:.0f}, cheek {:.0f} — softening a feature is the exact '
      'identity loss this routes around'.format(eye_delta, cheek_delta))

empty = to_lab(crop)
check('a fully masked-out frame comes back exactly untouched',
      np.array_equal(
          compositor._scatter(empty.copy(), np.zeros_like(full_mask),
                              scatter_face, scatter_matrix),
          empty),
      'no weight means no work at all, in the space it was handed')

half_mask = np.zeros_like(full_mask)
half_mask[:, :ALIGNED // 2] = 1.0
gated = scatter(0.9, mask=half_mask)
outside = np.abs(gated[:, ALIGNED // 2 + 40:].astype(np.int16)
                 - crop[:, ALIGNED // 2 + 40:].astype(np.int16)).max()
inside = np.abs(gated[:, :40].astype(np.int16)
                - crop[:, :40].astype(np.int16)).max()
# Sampled 40px clear of the midline, well past the weight blur's 3-sigma of
# ~15px, so what is left is the uint8 BGR->LAB->BGR round trip the test itself
# performs — the stage no longer does one of its own.
check('the mask gates the softening',
      outside <= 2 < inside,
      'outside moved {} (the test\'s own colour round trip), inside moved '
      '{}'.format(int(outside), int(inside)))

check('unusable keypoints decline rather than soften blind',
      np.array_equal(
          compositor._scatter(crop, full_mask, MagicMock(kps=None),
                              scatter_matrix),
          crop))

# The band split is what keeps scatter and texture from being the same lever.
check('scatter sits above the texture band, not inside it',
      FaceCompositor._SCATTER_SIGMA > DETAIL_SIGMA,
      '{:.1f} against {:.1f} at the same 256 reference — a pass reaching into '
      'the texture band would undo the stage after it'.format(
          FaceCompositor._SCATTER_SIGMA, DETAIL_SIGMA))
check('the sigma scales with the working resolution',
      FaceCompositor._SCATTER_REFERENCE == DETAIL_SIGMA_REFERENCE,
      'otherwise "soft" is a different distance when the operator leans in')

config.diffuse_strength = 0.0

# -- Grain noise ----------------------------------------------------------
print('\nGrain noise')

field_a = compositor._noise_field((64, 64))
field_b = compositor._noise_field((64, 64))
check('the noise field is unit variance',
      abs(float(field_a.std()) - 1.0) < 0.15,
      'std {:.3f} — the caller scales it by the measured sigma'.format(
          float(field_a.std())))
check('consecutive frames do not share a pattern',
      not np.array_equal(field_a, field_b),
      'a fixed grain overlay is a worse artefact than none')
check('the tile is cached, not regenerated',
      compositor._noise is not None
      and compositor._noise.shape[0] >= 128,
      'twice the largest region asked for, so there are windows to choose from')

before = compositor._noise
compositor._noise_field((40, 40))
check('a smaller region reuses the same tile', compositor._noise is before)
compositor._noise_field((400, 400))
check('a larger region grows it', compositor._noise is not before)

compositor._prev_fake = np.zeros((4, 4, 3), dtype=np.uint8)
compositor.reset()
check('the tile survives reset — it is a cache, not temporal state',
      compositor._noise is not None)

# ── Stale readings ─────────────────────────────────────────────────────
print('\nStale readings')

compositor.last_detail_ratio = 1.9
compositor.last_texture_headroom = 3.0
compositor.last_texture_confidence = 0.5
compositor.last_stage_ms['restore'] = 12.0
compositor.clear_readings()

check('a frame that never composites leaves nothing behind',
      compositor.last_detail_ratio is None
      and compositor.last_texture_headroom is None
      and compositor.last_texture_confidence is None
      and not compositor.last_stage_ms,
      'the guarded path still calls _log_timing, so stale numbers would be '
      'recorded a second time and inflate every distribution')

compositor._prev_fake = np.zeros((4, 4, 3), dtype=np.uint8)
compositor.clear_readings()
check('clearing readings does not drop temporal state',
      compositor._prev_fake is not None,
      'reset() is per source change; this is per frame — conflating them '
      'would drop the smoothing buffer thirty times a second')
compositor._prev_fake = None

# ── Mask erode ─────────────────────────────────────────────────────────
print('\nMask erode')

mask_config = FaceSwapConfig()
mask_config.occluder = False
masker = FaceMasker(mask_config)
real_crop = np.full((256, 256, 3), 128, dtype=np.uint8)
mask_matrix = canonical_from_frame(TARGET, 256).astype(np.float32)

mask_config.mask_erode = 0.0
loose = masker.build(TARGET, mask_matrix, real_crop, (400, 400)).sum()
mask_config.mask_erode = 0.03
tight = masker.build(TARGET, mask_matrix, real_crop, (400, 400)).sum()

check('eroding before the feather pulls the mask in',
      tight < loose, '{:.0f} vs {:.0f} covered'.format(tight, loose))
check('it pulls in, it does not erase',
      tight > 0.5 * loose,
      'the transition moves onto skin; the coverage stays')


# ── The desktop panel and the pipeline must agree ────────────────────────────
#
# The tuning panel is the only appearance control the desktop can move while a
# stream runs, and it deliberately does not push its values on connect: doing so
# would revert a pipeline launched with TEXTURE_STRENGTH in `.env`, which is the
# `set_enhance` mistake CLAUDE.md records. The other half of that bargain is
# that the desktop must READ the values back, or the slider and the layer
# disagree in silence — a panel reading 0.00 over a running layer turns an A/B
# into a measurement of nothing.

from pipeline.api.handlers import handle_get_state  # noqa: E402

_state_config = FaceSwapConfig()
_state_config.texture_strength = 0.42
_state_config.diffuse_strength = 0.27
_state = handle_get_state(_state_config, None).data

check('get_state reports texture_strength',
      _state.get('texture_strength') == 0.42)
check('get_state reports diffuse_strength',
      _state.get('diffuse_strength') == 0.27)

# `texture_band` is on the panel too, so it is under the same bargain: the
# desktop does not assert it on connect, therefore it must be able to read it
# back. The other two shaping knobs are reported for tools/stats.py rather than
# for a slider, and cost nothing to include.
_state_config.texture_band = 3.0
_band_state = handle_get_state(_state_config, None).data
check('get_state reports the shaping knobs the panel can move',
      _band_state.get('texture_band') == 3.0,
      'a BAND slider reading 2.0 over a pipeline running 3.0 turns an A/B '
      'into a measurement of nothing')
check('get_state reports the shaping knobs it cannot',
      _band_state.get('texture_relief') is not None
      and _band_state.get('texture_contrast') is not None,
      'set over set_realism, so the only way to see them is to read them back')
check('get_state still reports what it did before',
      all(key in _state for key in
          ('quality', 'enhance', 'restoration_preset', 'source_loaded')))

# -- End to end: do both layers survive into the finished frame? ----------
#
# Everything above tests the two stages in isolation, which proves they compute
# something. It does not prove the result reaches the output: texture is added
# inside `_paste` and scatter runs before `_match_color`, so either could be
# computed and then overwritten, warped away, or masked to nothing without a
# single stage-level check noticing.
#
# That gap is not hypothetical for texture in particular. It is applied AFTER
# `_match_detail`, so no reading downstream sees it — the pod sweep measured
# identical `detail_ratio` at texture 0.0, 0.3 and 0.5, and could not have told
# a working layer from a dead one.
print('')
print('End to end through composite()')

e2e_face = make_face(x=150.0, y=150.0, size=160.0)
e2e_frame = np.clip(
    rng.normal(120, 10, (360, 640, 3)), 0, 255).astype(np.uint8)
# A smooth swap, the way a 128-native swapper actually delivers one.
e2e_swap = cv2.GaussianBlur(
    rng.normal(130, 6, (256, 256, 3)).astype(np.float32), (0, 0), 3.0)
e2e_swap = np.clip(e2e_swap, 0, 255).astype(np.uint8)
e2e_matrix = canonical_from_frame(e2e_face, 256).astype(np.float32)


def composite_with(texture_strength, diffuse_strength):
    """A full composite at these two strengths, everything else held still."""
    cfg = FaceSwapConfig()
    cfg.texture_strength = texture_strength
    cfg.diffuse_strength = diffuse_strength
    cfg.grain = False          # random per frame; would swamp the difference
    cfg.temporal_alpha = 1.0   # no EMA, so one call is one call
    cfg.enhance = False        # no model weights in this environment
    # A masker that returns a real soft-edged mask rather than a MagicMock:
    # `composite` warps and multiplies by it, so a mock would raise instead of
    # compositing. Full coverage, so the occlusion guard passes.
    masker = MagicMock()
    masker.last_coverage = 1.0

    def _build(_face, _matrix, real_crop, _shape):
        edge = real_crop.shape[0]
        m = np.zeros((edge, edge), dtype=np.float32)
        cv2.circle(m, (edge // 2, edge // 2), int(edge * 0.42), 1.0, -1)
        return cv2.GaussianBlur(m, (0, 0), edge * 0.03)

    masker.build.side_effect = _build

    comp = FaceCompositor(cfg, MagicMock(available=False), masker)
    comp.source_texture = extracted
    out = comp.composite(e2e_frame.copy(), e2e_face, e2e_swap.copy(),
                         e2e_matrix.copy())
    return out, comp.last_stage_ms


e2e_base, e2e_base_ms = composite_with(0.0, 0.0)
check('the baseline composites at all', e2e_base is not None,
      'nothing below means anything if this is None')

if e2e_base is not None:
    e2e_tex, e2e_tex_ms = composite_with(0.5, 0.0)
    e2e_dif, e2e_dif_ms = composite_with(0.0, 0.3)
    e2e_both, _ = composite_with(0.5, 0.3)

    def face_box(f, pad=10):
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        return (max(0, y1 - pad), min(e2e_frame.shape[0], y2 + pad),
                max(0, x1 - pad), min(e2e_frame.shape[1], x2 + pad))

    top, bottom, left, right = face_box(e2e_face)

    def spread(other):
        """Mean |difference| from the baseline, inside the face and outside."""
        d = cv2.absdiff(e2e_base, other).astype(np.float32).mean(axis=2)
        inside = float(d[top:bottom, left:right].mean())
        blanked = d.copy()
        blanked[top:bottom, left:right] = 0.0
        outside = float(blanked.sum() / max(1, (blanked > 0).sum()))
        return inside, outside

    tex_in, tex_out = spread(e2e_tex)
    dif_in, dif_out = spread(e2e_dif)

    check('texture changes the finished frame', tex_in > 0.05,
          'mean |diff| {:.3f} units inside the face'.format(tex_in))
    check('scatter changes the finished frame', dif_in > 0.05,
          'mean |diff| {:.3f} units inside the face'.format(dif_in))
    check('texture changes the face, not the frame around it',
          tex_out < tex_in,
          'inside {:.3f} vs outside {:.3f} - a difference spread over the '
          'whole frame would be a paste bug'.format(tex_in, tex_out))
    check('scatter changes the face, not the frame around it',
          dif_out < dif_in,
          'inside {:.3f} vs outside {:.3f}'.format(dif_in, dif_out))
    check('they are not each other',
          float(cv2.absdiff(e2e_tex, e2e_dif).mean()) > 0.01,
          'two layers in two bands must not produce the same frame')
    check('together they differ from either alone',
          float(cv2.absdiff(e2e_both, e2e_tex).mean()) > 0.005
          and float(cv2.absdiff(e2e_both, e2e_dif).mean()) > 0.005)
    check('each bills itself only when it runs',
          'texture' in e2e_tex_ms and 'texture' not in e2e_base_ms
          and e2e_dif_ms.get('scatter', 0.0) > e2e_base_ms.get('scatter', 0.0),
          'a stage that reports time while off would hide a no-op')

print('=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
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
