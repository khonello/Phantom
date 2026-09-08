"""
The identity work: measuring likeness, pushing it, and not clipping it off.

Four separate pieces share this file because they share one failure they exist
to prevent — *the output is confidently the wrong-ish person and nothing says
so*. Each is pinned at the point where it would silently become a no-op:

1. **`IdentityProbe`** re-frames a crop from the swapper's alignment into the
   recognition model's. Get that transform wrong and every reading is a
   plausible-looking number measured on a mis-framed face, which is worse than
   no reading at all — a mis-framed crop still returns a cosine.

2. **`identity_push`** extrapolates away from the target. The two things that
   must hold are that zero is bit-identical to not having the feature, and that
   the result is a unit vector pointing further from the target than the source
   did. A push that quietly normalised back to the source would look like a
   working knob and do nothing.

3. **The shape mask** must be a no-op when the generated face has the target's
   outline — that is what makes it safe to leave on for `inswapper` — and must
   admit growth *only* below the eye line, bounded, where the generated face
   actually is.

4. **Pose-weighted averaging** must degrade to the flat mean it replaces when
   the inputs carry no pose or score, and must keep the raw embedding that a
   converter-based model needs.
"""

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

import types

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.processing import geometry
from pipeline.services import identity, swapper_models
from pipeline.processing.compositor import FaceCompositor
from pipeline.services.database import SOURCE_BLENDS, FaceDatabase
from pipeline.services.face_swapping import FaceSwapper, PUSH_MAX
from pipeline.services.masking import FaceMasker

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


print('=' * 70)
print('Identity: measuring it, pushing it, and keeping its shape')
print('=' * 70)


# ── 1. The probe's re-framing ──────────────────────────────────────────────
print('\nIdentityProbe re-frames aligned crops for recognition')

# A recognition model that returns the crop itself, flattened and padded. That
# makes the *pixels it was given* observable, which is the only thing worth
# testing here — the real model is a black box and stubbing its weights would
# test nothing.
seen: list = []


class RecordingRecognition:
    """Stands in for w600k_r50, recording what it was asked to embed."""

    @staticmethod
    def get_feat(crop):
        seen.append(crop.copy())
        return np.arange(512, dtype=np.float32).reshape(1, 512)


probe = identity.IdentityProbe(
    types.SimpleNamespace(recognition_model=lambda: RecordingRecognition()))

check('a probe reports itself available when the pack has recognition',
      probe.available)

# Paint the five template points into an arcface-aligned crop, then check they
# land on the recognition template's points after re-framing. This is the whole
# correctness claim: the transform is between two five-point templates.
SIZE = 256
crop = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
for x, y in (geometry.ARCFACE_128_TEMPLATE * SIZE):
    cv2.circle(crop, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)

seen.clear()
probe.embed_aligned(crop, geometry.ARCFACE_128_TEMPLATE)
check('embedding an aligned crop reaches the recognition model', len(seen) == 1)

warped = seen[0]
check('the recognition crop is 112x112',
      warped.shape[:2] == (112, 112), str(warped.shape))

# Each landmark should now sit within a pixel or two of arcface_dst.
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
found = []
for x, y in identity.ARCFACE_112:
    window = gray[max(0, int(y) - 4):int(y) + 5, max(0, int(x) - 4):int(x) + 5]
    found.append(window.max() if window.size else 0)

check('every template point lands on the recognition template',
      all(value > 128 for value in found), str(found))

# The same claim for the other template. A model aligned to mtcnn_512 puts the
# face 6% lower in the crop; if the probe ignored the template it was given,
# these points would land ~15px off in a 112 crop and this check would fail.
crop2 = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
for x, y in (geometry.MTCNN_512_TEMPLATE * SIZE):
    cv2.circle(crop2, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)

seen.clear()
probe.embed_aligned(crop2, geometry.MTCNN_512_TEMPLATE)
gray2 = cv2.cvtColor(seen[0], cv2.COLOR_BGR2GRAY)
found2 = []
for x, y in identity.ARCFACE_112:
    window = gray2[max(0, int(y) - 4):int(y) + 5, max(0, int(x) - 4):int(x) + 5]
    found2.append(window.max() if window.size else 0)

check('a mtcnn-aligned crop re-frames to the same recognition template',
      all(value > 128 for value in found2), str(found2))

# And the negative: reading an mtcnn crop as though it were arcface must NOT
# land on target. Without this, the check above would pass for a probe that
# ignored its template argument entirely.
seen.clear()
probe.embed_aligned(crop2, geometry.ARCFACE_128_TEMPLATE)
gray3 = cv2.cvtColor(seen[0], cv2.COLOR_BGR2GRAY)
misread = []
for x, y in identity.ARCFACE_112:
    window = gray3[max(0, int(y) - 4):int(y) + 5, max(0, int(x) - 4):int(x) + 5]
    misread.append(window.max() if window.size else 0)

check('reading a crop with the wrong template does NOT land on target',
      not all(value > 128 for value in misread), str(misread))

# Degradation: a pack without recognition goes quiet rather than raising.
blind = identity.IdentityProbe(types.SimpleNamespace(
    recognition_model=lambda: None))
check('a pack without recognition reports unavailable', not blind.available)
check('and embedding returns None rather than raising',
      blind.embed_aligned(crop, geometry.ARCFACE_128_TEMPLATE) is None)

# Cosine's own guards.
check('cosine of a vector with itself is 1',
      abs(identity.cosine(np.ones(8), np.ones(8)) - 1.0) < 1e-9)
check('cosine against None is None', identity.cosine(np.ones(8), None) is None)
check('cosine of mismatched lengths is None',
      identity.cosine(np.ones(8), np.ones(4)) is None)
check('cosine of a zero vector is None',
      identity.cosine(np.zeros(8), np.ones(8)) is None)


# ── 2. identity_push ───────────────────────────────────────────────────────
print('\nidentity_push moves the conditioning vector away from the target')


def unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


SOURCE = unit([1.0, 0.0, 0.0, 0.0])
TARGET = unit([0.6, 0.8, 0.0, 0.0])

config = FaceSwapConfig()
swapper = FaceSwapper(config)

source_face = types.SimpleNamespace(
    normed_embedding=SOURCE, embedding=SOURCE * 21.0)
target_face = types.SimpleNamespace(normed_embedding=TARGET)

config.set('identity_push', 0.0)
check('at zero the source face is passed through unchanged, by identity',
      swapper.pushed_source(source_face, target_face) is source_face)

config.set('identity_push', 0.3)
pushed = swapper.pushed_source(source_face, target_face)
check('a push returns a stand-in rather than mutating the source',
      pushed is not source_face
      and np.allclose(source_face.normed_embedding, SOURCE))

vector = np.asarray(pushed.normed_embedding, dtype=np.float64)
check('the pushed vector is still a unit vector',
      abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6,
      str(float(np.linalg.norm(vector))))

before = float(np.dot(SOURCE, TARGET))
after = float(np.dot(vector, TARGET))
check('the pushed vector sits further from the target than the source did',
      after < before, '{:.4f} -> {:.4f}'.format(before, after))

# Monotonic, so the knob means something across its range rather than at one
# point of it.
similarities = []
for amount in (0.0, 0.1, 0.2, 0.4, PUSH_MAX):
    config.set('identity_push', amount)
    moved = swapper.pushed_source(source_face, target_face)
    moved_vector = np.asarray(
        getattr(moved, 'normed_embedding'), dtype=np.float64)
    similarities.append(float(np.dot(moved_vector, TARGET)))

check('similarity to the target falls monotonically with the push',
      all(b <= a + 1e-9 for a, b in zip(similarities, similarities[1:])),
      str([round(s, 4) for s in similarities]))

config.set('identity_push', 0.3)
raw = np.asarray(pushed.embedding, dtype=np.float64)
check('the raw embedding keeps its magnitude and only changes direction',
      abs(float(np.linalg.norm(raw)) - 21.0) < 1e-3,
      str(float(np.linalg.norm(raw))))

# A target with no embedding is a capability gap, not a failure.
check('a target with no embedding leaves the source untouched',
      swapper.pushed_source(
          source_face, types.SimpleNamespace()) is source_face)

check('the push is clamped to PUSH_MAX',
      (config.set('identity_push', 10.0) or swapper.identity_push()) == PUSH_MAX)
config.set('identity_push', 0.0)


# ── 3. The shape-following mask ────────────────────────────────────────────
print('\nThe mask may follow the generated outline, within bounds')

MASK_SIZE = 256


def landmarks_for(scale_x: float) -> np.ndarray:
    """106 points on the rim of an ellipse, `scale_x` wide."""
    angles = np.linspace(0.0, 2.0 * np.pi, 106, endpoint=False)
    return np.stack([
        MASK_SIZE / 2 + np.cos(angles) * MASK_SIZE * 0.32 * scale_x,
        MASK_SIZE * 0.52 + np.sin(angles) * MASK_SIZE * 0.42,
    ], axis=1).astype(np.float32)


def hull_for(scale_x: float) -> np.ndarray:
    """
    The filled hull of those landmarks, built the way `_hull_mask` builds it.

    Through the masker's own `_expand_hull` rather than as a bare ellipse, and
    that matters more than it looks: the shape term expands the *generated*
    landmarks the same way, so a target hull that skipped the expansion would
    be 10% smaller than an identical generated one and every check below would
    measure the expansion instead of the shape.
    """
    points = cv2.convexHull(landmarks_for(scale_x))
    points = FaceMasker(FaceSwapConfig())._expand_hull(points)
    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    cv2.fillConvexPoly(
        mask, cv2.convexHull(points).astype(np.int32), (1.0,))
    return mask


def masker_with(growth: float, generated_scale: float) -> tuple:
    """A masker whose landmark model reports a face `generated_scale` wide."""
    cfg = FaceSwapConfig()
    cfg.set('mask_shape_growth', growth)

    model = MagicMock()
    model.get.return_value = landmarks_for(generated_scale)
    detector = types.SimpleNamespace(landmark_model=lambda: model)

    masker = FaceMasker(cfg, detector)
    face = types.SimpleNamespace(
        bbox=np.array([10.0, 10.0, 90.0, 110.0], dtype=np.float32))
    matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    swapped = np.zeros((MASK_SIZE, MASK_SIZE, 3), dtype=np.uint8)

    shaped, admitted = masker._shape(
        hull_for(1.0), swapped, face, matrix, MASK_SIZE,
        geometry.ARCFACE_128_TEMPLATE)
    return shaped, admitted


base = hull_for(1.0)

off_mask, off_growth = masker_with(0.0, 1.20)
check('with the knob at zero the mask is the target hull, exactly',
      np.array_equal(off_mask, base) and off_growth is None)

same_mask, same_growth = masker_with(0.08, 1.0)
check('a generated face with the target outline changes nothing',
      float(np.abs(same_mask - base).max()) < 1e-6,
      'max delta {:.4f}'.format(float(np.abs(same_mask - base).max())))
check('and it admits no growth region',
      same_growth is not None and float(same_growth.sum()) < 1.0,
      'sum {:.1f}'.format(float(same_growth.sum())))

wide_mask, wide_growth = masker_with(0.08, 1.25)
check('a wider generated face grows the mask',
      float(wide_mask.sum()) > float(base.sum()),
      '{:.0f} -> {:.0f}'.format(float(base.sum()), float(wide_mask.sum())))

# The two bounds that make growth safe.
rows = np.nonzero(wide_growth.sum(axis=1) > 0.0)[0]
eye_row = geometry.ARCFACE_128_TEMPLATE[0:2, 1].mean() * MASK_SIZE
check('growth never appears above the eye line',
      rows.size > 0 and float(rows.min()) >= eye_row - 1.0,
      'topmost grown row {} vs eye line {:.0f}'.format(
          int(rows.min()) if rows.size else -1, eye_row))

distance = cv2.distanceTransform(
    (base < 0.5).astype(np.uint8), cv2.DIST_L2, 3)
reach = float(distance[wide_growth > 0.01].max()) if np.any(
    wide_growth > 0.01) else 0.0
limit = MASK_SIZE * 0.08
check('growth stays within mask_shape_growth of the target hull',
      reach <= limit + 1.5, '{:.1f}px against a {:.1f}px bound'.format(
          reach, limit))

# A pack with no landmark model must not silently change the mask.
blind_cfg = FaceSwapConfig()
blind_cfg.set('mask_shape_growth', 0.08)
blind_masker = FaceMasker(
    blind_cfg, types.SimpleNamespace(landmark_model=lambda: None))
blind_mask, blind_growth = blind_masker._shape(
    base, np.zeros((MASK_SIZE, MASK_SIZE, 3), dtype=np.uint8),
    types.SimpleNamespace(bbox=np.array([10.0, 10.0, 90.0, 110.0])),
    np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
    MASK_SIZE, geometry.ARCFACE_128_TEMPLATE)
check('no landmark model means the target hull, unchanged',
      np.array_equal(blind_mask, base) and blind_growth is None)

# The eye line is read from the template, not assumed. mtcnn puts it lower, so
# the ramp must start lower too.
arc_weight = FaceMasker(FaceSwapConfig())._lower_face(
    MASK_SIZE, geometry.ARCFACE_128_TEMPLATE)
mtcnn_weight = FaceMasker(FaceSwapConfig())._lower_face(
    MASK_SIZE, geometry.MTCNN_512_TEMPLATE)
check('the shape ramp follows the template it was given',
      float(mtcnn_weight.sum()) < float(arc_weight.sum()),
      'mtcnn {:.0f} < arcface {:.0f}'.format(
          float(mtcnn_weight.sum()), float(arc_weight.sum())))


# ── 4. Pose-weighted source averaging ──────────────────────────────────────
print('\nSource averaging weights the photographs it trusts')

database = FaceDatabase(MagicMock(), FaceSwapConfig())

FRONTAL = unit([1.0, 0.0, 0.0, 0.0])
ANGLED = unit([0.0, 1.0, 0.0, 0.0])


def photo(vector, yaw=None, score=None):
    """A source face carrying just what the weighting reads."""
    face = types.SimpleNamespace(
        normed_embedding=vector, embedding=vector * 21.0, kps=None)
    if yaw is not None:
        # `pose` only — deliberately no bbox, so this also pins that the
        # frontality term survives a face that cannot become a `Detection`.
        face.pose = np.array([0.0, yaw, 0.0], dtype=np.float32)
    if score is not None:
        face.det_score = score
    return face


flat = database._average_faces([photo(FRONTAL), photo(ANGLED)])
check('with no pose or score the average is the flat mean it always was',
      abs(float(np.dot(flat.normed_embedding, FRONTAL))
          - float(np.dot(flat.normed_embedding, ANGLED))) < 1e-6)

check('the averaged face keeps a raw embedding for converter models',
      flat.embedding is not None and flat.embedding.shape == FRONTAL.shape)

weighted = database._average_faces(
    [photo(FRONTAL, yaw=0.0), photo(ANGLED, yaw=55.0)])
check('a frontal photograph outweighs a turned one',
      float(np.dot(weighted.normed_embedding, FRONTAL))
      > float(np.dot(weighted.normed_embedding, ANGLED)),
      'frontal {:.3f} vs angled {:.3f}'.format(
          float(np.dot(weighted.normed_embedding, FRONTAL)),
          float(np.dot(weighted.normed_embedding, ANGLED))))

# The floor: one good photo must not erase three others, or multi-photo
# averaging has been switched off by accident.
dominant = database._average_faces(
    [photo(FRONTAL, yaw=0.0)] + [photo(ANGLED, yaw=80.0)] * 3)
check('the weight floor keeps turned photographs contributing',
      float(np.dot(dominant.normed_embedding, ANGLED)) > 0.3,
      '{:.3f}'.format(float(np.dot(dominant.normed_embedding, ANGLED))))

mixed = database._average_faces(
    [photo(FRONTAL), types.SimpleNamespace(normed_embedding=ANGLED)])
check('a .npy source without a raw embedding drops it rather than faking one',
      mixed is not None and mixed.embedding is None)

check('nothing usable gives None rather than an empty face',
      database._average_faces([types.SimpleNamespace()]) is None)


# ── 5. The registry entry ──────────────────────────────────────────────────
print('\nhififace is registered with what it actually needs')

model = swapper_models.resolve('hififace_unofficial_256')
check('hififace is its own inference kind', model.kind == 'hififace')
check('it aligns to mtcnn_512, not arcface', model.template == 'mtcnn_512')
check('the template it names resolves to a real one',
      geometry.alignment_template(model.template) is
      geometry.MTCNN_512_TEMPLATE)
check('an unknown template falls back rather than raising',
      geometry.alignment_template('nonsense') is
      geometry.ARCFACE_128_TEMPLATE)
check('it declares an embedding converter',
      bool(model.converter_filename) and bool(model.converter_url))
check('the converter is fetched from the tag that actually carries it',
      'models-3.1.0' in model.url and 'models-3.1.0' in model.converter_url,
      model.url)
check('every registered model still declares a template that resolves',
      all(geometry.alignment_template(swapper_models.resolve(n).template)
          is not None for n in swapper_models.names()))
check('the models that take a bare ArcFace vector declare no converter',
      not swapper_models.resolve('inswapper_128').converter_filename
      and not swapper_models.resolve('hyperswap_1a_256').converter_filename)


# ── 6. Keeping some of the source's complexion ────────────────────────────
print('\nThe colour match may leave some of the source\'s own skin tone')


def compositor_for(keep: float) -> FaceCompositor:
    """A compositor with just `complexion_keep` set."""
    cfg = FaceSwapConfig()
    cfg.set('complexion_keep', keep)
    return FaceCompositor(cfg, MagicMock(available=False), MagicMock())


def complexion(keep: float, gap: float) -> float:
    """The a/b correction multiplier at this keep and this tone difference."""
    # meanStdDev returns (3, 1) columns; only a/b are read.
    fake_mean = np.array([[128.0], [128.0], [128.0]])
    real_mean = np.array([[128.0], [128.0 + gap], [128.0]])
    return compositor_for(keep)._complexion_scale(fake_mean, real_mean)


check('at keep=0 the chroma correction is exactly 1.0, the shipped behaviour',
      complexion(0.0, 10.0) == 1.0)

off = compositor_for(0.0)
off._complexion_scale(np.array([[128.0], [128.0], [128.0]]),
                      np.array([[128.0], [138.0], [128.0]]))
check('and nothing is recorded, so the reading stays absent rather than zero',
      off.last_complexion_kept is None)

on = compositor_for(0.4)
on._complexion_scale(np.array([[128.0], [128.0], [128.0]]),
                     np.array([[128.0], [132.0], [128.0]]))
check('what it withheld is published for the readings',
      on.last_complexion_kept is not None
      and abs(on.last_complexion_kept - 1.6) < 1e-6,
      str(on.last_complexion_kept))

# Below the cap the knob is the fraction it says it is.
scale = complexion(0.4, 4.0)
check('below the cap, keep=0.4 withholds 40% of the difference',
      abs(scale - 0.6) < 1e-6, '{:.4f}'.format(scale))

# Above it, the bound takes over and the knob stops mattering.
wide = complexion(0.4, 40.0)
withheld = 40.0 * (1.0 - wide)
check('a large tone gap is capped rather than scaled',
      abs(withheld - FaceCompositor._COMPLEXION_RESIDUAL) < 1e-3,
      '{:.2f} LAB units withheld'.format(withheld))

check('the withheld amount never exceeds the cap, at any keep or gap',
      all(gap * (1.0 - complexion(k, gap))
          <= FaceCompositor._COMPLEXION_RESIDUAL + 1e-6
          for k in (0.2, 0.5, 1.0) for gap in (1.0, 5.0, 20.0, 60.0)))

# The failure this guards against: correction giving way as the two faces
# diverge, rather than leaving a bigger and bigger step.
scales = [complexion(0.5, gap) for gap in (2.0, 6.0, 12.0, 30.0, 60.0)]
check('correction returns toward full as the complexions diverge',
      all(a <= b + 1e-9 for a, b in zip(scales, scales[1:])),
      str([round(s, 3) for s in scales]))

check('matching tones spend nothing and say so',
      complexion(0.5, 0.0) == 1.0)


# ── 7. Source blending strategies ─────────────────────────────────────────
print('\nsource_blend chooses how photographs become one identity')

check('every advertised strategy is a real one',
      set(SOURCE_BLENDS) == {'mean', 'weighted', 'norm', 'best', 'median'})


def blended(strategy: str, faces: list):
    cfg = FaceSwapConfig()
    cfg.set('source_blend', strategy)
    return FaceDatabase(MagicMock(), cfg)._average_faces(faces)


tilted = [photo(FRONTAL, yaw=0.0), photo(ANGLED, yaw=60.0),
          photo(ANGLED, yaw=60.0)]

flat_blend = blended('mean', tilted)
weighted_blend = blended('weighted', tilted)
check('mean and weighted disagree when the photographs differ in pose',
      abs(float(np.dot(flat_blend.normed_embedding, FRONTAL))
          - float(np.dot(weighted_blend.normed_embedding, FRONTAL))) > 1e-3)

check('mean reproduces the flat average exactly',
      abs(float(np.dot(flat_blend.normed_embedding, FRONTAL))
          - float(np.dot(
              blended('mean', tilted).normed_embedding, FRONTAL))) < 1e-12)

best = blended('best', tilted)
check('best picks one photograph rather than blending',
      abs(float(np.dot(best.normed_embedding, FRONTAL)) - 1.0) < 1e-6,
      '{:.4f}'.format(float(np.dot(best.normed_embedding, FRONTAL))))

# `norm` reads the embedding's own magnitude. Give one photo a much larger raw
# vector and it should dominate, which is the whole claim.
loud = types.SimpleNamespace(
    normed_embedding=FRONTAL, embedding=FRONTAL * 40.0, kps=None)
quiet = types.SimpleNamespace(
    normed_embedding=ANGLED, embedding=ANGLED * 8.0, kps=None)
by_norm = blended('norm', [loud, quiet])
by_score = blended('weighted', [loud, quiet])
check('norm weights by embedding magnitude, det_score does not',
      float(np.dot(by_norm.normed_embedding, FRONTAL))
      > float(np.dot(by_score.normed_embedding, FRONTAL)),
      'norm {:.3f} vs weighted {:.3f}'.format(
          float(np.dot(by_norm.normed_embedding, FRONTAL)),
          float(np.dot(by_score.normed_embedding, FRONTAL))))

# The median's whole purpose: one distant point must move it less than it
# moves the mean.
OTHER = unit([0.0, 0.0, 1.0, 0.0])
cluster = [photo(FRONTAL), photo(FRONTAL), photo(FRONTAL), photo(OTHER)]
mean_pull = float(np.dot(blended('mean', cluster).normed_embedding, OTHER))
median_pull = float(np.dot(blended('median', cluster).normed_embedding, OTHER))
check('the geometric median resists an outlier the mean follows',
      median_pull < mean_pull,
      'median {:.3f} < mean {:.3f}'.format(median_pull, mean_pull))

check('the median still lands on the cluster it came from',
      float(np.dot(blended('median', cluster).normed_embedding, FRONTAL)) > 0.9)

check('an unknown strategy falls back rather than raising',
      blended('nonsense', tilted) is not None)

check('every strategy returns a unit vector',
      all(abs(float(np.linalg.norm(
          blended(s, tilted).normed_embedding)) - 1.0) < 1e-5
          for s in ('mean', 'weighted', 'norm', 'best', 'median')))

check('every strategy keeps the raw embedding a converter model needs',
      all(blended(s, tilted).embedding is not None
          for s in ('mean', 'weighted', 'norm', 'best', 'median')))

check('a single photograph is unchanged by every strategy',
      all(abs(float(np.dot(
          blended(s, [photo(FRONTAL)]).normed_embedding, FRONTAL)) - 1.0) < 1e-6
          for s in ('mean', 'weighted', 'norm', 'best', 'median')))


print('\n' + '=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
print('=' * 70)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
