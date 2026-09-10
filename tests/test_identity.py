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

5. **The shape metric** must be invariant to a similarity transform of any of
   its three inputs — they are read off three different images at three
   different scales, so a reading that moved with the crop would be measuring
   the camera. Its two endpoints must be exact, and its outline subset must
   land on the silhouette rather than on whichever points happen to sit far
   from the centroid.
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
from pipeline.types import Bbox
from pipeline.processing import geometry
from pipeline.services import identity, swapper_models
from pipeline.processing.compositor import FaceCompositor
from pipeline.services import database as db
from pipeline.services.database import SOURCE_BLENDS, FaceDatabase
from pipeline.services.face_swapping import FaceSwapper, PUSH_MAX
from pipeline.services.masking import FaceMasker
from pipeline.services import shape as shape_metric
from pipeline.services.readings import Readings

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
check('the model is fetched from the tag that actually carries it',
      'models-3.1.0' in model.url,
      model.url)
check('the converter is the maintained crossface one, on its own tag',
      'crossface' in model.converter_filename
      and 'models-3.4.0' in model.converter_url,
      model.converter_url)
check('every converter model declares the size of its converter, so a '
      'truncated download is visible',
      all(swapper_models.resolve(n).converter_size_bytes > 0
          for n in swapper_models.names()
          if swapper_models.resolve(n).converter_filename))
check('every registered model still declares a template that resolves',
      all(geometry.alignment_template(swapper_models.resolve(n).template)
          is not None for n in swapper_models.names()))
check('the models that take a bare ArcFace vector declare no converter',
      not swapper_models.resolve('inswapper_128').converter_filename
      and not swapper_models.resolve('alphaface_256').converter_filename)


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

# ── The averaged raw vector keeps a real embedding's magnitude ─────────
#
# Averaging vectors that disagree yields a resultant shorter than any input,
# so the plain mean of raw ArcFace vectors came out 5-20% under the norm of a
# real embedding — worse the more photographs were added. Invisible under
# inswapper, which divides by the norm; a different input entirely to
# `alphaface` and to the `crossface` converters, which were fitted on
# ArcFace's own output scale. It degrades to a blander identity, never an
# error, which is why it needs a test rather than a look.
print()
print('The averaged raw vector keeps a usable magnitude')

_photo_norm = 21.0
for _strategy in ('mean', 'weighted', 'norm', 'best', 'median'):
    _averaged = blended(_strategy, tilted)
    _magnitude = float(np.linalg.norm(
        np.asarray(_averaged.embedding, dtype=np.float64)))
    check('{}: the raw vector still has a real embedding norm'.format(
              _strategy),
          abs(_magnitude - _photo_norm) < 1e-3,
          'got {:.3f}, photographs carry {:.1f}'.format(
              _magnitude, _photo_norm))

_averaged = blended('weighted', tilted)
check('the raw and normalised vectors point the same way',
      float(np.dot(
          np.asarray(_averaged.embedding, dtype=np.float64)
          / np.linalg.norm(_averaged.embedding),
          np.asarray(_averaged.normed_embedding, dtype=np.float64),
      )) > 1.0 - 1e-6,
      'they are one identity at two scales; consumers assume that')

check('averaging more photographs does not shrink the magnitude',
      abs(float(np.linalg.norm(np.asarray(
          blended('mean', tilted + [photo(FRONTAL)]).embedding,
          dtype=np.float64))) - _photo_norm) < 1e-3,
      'uploading a fourth photograph used to weaken the conditioning vector')


# ── 5. The shape metric ────────────────────────────────────────────────────
print("\nShape: did the output take the source's head shape or the target's")

_N_OUTLINE = 36


def head(jaw: float = 1.0, eyes: float = 1.0) -> np.ndarray:
    """
    A 106-point face: an outline arc plus interior features.

    `jaw` widens the contour, `eyes` moves the interior features apart. They are
    separate levers so that a change to one can be checked *not* to register as
    the other, which is the whole claim the outline subset makes.
    """
    t = np.linspace(-1.35, 1.35, _N_OUTLINE)
    outline = np.stack([np.sin(t) * 0.50 * jaw, -np.cos(t) * 0.62 + 0.16], 1)

    feats = []
    for cx in (-0.20 * eyes, 0.20 * eyes):
        feats += [[cx + dx, 0.10] for dx in (-0.07, 0.0, 0.07)]
    for cx in (-0.22 * eyes, 0.22 * eyes):
        feats += [[cx + dx, 0.22] for dx in (-0.08, 0.0, 0.08)]
    feats += [[0.0, y] for y in (0.06, -0.02, -0.10)]
    feats += [[dx, -0.16] for dx in (-0.06, 0.0, 0.06)]
    feats += [[dx, -0.34] for dx in (-0.13, -0.06, 0.0, 0.06, 0.13)]
    feats += [[dx, -0.28] for dx in (-0.10, 0.0, 0.10)]
    inner = np.array(feats, dtype=float)
    pad = np.repeat(inner[-1:], 70 - inner.shape[0], axis=0) + np.linspace(
        0, 0.02, 70 - inner.shape[0])[:, None]
    return np.concatenate([outline, inner, pad], 0)


_SRC = head(jaw=0.78, eyes=1.18)
_TGT = head(jaw=1.25, eyes=0.85)

# Both endpoints have to be exact, or the scale in between means nothing.
_kept = shape_metric.compare(_SRC, _TGT, _TGT)
check("an output with the target's shape reads 0.000",
      _kept is not None and abs(_kept.shift) < 1e-9,
      '{:+.6f}'.format(_kept.shift))

_took = shape_metric.compare(_SRC, _TGT, _SRC)
check("an output with the source's shape reads 1.000",
      _took is not None and abs(_took.shift - 1.0) < 1e-9,
      '{:+.6f}'.format(_took.shift))
check('and the outline reading agrees at both ends',
      abs(_kept.outline_shift) < 1e-9
      and abs(_took.outline_shift - 1.0) < 1e-9)

# Similarity invariance. The three landmark sets are read off three different
# images at three different scales, so a reading that moved with the crop would
# be measuring the camera rather than the face.
_theta = 0.4
_rot = np.array([[np.cos(_theta), -np.sin(_theta)],
                 [np.sin(_theta), np.cos(_theta)]])
_moved = (_TGT * 3.7) @ _rot.T + np.array([120.0, -45.0])
_invariant = shape_metric.compare(_SRC, _TGT, _moved)
check('scaling, rotating and moving the output changes nothing',
      abs(_invariant.shift) < 1e-9,
      'shift {:+.8f} after x3.7 and 23 degrees'.format(_invariant.shift))

_far = shape_metric.compare(_SRC * 9.0 + 500.0, _TGT, _TGT)
check('and neither does rescaling the source photograph',
      abs(_far.shift) < 1e-9, '{:+.8f}'.format(_far.shift))

# The outline subset must be the silhouette. A radial ranking from the centroid
# was tried first and picked the brow ends over the chin — a face is taller than
# it is wide — so this pins that the hull-distance ranking does not.
_split = shape_metric.outline_indices(_SRC)
check('the outline subset is resolvable', _split is not None)
_picked = set(_split[0].tolist())
_arc = set(range(_N_OUTLINE))
check('every point it picks is genuine contour',
      _picked <= _arc,
      '{} of {} on the arc'.format(len(_picked & _arc), len(_picked)))
check('and it picks nearly all of the contour',
      len(_picked & _arc) >= int(_N_OUTLINE * 0.9),
      '{} of {}'.format(len(_picked & _arc), _N_OUTLINE))

# Why the outline reading exists: it separates a model that moved the contour
# from one that only repainted the interior. inswapper and alphaface are the
# second kind, and the whole-face reading understates the difference.
_contour = _TGT.copy()
_contour[:_N_OUTLINE] = _SRC[:_N_OUTLINE]
_interior = _TGT.copy()
_interior[_N_OUTLINE:] = _SRC[_N_OUTLINE:]

_a = shape_metric.compare(_SRC, _TGT, _contour)
_b = shape_metric.compare(_SRC, _TGT, _interior)
check('a moved contour reads high at the outline',
      _a.outline_shift > 0.7, '{:+.3f}'.format(_a.outline_shift))
check('an unmoved contour reads near zero at the outline',
      _b.outline_shift < 0.15, '{:+.3f}'.format(_b.outline_shift))
check('and the outline reading separates them better than the whole face',
      (_a.outline_shift - _b.outline_shift) > (_a.shift - _b.shift),
      'outline {:.3f} vs whole-face {:.3f}'.format(
          _a.outline_shift - _b.outline_shift, _a.shift - _b.shift))

# Two heads that already agree have no disagreement to resolve, and the ratio
# would be dividing noise by noise.
_matched = shape_metric.compare(_SRC, _SRC, _SRC)
check('matching head shapes withhold the ratio rather than reporting one',
      _matched is not None and _matched.shift is None
      and _matched.mismatch < 1e-9)

# Direction has to survive. Moving away from the source is a different finding
# from not moving, and must not be clipped to zero.
_away = shape_metric.compare(_SRC, _TGT, _TGT + (_TGT - _SRC) * 0.5)
check('an output further from the source than the target reads negative',
      _away.shift < -0.05, '{:+.3f}'.format(_away.shift))

# Absent readings are omitted rather than zeroed: zero means "kept the target's
# shape", which is a measurement, not a failure to measure.
check('unmeasurable readings are omitted, not zeroed',
      'shape_shift' not in _matched.as_readings()
      and 'shape_mismatch' in _matched.as_readings())

# A pack with no landmark model leaves the probe silent rather than raising.
_blind = shape_metric.ShapeProbe(
    types.SimpleNamespace(landmark_model=lambda: None))
check('no landmark model means no shape probe, and no exception',
      not _blind.available
      and _blind.landmarks(np.zeros((64, 64, 3), np.uint8),
                           [0.0, 0.0, 60.0, 60.0]) is None)

# Mismatched point counts are a wiring mistake, not a shape difference.
check('point sets of different sizes are refused',
      shape_metric.compare(_SRC, _TGT, _TGT[:-4]) is None)

# The attribution split. A crop in aligned space has to be comparable with
# landmarks in frame space without anything being re-derived, or the generated
# contour cannot be measured at all — it exists in no other space.
_ALIGNED = 256
_probe_model = MagicMock()
_probe_model.get.return_value = _CONTOUR_IN_CROP = (
    _SRC * 60.0 + np.array([128.0, 128.0]))
_probe = shape_metric.ShapeProbe(
    types.SimpleNamespace(landmark_model=lambda: _probe_model))

# `Face` comes from the stubbed insightface, so calling it returns a MagicMock
# and the bbox never reaches an attribute anything can read back. Stand a real
# object in its place for the duration, or the box is untestable — and the box
# is the part of this that can silently be wrong.
_saved_face = shape_metric.Face
shape_metric.Face = lambda **kw: types.SimpleNamespace(**kw)

_crop = np.zeros((_ALIGNED, _ALIGNED, 3), dtype=np.uint8)
_to_aligned = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
_in_crop = _probe.landmarks_aligned(
    _crop, [10.0, 10.0, 100.0, 120.0], _to_aligned)
check('an aligned crop yields landmarks in crop coordinates',
      _in_crop is not None and _in_crop.shape == (106, 2))

_box = np.asarray(_probe_model.get.call_args[0][1].bbox, dtype=float)
check('the box handed to the model is the target box warped into the crop',
      abs(_box[0] - 20.0) < 1e-3 and abs(_box[3] - 240.0) < 1e-3,
      'got {}'.format([round(float(v), 1) for v in _box]))

# A box warped outside the crop must be clipped, not passed through negative.
_probe_model.get.reset_mock()
_probe.landmarks_aligned(_crop, [-400.0, -400.0, 40.0, 40.0], _to_aligned)
_clipped = np.asarray(_probe_model.get.call_args[0][1].bbox, dtype=float)
check('a box reaching outside the crop is clipped to it',
      _clipped[0] >= 0.0 and _clipped[2] <= _ALIGNED,
      'got {}'.format([round(float(v), 1) for v in _clipped]))

shape_metric.Face = _saved_face

# The claim the whole attribution rests on: a crop measured in aligned space is
# directly comparable with a frame measured in frame space, because each set is
# normalised independently before the fit.
_aligned_contour = _CONTOUR_IN_CROP
_frame_contour = _SRC * 11.0 + np.array([900.0, 40.0])
check('the same shape read in two different spaces reads identically',
      abs(shape_metric.compare(_SRC, _TGT, _aligned_contour).outline_shift
          - shape_metric.compare(_SRC, _TGT, _frame_contour).outline_shift)
      < 1e-9)

# Attribution has to survive into the report, and the two causes of a low
# shift must produce different recommendations.
_clipped_report = Readings()
_ceiling_report = Readings()
for _ in range(12):
    # Generated a contour, then lost it downstream -> mask_shape_growth.
    _clipped_report.record('shape_mismatch', 0.06)
    _clipped_report.record('outline_mismatch', 0.08)
    _clipped_report.record('outline_swap', 0.40)
    _clipped_report.record('outline_final', 0.36)
    _clipped_report.record('outline_shift', 0.05)
    # Never generated one -> a different model, and the mask is innocent.
    _ceiling_report.record('shape_mismatch', 0.06)
    _ceiling_report.record('outline_mismatch', 0.08)
    _ceiling_report.record('outline_swap', 0.01)
    _ceiling_report.record('outline_shift', 0.01)

_clipped_text = _clipped_report.format_report()
_ceiling_text = _ceiling_report.format_report()
check('a clipped contour reports what was generated against what survived',
      'the generator produced' in _clipped_text
      and 'taken back downstream' in _clipped_text)
check('an unmoved contour says the mask is not the lever',
      'mask_shape_growth is not' in _ceiling_text
      and 'GENERATOR never moved' in _ceiling_text)
check('and the two do not give the same advice',
      ('GENERATOR never moved' in _ceiling_text)
      != ('GENERATOR never moved' in _clipped_text))

# The stage that took the most is named, rather than the mask being assumed.
# Restoration can eat a contour too — it regresses a face toward its training
# manifold — and recommending the wrong knob is worse than recommending none.
_restore_report = Readings()
for _ in range(12):
    _restore_report.record('shape_mismatch', 0.06)
    _restore_report.record('outline_swap', 0.40)
    _restore_report.record('outline_final', 0.09)   # restoration took 0.31
    _restore_report.record('outline_shift', 0.08)   # the mask took 0.01
_restore_text = _restore_report.format_report()
check('a contour lost to restoration blames restoration, not the mask',
      'most of it by restoration' in _restore_text
      and 'Sweep enhance_strength' in _restore_text)
check('and the mask-clipped case still blames the mask',
      'most of it by mask and paste' in _clipped_text
      and 'Sweep mask_shape_growth' in _clipped_text)

# Without the intermediate reading there is nothing to attribute, and the
# report must fall back rather than invent a stage.
_bare = Readings()
for _ in range(12):
    _bare.record('shape_mismatch', 0.06)
    _bare.record('outline_shift', 0.02)
check('no intermediate reading falls back instead of attributing',
      'the generator produced' not in _bare.format_report()
      and 'entirely the target' in _bare.format_report())

# And a pairing whose heads already agree must not recommend anything at all.
_close = Readings()
for _ in range(12):
    _close.record('shape_mismatch', 0.008)
    _close.record('outline_shift', 0.01)
check('heads that already match recommend no lever',
      'mask_shape_growth' not in _close.format_report())


# ── 6. The shape reference is chosen for geometry, not for pores ───────────
print('\nThe shape reference prefers a frontal photograph')

# Measured on a real source set: the texture pick came back at -28 degrees of
# yaw, because sharpness carries twice the weight of frontality there. That is
# correct for pores and wrong for a silhouette, so the two picks are separate.


def _shot(sharpness: float, yaw: float, pitch: float = 0.0,
          extent: float = 300.0) -> tuple:
    """A frame and detection with the given sharpness, pose and face size."""
    rng = np.random.default_rng(int(abs(yaw) * 7 + sharpness))
    frame = np.full((640, 640, 3), 120, dtype=np.uint8)
    box = (170.0, 170.0, 170.0 + extent, 170.0 + extent)
    # Laplacian variance follows the amplitude of the noise painted into the
    # face, which is what `guards.sharpness` actually measures.
    patch = rng.normal(120, sharpness, (int(extent), int(extent), 3))
    frame[170:170 + int(extent), 170:170 + int(extent)] = np.clip(
        patch, 0, 255).astype(np.uint8)

    face = types.SimpleNamespace(
        pose=np.array([pitch, yaw, 0.0], dtype=np.float32),
        kps=np.array([[220.0, 250.0], [400.0, 250.0], [310.0, 330.0],
                      [240.0, 400.0], [380.0, 400.0]], dtype=np.float32),
        landmark_2d_106=_SRC.copy(),
    )
    # A real Bbox rather than a stand-in: `guards.sharpness` and
    # `_exposure_score` both read it, and a namespace that satisfied one would
    # quietly fail the other.
    detection = types.SimpleNamespace(
        face=face, kps=face.kps,
        bbox=Bbox(x=int(box[0]), y=int(box[1]),
                  w=int(extent), h=int(extent)))
    return frame, detection


# Off-axis combines yaw and pitch, and ignores roll — the shape metric fits
# rotation away before measuring, so a tilted head costs nothing.
check('off_axis combines yaw and pitch in quadrature',
      abs(db.off_axis(_shot(20.0, 30.0, 40.0)[1]) - 50.0) < 1e-6,
      '{:.2f}'.format(db.off_axis(_shot(20.0, 30.0, 40.0)[1])))
check('off_axis ignores roll',
      abs(db.off_axis(_shot(20.0, 12.0, 0.0)[1]) - 12.0) < 1e-6)
check('off_axis is None when pose cannot be read',
      db.off_axis(types.SimpleNamespace(
          face=types.SimpleNamespace(pose=None), kps=None)) is None)

# The decisive case: a sharp angled photograph against a softer frontal one.
# The texture picker must keep preferring the sharp one, and the shape picker
# must not — that divergence is the entire reason for a second scorer.
_sharp_angled = _shot(sharpness=60.0, yaw=28.0)
_soft_frontal = _shot(sharpness=14.0, yaw=2.0)

_t_angled = db._texture_score(*_sharp_angled)
_t_frontal = db._texture_score(*_soft_frontal)
_s_angled = db._shape_score(*_sharp_angled)
_s_frontal = db._shape_score(*_soft_frontal)

check('the texture score still prefers the sharp angled photograph',
      _t_angled > _t_frontal,
      'angled {:.3f} vs frontal {:.3f}'.format(_t_angled, _t_frontal))
check('the shape score prefers the frontal one instead',
      _s_frontal > _s_angled,
      'frontal {:.3f} vs angled {:.3f}'.format(_s_frontal, _s_angled))
check('and the two therefore disagree, which is why there are two',
      (_t_angled > _t_frontal) != (_s_frontal < _s_angled))

# Frontality has to dominate the shape score, or the fix does not hold when a
# very sharp angled photograph is in the set.
_very_sharp_angled = _shot(sharpness=200.0, yaw=28.0)
check('frontality outweighs sharpness in the shape score',
      db._shape_score(*_soft_frontal)
      > db._shape_score(*_very_sharp_angled),
      'frontal {:.3f} vs very sharp angled {:.3f}'.format(
          db._shape_score(*_soft_frontal),
          db._shape_score(*_very_sharp_angled)))

# A pitched-down photograph is as bad a shape reference as a turned one, and
# the texture picker cannot see that at all — it reads yaw only.
_pitched = _shot(sharpness=60.0, yaw=0.0, pitch=30.0)
check('a pitched-down photograph is penalised by the shape score',
      db._shape_score(*_soft_frontal) > db._shape_score(*_pitched),
      'frontal {:.3f} vs pitched {:.3f}'.format(
          db._shape_score(*_soft_frontal),
          db._shape_score(*_pitched)))

# An unreadable pose scores neutral, never frontal: a trimmed pack must not
# silently promote every photograph to the top of the dominant term.
_no_pose = _shot(sharpness=60.0, yaw=0.0)
_no_pose[1].face.pose = None
_no_pose[1].kps = None
check('an unreadable pose scores neutral rather than frontal',
      db._shape_score(*_no_pose) < db._shape_score(*_soft_frontal))


# ── 7. mask_erode has a quantisation floor ─────────────────────────────────
print('\nmask_erode survives rounding at every working size')

# `_build` computes `erode_px = int(round(size * mask_erode))`, a constant
# number of pixels, while `_expand_hull` grows the hull by a *fraction* of its
# radius. The two only stay in proportion while the rounding does not dominate:
# below about 0.008 the erode collapses to 1px at several sizes at once, so the
# same setting removes a quarter of the expansion at one aligned size and a
# sixth at another — behaviour that would then depend on the preset and on how
# close the operator is sitting.
_SIZES = (128, 192, 256, 320)   # _ALIGNED_MIN through the largest ceiling
_default_erode = FaceSwapConfig().mask_erode


def _erode_px(size: int, erode: float) -> int:
    """The pixel erosion `FaceMasker._build` would apply."""
    return int(round(size * max(0.0, min(erode, 0.25))))


_at_default = [_erode_px(s, _default_erode) for s in _SIZES]
check('the default erode acts at every aligned size, including the smallest',
      all(px >= 1 for px in _at_default),
      'px {} at sizes {}'.format(_at_default, list(_SIZES)))

# Proportional means the pixel count tracks the size rather than flattening.
check('and it scales with the crop rather than flattening into rounding',
      len(set(_at_default)) == len(_SIZES) and _at_default == sorted(_at_default),
      'px {}'.format(_at_default))

# The budget-against-spend verdict. A layer that reserves and then under-fills
# leaves the face softer than with texture off, and that state was previously
# invisible: the readings carried the numbers and nothing compared them.
_starved = Readings()
_filled = Readings()
for _ in range(12):
    _starved.record('texture_headroom', 6.5)
    _starved.record('texture_delivered', 1.4)      # 22% of budget
    _filled.record('texture_headroom', 6.5)
    _filled.record('texture_delivered', 5.9)       # 91% of budget

_starved_text = _starved.format_report()
check('an under-filled texture reservation says the face is softer',
      'SOFTER than with texture off' in _starved_text
      and 'Do not raise texture_strength' in _starved_text)
check('and a filled one says the reservation is being filled',
      'reservation is being filled' in _filled.format_report()
      and 'SOFTER' not in _filled.format_report())

# Neither reading alone can produce the verdict — that is the point of the pair.
_alone = Readings()
for _ in range(12):
    _alone.record('texture_headroom', 6.5)
check('headroom without delivered produces no budget verdict',
      'budget' not in _alone.format_report())


# The value below the floor is the one this is protecting against: it must be
# visibly worse on that same test, or the floor is not where it is claimed.
_below = [_erode_px(s, 0.0075) for s in _SIZES]
check('a value below the floor does flatten, which is why 0.015 is the floor',
      len(set(_below)) < len(_SIZES), 'px {}'.format(_below))

# And the old default cancelled the hull expansion outright, which is the
# defect that was measured — kept as the explanation for why it moved.
check('the old 0.03 removed roughly the whole 10% hull expansion',
      all(abs(_erode_px(s, 0.03) - 0.10 * 0.33 * s) < 1.0 for s in _SIZES),
      'px {} against expansion {}'.format(
          [_erode_px(s, 0.03) for s in _SIZES],
          [round(0.10 * 0.33 * s, 1) for s in _SIZES]))


print('\n' + '=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
print('=' * 70)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
