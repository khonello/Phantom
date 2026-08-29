"""
The restoration-resolution levers.

Restoration is three quarters of a frame and the only stage whose cost ignores
how big the face is: it runs on a fixed FFHQ crop whether the face covers 101
pixels or 500. These tests cover the two levers aimed at that, and — more
importantly — the ways each one is allowed to *decline*.

Both fail toward the current behaviour. A model that cannot accept a smaller
crop keeps its own size, and a face whose size cannot be measured is restored.
Restoration decides whether output reads as a call or as AI, so a lever that
silently turned it off would change what every participant sees for a reason
nobody could see.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.config import FaceSwapConfig
from pipeline.processing.compositor import FaceCompositor, _FFHQ_ERODE, _FFHQ_FEATHER
from pipeline.services.enhancement import CROP_SIZE, Enhancer, _spatial_size


# ── The model decides, not the config ──────────────────────────────────


@pytest.mark.parametrize('shape, expected', [
    ([1, 3, 512, 512], 512),           # fixed square export
    ([1, 3, 256, 256], 256),
    ([1, 3, 'height', 'width'], None),  # symbolic dims
    (['batch', 3, None, None], None),
    ([1, 3, 512, 256], None),           # fixed but not square: not a size
    (None, CROP_SIZE),                  # no shape declared at all
])
def test_spatial_size_reads_the_graph(shape, expected):
    """A size is only 'fixed' when both spatial dims are equal ints."""
    assert _spatial_size(shape) == expected


def _enhancer(native, **settings):
    """An Enhancer with its backend pre-set, so nothing is loaded."""
    enhancer = Enhancer(FaceSwapConfig(**settings))
    enhancer._backend = SimpleNamespace(native_size=native)
    enhancer._loaded = True
    return enhancer


def test_dynamic_model_honours_restore_size():
    assert _enhancer(None, restore_size=256).crop_size == 256


def test_fixed_model_overrides_restore_size():
    """
    A graph exported at 512 wins, because feeding it 256 throws once per frame
    rather than running faster. This is the answer the pod session is looking
    for, and it has to arrive as a number rather than a crash.
    """
    assert _enhancer(512, restore_size=256).crop_size == 512


def test_the_override_is_said_once(monkeypatch):
    said = []
    monkeypatch.setattr(
        'pipeline.services.enhancement.emit_warning',
        lambda msg, **kw: said.append(msg),
    )
    enhancer = _enhancer(512, restore_size=256)
    for _ in range(5):
        enhancer.crop_size
    assert len(said) == 1, 'this sits on the live path; it must not warn per frame'
    assert '256' in said[0] and '512' in said[0]


def test_a_new_backend_may_be_asked_again(monkeypatch):
    """`clear()` drops the warning, since the next backend may accept the size."""
    said = []
    monkeypatch.setattr(
        'pipeline.services.enhancement.emit_warning',
        lambda msg, **kw: said.append(msg),
    )
    enhancer = _enhancer(512, restore_size=256)
    enhancer.crop_size
    enhancer.clear()
    enhancer._backend = SimpleNamespace(native_size=512)
    enhancer._loaded = True
    enhancer.crop_size
    assert len(said) == 2


@pytest.mark.parametrize('requested, expected', [
    (1024, 512),   # clamped to the training size
    (64, 128),     # clamped to the floor: below the swap it is restoring
    (250, 248),    # snapped down to a multiple of 8
    (0, 512),      # unset falls back to the default rather than to zero
])
def test_restore_size_is_clamped_and_even(requested, expected):
    assert _enhancer(None, restore_size=requested).crop_size == expected


def test_default_is_unchanged_behaviour():
    """The out-of-the-box path must be the 512 crop it has always been."""
    assert _enhancer(None).crop_size == CROP_SIZE
    assert FaceSwapConfig().restore_size == CROP_SIZE
    assert FaceSwapConfig().restore_min_face == 0


# ── Skipping restoration for a small face ──────────────────────────────


def _compositor(**settings):
    return FaceCompositor(FaceSwapConfig(**settings), Enhancer(FaceSwapConfig()), None)


def _face(width, height):
    return SimpleNamespace(bbox=np.array([10.0, 20.0, 10.0 + width, 20.0 + height]))


def test_threshold_off_restores_everything():
    assert _compositor(restore_min_face=0)._restore_worthwhile(_face(20, 20))


def test_small_face_is_skipped():
    assert not _compositor(restore_min_face=120)._restore_worthwhile(_face(101, 129))


def test_large_face_is_restored():
    assert _compositor(restore_min_face=120)._restore_worthwhile(_face(200, 260))


def test_measured_on_the_shorter_side():
    """Same measure as guard_min_frame_px, so the two are set in one unit."""
    compositor = _compositor(restore_min_face=120)
    assert not compositor._restore_worthwhile(_face(400, 100))


@pytest.mark.parametrize('face', [
    SimpleNamespace(bbox=None),
    SimpleNamespace(bbox=np.array([1.0, 2.0])),
    SimpleNamespace(bbox=['x', 'y', 'z', 'w']),
    SimpleNamespace(),
])
def test_an_unmeasurable_face_is_restored(face):
    """Fail toward restoring: skipping is what the operator would see."""
    assert _compositor(restore_min_face=120)._restore_worthwhile(face)


# ── The seam does not change with the crop size ────────────────────────


def test_feathering_reproduces_the_previous_512_constants():
    """
    The feather is a fraction of the crop now, not a pixel count. At 512 it
    must still be the 5px erode and 6.0 sigma it was, or every existing
    judgement about the seam was made against different output.
    """
    assert round(CROP_SIZE * _FFHQ_ERODE) == 5
    assert CROP_SIZE * _FFHQ_FEATHER == pytest.approx(6.0)
