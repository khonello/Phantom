"""
The background layer, and the three ways it can silently do nothing.

This is a decorative stage on the display path, so almost every failure it has
is quiet by construction: a missing model, an unknown key, a layer left out of
the decoration gate. Each of those produces a frame that looks exactly like a
frame with the feature switched off, which is why they are pinned here rather
than left to be noticed on a call.

Four claims carry the layer and are each covered below:

1. **The matte decides which pixels survive.** Person pixels come through
   unchanged and background pixels become the replacement — otherwise this is
   an expensive blur of the whole picture.
2. **Two streams do not share one smoothed mask.** `Bridge._decorate` runs from
   the display timer and the webcam thread over two different videos, so a
   shared EMA would blend the operator's room against their swapped self at the
   sum of the two rates.
3. **A missing model degrades to the input frame**, never to an exception and
   never to a half-composited picture.
4. **The order in `_decorate` is background, then filter**, because a filter
   regrades the whole picture and an ungraded background under a graded person
   is the thing this ordering exists to prevent.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import types

import numpy as np
import pytest

from desktop import backgrounds


# ── A stand-in for the ONNX net ──────────────────────────────────────────────
#
# The real segmenter is `cv2.dnn` over a model file that is not in the
# repository, so every test here supplies its own matte. That is the right seam
# anyway: what is under test is the compositing, the smoothing and the
# degradation, none of which is the model's behaviour.

class _StubSegmenter:
    """A segmenter returning a fixed matte, or None to stand for no model."""

    def __init__(self, matte=None, available=True):
        self._matte = matte
        self.available = available
        self.failure = '' if available else 'no segmentation model found'
        self.calls = 0

    def matte(self, frame):
        self.calls += 1
        if self._matte is None:
            return None
        return self._matte.astype(np.float32)


def _frame(width=64, height=48, value=200):
    """A flat BGR frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _half_matte(width=64, height=48):
    """Person on the left half, background on the right."""
    matte = np.zeros((height, width), dtype=np.float32)
    matte[:, : width // 2] = 1.0
    return matte


# ── The registry ─────────────────────────────────────────────────────────────

def test_registry_has_none_first_and_blur_before_colour():
    """
    Order is part of the design, not presentation.

    Blur is the forgiving mode and a solid colour is the one that turns every
    matte error into a high-contrast fringe. The picker gives no other steer, so
    the list order is the steer.
    """
    keys = [b.key for b in backgrounds.BACKGROUNDS]
    assert keys[0] == 'none'

    kinds = [backgrounds.get(k).kind for k in keys]
    first_colour = kinds.index('colour')
    assert 'blur' in kinds[:first_colour], 'blur must precede the colours'


def test_names_are_key_and_label_pairs():
    listed = backgrounds.names()
    assert listed == [{'key': b.key, 'name': b.name}
                      for b in backgrounds.BACKGROUNDS]
    assert all(set(entry) == {'key', 'name'} for entry in listed)


def test_unknown_key_is_a_no_op_not_an_error():
    """
    A background removed in a later version must degrade, not raise.

    Same contract as `desktop/filters.py`: this runs on a live call, and a key
    that no longer resolves is a stale preference rather than a fault.
    """
    renderer = backgrounds.Renderer(_StubSegmenter(_half_matte()))
    frame = _frame()
    for key in ('', 'none', 'no_such_background'):
        assert np.array_equal(renderer.render(frame, key), frame)


# ── The matte actually decides what survives ────────────────────────────────

def test_colour_fills_background_and_leaves_the_person():
    """
    The half the matte claims comes through; the half it does not is replaced.

    Sampled away from the boundary, because the mask is feathered on purpose
    and the transition column is meant to be neither.
    """
    stub = _StubSegmenter(_half_matte())
    renderer = backgrounds.Renderer(stub)

    out = renderer.render(_frame(value=200), 'white')
    white = backgrounds.get('white').colour

    # Person side, clear of the feather.
    assert np.allclose(out[:, 4], 200, atol=2)
    # Background side, clear of the feather.
    assert np.allclose(out[:, -4], white, atol=2)


def test_blur_replaces_the_background_with_the_scene_not_a_colour():
    """
    Blur has to carry the room's own tones, or it is just a grey card.

    A frame with a bright right half is blurred behind a person on the left; the
    background must stay bright rather than becoming a constant.
    """
    frame = _frame(value=40)
    frame[:, 32:] = 240

    renderer = backgrounds.Renderer(_StubSegmenter(_half_matte()))
    out = renderer.render(frame, 'blur')

    assert out[:, -4].mean() > 150, 'blurred background lost the scene'
    assert np.allclose(out[:, 4], 40, atol=3), 'person side was disturbed'


def test_output_keeps_the_input_dtype_and_shape():
    renderer = backgrounds.Renderer(_StubSegmenter(_half_matte()))
    frame = _frame()
    out = renderer.render(frame, 'blur')
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


# ── Degradation ─────────────────────────────────────────────────────────────

def test_missing_model_returns_the_frame_untouched():
    """
    No model is the shipping state until someone puts the file in place.

    It must look like the feature is off, not like the picture is broken.
    """
    renderer = backgrounds.Renderer(_StubSegmenter(None, available=False))
    frame = _frame()
    assert np.array_equal(renderer.render(frame, 'blur'), frame)
    assert not renderer.available
    assert renderer.failure


def test_a_dead_net_stops_the_layer_instead_of_freezing_the_mask():
    """
    A net that starts failing must stop the layer, not hold its last mask.

    The matte is only recomputed every `_MATTE_INTERVAL` frames, so a frame or
    two after the failure is still composited from a mask at most one interval
    old — that is what decimation means and it is fine. What must not happen is
    the mask being held indefinitely: `Segmenter` nulls its net on any inference
    failure, so the failure is permanent, and holding would paint a frozen
    silhouette over moving video for the rest of the session.
    """
    stub = _StubSegmenter(_half_matte())
    renderer = backgrounds.Renderer(stub)
    assert not np.array_equal(renderer.render(_frame(), 'white'), _frame())

    stub._matte = None
    frame = _frame()
    outputs = [renderer.render(frame, 'white')
               for _ in range(backgrounds._MATTE_INTERVAL + 2)]

    assert np.array_equal(outputs[-1], frame), 'the layer never gave up'
    assert all(np.array_equal(o, frame)
               for o in outputs[backgrounds._MATTE_INTERVAL:]), (
        'the layer should be off within one matte interval of the failure')


def test_the_matte_is_recomputed_only_once_per_interval():
    """
    The model is the whole cost of this layer, so the schedule is the design.

    Pinned because it is invisible from the output: a renderer calling the net
    every frame produces exactly the same picture and three times the CPU.
    """
    stub = _StubSegmenter(_half_matte())
    renderer = backgrounds.Renderer(stub)

    frames = 8
    for _ in range(frames):
        renderer.render(_frame(), 'white')

    expected = frames // backgrounds._MATTE_INTERVAL
    assert stub.calls == expected, (
        'expected {} inferences over {} frames at interval {}, got {}'.format(
            expected, frames, backgrounds._MATTE_INTERVAL, stub.calls))


def test_a_decimated_frame_still_composites():
    """
    Skipping the model must not skip the background.

    The cheap mistake is to decimate the whole layer rather than just the
    inference, which would flash the real room every other frame.
    """
    renderer = backgrounds.Renderer(_StubSegmenter(_half_matte()))
    white = backgrounds.get('white').colour
    for _ in range(6):
        out = renderer.render(_frame(value=200), 'white')
        assert np.allclose(out[:, -4], white, atol=2)


def test_non_bgr_input_is_refused_rather_than_reshaped():
    renderer = backgrounds.Renderer(_StubSegmenter(_half_matte()))
    grey = np.full((48, 64), 128, dtype=np.uint8)
    assert np.array_equal(renderer.render(grey, 'blur'), grey)


# ── Temporal state, which is the whole reason Renderer is a class ───────────

def test_two_renderers_keep_independent_masks():
    """
    The display timer and the webcam thread must not share smoothing state.

    They carry two different videos. One shared EMA would be advanced by both
    callers at the sum of their rates, blending the operator's own room against
    their swapped self — the failure `desktop/effects.py` avoids by being a pure
    function of the clock, which a matte cannot be.
    """
    left = _half_matte()
    right = 1.0 - left

    a = backgrounds.Renderer(_StubSegmenter(left))
    b = backgrounds.Renderer(_StubSegmenter(right))

    for _ in range(4):
        a.render(_frame(), 'white')
        b.render(_frame(), 'white')

    out_a = a.render(_frame(value=200), 'white')
    out_b = b.render(_frame(value=200), 'white')

    # Each kept its own side of the picture.
    assert out_a[:, 4].mean() < out_a[:, -4].mean()
    assert out_b[:, 4].mean() > out_b[:, -4].mean()


def test_selecting_none_drops_the_smoothed_mask():
    """
    Otherwise the previous mode's edge bleeds into the first frames of the next.
    """
    renderer = backgrounds.Renderer(_StubSegmenter(_half_matte()))
    renderer.render(_frame(), 'white')
    renderer.render(_frame(), 'none')
    assert renderer._previous is None


def test_a_resolution_change_does_not_blend_against_the_old_shape():
    """A preset change resizes the frame; the stored mask no longer applies."""
    stub = _StubSegmenter(_half_matte(64, 48))
    renderer = backgrounds.Renderer(stub)
    renderer.render(_frame(64, 48), 'white')

    stub._matte = _half_matte(32, 24)
    out = renderer.render(_frame(32, 24), 'white')
    assert out.shape == (24, 32, 3)


def test_smoothing_converges_rather_than_dissolving_the_subject():
    """
    The stored mask is the estimate, not an ever-more-blurred copy of itself.

    Feathering the state instead of the output would compound the blur every
    frame; over a few seconds the subject would fade into the background. The
    person's own pixels must stay put under a constant matte.
    """
    renderer = backgrounds.Renderer(_StubSegmenter(_half_matte()))
    out = None
    for _ in range(60):
        out = renderer.render(_frame(value=200), 'white')
    assert np.allclose(out[:, 4], 200, atol=2)


# ── Model resolution ────────────────────────────────────────────────────────

def test_model_path_honours_the_override_and_refuses_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv('PHANTOM_SEGMENT_MODEL', str(tmp_path / 'absent.onnx'))
    assert backgrounds.model_path() == ''

    present = tmp_path / 'present.onnx'
    present.write_bytes(b'not really onnx')
    monkeypatch.setenv('PHANTOM_SEGMENT_MODEL', str(present))
    assert backgrounds.model_path() == str(present)


def test_a_model_that_will_not_load_is_reported_not_raised(tmp_path, monkeypatch):
    """
    `Segmenter` records the reason rather than throwing on the display path.
    """
    bad = tmp_path / 'bad.onnx'
    bad.write_bytes(b'definitely not onnx')
    monkeypatch.setenv('PHANTOM_SEGMENT_MODEL', str(bad))

    segmenter = backgrounds.Segmenter()
    assert not segmenter.available
    assert 'could not be loaded' in segmenter.failure
    assert segmenter.matte(_frame()) is None


# ── Output-shape handling, which decides matte polarity ─────────────────────

@pytest.mark.parametrize('output, description', [
    (np.zeros((1, 1, 8, 8), dtype=np.float32), 'single-channel sigmoid'),
    (np.zeros((1, 2, 8, 8), dtype=np.float32), 'two-channel softmax'),
    (np.zeros((1, 8, 8, 1), dtype=np.float32), 'channels-last single'),
])
def test_known_output_shapes_reduce_to_one_plane(output, description):
    mask = backgrounds._to_mask(output, 16, 12)
    assert mask is not None, description
    assert mask.shape == (12, 16)
    assert mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


def test_a_logit_output_is_squashed_not_clipped():
    """
    Clipping a logit map would turn a usable matte into a hard binary one.

    The give-away is the midpoint: a logit of 0 is probability 0.5, and a clip
    would leave it at 0.0.
    """
    logits = np.array([[[-4.0, 0.0], [4.0, 0.0]]], dtype=np.float32)
    mask = backgrounds._to_mask(logits, 2, 2)
    assert mask is not None
    assert 0.4 < float(mask.max()) <= 1.0
    assert float(mask.min()) < 0.1


def test_an_unreadable_output_shape_returns_none():
    assert backgrounds._to_mask(np.zeros((2, 3, 4, 5, 6), dtype=np.float32), 8, 8) is None


# ── The order in the chain, on the real Bridge method ───────────────────────

def test_decorate_applies_the_background_before_the_filter():
    """
    Background, then grade — and the ordering is correctness, not arrangement.

    A filter regrades the whole picture. A background replaced first is graded
    along with everything else; grading first would leave an ungraded background
    behind a graded person, which reads as pasted on. The rail that sets the
    background used to hold the effects, whose slot is *after* the filter, so
    "reuse the rail, reuse the slot" is the natural mistake and this is what
    catches it.

    Driven through `Bridge._decorate` itself rather than a reimplementation of
    it: the claim is about that method's order of operations, and a test that
    restates the order proves only that it was restated.
    """
    pytest.importorskip('PySide6', reason='the bridge needs Qt')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from desktop.bridge import Bridge

    seen = []

    class _MarkerBackground:
        """Stands in for a renderer, recording when it ran."""

        def render(self, frame, key):
            seen.append('background')
            return frame

    bridge = Bridge.__new__(Bridge)
    bridge._filters_enabled = True
    bridge._filter = 'mono'
    bridge._effect = 'none'
    bridge._background = 'blur'
    bridge._bg_display = _MarkerBackground()

    import desktop.bridge as bridge_module

    real_apply = bridge_module.look_filters.apply

    def _watched_apply(frame, key):
        seen.append('filter')
        return real_apply(frame, key)

    bridge_module.look_filters.apply = _watched_apply
    try:
        bridge._decorate(_frame(), bridge._bg_display)
    finally:
        bridge_module.look_filters.apply = real_apply

    assert seen == ['background', 'filter'], (
        'expected background then filter, got {}'.format(seen))


def test_the_decoration_gate_names_the_background():
    """
    A layer missing from `_has_decoration` does nothing whenever no other layer
    is on, then starts working the moment one is — silent, and reproducible only
    by accident. The same class as the webp mimetype bug.
    """
    pytest.importorskip('PySide6', reason='the bridge needs Qt')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    from desktop.bridge import Bridge

    bridge = Bridge.__new__(Bridge)
    bridge._filters_enabled = True
    bridge._filter = 'none'
    bridge._effect = 'none'
    bridge._background = 'blur'

    assert bridge._decorating(), (
        'a background alone must put the frame on the decorated path')

    # With APPLY off nothing is decorated, whatever is selected. This is the
    # half of the gate that actually distinguishes the two paths.
    bridge._filters_enabled = False
    assert not bridge._decorating(), 'APPLY is off; the cheap path should hold'

    # Note what is *not* asserted, because it is not true and was not true
    # before this layer existed: with APPLY on and everything set to 'none',
    # `_decorating()` is still True. `_filter_key()` returns the selected key
    # rather than '', and 'none' is a non-empty string. The cost is one JPEG
    # decode per frame that changes nothing, so it is a small waste rather than
    # a defect, and tightening it would change `_filter_key`'s contract for
    # every caller. Recorded here so the next reader does not take the gate to
    # mean more than it does.
    bridge._filters_enabled = True
    bridge._background = 'none'
    assert bridge._decorating()


def test_picking_a_look_engages_the_panel(monkeypatch):
    """
    A picker whose chips light up while the picture does not move is broken.

    Selecting a background changed nothing until ENABLE was pressed
    separately, and the reported symptom was exactly that: "clicking any of
    the vertical options doesn't seem to work". The chip highlighted, the
    frame did not, and there was no way to tell a pending look from a dead
    feature.

    Note which resolution this is. The other one — render the pick locally
    while the virtual camera stays ungraded — would make the operator's
    preview disagree with what the call sees, which is the failure the single
    accessor exists to prevent. So picking engages everything at once, and
    ENABLE stays a master switch rather than becoming a commit step.
    """
    from desktop.bridge import Bridge

    seen = []
    monkeypatch.setattr(Bridge, 'filtersEnabledChanged',
                        types.SimpleNamespace(emit=seen.append))
    for name in ('filterChanged', 'effectChanged', 'backgroundChanged'):
        monkeypatch.setattr(Bridge, name,
                            types.SimpleNamespace(emit=lambda _v: None))

    bridge = Bridge.__new__(Bridge)
    bridge._filters_enabled = False
    bridge._filter = 'none'
    bridge._effect = 'none'
    bridge._background = 'none'
    bridge._bg_display = backgrounds.Renderer()
    bridge._bg_webcam = backgrounds.Renderer()
    bridge._warned_no_segmenter = False

    assert not bridge._decorating(), 'nothing picked, nothing decorated'

    bridge.selectBackground('blur')
    assert bridge._filters_enabled, (
        'picking a background must engage the panel, or the click does nothing')
    assert bridge._decorating(), 'and the frame must reach the decorated path'
    assert seen == [True], 'the UI has to be told once, not per click'

    # Picking `none` for one layer says nothing about the other two, so it
    # must not tear the whole panel down.
    bridge.selectBackground('none')
    assert bridge._filters_enabled, (
        'turning one layer off is not a statement about the others')

    # ENABLE keeps its meaning: everything off, picks retained.
    bridge.toggleFilters()
    assert not bridge._filters_enabled
    assert not bridge._decorating()

    # And a filter engages it just the same, so the two pickers agree.
    bridge.selectFilter('warm')
    assert bridge._filters_enabled, 'the filter strip must behave as the rail'
