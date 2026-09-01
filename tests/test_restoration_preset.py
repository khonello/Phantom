"""
Restoration strength as named steps.

The header once had an ENHANCE *toggle*, removed because "off" was never "less
plastic" — it was no restoration at all, a 128-native swap dropped into a sharp
frame. A switch across an axis that is not binary.

A list is not the same object. `off` as the bottom of a scale reads as one end
of a range, which is what it is. These pin that, and the one interaction that
would otherwise undo the operator silently.
"""

import pytest

from pipeline.api.schema import COMMANDS, RESTORATION_PRESETS
from pipeline.config import FaceSwapConfig


def _config(model='inswapper_128'):
    c = FaceSwapConfig()
    c.apply_preset('optimal')
    c.apply_model_profile(model)
    return c


# ── The presets themselves ─────────────────────────────────────────────


def test_auto_is_the_default():
    """
    The behaviour that existed before this was a control: the swap model's
    profile decides. Anything else is an explicit operator choice.
    """
    assert FaceSwapConfig().restoration_preset == 'auto'
    assert RESTORATION_PRESETS['auto'] is None


def test_off_is_the_bottom_of_the_scale_not_a_switch():
    assert RESTORATION_PRESETS['off'] == {'enhance': False}
    ordered = [n for n in RESTORATION_PRESETS if n != 'auto']
    assert ordered[0] == 'off', 'off belongs at one end, not in the middle'


def test_strength_rises_across_the_named_steps():
    steps = [RESTORATION_PRESETS[n]['enhance_strength']
             for n in ('subtle', 'balanced', 'full')]
    assert steps == sorted(steps)
    assert steps[-1] == 1.0


def test_every_preset_but_auto_sets_enhance():
    for name, values in RESTORATION_PRESETS.items():
        if values is None:
            continue
        assert 'enhance' in values, name


# ── Applying them ──────────────────────────────────────────────────────


@pytest.mark.parametrize('name', sorted(RESTORATION_PRESETS))
def test_every_preset_applies(name):
    c = _config()
    assert c.apply_restoration_preset(name)
    assert c.restoration_preset == name


def test_off_disables_restoration():
    c = _config()
    c.apply_restoration_preset('off')
    assert c.enhance is False


def test_full_keeps_all_of_the_restored_face():
    c = _config()
    c.apply_restoration_preset('full')
    assert c.enhance is True
    assert c.enhance_strength == 1.0


def test_auto_returns_to_the_model_profile():
    """`auto` is only meaningful as the model's own value."""
    c = _config('hyperswap_1a_256')
    profile_strength = c.enhance_strength

    c.apply_restoration_preset('full')
    assert c.enhance_strength == 1.0

    c.apply_restoration_preset('auto')
    assert c.enhance_strength == profile_strength


def test_an_unknown_preset_is_refused_and_changes_nothing():
    c = _config()
    before = (c.enhance, c.enhance_strength, c.restoration_preset)
    assert c.apply_restoration_preset('cinematic') is False
    assert (c.enhance, c.enhance_strength, c.restoration_preset) == before


# ── The interaction that would otherwise revert the operator ───────────


def test_a_model_change_does_not_undo_an_explicit_choice():
    """
    `apply_model_profile` sets enhance_strength per model — 0.7 for inswapper,
    0.5 for hyperswap — and the desktop applies a profile on start. Without
    this, picking a strength and then changing swapper would revert it with
    nothing said: the same shape as `startPipeline` firing `set_enhance` over a
    pipeline launched with `--no-enhance`.
    """
    c = _config('inswapper_128')
    c.apply_restoration_preset('subtle')
    chosen = c.enhance_strength

    c.apply_model_profile('hyperswap_1a_256')

    assert c.enhance_strength == chosen, 'the model profile overrode the operator'
    assert c.swapper_model == 'hyperswap_1a_256', 'the model itself must still change'


def test_auto_still_follows_the_model_across_a_change():
    """The default must keep deferring, or `auto` means nothing."""
    c = _config('inswapper_128')
    assert c.restoration_preset == 'auto'

    c.apply_model_profile('hyperswap_1a_256')

    from pipeline.services import swapper_models
    assert c.enhance_strength == swapper_models.resolve(
        'hyperswap_1a_256').enhance_strength


def test_a_model_change_still_moves_everything_else():
    """Only the restoration fields are pinned; aligned_min must still follow."""
    c = _config('inswapper_128')
    c.apply_restoration_preset('subtle')

    c.apply_model_profile('hyperswap_1a_256')

    assert c.aligned_min == 256


# ── Reachable from outside ─────────────────────────────────────────────


def test_the_command_is_registered():
    assert 'set_restoration' in COMMANDS


def test_set_realism_also_accepts_it():
    """So tools/realism.py can drive it without a second command."""
    from pipeline.api.handlers import _REALISM_FIELDS

    validator = _REALISM_FIELDS['restoration_preset']
    assert validator('subtle') == 'subtle'
    assert validator('nope') is None


def test_get_state_reports_it():
    """A reconnecting desktop must show the strength actually in force."""
    from pipeline.api.handlers import handle_get_state

    c = _config()
    c.apply_restoration_preset('full')
    assert handle_get_state(c, None).data['restoration_preset'] == 'full'
