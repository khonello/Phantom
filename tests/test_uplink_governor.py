"""
The rules the uplink governor must not break.

Two of them are the reason it exists rather than details of it. The dropdown is
a **ceiling** — automation may protect the operator from a link that cannot
carry their choice and may never hand them something they did not ask for. And
adaptation is **asymmetric**: one bad window drops a gear, because a saturated
uplink is already costing seconds, while climbing back needs a sustained clean
run, because a gear change nobody needed is a visible change for nothing.

The rest pin the arithmetic that decides those: that the block-time threshold is
a fraction of the send interval rather than a millisecond count (so it means the
same thing at 15fps and at 20), that an idle window is not read as a healthy
one, and that a single capture resolution serves the whole ladder — which is
what lets a gear change avoid reconfiguring the camera at all.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from desktop.uplink import (                                    # noqa: E402
    LADDER,
    UplinkGovernor,
    capture_ceiling,
    gear_for,
)
from pipeline.api.schema import PRESETS                         # noqa: E402


def _saturated(gov: UplinkGovernor, now: float) -> object:
    """One window of a link that cannot carry the current gear."""
    return gov.observe(sent=30, delivered_ratio=0.61,
                       block_ms_per_frame=40.0, now=now)


def _clean(gov: UplinkGovernor, now: float) -> object:
    """One window of a link with headroom."""
    return gov.observe(sent=30, delivered_ratio=1.0,
                       block_ms_per_frame=0.2, now=now)


# ── The ladder itself ────────────────────────────────────────────────────────


def test_ladder_names_are_real_presets():
    for name in LADDER:
        assert name in PRESETS


def test_one_capture_resolution_serves_every_gear():
    """
    The whole decoupling rests on this: if any gear were larger than the
    capture, adapting up would need the camera reconfigured — ~3.9s to first
    frame — and the governor would be unaffordable.
    """
    width, height = capture_ceiling()
    for name in LADDER:
        gear = gear_for(name)
        assert gear.width <= width
        assert gear.height <= height


def test_gear_carries_no_appearance_settings():
    """
    A gear change must not alter how the face looks. `det_size`,
    `aligned_size` and `occluder` decide that, so they must not be reachable
    from a Gear at all.
    """
    gear = gear_for('optimal')
    for field in ('det_size', 'aligned_size', 'occluder'):
        assert not hasattr(gear, field)


def test_unknown_preset_falls_back_rather_than_raising():
    assert gear_for('nonsense').name == 'optimal'
    assert UplinkGovernor('nonsense').ceiling == 'optimal'


# ── The ceiling ──────────────────────────────────────────────────────────────


def test_never_climbs_above_the_operators_choice():
    gov = UplinkGovernor('fast')
    now = 0.0
    for _ in range(60):
        now += 2.0
        _clean(gov, now)
    assert gov.gear.name == 'fast'
    assert gov.shifts_up == 0


def test_climbs_back_to_the_ceiling_and_stops():
    gov = UplinkGovernor('production')
    now = 100.0
    assert _saturated(gov, now) == 'optimal'
    now += 10.0
    assert _saturated(gov, now) == 'fast'

    # Clean from here on. It should reach `production` and stay there.
    for _ in range(80):
        now += 2.0
        _clean(gov, now)
    assert gov.gear.name == 'production'
    assert gov.adapted is False


def test_operator_choice_applies_immediately_in_both_directions():
    """
    Someone who picks a gear expects to see it, not to be climbed towards it.
    """
    gov = UplinkGovernor('production')
    _saturated(gov, 100.0)
    assert gov.gear.name == 'optimal'

    assert gov.set_ceiling('production', now=101.0) is True
    assert gov.gear.name == 'production'

    assert gov.set_ceiling('fast', now=102.0) is True
    assert gov.gear.name == 'fast'


def test_adapted_reports_disagreement_with_the_dropdown():
    gov = UplinkGovernor('optimal')
    assert gov.adapted is False
    _saturated(gov, 100.0)
    assert gov.adapted is True
    assert gov.gear.name == 'fast'


# ── Asymmetry ────────────────────────────────────────────────────────────────


def test_one_bad_window_drops_a_gear():
    gov = UplinkGovernor('optimal')
    assert _saturated(gov, 100.0) == 'fast'
    assert gov.shifts_down == 1


def test_one_good_window_does_not_raise_one():
    gov = UplinkGovernor('optimal')
    _saturated(gov, 100.0)
    assert _clean(gov, 110.0) is None
    assert gov.gear.name == 'fast'


def test_climbing_needs_a_sustained_clean_run():
    gov = UplinkGovernor('optimal')
    _saturated(gov, 100.0)

    now = 110.0
    # Clean, but interrupted before UP_AFTER_S elapses. The run restarts, so
    # nothing climbs.
    for _ in range(5):
        now += 2.0
        assert _clean(gov, now) is None
    now += 2.0
    gov.observe(sent=30, delivered_ratio=0.90, block_ms_per_frame=1.0, now=now)
    for _ in range(5):
        now += 2.0
        assert _clean(gov, now) is None
    assert gov.gear.name == 'fast'


def test_cooldown_prevents_consecutive_shifts():
    gov = UplinkGovernor('production')
    assert _saturated(gov, 100.0) == 'optimal'
    # Inside COOLDOWN_S, however bad the window looks.
    assert _saturated(gov, 100.0 + UplinkGovernor.COOLDOWN_S / 2) is None
    assert gov.gear.name == 'optimal'


def test_never_drops_below_the_bottom_of_the_ladder():
    gov = UplinkGovernor('optimal')
    now = 100.0
    for _ in range(20):
        now += 10.0
        _saturated(gov, now)
    assert gov.gear.name == LADDER[0]


# ── The signals ──────────────────────────────────────────────────────────────


def test_blocking_alone_drops_a_gear_even_with_full_delivery():
    """
    Send-buffer backpressure is the earlier signal. Every frame can still be
    coming back while the sender sits waiting on the socket, and that is a link
    already over budget.
    """
    gov = UplinkGovernor('optimal')
    changed = gov.observe(sent=30, delivered_ratio=1.0,
                          block_ms_per_frame=40.0, now=100.0)
    assert changed == 'fast'


def test_block_threshold_is_a_fraction_of_the_interval():
    """
    The same absolute block time is over budget at 20fps and inside it at 15,
    because the interval differs. A millisecond constant would mean two
    different things on two rungs of the same ladder.
    """
    interval_20 = gear_for('production').interval_ms   # 50ms
    interval_15 = gear_for('optimal').interval_ms      # 66.7ms
    assert interval_20 < interval_15

    block = 0.30 * interval_20   # over budget at 20fps, under it at 15fps
    assert block / interval_20 > UplinkGovernor.DOWN_BLOCK_FRACTION
    assert block / interval_15 < UplinkGovernor.DOWN_BLOCK_FRACTION

    at_20 = UplinkGovernor('production')
    assert at_20.observe(sent=30, delivered_ratio=1.0,
                         block_ms_per_frame=block, now=100.0) == 'optimal'

    at_15 = UplinkGovernor('optimal')
    assert at_15.observe(sent=30, delivered_ratio=1.0,
                         block_ms_per_frame=block, now=100.0) is None


def test_an_idle_window_is_not_evidence_of_health():
    """
    A paused or starting stream produces ratios computed on two or three
    frames. Crediting those towards a climb would have the governor raise the
    gear precisely when it has learned nothing.
    """
    gov = UplinkGovernor('optimal')
    _saturated(gov, 100.0)

    now = 110.0
    for _ in range(40):
        now += 2.0
        assert gov.observe(sent=1, delivered_ratio=1.0,
                           block_ms_per_frame=0.0, now=now) is None
    assert gov.gear.name == 'fast'


def test_disabled_governor_observes_but_never_steers():
    """
    `PHANTOM_UPLINK_ADAPT=0` is for a measurement run: two sessions cannot be
    compared if the thing under test chose itself differently in each.
    """
    gov = UplinkGovernor('optimal', enabled=False)
    now = 100.0
    for _ in range(10):
        now += 10.0
        assert _saturated(gov, now) is None
    assert gov.gear.name == 'optimal'
    assert gov.shifts_down == 0


def test_stats_report_the_gear_and_the_ceiling_separately():
    gov = UplinkGovernor('production')
    _saturated(gov, 100.0)
    stats = gov.stats()
    assert stats['ceiling'] == 'production'
    assert stats['gear'] == 'optimal'
    assert stats['adapted'] is True
    assert stats['reason']
