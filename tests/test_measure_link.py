"""
Cover the parts of tools/measure_link.py that decide what a run *means*.

The socket half needs a live pipeline and is not tested here. The reading is,
because that is the half a person acts on: it is the difference between "the
link is the constraint" and "the pipeline is", and getting it backwards sends
someone off renting a faster GPU to fix their broadband.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, 'tools')):
    if path not in sys.path:
        sys.path.insert(0, path)

import measure_link  # noqa: E402
from pipeline.api.schema import PRESETS  # noqa: E402


def _run(preset, mbps, delivered, p50, minimum=200.0):
    return {
        'preset': preset, 'uplink_mbps': mbps, 'delivered_pct': delivered,
        'p50_ms': p50, 'p95_ms': p50 * 2, 'min_ms': minimum,
        'received': 100 if delivered else 0, 'sent': 100,
    }


NETWORK = {'p50': 213.0, 'p95': 320.0, 'min': 192.0, 'max': 400.0, 'samples': 15.0}


# ── The reading ──────────────────────────────────────────────────────────────

def test_a_large_saving_from_less_bitrate_names_bandwidth():
    """
    The measurement this tool exists for. CLAUDE.md predicted it and the first
    run confirmed it: optimal 3.96 Mbps at p50 1222ms against fast 1.58 Mbps at
    p50 366ms. A preset change worth ~10ms of compute moved 856ms.
    """
    text = '\n'.join(measure_link._verdict(NETWORK, [
        _run('optimal', 3.96, 61.0, 1222.0),
        _run('fast', 1.58, 94.0, 366.0),
    ]))
    assert 'constraint is bandwidth' in text
    assert 'No GPU and no datacenter changes it' in text


def test_a_small_saving_clears_the_uplink():
    """The opposite conclusion has to be reachable, or the tool only ever agrees."""
    text = '\n'.join(measure_link._verdict(NETWORK, [
        _run('optimal', 3.96, 99.0, 320.0),
        _run('fast', 1.58, 99.0, 310.0),
    ]))
    assert 'not the constraint' in text
    assert 'constraint is bandwidth' not in text


def test_lost_frames_are_called_out_even_when_the_p50_looks_fine():
    """
    A saturated uplink shows up first as frames that never return, and a p50
    over the survivors flatters it. Delivery has to be stated in its own right.
    """
    text = '\n'.join(measure_link._verdict(NETWORK, [_run('optimal', 3.96, 61.0, 300.0)]))
    assert 'lost 39% of frames' in text
    assert 'survivors' in text


def test_the_pipeline_share_is_separated_from_the_network():
    """min frame RTT - min network RTT is the pipeline plus both JPEG hops."""
    text = '\n'.join(measure_link._verdict(NETWORK, [_run('fast', 1.58, 94.0, 366.0, 247.0)]))
    assert 'about 55ms' in text


def test_nothing_returned_is_not_reported_as_a_link_result():
    """
    Zero delivery usually means the test frame had no face - the live path
    emits only swapped frames. Calling that a bad link would be wrong.
    """
    text = '\n'.join(measure_link._verdict(NETWORK, [
        {'preset': 'fast', 'uplink_mbps': 1.58, 'delivered_pct': 0.0, 'received': 0},
    ]))
    assert 'Not a link result' in text


# ── Arguments ────────────────────────────────────────────────────────────────

def test_presets_come_from_the_schema_so_they_cannot_drift():
    for name in ('fast', 'optimal', 'production'):
        assert name in PRESETS


def test_an_unknown_preset_is_refused_before_anything_is_rented():
    assert measure_link.main(['--presets', 'ludicrous']) == 1


def test_percentile_holds_at_the_edges():
    assert measure_link._percentile([], 0.95) == 0.0
    assert measure_link._percentile([5.0], 0.95) == 5.0
    assert measure_link._percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0


def test_a_frame_without_a_readable_file_fails_loudly():
    with pytest.raises(SystemExit):
        measure_link._encode_frame(os.path.join(REPO_ROOT, 'no-such-image.jpg'),
                                   480, 270, 60)
