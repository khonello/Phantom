"""
`get_stats` — what this pipeline is actually running.

Every question it answers has cost real time to answer another way: which model
loaded (read the log, know what to grep), whether the GPU is in use (read the
log), how much of the paid hour is left (guess from when you started it). The
value is in reporting *resolved* values, since both registries fall back on an
unknown name and the gap between requested and loaded is usually the bug.
"""

import sys

import pytest

from pipeline.api.handlers import HandlerContext, handle_get_stats
from pipeline.api.schema import COMMANDS
from pipeline.config import FaceSwapConfig

sys.path.insert(0, 'tools')


def _stats(config=None, ctx=None):
    config = config or FaceSwapConfig()
    ctx = ctx or HandlerContext(pipeline=None, shutdown_event=None)
    reply = handle_get_stats(config, ctx)
    assert reply.success
    return reply.data


# ── It is reachable ────────────────────────────────────────────────────


def test_registered_as_a_command():
    """`COMMANDS` is checked against dispatch both ways, so this is the seam."""
    assert 'get_stats' in COMMANDS


def test_answers_with_no_pipeline_attached():
    """
    The most likely moment to ask is right after a deploy, before a stream has
    ever started. Requiring a running pipeline would make it useless exactly
    then.
    """
    data = _stats()
    assert data['ready'] is True
    assert data['pipeline_running'] is False


# ── It reports what loaded, not what was asked for ─────────────────────


def test_reports_resolved_models_not_requested_ones():
    """
    Both registries fall back rather than fail, so a typo in `.env` runs the
    default. Reporting the requested string would hide the one thing someone
    is looking for.
    """
    config = FaceSwapConfig()
    config.set('enhancer_model', 'coformer')     # typo
    config.set('swapper_model', 'hyperswapp')    # typo

    data = _stats(config)
    assert data['enhancer']['model'] == 'gpen_bfr_256'
    assert data['swapper']['model'] == 'inswapper_128'


def test_reports_the_swapper_native_size():
    config = FaceSwapConfig()
    config.set('swapper_model', 'hyperswap_1a_256')
    assert _stats(config)['swapper']['native_size'] == 256


def test_reports_the_restoration_crop():
    assert _stats()['enhancer']['crop'] == 256


def test_says_when_restoration_is_off():
    config = FaceSwapConfig()
    config.set('enhance', False)
    data = _stats(config)
    assert data['enhancer']['enabled'] is False
    # Still names the model, so "off" and "misconfigured" are distinguishable.
    assert data['enhancer']['model']


def test_flags_whether_the_fidelity_weight_does_anything():
    """
    `enhancer_weight` is CodeFormer's input and inert on every other model. A
    plausible-looking number with no qualifier is how someone spends an
    afternoon tuning a dead knob.
    """
    config = FaceSwapConfig()
    config.set('enhancer_model', 'gpen_bfr_256')
    assert _stats(config)['realism']['enhancer_weight_active'] is False

    config.set('enhancer_model', 'codeformer')
    assert _stats(config)['realism']['enhancer_weight_active'] is True


# ── The provider check ─────────────────────────────────────────────────


def test_reports_requested_and_available_providers():
    """
    The pair is the point. A requested accelerator missing from `available` is
    the silent CPU fallback that bills a GPU hour and produces nothing usable.
    """
    data = _stats()
    assert 'execution_providers' in data
    assert 'available_providers' in data


def test_the_tool_warns_on_a_provider_that_is_not_available():
    from stats import _warnings

    problems = _warnings({
        'execution_providers': ['CUDAExecutionProvider'],
        'available_providers': ['CPUExecutionProvider'],
        'source_loaded': True,
    })
    assert any('CUDA' in p for p in problems)


def test_the_tool_is_quiet_when_all_is_well():
    from stats import _warnings

    assert _warnings({
        'execution_providers': ['CUDAExecutionProvider'],
        'available_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'source_loaded': True,
    }) == []


def test_cpu_only_is_not_a_warning():
    """`--execution-provider cpu` is supported, not a mistake."""
    from stats import _warnings

    problems = _warnings({
        'execution_providers': ['CPUExecutionProvider'],
        'available_providers': ['CPUExecutionProvider'],
        'source_loaded': True,
    })
    assert problems == []


# ── The server's own clock ─────────────────────────────────────────────


def test_server_facts_are_included_when_offered():
    ctx = HandlerContext(
        pipeline=None,
        shutdown_event=None,
        server_stats=lambda: {'uptime_seconds': 61.0,
                              'auto_stop_remaining_seconds': 900.0,
                              'auto_stop_minutes': 120,
                              'clients': 1},
    )
    server = _stats(ctx=ctx)['server']
    assert server['uptime_seconds'] == 61.0
    assert server['auto_stop_remaining_seconds'] == 900.0


def test_a_failing_server_probe_does_not_take_the_report_down():
    """
    The rest of the report is still worth having. A stats command that raises
    is worse than one that says which part it could not read.
    """
    def boom():
        raise RuntimeError('nope')

    data = _stats(ctx=HandlerContext(pipeline=None, shutdown_event=None,
                                     server_stats=boom))
    assert 'error' in data['server']
    assert data['gpu']


def test_server_section_absent_when_not_offered():
    assert 'server' not in _stats()


# ── The report renders ─────────────────────────────────────────────────


@pytest.mark.parametrize('enhance', [True, False])
def test_render_covers_a_full_report(enhance):
    from stats import _render

    config = FaceSwapConfig()
    config.set('enhance', enhance)
    text = '\n'.join(_render(_stats(config)))

    for expected in ('PIPELINE', 'HARDWARE', 'MODELS', 'LOOK', 'CAPTURE'):
        assert expected in text
    assert ('OFF' in text) is (not enhance)
