"""
The GPU speed tier and the bounded wait.

Auto-discovery used to take the fastest *available* card, which silently
accepts a rank-34 L4 when the rank-100 4090 is busy. That is how a whole
measurement session came back with numbers against an architecture nothing
else was measured on. These cover the floor that refuses it, and the wait that
makes refusing affordable — billing starts when a pod runs, not while you
queue for one.

The orchestrator is loaded by path: `runpod/` has no `__init__.py`, and the
name collides with the RunPod SDK it imports. The SDK, `requests` and the SSH
layer are stubbed, since nothing here touches the network.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_STUBS = ('requests', 'runpod', 'paramiko', 'scp', 'dotenv')


def _load_orchestrator():
    """
    Import runpod/orchestrator.py by path, with its network deps stubbed.

    `dotenv` is stubbed even when it is installed, so `load_dotenv` at import
    time cannot read the developer's real `.env` into the process — these
    tests read RUNPOD_* settings, and a machine with a configured pod would
    otherwise get different results from a clean checkout.

    The stubs are removed afterwards. The module keeps its own references, so
    nothing here leaks into the rest of the suite.
    """
    saved = {name: sys.modules.get(name) for name in _STUBS}
    for name in _STUBS:
        sys.modules[name] = MagicMock()
    try:
        path = os.path.join(_REPO_ROOT, 'runpod', 'orchestrator.py')
        spec = importlib.util.spec_from_file_location('phantom_orchestrator', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


orch = _load_orchestrator()


# ── The tier is the five cards it claims to be ─────────────────────────


def test_the_documented_tier_is_what_the_floor_selects():
    """
    The default floor and the names in the docs must not drift apart. If a
    rank moves in _GPU_PERF, this is what says the prose is now wrong.
    """
    tier = sorted(
        name for name, perf in orch._GPU_PERF.items()
        if perf >= orch._DEFAULT_MIN_GPU_PERF
    )
    assert tier == ['H100', 'H200', 'L40S', 'RTX 4090', 'RTX 6000 Ada']


def test_the_l4_that_cost_a_session_is_below_the_floor():
    assert orch._gpu_perf('NVIDIA L4') < orch._DEFAULT_MIN_GPU_PERF


def test_the_4090_is_in_the_tier():
    assert orch._gpu_perf('NVIDIA GeForce RTX 4090') >= orch._DEFAULT_MIN_GPU_PERF


# ── The floor actually filters ─────────────────────────────────────────


_FLEET = [
    {'id': 'NVIDIA GeForce RTX 4090', 'displayName': 'RTX 4090', 'memoryInGb': 24},
    {'id': 'NVIDIA L4', 'displayName': 'L4', 'memoryInGb': 24},
    {'id': 'NVIDIA RTX A4000', 'displayName': 'RTX A4000', 'memoryInGb': 16},
]


@pytest.fixture
def fleet(monkeypatch):
    monkeypatch.setattr(orch, '_get_gpu_types', lambda key: list(_FLEET))
    monkeypatch.setattr(orch, '_get_cheapest_price', lambda gpu: 0.34)
    return _FLEET


def test_no_floor_keeps_everything(fleet):
    names = [n for n, _, _, _ in orch._discover_gpus('k', 16, 1.0, min_perf=0)]
    assert names == ['RTX 4090', 'L4', 'RTX A4000']


def test_floor_drops_the_slow_cards(fleet):
    names = [n for n, _, _, _ in orch._discover_gpus('k', 16, 1.0, min_perf=85)]
    assert names == ['RTX 4090']


def test_an_impossible_floor_returns_nothing(fleet):
    assert orch._discover_gpus('k', 16, 1.0, min_perf=101) == []


def test_still_fastest_first_within_the_tier(fleet):
    """Price is a tiebreak, not the sort. Everything here is already affordable."""
    ranked = orch._discover_gpus('k', 16, 1.0, min_perf=0)
    perfs = [orch._gpu_perf(gpu_id) for _, gpu_id, _, _ in ranked]
    assert perfs == sorted(perfs, reverse=True)


# ── Only capacity is worth waiting out ─────────────────────────────────


@pytest.mark.parametrize('message', [
    'There are not enough free GPUs',
    'no free GPUs available',
    'insufficient capacity in this region',
    'that machine is no longer available',
])
def test_capacity_errors_are_retryable(message):
    assert orch._is_capacity_error(message)


@pytest.mark.parametrize('message', [
    'network volume abc123 not found',
    'invalid image name',
    'unauthorized',
])
def test_other_errors_are_not(message):
    """
    A bad volume id fails identically every sixty seconds. Spending the whole
    wait budget proving that is worse than saying so immediately.
    """
    assert not orch._is_capacity_error(message)


# ── The fallback switch ────────────────────────────────────────────────


@pytest.mark.parametrize('raw, expected', [
    ('true', True), ('True', True), ('1', True), ('yes', True), ('on', True),
    ('false', False), ('0', False), ('no', False), ('', False),
    ('maybe', False),
])
def test_env_flag_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv('RUNPOD_GPU_FALLBACK', raw)
    assert orch._env_flag('RUNPOD_GPU_FALLBACK') is expected


def test_unset_means_fail_rather_than_fall_back(monkeypatch):
    """
    The default is the measurement default: refuse a slower card, because
    numbers from two architectures are not a comparison. A customer session
    opts into falling back.
    """
    monkeypatch.delenv('RUNPOD_GPU_FALLBACK', raising=False)
    assert orch._env_flag('RUNPOD_GPU_FALLBACK') is False


# ── The wait is bounded, and its arithmetic is right ───────────────────


def test_the_budget_allows_the_documented_number_of_retries():
    """
    300s of budget with a 60s interval sleeps five times, so the whole
    candidate list is tried six times in all. The loop stops once less than
    one interval remains rather than sleeping past the deadline.
    """
    budget, interval = orch._DEFAULT_GPU_WAIT, orch._GPU_RETRY_INTERVAL
    remaining, retries = float(budget), 0
    while remaining >= interval:
        remaining -= interval
        retries += 1
    assert (budget, interval, retries) == (300, 60, 5)
