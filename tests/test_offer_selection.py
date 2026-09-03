"""
How `start` chooses a machine, and the two mistakes it must not make.

This replaces `test_gpu_tier.py`, which pinned a hand-typed `_GPU_PERF` table
against the prose that described it. Vast publishes `dlperf`, a measured score
on every offer, so there is no table left to drift — but the reason the table
existed has not gone anywhere: auto-discovery accepting a slow card because the
fast one was busy cost a whole measurement session, and every number in it was
against an architecture nothing else was measured on.

Two properties, and they pull in opposite directions on purpose:

  1. **A speed floor still exists**, so nothing below a 4090 is ever taken
     silently.
  2. **Ordering is by distance, then price** — not by speed. That is only safe
     *because* of (1): once the floor has removed everything slow, the
     remaining offers differ in the two things that actually decide this
     product's felt quality, and speed is no longer one of them.

The orchestrator is loaded by path: `vast/` has no `__init__.py`. `dotenv` is
stubbed even when installed, so `load_dotenv` at import cannot read the
developer's real `.env` — a machine with a configured instance would otherwise
get different results from a clean checkout.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_STUBS = ('requests', 'paramiko', 'dotenv')


def _load_orchestrator():
    saved = {name: sys.modules.get(name) for name in _STUBS}
    for name in _STUBS:
        sys.modules[name] = MagicMock()
    try:
        path = os.path.join(_REPO_ROOT, 'vast', 'orchestrator.py')
        spec = importlib.util.spec_from_file_location('phantom_vast_orch', path)
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

GEO = ['GB', 'IE', 'FR', 'NL', 'BE', 'DE']


def offer(gpu, country, price, dlperf=97.0, **extra):
    body = {
        'id': abs(hash((gpu, country, price))) % 10 ** 8,
        'machine_id': abs(hash((gpu, country))) % 10 ** 6,
        'gpu_name': gpu,
        'geolocation': '{}, {}'.format({'GB': 'United Kingdom', 'FR': 'France',
                                        'DE': 'Germany', 'NL': 'The Netherlands',
                                        'RO': 'Romania'}.get(country, country), country),
        'dph_total': price,
        'dlperf': dlperf,
    }
    body.update(extra)
    return body


# ── The country code is read, not the label ──────────────────────────────────

def test_country_is_parsed_from_the_label():
    """
    Vast returns "United Kingdom, GB" but filters on "GB". Reading the label
    whole would put every offer in the same unknown bucket and silently turn
    the priority ordering back into price ordering.
    """
    assert orch._country(offer('RTX 4090', 'GB', 0.3)) == 'GB'
    assert orch._country(offer('RTX 4090', 'NL', 0.3)) == 'NL'


def test_an_unknown_country_sorts_last_rather_than_first():
    ranked = orch._rank(
        [offer('RTX 4090', 'RO', 0.10), offer('RTX 4090', 'GB', 0.90)], GEO)
    assert orch._country(ranked[0]) == 'GB', 'a cheap distant host must not win'


def test_a_missing_geolocation_does_not_crash():
    bad = offer('RTX 4090', 'GB', 0.30)
    bad['geolocation'] = None
    assert orch._rank([bad], GEO) == [bad]


# ── Distance beats price; price breaks ties ──────────────────────────────────

def test_the_nearer_country_wins_even_when_dearer():
    """
    The bug this ordering was written for: sorting on price alone had a French
    host at $0.336 beating a British one at $0.350 — three cents to give back
    part of the round trip the migration exists to remove.
    """
    ranked = orch._rank(
        [offer('RTX 4090', 'FR', 0.336), offer('RTX 4090', 'GB', 0.350)], GEO)
    assert orch._country(ranked[0]) == 'GB'


def test_price_decides_inside_one_country():
    ranked = orch._rank(
        [offer('RTX 4090', 'GB', 0.90), offer('RTX 4090', 'GB', 0.31)], GEO)
    assert ranked[0]['dph_total'] == pytest.approx(0.31)


def test_the_whole_priority_order_is_respected():
    offers = [offer('RTX 4090', c, 0.5) for c in ('DE', 'NL', 'FR', 'GB')]
    assert [orch._country(o) for o in orch._rank(offers, GEO)] == \
        ['GB', 'FR', 'NL', 'DE']


def test_reordering_the_list_reorders_the_result():
    """The setting is the operator's statement of preference, not a constant."""
    offers = [offer('RTX 4090', 'GB', 0.9), offer('RTX 4090', 'FR', 0.3)]
    assert orch._country(orch._rank(offers, ['FR', 'GB'])[0]) == 'FR'


# ── The floor that makes price-ordering safe ─────────────────────────────────

def test_the_speed_floor_is_a_4090():
    """
    Measured in western Europe on 2026-09-03: RTX 6000 Ada 113, RTX 4090 97.2,
    RTX 5080 83.8, A100 SXM4 83.3, RTX 3090 44.5. The default must sit between
    the 4090 and the next card down, or it is not the tier the docs claim.
    """
    assert 83.8 < orch._DEFAULT_MIN_DLPERF <= 97.2


def test_the_floor_reaches_the_search_as_a_filter(monkeypatch):
    """
    A floor that is computed and not sent is no floor. This asserts the value
    is in the request body, which is the only place it does anything.
    """
    captured = {}

    def fake_request(method, path, auth=True, raise_on_error=False, **kwargs):
        captured.update(kwargs.get('json') or {})
        return {'offers': []}

    monkeypatch.setattr(orch, '_request', fake_request)
    orch._search_offers(
        geolocations=GEO, min_dlperf=90.0, min_vram=16, max_price=1.0,
        min_reliability=0.98, min_inet_up=100.0, min_ports=32,
        max_compute_cap=900, verified_only=True, disk=25)

    assert captured['dlperf'] == {'gte': 90.0}
    assert captured['compute_cap'] == {'lte': 900}
    assert captured['verified'] == {'eq': True}
    assert captured['geolocation'] == {'in': GEO}


def test_non_nvidia_is_excluded(monkeypatch):
    """
    Every ONNX model runs on CUDAExecutionProvider, so a card without it is not
    a slower option — it is no option. MI300X listed at $0.50/hr with 192GB on
    RunPod and passed every other filter, which is exactly what a
    cheapest-first search reaches for.
    """
    captured = {}

    def fake_request(method, path, auth=True, raise_on_error=False, **kwargs):
        captured.update(kwargs.get('json') or {})
        return {'offers': []}

    monkeypatch.setattr(orch, '_request', fake_request)
    orch._search_offers(
        geolocations=GEO, min_dlperf=90.0, min_vram=16, max_price=1.0,
        min_reliability=0.98, min_inet_up=100.0, min_ports=32,
        max_compute_cap=900, verified_only=True, disk=25)
    assert captured['gpu_arch'] == {'eq': 'nvidia'}


def test_a_pinned_host_still_cannot_bring_an_amd_card(monkeypatch):
    """
    Pinning a host deliberately bypasses the quality floors — the host was
    chosen on purpose. It must not bypass the two filters that are about
    whether the software can run at all.
    """
    captured = {}

    def fake_request(method, path, auth=True, raise_on_error=False, **kwargs):
        captured.update(kwargs.get('json') or {})
        return {'offers': []}

    monkeypatch.setattr(orch, '_request', fake_request)
    orch._search_offers(
        geolocations=GEO, min_dlperf=90.0, min_vram=16, max_price=1.0,
        min_reliability=0.98, min_inet_up=100.0, min_ports=32,
        max_compute_cap=900, verified_only=True, disk=25, host_id=135666)

    assert captured['host_id'] == {'eq': 135666}
    assert captured['gpu_arch'] == {'eq': 'nvidia'}
    assert captured['compute_cap'] == {'lte': 900}
    assert 'dlperf' not in captured, 'a pinned host is exempt from the speed floor'


# ── Resume falls back only on capacity ───────────────────────────────────────

def test_capacity_errors_are_recognised():
    for message in (
        'There are no GPUs available on this machine',
        'Insufficient capacity for this offer',
        'instance is already rented',
        'cannot start: machine unavailable',
    ):
        assert orch._is_capacity_error(message), message


def test_other_failures_are_not_treated_as_capacity():
    """
    The gate is the point. Falling back on any resume failure would rent a
    billing instance in response to a typo'd id or a rejected key.
    """
    for message in (
        'HTTP 401: invalid api key',
        'HTTP 404: instance not found',
        'HTTP 422: bad request body',
    ):
        assert not orch._is_capacity_error(message), message


# ── The port map is read the way docker writes it ────────────────────────────

def test_the_websocket_address_comes_from_the_published_mapping():
    instance = {
        'public_ipaddr': '192.0.2.45',
        'ports': {'9000/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '33526'}]},
    }
    assert orch._ws_address(instance) == '192.0.2.45:33526'


def test_no_mapping_yet_is_not_an_address():
    """Only running instances publish `ports`; None here means "not yet"."""
    assert orch._ws_address({'public_ipaddr': '192.0.2.45'}) is None
    assert orch._ws_address({'ports': {'9000/tcp': []}}) is None


def test_ssh_prefers_the_direct_mapping_over_the_proxy():
    """
    `ssh_direct` is what makes exec_command and SFTP work. Falling back to the
    proxy silently would reintroduce every RunPod workaround.
    """
    instance = {
        'public_ipaddr': '192.0.2.45',
        'ports': {'22/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '20000'}]},
        'ssh_host': 'ssh123.vast.ai', 'ssh_port': 10600,
    }
    assert orch._ssh_target(instance) == ('192.0.2.45', 20000)


def test_ssh_falls_back_to_the_proxy_when_there_is_no_direct_port():
    instance = {'ssh_host': 'ssh123.vast.ai', 'ssh_port': 10600}
    assert orch._ssh_target(instance) == ('ssh123.vast.ai', 10600)
