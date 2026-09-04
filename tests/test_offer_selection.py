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
        # A mid-range clock by default, so a test that says nothing about the
        # CPU is ranked purely on country and price, as it was before.
        'cpu_ghz': 4.5,
        'cpu_cores_effective': 16.0,
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
        min_cpu_ghz=3.5, min_cpu_cores=8.0,
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
        min_cpu_ghz=3.5, min_cpu_cores=8.0,
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
        min_cpu_ghz=3.5, min_cpu_cores=8.0,
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


# ── Secrets must not reach a terminal ────────────────────────────────────────
# Two credentials travel through `_ssh_run` and both were printed in full:
# startup.sh reports `API_TOKEN <hex>` on stdout for the orchestrator to parse,
# and the launch command carries PHANTOM_API_TOKEN plus a Vast API key — the
# account-wide one whenever VAST_SCOPED_API_KEY is unset. `_update_env_key`
# already masked them on the way into .env, which is what made the louder path
# an oversight rather than a decision.

TOKEN = 'deadbeef' * 8


def test_a_known_secret_is_masked_by_value():
    out = orch._redact('export PHANTOM_API_TOKEN={};'.format(TOKEN), [TOKEN])
    assert TOKEN not in out and '<redacted>' in out


def test_the_token_line_is_masked_before_its_value_is_known():
    """
    The first time the token appears is the line we are reading it from, so
    redaction by value cannot help there. That line is matched by shape.
    """
    assert TOKEN not in orch._redact('API_TOKEN {}'.format(TOKEN))


def test_an_api_key_on_the_launch_command_is_masked():
    key = 'vast-' + 'a' * 40
    launch = "export VAST_API_KEY='{}'; nohup python pipeline.py &".format(key)
    assert key not in orch._redact(launch, [key])


def test_redaction_leaves_ordinary_output_alone():
    """
    Short values are skipped: redacting every occurrence of a three-letter
    string would blank most of a deploy transcript.
    """
    assert orch._redact('the cat sat on the mat', ['cat']) == 'the cat sat on the mat'


def test_redaction_survives_an_empty_secret_list():
    assert orch._redact('nothing to hide') == 'nothing to hide'
    assert orch._redact('nothing to hide', []) == 'nothing to hide'


# ── Shell quoting ────────────────────────────────────────────────────────────

def test_values_are_quoted_against_the_shell():
    assert orch._shell_quote('a b; rm -rf /') == "'a b; rm -rf /'"
    # Raw strings on both sides. In a POSIX shell a single quote inside
    # single quotes is close, escaped-quote, reopen. Without r"" Python eats
    # the backslash and the expectation quietly becomes a different string —
    # which is how this test first failed against a correct function.
    assert orch._shell_quote("it's") == r"'it'\''s'"


# ── Writing .env cannot crash on the value it is given ───────────────────────

def test_env_write_treats_the_value_as_literal(tmp_path, monkeypatch):
    """
    `re.sub` interprets its replacement, so a value containing a backslash
    escape raised "invalid group reference" and took the call down.

    That call runs *after* an instance is rented and running, so the crash
    would strand a billing machine with its address never written to .env —
    which is the expensive way to find out.
    """
    env = tmp_path / '.env'
    env.write_text('PHANTOM_API_URL=ws://localhost:9000/ws\nOTHER=1\n')
    monkeypatch.setattr(orch, '_ENV_PATH', env)

    for value in (r'x\g<0>y\1z', 'wss://1.2.3.4:33526/ws', 'a1b2' * 16):
        orch._update_env_key('PHANTOM_API_URL', value)
        line = [ln for ln in env.read_text().splitlines()
                if ln.startswith('PHANTOM_API_URL=')]
        assert line == ['PHANTOM_API_URL=' + value]

    # The rest of the file is left alone, and a new key is appended.
    orch._update_env_key('BRAND_NEW', 'x')
    text = env.read_text()
    assert 'OTHER=1' in text and 'BRAND_NEW=x' in text


# ── The API contract, as documented ──────────────────────────────────────────

def test_a_200_carrying_success_false_is_a_failure(monkeypatch):
    """
    Vast answers state changes with a boolean envelope, so a refusal can arrive
    as HTTP 200 with {"success": false}. Checking only the status code read
    that as success — worst in `resume`, whose capacity fallback exists
    precisely to notice a refusal and would instead have booted an instance
    that never started.
    """
    class Resp:
        status_code = 200
        text = '{"success": false, "msg": "no gpus available"}'

        @staticmethod
        def json():
            return {'success': False, 'msg': 'no gpus available'}

    monkeypatch.setattr(orch.requests, 'request', lambda *a, **k: Resp())
    monkeypatch.setenv('VAST_API_KEY', 'k')

    with pytest.raises(orch.VastAPIError) as caught:
        orch._request('PUT', '/instances/1/', raise_on_error=True, json={'state': 'running'})
    assert orch._is_capacity_error(str(caught.value)), \
        'the message must still reach the capacity gate, or resume cannot fall back'


def test_a_200_carrying_success_true_is_returned(monkeypatch):
    class Resp:
        status_code = 200
        text = '{"success": true}'

        @staticmethod
        def json():
            return {'success': True, 'new_contract': 42}

    monkeypatch.setattr(orch.requests, 'request', lambda *a, **k: Resp())
    monkeypatch.setenv('VAST_API_KEY', 'k')
    assert orch._request('PUT', '/asks/1/', json={})['new_contract'] == 42


# ── The CPU, which decides the one stage a faster GPU does not ───────────────
# The compositor is OpenCV on the CPU: ~10.3ms of a ~29ms frame on a 4090 host
# and ~20ms on an L4 host with identical models. Clock across eligible western
# European offers ranged 3.5 to 6.0 GHz on 2026-09-04, so the host CPU swings
# roughly 4-5ms of frame time -- more than switching restoration models bought
# on detect, and free to choose.


def test_the_cpu_floors_reach_the_search(monkeypatch):
    captured = {}

    def fake_request(method, path, auth=True, raise_on_error=False, **kwargs):
        captured.update(kwargs.get('json') or {})
        return {'offers': []}

    monkeypatch.setattr(orch, '_request', fake_request)
    orch._search_offers(
        geolocations=GEO, min_dlperf=90.0, min_vram=16, max_price=1.0,
        min_reliability=0.98, min_inet_up=100.0, min_ports=32,
        min_cpu_ghz=4.5, min_cpu_cores=8.0,
        max_compute_cap=900, verified_only=True, disk=25)
    assert captured['cpu_ghz'] == {'gte': 4.5}
    assert captured['cpu_cores_effective'] == {'gte': 8.0}


def test_the_cpu_floor_is_loose_on_purpose():
    """
    A slow CPU still holds the 50ms deadline at 20fps, so clock is a preference
    rather than a requirement -- and a tight floor collapses availability. At
    4.5 GHz only two offers in all of western Europe survived. `_rank` is what
    selects on it; the floor only excludes the genuinely starved.
    """
    assert orch._DEFAULT_MIN_CPU_GHZ <= 4.0, \
        'a high floor trades availability for a preference'
    assert orch._DEFAULT_MIN_CPU_CORES >= 4


def test_a_faster_cpu_wins_inside_a_country():
    slow = offer('RTX 4090', 'GB', 0.29, cpu_ghz=3.5)
    fast = offer('RTX 4090', 'GB', 0.45, cpu_ghz=5.7)
    assert orch._rank([slow, fast], GEO)[0] is fast, \
        'the compositor is CPU-bound; clock is worth more than the price gap here'


def test_country_still_beats_the_cpu():
    """Distance is the term this deployment cannot buy back. Nothing outranks it."""
    near_slow = offer('RTX 4090', 'GB', 0.29, cpu_ghz=3.5)
    far_fast = offer('RTX 4090', 'DE', 0.29, cpu_ghz=6.0)
    assert orch._country(orch._rank([far_fast, near_slow], GEO)[0]) == 'GB'


def test_price_decides_inside_a_clock_band():
    """
    The reason clock is banded rather than sorted raw. On raw clock a 5.6 GHz
    box at $0.90 would beat a 5.7 GHz box at $0.29 -- a difference inside the
    noise of what this workload can feel, bought at 3x.
    """
    cheap = offer('RTX 4090', 'GB', 0.29, cpu_ghz=5.7)
    dear = offer('RTX 4090', 'GB', 0.90, cpu_ghz=5.6)
    assert orch._cpu_band(cheap) == orch._cpu_band(dear), 'test premise: same band'
    assert orch._rank([dear, cheap], GEO)[0] is cheap


def test_banding_still_has_boundaries_and_that_is_accepted():
    """
    Honest about the limit: 5.7 and 5.8 GHz straddle a band edge, so clock
    decides between them even though the gap is meaningless. Any bucketing has
    edges, and this one is accepted rather than fixed, because the differences
    that actually matter here are large -- 3.5 GHz against 5.7 is 1.6x, several
    bands wide, and no edge case hides it. A narrower band would move the
    problem, not remove it.
    """
    a = offer('RTX 4090', 'GB', 0.29, cpu_ghz=5.7)
    b = offer('RTX 4090', 'GB', 0.90, cpu_ghz=5.8)
    assert orch._cpu_band(a) != orch._cpu_band(b)
    assert orch._rank([a, b], GEO)[0] is b  # the edge case, documented not fixed


def test_a_large_clock_difference_does_override_price():
    cheap = offer('RTX 4090', 'GB', 0.29, cpu_ghz=3.5)
    dear = offer('RTX 4090', 'GB', 0.45, cpu_ghz=5.5)
    assert orch._rank([dear, cheap], GEO)[0] is dear


def test_the_band_rounds_to_half_a_gigahertz():
    assert orch._cpu_band({'cpu_ghz': 5.7}) == 5.5
    assert orch._cpu_band({'cpu_ghz': 5.8}) == 6.0
    assert orch._cpu_band({'cpu_ghz': 3.5}) == 3.5
    assert orch._cpu_band({}) == 0.0, 'a missing clock must not crash the sort'


def test_gpu_speed_is_deliberately_not_in_the_ranking():
    """
    VAST_MIN_DLPERF has already removed everything below a 4090, so every
    surviving offer is fast enough on the GPU. Ranking on it again would buy
    headroom that goes unused at 20fps, at the cost of the CPU term that does
    not go unused.
    """
    weaker_gpu_faster_cpu = offer('RTX 4090', 'GB', 0.29, dlperf=97, cpu_ghz=5.7)
    stronger_gpu_slower_cpu = offer('RTX 6000Ada', 'GB', 0.29, dlperf=113, cpu_ghz=3.5)
    ranked = orch._rank([stronger_gpu_slower_cpu, weaker_gpu_faster_cpu], GEO)
    assert ranked[0] is weaker_gpu_faster_cpu


def test_a_missing_cpu_field_sorts_last_rather_than_crashing():
    known = offer('RTX 4090', 'GB', 0.50, cpu_ghz=5.0)
    unknown = offer('RTX 4090', 'GB', 0.10)
    unknown['cpu_ghz'] = None
    assert orch._rank([unknown, known], GEO)[0] is known


# ── The GPU is gated AND ranked, not just gated ──────────────────────────────
# Four GPU filters reach the search: dlperf, gpu_ram, compute_cap and gpu_arch,
# plus the price ceiling. What was missing was the GPU in the *ordering* — at
# equal country and CPU, a few cents decided between cards, which is the wrong
# tiebreak when the frame is roughly 19ms of GPU work against 10.3ms of CPU.


def test_every_gpu_gate_reaches_the_search(monkeypatch):
    captured = {}

    def fake_request(method, path, auth=True, raise_on_error=False, **kwargs):
        captured.update(kwargs.get('json') or {})
        return {'offers': []}

    monkeypatch.setattr(orch, '_request', fake_request)
    orch._search_offers(
        geolocations=GEO, min_dlperf=90.0, min_vram=16, max_price=1.0,
        min_reliability=0.98, min_inet_up=100.0, min_ports=32,
        min_cpu_ghz=3.5, min_cpu_cores=8.0,
        max_compute_cap=900, verified_only=False, disk=25)

    assert captured['dlperf'] == {'gte': 90.0}, 'speed'
    assert captured['gpu_ram'] == {'gte': 16 * 1024}, 'VRAM'
    assert captured['compute_cap'] == {'lte': 900}, 'architecture the image supports'
    assert captured['gpu_arch'] == {'eq': 'nvidia'}, 'CUDAExecutionProvider exists'
    assert captured['dph_total'] == {'lte': 1.0}, 'budget'


def test_the_better_gpu_wins_at_equal_country_and_cpu():
    slow_gpu = offer('RTX 4090', 'GB', 0.29, dlperf=91, cpu_ghz=5.7)
    fast_gpu = offer('RTX 6000Ada', 'GB', 0.45, dlperf=113, cpu_ghz=5.7)
    assert orch._rank([slow_gpu, fast_gpu], GEO)[0] is fast_gpu


def test_the_cpu_outranks_the_gpu():
    """
    Not because the CPU is the larger term — it is not, the frame is roughly
    19ms GPU against 10.3ms CPU. Because VAST_MIN_DLPERF has already guaranteed
    the GPU is adequate, while the CPU floor is deliberately loose and
    guarantees nothing. CPU ordering does real work; GPU ordering refines
    between cards that all already hold the deadline.
    """
    fast_cpu_slow_gpu = offer('RTX 4090', 'GB', 0.29, dlperf=91, cpu_ghz=5.7)
    slow_cpu_fast_gpu = offer('RTX 6000Ada', 'GB', 0.29, dlperf=113, cpu_ghz=3.5)
    assert orch._rank([slow_cpu_fast_gpu, fast_cpu_slow_gpu], GEO)[0] \
        is fast_cpu_slow_gpu


def test_price_still_decides_when_both_bands_tie():
    cheap = offer('RTX 4090', 'GB', 0.29, dlperf=97, cpu_ghz=5.7)
    dear = offer('RTX 4090', 'GB', 0.80, dlperf=97, cpu_ghz=5.7)
    assert orch._rank([dear, cheap], GEO)[0] is cheap


def test_the_gpu_band_is_coarse_enough_to_ignore_noise():
    """A few dlperf points must not override a price gap, same as clock."""
    assert orch._dlperf_band({'dlperf': 91}) == orch._dlperf_band({'dlperf': 97})
    assert orch._dlperf_band({'dlperf': 97}) != orch._dlperf_band({'dlperf': 113})
    assert orch._dlperf_band({}) == 0.0, 'a missing score must not crash the sort'


# ── Verification is a badge, reliability is a measurement ────────────────────

def test_verification_is_not_filtered_on_by_default(monkeypatch):
    """
    Vast's "verified" is a datacenter badge. Left on it excluded the best UK
    box on every axis that matters — $0.294 against $0.737, a 5.7GHz Ryzen
    against a 3.5GHz EPYC, $1.70/month storage against $8.30 — for a measured
    reliability difference of 0.993 against 0.997.
    """
    for name in ('VAST_VERIFIED_ONLY', 'VAST_GEOLOCATIONS'):
        monkeypatch.delenv(name, raising=False)
    assert orch._selection_settings()['verified_only'] is False


def test_verification_can_still_be_demanded(monkeypatch):
    monkeypatch.setenv('VAST_VERIFIED_ONLY', 'true')
    assert orch._selection_settings()['verified_only'] is True


def test_reliability_remains_a_hard_floor(monkeypatch):
    """Dropping the badge must not drop the measurement it was standing in for."""
    monkeypatch.delenv('VAST_MIN_RELIABILITY', raising=False)
    assert orch._selection_settings()['min_reliability'] >= 0.95


# ── Relaxation, bounded by the one number that is not negotiable ─────────────
# 20fps is a hard target, so every rung has to fit a frame into 50ms. GPU work
# scales about inversely with dlperf from ~19ms at 97, so dlperf 45 (an RTX
# 3090) lands at ~51ms and misses. The ladder stops above that rather than
# continuing.

def test_the_ladder_goes_from_strict_to_loose():
    names = [name for name, _, _ in orch._RELAXATION_LADDER]
    assert names[0] == 'preferred'
    dlperfs = [d for _, d, _ in orch._RELAXATION_LADDER]
    ghz = [g for _, _, g in orch._RELAXATION_LADDER]
    assert min(dlperfs) < max(dlperfs), 'the GPU floor must actually relax'
    assert min(ghz) < max(ghz), 'the CPU floor must actually relax'


def test_the_cpu_relaxes_before_the_gpu():
    """It costs less: a CPU rung is worth a few ms, a GPU rung about seven."""
    names = [name for name, _, _ in orch._RELAXATION_LADDER]
    assert names.index('relaxed CPU') < names.index('relaxed GPU')


def test_no_rung_would_miss_twenty_fps():
    """
    The invariant the ladder exists to protect. ~19ms of GPU work at dlperf 97,
    scaling inversely, plus ~10.3ms of CPU work, must stay inside 50ms.
    """
    for name, min_dlperf, _ghz in orch._RELAXATION_LADDER:
        gpu_ms = 19.0 * 97.0 / min_dlperf
        frame_ms = gpu_ms + 10.3
        assert frame_ms < 50.0, \
            '{} allows dlperf {:.0f} -> ~{:.0f}ms frame, misses 20fps'.format(
                name, min_dlperf, frame_ms)


def test_the_bottom_rung_still_refuses_a_3090():
    """
    dlperf 44.5 lands at ~51ms — over the 50ms budget. An unbounded fallback
    would have taken it, and worse: the old one dropped the floor to zero,
    which admits a GTX 1080 at dlperf 3.4.
    """
    floor = min(d for _, d, _ in orch._RELAXATION_LADDER)
    assert floor > 44.5, 'an RTX 3090 must not be reachable'
    assert floor > 3.4, 'a GTX 1080 must certainly not be reachable'


def test_relaxation_can_be_pinned_off_for_a_measurement(monkeypatch):
    """Numbers from two architectures are not a comparison."""
    monkeypatch.setenv('VAST_RELAX', 'false')
    relax = (os.environ.get('VAST_RELAX') or 'true').strip().lower() \
        not in ('0', 'false', 'no', 'off')
    assert relax is False
