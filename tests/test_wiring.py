"""
Cross-module wiring contracts.

Unit tests cover behaviour inside a module. These cover the seams *between*
modules and files, which is where this project has actually broken: a dead
image tag in four files, a `requirements-ci.txt` that never existed, an
`Enhancer()` call with the wrong arity that a bare `except` swallowed, and
config settings the orchestrator never forwarded to the pod.

None of those were caught by tests, because each lived in the gap between two
things that were individually fine. Everything here asserts that two files
still agree.

Deliberately reads source text in places rather than importing. `runpod/`
imports the RunPod SDK and paramiko, neither of which belongs in a test
environment, and the point is the literal contents anyway.
"""

import sys

# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ast
import io
import re
from unittest.mock import MagicMock


class StubModule(MagicMock):
    __path__: list = []


for _name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'onnxruntime', 'torch', 'torchvision', 'psutil',
    'tensorflow', 'opennsfw2', 'gfpgan', 'onnx',
):
    sys.modules.setdefault(_name, StubModule())

import logging

from pipeline.config import FaceSwapConfig
from pipeline.api.schema import PRESETS
from pipeline.services import guards, swapper_models

logging.disable(logging.ERROR)

PASS, FAIL = [], []


def check(label, condition, detail=''):
    (PASS if condition else FAIL).append(label)
    print('  [{}] {}'.format('PASS' if condition else 'FAIL', label)
          + (' - {}'.format(detail) if detail else ''))


def read(*parts):
    return io.open(_os.path.join(_REPO_ROOT, *parts), encoding='utf-8').read()


CONFIG_FIELDS = set(FaceSwapConfig().__dict__)

print('=' * 70)
print('Cross-module wiring')
print('=' * 70)

# ── Config fields referenced from elsewhere must exist ─────────────────
print('\nConfig field references')

preset_fields = {k for preset in PRESETS.values() for k in preset}
missing = sorted(preset_fields - CONFIG_FIELDS)
check('every preset key is a real config field', not missing, str(missing))

profile_fields = set()
for model in swapper_models.SWAPPER_MODELS.values():
    profile_fields |= set(model.look())
missing = sorted(profile_fields - CONFIG_FIELDS)
check('every model profile key is a real config field', not missing, str(missing))

missing = sorted(set(guards.GUARD_FIELDS) - CONFIG_FIELDS)
check('every guard threshold is a real config field', not missing, str(missing))

handlers_src = read('pipeline', 'api', 'handlers.py')
realism = re.search(r'_REALISM_FIELDS[^{]*\{(.*?)\n\}', handlers_src, re.S)
realism_keys = set(re.findall(r"^\s*'([a-z_]+)':", realism.group(1), re.M))
missing = sorted(realism_keys - CONFIG_FIELDS)
check('every set_realism field is a real config field', not missing, str(missing))

check('preset and model profile do not both own a knob',
      not (preset_fields & profile_fields),
      'overlap: {}'.format(sorted(preset_fields & profile_fields)))

# ── Orchestrator forwards only names the pipeline reads ────────────────
print('\nOrchestrator env forwarding')

orch_src = read('runpod', 'orchestrator.py')
forwarded = re.search(r'_FORWARDED_ENV = \((.*?)\n\)', orch_src, re.S)
forwarded_names = set(re.findall(r'"([A-Z_]+)"', forwarded.group(1)))
check('the forwarded list is non-empty', bool(forwarded_names),
      '{} names'.format(len(forwarded_names)))

# Any module in the package may read an env var, not just core.py.
package_src = ''
for root, _dirs, files in _os.walk(_os.path.join(_REPO_ROOT, 'pipeline')):
    if '__pycache__' in root:
        continue
    for name in files:
        if name.endswith('.py'):
            package_src += io.open(_os.path.join(root, name), encoding='utf-8').read()

unread = sorted(n for n in forwarded_names if n not in package_src)
check('every forwarded var is actually read by the pipeline', not unread,
      'forwarded but never read: {}'.format(unread))

# The reverse: settings that exist but never reach the pod are the failure
# that made a measurement session manual.
declared = set(re.findall(r"os\.environ\.get\('([A-Z_]+)'", package_src))
declared |= set(re.findall(r"_env_(?:float|int|bool)\('([A-Z_]+)'\)", package_src))
# Names the pod must not or need not receive.
exempt = {
    'RUNPOD_API_KEY', 'RUNPOD_MAX_UPTIME', 'RUNPOD_STOP_WARNING',
    'API_PORT', 'PHANTOM_API_URL', 'EXECUTION_PROVIDER',
    'PHANTOM_SESSION_ID', 'TF_CPP_MIN_LOG_LEVEL', 'OMP_NUM_THREADS',
    'INSIGHTFACE_HOME',
}
stranded = sorted(declared - forwarded_names - exempt)
check('no pipeline setting is stranded on the local machine', not stranded,
      'read by the pipeline but never forwarded: {}'.format(stranded))

# ── The orchestrator's own settings are documented ─────────────────────
# The forwarding checks above cover settings that travel *to* the pod. These
# cover the ones that configure the orchestrator itself, which drifted the other
# way: RUNPOD_DATACENTERS, the auto-discovery bounds and the auto-stop timers
# were all read by the code and named in neither .env.example nor the guide, so
# the documented setup silently pinned GPUs and had no billing cap.
print('\nOrchestrator settings are documented')

env_example_src = read('.env.example')
deploy_doc = read('RUNPOD_DEPLOYMENT.md')

orch_settings = set(re.findall(r'os\.getenv\("(RUNPOD_[A-Z_]+)"', orch_src))
check('the orchestrator reads a plausible number of settings',
      len(orch_settings) >= 10, '{} found'.format(len(orch_settings)))

undocumented = sorted(n for n in orch_settings if n not in env_example_src)
check('.env.example documents every setting the orchestrator reads',
      not undocumented, str(undocumented))

unexplained = sorted(n for n in orch_settings if n not in deploy_doc)
check('RUNPOD_DEPLOYMENT.md documents every setting the orchestrator reads',
      not unexplained, str(unexplained))

# Auto-discovery is the default only while nothing pins it. A shipped value
# turns off the VRAM, price and compute-capability filtering in one go.
gpu_pin = re.search(r'^RUNPOD_GPU_TYPES=(.*)$', env_example_src, re.M)
check('.env.example leaves GPU auto-discovery on',
      gpu_pin is not None and not gpu_pin.group(1).strip(),
      'a shipped RUNPOD_GPU_TYPES disables VRAM, price and architecture filtering')

# ── The deploy guide describes the code that exists ────────────────────
print('\nDeploy guide matches the code')

# The word itself is allowed — both files say "not tmux" to correct the old
# instructions. What must not survive is anything that treats it as live: a
# session to attach to, or a claim that the pipeline is started inside one.
TMUX_CLAIMS = ('tmux attach', 'tmux new', 'tmux kill', 'in tmux', 'tmux session')
for label, src in (('the guide', deploy_doc),
                   ('the orchestrator', orch_src),
                   ('.env.example', env_example_src),
                   ('startup.sh', read('runpod', 'startup.sh'))):
    stale = [c for c in TMUX_CLAIMS if c in src]
    check('{} does not present tmux as live'.format(label), not stale,
          'the pipeline has run under nohup since the tmux dependency was '
          'dropped; found {}'.format(stale))
pipeline_log = re.search(r'^_PIPELINE_LOG = "([^"]+)"', orch_src, re.M)
check('the guide names the log the pipeline actually writes',
      pipeline_log is not None and pipeline_log.group(1) in deploy_doc,
      'the log is the only view of a nohup pipeline')
check('the guide gives the proxy WebSocket address, not a raw IP',
      'proxy.runpod.net' in deploy_doc)
check('the guide gives the proxy SSH address, not root@ip',
      'ssh.runpod.io' in deploy_doc and 'root@' not in deploy_doc)
check('the guide covers forwarding pipeline settings into the pod',
      '_FORWARDED_ENV' in deploy_doc,
      'without it, configuring a run means SSHing in by hand')
check('the retired DEPLOYMENT.md is gone rather than left to rot',
      not _os.path.isfile(_os.path.join(_REPO_ROOT, 'DEPLOYMENT.md')),
      'it predated the orchestrator and pinned a dead image tag')

# ── CLI exposes every registered model ─────────────────────────────────
print('\nModel registry reaches the interfaces')

core_src = read('pipeline', 'core.py')
check('the CLI offers the registry rather than a hardcoded list',
      'choices=list(swapper_models.names())' in core_src)
check('set_realism validates against the registry',
      'swapper_models.names()' in handlers_src)
check('the CLI reads SWAPPER_MODEL from the environment',
      "os.environ.get('SWAPPER_MODEL')" in core_src)
check('.env.example documents every registered model',
      all(name in read('.env.example') for name in swapper_models.names()),
      str([n for n in swapper_models.names() if n not in read('.env.example')]))

# ── Order of application ───────────────────────────────────────────────
print('\nPreset then profile, in that order')

preset_at = core_src.find('CONFIG.apply_preset(')
profile_at = core_src.find('CONFIG.apply_model_profile(')
check('both are applied at startup', preset_at > 0 and profile_at > 0)
check('the model profile is applied after the preset',
      preset_at < profile_at,
      'the preset owns compute; the model owns appearance')

overrides = core_src.find('for field, value in (')
check('explicit flags are applied after both',
      overrides > profile_at,
      'an operator override must win over either')

# ── Image tag agreement ────────────────────────────────────────────────
print('\nDeploy image agreement')

dockerfile = read('Dockerfile')
env_example = read('.env.example')
docker_tag = re.search(r'^FROM (\S+)', dockerfile, re.M).group(1)
env_tag = re.search(r'^RUNPOD_IMAGE=(\S+)', env_example, re.M).group(1)
check('Dockerfile and .env.example pin the same image',
      docker_tag == env_tag,
      '{} vs {}'.format(docker_tag, env_tag))
check('the image is a devel tag',
      'devel' in docker_tag,
      'runtime is not published for runpod/pytorch - TROUBLESHOOTING section 5')
check('no file still references the dead runtime tag',
      'cuda12.4.1-runtime' not in dockerfile + env_example)

# ── Both deploy paths cover the same setup ─────────────────────────────
print('\nSSH and Docker parity')

startup = read('runpod', 'startup.sh')
check('startup.sh runs the pre-warm script',
      'prewarm.py' in startup)
check('the pre-warm script exists',
      _os.path.isfile(_os.path.join(_REPO_ROOT, 'runpod', 'prewarm.py')))
check('startup.sh fails hard when cuDNN cannot load',
      'libcudnn.so.9' in startup and 'exit 1' in startup,
      'a CPU fallback wastes a paid GPU hour')
check('the Docker build fails hard when cuDNN cannot load',
      'libcudnn.so.9' in dockerfile,
      'the SSH path checks at setup; the image must check at build')
check('both paths resolve the cuDNN directory with the same helper',
      'cudnn_path.py' in dockerfile and 'cudnn_path.py' in startup,
      'they previously had the same bug and fixed it in neither')
check('neither path still uses nvidia.cudnn.__file__',
      '__file__' not in dockerfile
      and 'nvidia.cudnn.__file__' not in startup,
      'it is None for a namespace package, which broke the Docker build')
check('startup.sh no longer swallows the cuDNN resolution error',
      'cudnn_path.py" || echo' in startup,
      'the old 2>/dev/null turned a TypeError into a misleading warning')
# Follow the chain rather than grepping one file: Dockerfile -> entrypoint ->
# prewarm. Asserting only on the Dockerfile would pass for the wrong reasons.
entrypoint_path = _os.path.join(_REPO_ROOT, 'runpod', 'entrypoint.sh')
check('the Docker entrypoint exists', _os.path.isfile(entrypoint_path))
entrypoint = read('runpod', 'entrypoint.sh') if _os.path.isfile(entrypoint_path) else ''
check('the Dockerfile runs that entrypoint',
      'entrypoint.sh' in dockerfile)
check('Docker pre-warms too, so first-frame cost matches SSH',
      'prewarm.py' in entrypoint,
      'otherwise a 384 MB model downloads on a customer first frame')
check('the entrypoint execs the pipeline as PID 1',
      'exec python' in entrypoint,
      'otherwise SIGTERM never reaches Python and a clean stop loses the reports')
check('the entrypoint does not abort on a pre-warm failure',
      '&& exec' not in entrypoint,
      'a slow first frame is not a broken pod')

prewarm = read('runpod', 'prewarm.py')
for label in ('detection', 'swap', 'restoration', 'occluder'):
    check('pre-warm covers {}'.format(label), "'{}'".format(label) in prewarm)
check('pre-warm constructs Enhancer with a config',
      'Enhancer(CONFIG)' in prewarm,
      'Enhancer() with no args raised TypeError and was silently swallowed')

# ── Pod environment actually reaches the processes ─────────────────────
print('\nPod environment delivery')

check('startup.sh sources the RunPod env file',
      '/etc/rp_environment' in startup,
      'inheritance via .bashrc is a convention, not a guarantee')
check('the Docker entrypoint sources it too',
      '/etc/rp_environment' in entrypoint)
check('the SSH pipeline launch sources it',
      '/etc/rp_environment' in orch_src,
      '_shell_run uses a subshell, so it must be part of the launch command')

# ── Phase timing agreement ─────────────────────────────────────────────
print('\nCold-start measurement')

check('startup.sh emits parseable phase lines',
      'PHASE ' in startup and 'VOLUME ' in startup)
check('the orchestrator parses that exact prefix',
      'PHASE ' in orch_src and 'VOLUME ' in orch_src)
check('startup.sh reports whether the volume was warm or empty',
      'VOLUME_STATE' in startup)
check('the orchestrator absorbs inner phases into remote-setup',
      "'remote-setup'" in orch_src or '"remote-setup"' in orch_src)

# ── Scratch space ──────────────────────────────────────────────────────
print('\nBatch scratch')

ffmpeg_src = read('pipeline', 'io', 'ffmpeg.py')
check('scratch prefers the network volume on a pod',
      "'/workspace'" in ffmpeg_src and 'PHANTOM_TEMP_DIR' in ffmpeg_src,
      'the container disk is 20 GB and shared with the OS')
check('PHANTOM_TEMP_DIR is forwarded to the pod',
      'PHANTOM_TEMP_DIR' in forwarded_names)

# ── Access codes ───────────────────────────────────────────────────────
# The gate spans four files and a Cloud Function, and the two rules that make
# it correct are both invisible from any one of them.
print('\nAccess codes')

from desktop import codes as access_codes  # noqa: E402

bridge_src = read('desktop', 'bridge.py')
auth_src = read('desktop', 'auth.py')
qml_src = read('desktop', 'main.qml')
functions_src = read('firebase', 'functions', 'main.py')

check('the checksum is not degenerate against a base-32 payload',
      access_codes._CHECK_MOD > 32 and access_codes._CHECK_MOD % 2 == 1,
      'a modulus <= 32 makes the check character a copy of the last payload one')

minted = [access_codes.generate() for _ in range(200)]
check('every minted code passes its own checksum',
      all(access_codes.is_valid(c) for c in minted))
check('a minted code survives display formatting',
      all(access_codes.is_valid(access_codes.format_code(c)) for c in minted))
check('codes stay alphanumeric so they can be typed and read aloud',
      all(c.isalnum() for c in minted))

_typos = [(c[:i] + ch + c[i + 1:])
          for c in minted[:20] for i in range(len(c))
          for ch in access_codes._ALPHABET if ch != c[i]]
check('every single-character typo is caught locally',
      not any(access_codes.is_valid(t) for t in _typos),
      '{} variants tested'.format(len(_typos)))

check('the alphabet excludes the characters that are misheard',
      not any(ch in access_codes._ALPHABET for ch in 'ILOU'),
      'these get read aloud')

# The rule the whole design rests on: typing a code must not spend it.
check('the code is spent from the pipeline-running transition',
      '_commit_pending_code' in bridge_src
      and bridge_src.find('def _set_pipeline_running') < bridge_src.find('_commit_pending_code'),
      'a pod that fails to come up must cost the customer nothing')
check('redemption is two steps, not one',
      'def begin(' in auth_src and 'def commit(' in auth_src)
check('a failed commit leaves the code unspent',
      '_committed = True' in auth_src and auth_src.count('_committed = True') == 1,
      'commit must only mark spent on a confirmed success')

# The other rule: the hour belongs to the machine, not to a launch of the app.
check('the desktop asks the server on every launch',
      'self.checkAuth()' in bridge_src)
check('the session is not cached in a local file',
      'sessions' in functions_src and 'expires_at' in functions_src)
check('the gate is skipped entirely when unconfigured',
      'def is_enabled(' in auth_src and 'auth.is_enabled()' in bridge_src,
      'no PHANTOM_AUTH_URL must mean no gate, for local development')
check('.env.example documents the auth URL',
      'PHANTOM_AUTH_URL' in read('.env.example'))

# Mock mode is a development aid. The failure that would matter is it becoming
# a *fallback* — an unreachable real server quietly letting everyone in.
check('mock mode triggers on an exact literal, not on failure',
      '_MOCK_URLS' in auth_src and '_auth_url().lower() in _MOCK_URLS' in auth_src,
      'a down server must report unreachable, never open the gate')
check('the mock is never consulted without an explicit URL',
      auth_src.find('if not base:') < auth_src.find('if _is_mock():'),
      'an unset URL means no gate at all, which is a different thing')
check('mock mode is documented where it would be looked for',
      'mock' in read('.env.example') and 'Mock mode' in read('docs', 'ACCESS_CODES.md'))

check('the redeem endpoint is idempotent for the same machine',
      'replayed' in functions_src,
      'a dropped reply must not cost a second hour')
check('a machine with time left cannot spend another code',
      'session_active' in functions_src)
check('the burn is transactional',
      'firestore.transactional' in functions_src,
      'two redemptions a second apart would otherwise both succeed')
check('Firestore denies every client',
      'allow read, write: if false' in read('firebase', 'firestore.rules'))

for member in ('authRequired', 'authChecking', 'authMinutes', 'authError',
               'checkAuth', 'submitCode'):
    check('QML and bridge agree on {}'.format(member),
          member in qml_src and member in bridge_src)

# ── The virtual camera ─────────────────────────────────────────────────
# The device is the one place the operator's real face could reach a call.
# Everything here exists to keep that impossible.
print('\nVirtual camera')

controller_src = read('desktop', 'controller.py')

# Only the pipeline's swapped stream may reach the device. The local webcam
# preview is decoded in the same function and must never take that turning.
# _push_to_vcam delegates to _push_frame_to_vcam, so that one call is plumbing
# rather than a second source. Count callers outside the two helpers.
_helpers = bridge_src[bridge_src.find('def _push_to_vcam'):]
_outside = bridge_src.replace(_helpers, '')
_vcam_writers = re.findall(r'^\s*self\._push(?:_frame)?_to_vcam\(', _outside, re.M)
check('the virtual camera has exactly two feed calls',
      len(_vcam_writers) == 2, '{} found'.format(len(_vcam_writers)))

_poll = bridge_src[bridge_src.find('def _poll_frames'):]
_poll = _poll[:_poll.find('\n    def ', 1)]
check('both feed calls live in the live-frame path',
      _poll.count('_to_vcam(') == 2,
      'a writer outside _poll_frames is a second source for the device')
check('the device is fed only from the jitter buffer',
      '_jitter_buffer.pop_eligible()' in _poll,
      'that buffer carries pipeline output; the raw webcam is a different buffer')
_before_pop = _poll[:_poll.find('pop_eligible')]
check('the raw webcam preview never reaches the device',
      'webcam_buffer' in _before_pop and '_to_vcam' not in _before_pop,
      'the operator is on the call precisely so their own face is not sent')

# Releasing the device mid-call is what makes a conferencing app go looking
# for another camera. Only closing the app may do it.
_releases = re.findall(r'^\s*self\._stop_vcam\(\)', bridge_src, re.M)
check('the device is released from at most three places',
      len(_releases) <= 3, '{} found'.format(len(_releases)))
for caller, allowed in (('def cleanup', True), ('def _start_vcam', True),
                        ('def toggleVirtualCam', True)):
    body = bridge_src[bridge_src.find(caller):]
    body = body[:body.find('\n    def ', 1)]
    check('{} may release the device'.format(caller.split()[1]),
          '_stop_vcam()' in body or not allowed)

for forbidden in ('def stopPipeline', 'def _end_session', 'def _on_ws_connected'):
    body = bridge_src[bridge_src.find(forbidden):]
    body = body[:body.find('\n    def ', 1)]
    check('{} does not release the device'.format(forbidden.split()[1]),
          '_stop_vcam()' not in body,
          'stopping, expiring or disconnecting must not drop the camera')

check('the camera is opened for the life of the app',
      '_ensure_vcam()' in bridge_src
      and bridge_src.find('self._ensure_vcam()') < bridge_src.find('self._start_webcam(0)'),
      'opened at startup, not on a mode or a session')
check('VCAM is no longer a clickable control',
      'toggleVirtualCam' not in qml_src,
      'its only distinct effect is the one failure that must never happen')

# ── Session shutdown ───────────────────────────────────────────────────
print('\nSession shutdown')

check('the desktop handles the auto_stop event',
      "'auto_stop'" in bridge_src,
      'it was broadcast and ignored, so expiry looked like a network fault')
check('an expected disconnect stops the reconnect loop',
      'expect_disconnect' in controller_src and 'expect_disconnect' in bridge_src)
check('the reconnect loop checks it before backing off',
      '_expected_disconnect.is_set()' in controller_src)
check('the reconnect docstring no longer claims a retry limit',
      'max 3 retries' not in controller_src,
      'the cap is on the delay, not the number of attempts')
check('expiry raises a card rather than the full gate',
      'sessionExpired' in bridge_src and 'sessionExpired' in qml_src)
check('the card offers a way back',
      'enterNewCode' in bridge_src and 'enterNewCode' in qml_src)

# ── Everything parses ──────────────────────────────────────────────────
print('\nSyntax')
for path in (('runpod', 'orchestrator.py'), ('runpod', 'prewarm.py'),
             ('firebase', 'functions', 'main.py'), ('desktop', 'auth.py'),
             ('desktop', 'codes.py'), ('tools', 'mint_codes.py')):
    name = '/'.join(path)
    try:
        ast.parse(read(*path))
        ok, detail = True, ''
    except SyntaxError as exc:
        ok, detail = False, str(exc)
    check('{} parses'.format(name), ok, detail)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
