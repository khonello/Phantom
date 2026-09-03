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

Deliberately reads source text in places rather than importing. `vast/`
imports paramiko, which does not belong in a test
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

orch_src = read('vast', 'orchestrator.py')
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
    # Set explicitly on the launch command rather than forwarded in bulk:
    # they configure the instance's relationship to Vast and to the desktop,
    # not the swap.
    'VAST_API_KEY', 'VAST_INSTANCE_ID', 'VAST_MAX_UPTIME', 'VAST_STOP_WARNING',
    'PHANTOM_TLS_CERT', 'PHANTOM_TLS_KEY', 'PHANTOM_API_TOKEN',
    'API_PORT', 'PHANTOM_API_URL', 'EXECUTION_PROVIDER',
    'PHANTOM_SESSION_ID', 'TF_CPP_MIN_LOG_LEVEL', 'OMP_NUM_THREADS',
    'INSIGHTFACE_HOME', 'PHANTOM_SESSION_GRACE',
}
stranded = sorted(declared - forwarded_names - exempt)
check('no pipeline setting is stranded on the local machine', not stranded,
      'read by the pipeline but never forwarded: {}'.format(stranded))

# ── The desktop's dependencies are declared ────────────────────────────
# The desktop had no requirements file at all: its packages lived only in one
# developer's venv, so "set it up on another machine" meant reading imports and
# guessing. A third-party import missing from the file is a machine that
# installs cleanly and then fails at startup.
print('\nDesktop dependencies are declared')


def _declared(filename):
    """Distribution names a requirements file actually declares.

    Parsed rather than substring-matched: these files explain themselves, and
    this one names onnxruntime and torch in a comment saying they do *not*
    belong. A check reading the prose would agree with the opposite of what the
    file says.
    """
    names = set()
    for line in read(filename).splitlines():
        line = line.split('#', 1)[0].strip()
        if not line or line.startswith('-'):
            continue
        names.add(re.split(r'[<>=!\[;]', line)[0].strip().lower())
    return names


_DESKTOP_REQS = _declared('requirements-desktop.txt')

# Import name -> distribution name, where they differ.
_DIST = {
    'cv2': 'opencv-python',
    'dotenv': 'python-dotenv',
    'parselmouth': 'praat-parselmouth',
}
# Standard library and first-party; never in a requirements file.
_NOT_A_DEPENDENCY = {
    'ssl', 'hashlib', 'hmac',
    'argparse', 'base64', 'collections', 'dataclasses', 'desktop', 'gc',
    'hashlib', 'json', 'math', 'os', 'pathlib', 'pipeline', 'platform',
    'queue', 'secrets', 'struct', 'subprocess', 'sys', 'threading', 'time',
    'two', 'typing', 'urllib', 'uuid', 'winreg',
}

_desktop_src = ''
for _name in _os.listdir(_os.path.join(_REPO_ROOT, 'desktop')):
    if _name.endswith('.py'):
        _desktop_src += read('desktop', _name)
_desktop_src += read('desktop.py')

_imported = set(re.findall(r'^\s*(?:import|from) ([a-zA-Z_][a-zA-Z0-9_]*)',
                           _desktop_src, re.M))
_undeclared = [m for m in sorted(_imported - _NOT_A_DEPENDENCY)
               if _DIST.get(m, m).lower() not in _DESKTOP_REQS]
check('requirements-desktop.txt covers every third-party import',
      not _undeclared, 'missing: {}'.format(_undeclared))

# The desktop must never pull in the face models. It sends frames and displays
# what comes back; the moment it imports onnxruntime the two halves have been
# confused and a laptop starts trying to be the GPU.
check('the desktop declares no ML runtime',
      not (_DESKTOP_REQS & {'onnxruntime', 'onnxruntime-gpu', 'insightface',
                            'torch', 'tensorflow'}),
      'the desktop displays frames; the pipeline swaps faces')

# Every tool that opens a socket to a running pipeline lives in the
# orchestrator environment, so that file has to carry websockets. It did not,
# which meant a fresh machine could manage pods and then fail on the first
# command sent to one.
check('requirements-orchestrator.txt carries websockets',
      'websockets' in _declared('requirements-orchestrator.txt'))


# ── The orchestrator's own settings are documented ─────────────────────
# The forwarding checks above cover settings that travel *to* the pod. These
# cover the ones that configure the orchestrator itself, which drifted the other
# way: the geography, the auto-discovery bounds and the auto-stop timers
# were all read by the code and named in neither .env.example nor the guide, so
# the documented setup silently pinned GPUs and had no billing cap.
print('\nOrchestrator settings are documented')

env_example_src = read('.env.example')
deploy_doc = read('VAST_DEPLOYMENT.md')

orch_settings = set(re.findall(r'os\.getenv\("(VAST_[A-Z_]+)"', orch_src))
orch_settings |= set(re.findall(r'_env_(?:float|int|flag)\("(VAST_[A-Z_]+)"', orch_src))
check('the orchestrator reads a plausible number of settings',
      len(orch_settings) >= 10, '{} found'.format(len(orch_settings)))

undocumented = sorted(n for n in orch_settings if n not in env_example_src)
check('.env.example documents every setting the orchestrator reads',
      not undocumented, str(undocumented))

unexplained = sorted(n for n in orch_settings if n not in deploy_doc)
check('VAST_DEPLOYMENT.md documents every setting the orchestrator reads',
      not unexplained, str(unexplained))

# Geography is the setting the migration exists for, so a shipped value that
# quietly widened it would undo the whole point.
geo = re.search(r'^VAST_GEOLOCATIONS=(.*)$', env_example_src, re.M)
check('.env.example ships a geography, and it starts in the UK',
      geo is not None and geo.group(1).strip().startswith('GB'),
      'RunPod had no UK datacenter; being able to ask for one is the reason to be here')
host_pin = re.search(r'^VAST_PREFERRED_HOST=(.*)$', env_example_src, re.M)
check('.env.example ships no pinned host',
      host_pin is not None and not host_pin.group(1).strip(),
      'a host id is account-specific and would send everyone to one machine')

# A stopped pod keeps its host, so resting can end with that host's GPUs taken.
# The fallback to a new pod is the only thing that recovers it, and it has to
# stay gated: falling back on *any* resume failure would create a billing pod in
# response to a typo'd ID or a bad key.
resume_body = orch_src.split('def cmd_resume')[1].split(chr(10) + 'def ')[0]
check('resume falls back to start when the host is full',
      'cmd_start()' in resume_body)
check('that fallback is gated, not unconditional',
      '_is_capacity_error' in resume_body)
check('the capacity markers are substrings, not exact messages',
      "_CAPACITY_MARKERS" in orch_src and 'no gpu' in orch_src.lower(),
      'the wording carries the machine own numbers, so it cannot be matched whole')

# insightface depends on the CPU `onnxruntime`, and both wheels write the same
# `onnxruntime/` directory, so the CPU one lands last and shadows the GPU build.
# The only symptom is an argparse error rejecting --execution-provider cuda, so
# the repair and its verification both have to survive.
startup_src = read('vast', 'startup.sh')
check('startup removes the CPU onnxruntime that shadows the GPU build',
      'uninstall -y onnxruntime' in startup_src)
# Deliberately NOT "reinstall right after the uninstall". Removing the CPU
# wheel deletes the files it overwrote, which are the GPU build's own, so a venv
# can arrive with no CPU wheel and a gutted GPU one. The repair has to key off
# the provider list, which is the condition that actually has to hold.
check('the GPU reinstall is driven by the provider list, not the uninstall',
      'force-reinstall' in startup_src
      and startup_src.index('CUDAExecutionProvider')
      < startup_src.index('force-reinstall'))
check('startup fails when CUDAExecutionProvider is missing',
      'CUDAExecutionProvider' in startup_src and 'get_available_providers' in startup_src)

# The pipeline is launched from a different subshell than startup.sh ran in, so
# the cuDNN LD_LIBRARY_PATH only reaches it by being sourced again.
check('the pipeline launch sources the cuDNN environment',
      '/etc/profile.d/cudnn.sh' in orch_src.split('launch = (')[1][:600])


# The architecture filter is the only thing between a Blackwell card and a paid
# hour on an image whose torch and ONNX cannot use it — a marketplace schedules
# one without complaint. On RunPod this was a hand-maintained keyword table that
# an RTX PRO 4000 walked straight through; here it is a server-side filter on a
# field Vast publishes, so what must not be lost is the filter being applied at
# all.
check('the compute-capability ceiling is applied as a search filter',
      '"compute_cap": {"lte": max_compute_cap}' in orch_src,
      'a keyword table of GPU names goes stale as NVIDIA ships more')
check('that ceiling is sm_90, matching the image',
      '_DEFAULT_MAX_COMPUTE_CAP = 900' in orch_src,
      'Blackwell reports 1200 and would fail only after billing started')
check('the ceiling is settable without editing code',
      'VAST_MAX_COMPUTE_CAP' in orch_src and 'VAST_MAX_COMPUTE_CAP' in env_example_src,
      'the image will move, and a constant nobody can reach is a constant nobody updates')

# ── Deploy guide describes the code that exists ────────────────────────

# startup.sh pulls the repo it is itself part of. Bash resumes a changed script
# at the old byte offset in the new content, so a pull that moves HEAD has to be
# followed by handing over to the new copy — otherwise a boot can run a spliced
# mixture of both and look like the fix simply did not work.
check('startup re-execs itself when the pull moves HEAD',
      'exec bash' in startup_src and 'PHANTOM_STARTUP_REEXEC' in startup_src)
check('that re-exec cannot loop',
      startup_src.count('PHANTOM_STARTUP_REEXEC') >= 2,
      'needs both the guard test and the export')

# ── Deploy guide describes the code that exists ────────────────────────

# Selection order, and why it is not what it used to be.
#
# The RunPod orchestrator sorted fastest-first, because a speed ranking was the
# only thing stopping it picking a weak card. `VAST_MIN_DLPERF` now removes
# everything below a 4090 before the sort runs, so every surviving offer is
# fast enough and the ordering is free to spend on what still differs.
#
# It spends it on distance. VAST_GEOLOCATIONS is documented as a priority
# order, and sorting on price alone had a French host at $0.336 beating a
# British one at $0.350 — three cents to give back part of the round trip this
# whole migration exists to remove.
check('offers are ranked by country priority before price',
      '_rank(offers, geolocations)' in orch_src and 'rank.get(_country(o)' in orch_src)
check('the ranking reads the country code, not the whole label',
      'def _country' in orch_src and 'rsplit(",", 1)' in orch_src,
      'geolocation is "United Kingdom, GB" but the filter matches GB')
check('a speed floor still exists, so ranking on price is safe',
      'VAST_MIN_DLPERF' in orch_src and '"dlperf": {"gte": min_dlperf}' in orch_src,
      'without it, cheapest-first is how a measurement session ends up on an L4')

# Every ONNX model runs on CUDAExecutionProvider, so a card without it is not a
# slower option, it is no option. MI300X listed at $0.50/hr with 192GB on
# RunPod and passed every other filter — exactly what a cheapest-first search
# reaches for.
check('non-NVIDIA cards are excluded at the search',
      '"gpu_arch": {"eq": "nvidia"}' in orch_src)
check('that exclusion covers the pinned-host path too',
      orch_src.count('"gpu_arch": {"eq": "nvidia"}') >= 2,
      'a pinned host bypasses the quality floors; it must not bypass this')

# ── The deploy guide describes the code that exists ────────────────────
print('\nDeploy guide matches the code')

# The word itself is allowed — both files say "not tmux" to correct the old
# instructions. What must not survive is anything that treats it as live: a
# session to attach to, or a claim that the pipeline is started inside one.
TMUX_CLAIMS = ('tmux attach', 'tmux new', 'tmux kill', 'in tmux', 'tmux session')
for label, src in (('the guide', deploy_doc),
                   ('the orchestrator', orch_src),
                   ('.env.example', env_example_src),
                   ('startup.sh', read('vast', 'startup.sh'))):
    stale = [c for c in TMUX_CLAIMS if c in src]
    check('{} does not present tmux as live'.format(label), not stale,
          'the pipeline has run under nohup since the tmux dependency was '
          'dropped; found {}'.format(stale))
pipeline_log = re.search(r'^_PIPELINE_LOG = "([^"]+)"', orch_src, re.M)
check('the guide names the log the pipeline actually writes',
      pipeline_log is not None and pipeline_log.group(1) in deploy_doc,
      'the log is the only view of a nohup pipeline')
check('the guide explains why the address is wss and pinned',
      'PHANTOM_TLS_FINGERPRINT' in deploy_doc and 'self-signed' in deploy_doc,
      'Vast terminates no TLS, so this is the difference between encrypted and not')
check('the guide says storage bills while stopped',
      'billed while stopped' in deploy_doc or 'billing for storage' in deploy_doc
      or 'Storage does not' in deploy_doc,
      'the disk is the only copy of the models, and it is not free to keep')
check('the guide covers forwarding pipeline settings into the instance',
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

# ── The instance image, and what startup.sh must do to it ───────
print('\nInstance setup')

env_example = read('.env.example')
startup = read('vast', 'startup.sh')

image = re.search(r'^VAST_IMAGE=(\S+)', env_example, re.M)
check('.env.example pins a base image', image is not None)
check('the image is a devel tag',
      image is not None and 'devel' in image.group(1),
      'the runtime tags omit the headers cuDNN and TensorRT builds need')
check('the orchestrator defaults to the same image .env.example ships',
      image is not None and image.group(1) in orch_src,
      'a default that disagrees with the documented one is two deployments')

# There is no Docker path any more, and that is a decision rather than an
# omission: docs/VAST_MIGRATION.md chose stop/start with a stock image, so
# every deploy goes through startup.sh over SSH. Nothing may quietly depend on
# an image that bakes the pipeline in.
check('no Dockerfile is left to rot',
      not _os.path.isfile(_os.path.join(_REPO_ROOT, 'Dockerfile')),
      'stop/start on a stock image is the deployment; a stale image build is a trap')

check('startup.sh runs the pre-warm script',
      'prewarm.py' in startup)
check('the pre-warm script exists',
      _os.path.isfile(_os.path.join(_REPO_ROOT, 'vast', 'prewarm.py')))
check('startup.sh fails hard when cuDNN cannot load',
      'libcudnn.so.9' in startup and 'exit 1' in startup,
      'a CPU fallback wastes a paid GPU hour')
check('startup.sh resolves the cuDNN directory with the shared helper',
      'cudnn_path.py' in startup)
check('startup.sh does not use nvidia.cudnn.__file__',
      'nvidia.cudnn.__file__' not in startup,
      'it is None for a namespace package')
check('startup.sh no longer swallows the cuDNN resolution error',
      'cudnn_path.py" || echo' in startup,
      'the old 2>/dev/null turned a TypeError into a misleading warning')

prewarm = read('vast', 'prewarm.py')
for label in ('detection', 'swap', 'restoration', 'occluder'):
    check('pre-warm covers {}'.format(label), "'{}'".format(label) in prewarm)
check('pre-warm constructs Enhancer with a config',
      'Enhancer(CONFIG)' in prewarm,
      'Enhancer() with no args raised TypeError and was silently swallowed')

# ── The transport is protected, and both ends agree how ──────────────
# Vast publishes a random port on a shared public IP and terminates no TLS, so
# every piece of this has to line up or the result is a working, readable
# connection carrying the operator's face.
print('\nTransport security')

server_src = read('pipeline', 'api', 'server.py')
controller_src = read('desktop', 'controller.py')
link_src = read('tools', 'pipeline_link.py')

check('startup.sh generates a certificate',
      'openssl req -x509' in startup)
check('it fingerprints the DER form, not the PEM',
      'outform DER' in startup and 'sha256sum' in startup,
      'Python getpeercert(binary_form=True) is DER; a PEM hash never matches')
check('startup.sh reports the fingerprint on a parseable line',
      'CERT_FINGERPRINT' in startup and 'CERT_FINGERPRINT' in orch_src,
      'the orchestrator reads it out of the transcript')
check('startup.sh generates an API token',
      'API_TOKEN' in startup and 'API_TOKEN' in orch_src)
check('the certificate is generated once, not per boot',
      '-f "${CERT_FILE}"' in startup,
      'a fingerprint that changes on restart reads as an attack')

check('the orchestrator writes both into .env',
      'PHANTOM_TLS_FINGERPRINT' in orch_src and 'PHANTOM_API_TOKEN' in orch_src)
check('the orchestrator refuses to continue without a fingerprint',
      'did not report a certificate fingerprint' in orch_src,
      'an unpinned wss to a self-signed cert is not better than cleartext')

check('the server can serve TLS',
      'PHANTOM_TLS_CERT' in server_src and '_build_ssl_context' in server_src)
check('an unreadable certificate stops the server rather than downgrading it',
      'Refusing to start in cleartext' in server_src)
check('the server authenticates before joining the broadcast set',
      server_src.index('_authenticate') < server_src.index('self._clients.add'),
      'frames go to every client in _clients')
check('the token comparison is constant-time',
      'hmac.compare_digest' in server_src,
      'otherwise a token can be found a character at a time')

check('the desktop pins the fingerprint',
      '_check_pin' in controller_src and 'PHANTOM_TLS_FINGERPRINT' in controller_src)
check('the desktop sends the token in its first frame',
      'PHANTOM_API_TOKEN' in controller_src)
check('the measurement tools share one connector',
      'pipeline_link' in read('tools', 'stats.py')
      and 'pipeline_link' in read('tools', 'realism.py')
      and 'pipeline_link' in read('tools', 'sweep_levers.py'),
      'three hand-rolled ws:// strings is how three answers to one question start')
check('that connector pins too',
      'load_verify_locations' in link_src and 'fingerprint mismatch' in link_src)

# ── Instance environment actually reaches the processes ───────────
print('\nInstance environment delivery')

check('startup.sh sources the instance env file',
      '/etc/environment' in startup,
      'Vast own docs warn those variables are not in an SSH session by default')
check('the pipeline launch carries the forwarded settings',
      '_remote_env_exports' in orch_src,
      'exec_command opens no login shell, so it must be part of the command')
check('the launch sources the cuDNN path file',
      'cudnn.sh' in orch_src,
      'without it ONNX falls back to CPU on a GPU that is billing')

# ── Phase timing agreement ─────────────────────────────────────────────
print('\nCold-start measurement')

check('startup.sh emits parseable phase lines',
      'PHASE ' in startup and 'DISK ' in startup)
check('the orchestrator parses that exact prefix',
      'PHASE ' in orch_src and 'DISK ' in orch_src)
check('startup.sh reports whether the disk was warm or empty',
      'DISK_STATE' in startup,
      'it asked about a network volume on RunPod; here it is the instance disk, '
      'which does not outlive terminate')
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

# ── QML bindings reach something ───────────────────────────────────────
# A `bridge.x` that does not exist is not a parse error: QML resolves context
# properties when the binding evaluates, so it surfaces as a silently empty
# value while the app runs. On a rented GPU that is an expensive way to find a
# typo, and it is the failure mode every new property this feature added is
# exposed to.
print('\nQML bindings reach something')

import ast as _ast  # noqa: E402
import re as _re  # noqa: E402

_qml_src = read('desktop', 'main.qml')
_used = sorted(set(_re.findall(r'\bbridge\.([A-Za-z_][A-Za-z0-9_]*)', _qml_src)))

_bridge_tree = _ast.parse(read('desktop', 'bridge.py'))
_bridge_cls = next(
    (n for n in _bridge_tree.body
     if isinstance(n, _ast.ClassDef) and n.name == 'Bridge'), None,
)
check('the Bridge class is where it is expected', _bridge_cls is not None)

_exposed = set()
for _node in (_bridge_cls.body if _bridge_cls else []):
    if isinstance(_node, _ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, _ast.Name):
                _exposed.add(_t.id)
    if isinstance(_node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        for _dec in _node.decorator_list:
            _name = ''
            if isinstance(_dec, _ast.Call) and isinstance(_dec.func, _ast.Name):
                _name = _dec.func.id
            elif isinstance(_dec, _ast.Name):
                _name = _dec.id
            if _name in ('Property', 'Slot'):
                _exposed.add(_node.name)

_missing = [n for n in _used if n not in _exposed]
check('every bridge.<member> the QML binds to exists',
      not _missing,
      'QML binds to nothing for: %s' % _missing)
check('the check actually looked at something',
      len(_used) > 50, '%d members referenced' % len(_used))

# ── The declared command surface ───────────────────────────────────────
# COMMANDS was read by nothing at all, so it had drifted both ways: five
# entries no handler answered - a client method written against one would get
# `Unknown command` - and five working commands missing entirely. It is only a
# contract if something checks it.
print('\nThe declared command surface')

from pipeline.api.schema import COMMANDS, SERVER_COMMANDS  # noqa: E402

_handlers_src = read('pipeline', 'api', 'handlers.py')
_dispatched = set(_re.findall(r"command_type == '([a-z_]+)'", _handlers_src))
_declared = set(COMMANDS)

check('every declared command has a dispatch branch',
      not (_declared - _dispatched),
      'declared but unanswered: %s' % sorted(_declared - _dispatched))
check('every dispatched command is declared',
      not (_dispatched - _declared),
      'dispatched but undeclared: %s' % sorted(_dispatched - _declared))

_server_src = read('pipeline', 'api', 'server.py')
for _name in SERVER_COMMANDS:
    check('%s is answered by the server itself' % _name,
          "action == '%s'" % _name in _server_src and _name not in _declared,
          'handled before dispatch, so it belongs in neither COMMANDS nor handlers')

# Every command the client can send must be one the server answers. This is the
# direction that actually bites: a client method is what someone reaches for.
_controller_src = read('desktop', 'controller.py')
_sent = set(_re.findall(r"_send\(\s*'([a-z_]+)'", _controller_src))
_sent |= set(_re.findall(r"_fire\(\s*'([a-z_]+)'", read('desktop', 'bridge.py')))
_answered = _declared | set(SERVER_COMMANDS)
check('every command the client can send is answered',
      not (_sent - _answered),
      'client would get Unknown command for: %s' % sorted(_sent - _answered))

check('many_faces is deliberately not settable at runtime',
      'set_many_faces' not in _declared
      and 'def set_many_faces' not in _controller_src,
      'it bypasses every runtime guard and both temporal EMAs; CLI only')
check('keep_frames is deliberately not settable at runtime',
      'set_keep_frames' not in _declared
      and 'def set_keep_frames' not in _controller_src,
      'debugging flag, and a disk filler on a pod; CLI only')

# ── Output format defaults ─────────────────────────────────────────────
# `--keep-fps` was store_true, so the default retimed every render to 30fps,
# and `--keep-audio` was store_true with default=True, which can only produce
# True - there was no way to drop audio at all.
print('\nOutput format defaults')

_core_src = read('pipeline', 'core.py')
check('keep_fps defaults to preserving the source rate',
      FaceSwapConfig().keep_fps is True,
      'retiming duplicates frames on 24fps and discards motion on 60fps')
for _flag, _why in (
    ('keep-fps', 'retiming duplicates frames at 24fps and discards motion at 60'),
    ('keep-audio', 'store_true with default=True could only ever be True'),
):
    _call = _re.search(
        r"add_argument\('--%s'.*?\)\n" % _flag, _core_src, _re.DOTALL,
    )
    check('--%s is declared once and parses' % _flag, _call is not None)
    _text = _call.group(0) if _call else ''
    check('--%s has a real off switch' % _flag,
          'BooleanOptionalAction' in _text, _why)
    check('--%s defaults to keeping the source' % _flag,
          'default=True' in _text,
          'the CLI sets this unconditionally, so it - not the dataclass - '
          'is the default that reaches a run')

# ── What counts as a photo ─────────────────────────────────────────────
# The file dialog and the check that follows it have to agree. They did not:
# the picker offered *.webp while `is_image` asked `mimetypes`, which only
# learned image/webp in Python 3.11 - and on Windows also consults the
# registry, so the same file resolved on one machine and not the next.
# Choosing a webp face did nothing at all, silently.
print('\nWhat counts as a photo')

from pipeline.io import ffmpeg as ff_helpers  # noqa: E402

bridge_photo_src = read('desktop', 'bridge.py')
ffmpeg_src = read('pipeline', 'io', 'ffmpeg.py')

check('the accepted formats are named in one place',
      'IMAGE_EXTENSIONS' in ffmpeg_src)
check('and webp is among them, since the picker offers it',
      '.webp' in ff_helpers.IMAGE_EXTENSIONS)
# is_image's own body only. Slicing to the next `def is_video` swept up
# whatever sat between them, and `is_video_name` legitimately uses mimetypes.
_is_image_body = ffmpeg_src.split('def is_image')[1].split(chr(10) + 'def ')[0]
check('the extension check is not a mimetype lookup',
      'mimetypes' not in _is_image_body,
      'mimetypes.guess_type is environment-dependent for webp and heic')

for ext in ff_helpers.IMAGE_EXTENSIONS:
    check('the dialog filter entry *%s passes the check' % ext,
          ff_helpers.has_image_extension('photo' + ext))

check('the dot is part of the match',
      not ff_helpers.has_image_extension('diagram-png'),
      'endswith("png") also accepts a file called diagram-png')

for ext in ('.gif', '.heic', '.mp4', '.txt'):
    check('%s is refused' % ext, not ff_helpers.has_image_extension('x' + ext))

# The list is a claim about what OpenCV can decode, and nothing was checking
# it. `.gif` is excluded on exactly this basis, so the basis is worth proving.
import numpy as _np  # noqa: E402
import cv2 as _cv2  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_probe = _np.zeros((64, 64, 3), _np.uint8)
_probe[:, :, 1] = 180
_work = _tempfile.mkdtemp()
for _ext in ff_helpers.IMAGE_EXTENSIONS:
    _path = _os.path.join(_work, 'probe' + _ext)
    _wrote = _cv2.imwrite(_path, _probe)
    _back = _cv2.imread(_path) if _wrote else None
    check('OpenCV round-trips %s' % _ext,
          _back is not None and _back.shape == _probe.shape,
          'the list claims these are readable; .gif is excluded for failing this')

# Hex rather than an escaped byte literal: this is the smallest valid GIF,
# and the point is that OpenCV refuses a *well-formed* one.
_gif = _os.path.join(_work, 'x.gif')
with open(_gif, 'wb') as _fh:
    _fh.write(bytes.fromhex(
        '47494638396101000100800000000000ffffff21f90401000000002c0000'
        '0000010001000002024401003b'
    ))
check('and still cannot read a gif, which is why it is not on the list',
      _cv2.imread(_gif) is None,
      'if this ever passes, gif can be added rather than refused')

check('both file dialogs offer the same list',
      bridge_photo_src.count('_IMAGE_FILTER') >= 3
      and "'Images (*.jpg" not in bridge_photo_src,
      'a literal filter string in either dialog is how they drifted apart')
check('an empty selection says why rather than returning quietly',
      '_unusable_reason' in bridge_photo_src)
check('a refusal is styled as one',
      'statusError' in bridge_photo_src
      and 'bridge.statusError' in read('desktop', 'main.qml'),
      'every refusal used to render in the same grey as idle')

# ── Naming a target face ───────────────────────────────────────────────
# The point crosses five files between the operator's click and the guard it
# answers, and a break anywhere in that chain is silent: the photo is simply
# refused for holding two faces, exactly as if nobody had chosen.
print('\nNaming a target face')

schema_src = read('pipeline', 'api', 'schema.py')
handlers_src = read('pipeline', 'api', 'handlers.py')
guards_src = read('pipeline', 'services', 'guards.py')
detect_src = read('pipeline', 'processing', 'frame_processor.py')
controller_src = read('desktop', 'controller.py')

check('the command is declared', "'set_target_faces'" in schema_src)
check('and dispatched, not left as an unknown command',
      "command_type == 'set_target_faces'" in handlers_src)
check('the desktop can send it', 'set_target_faces' in controller_src)
check('upload reports the faces it found',
      '_count_target_faces' in handlers_src and 'face_boxes' in handlers_src)
check('a new upload drops the previous choice',
      "config.set('target_face_points', [])" in handlers_src,
      'a stale point would name a face in a photo nobody looked at')
check('the guard stands down for a named face',
      'face_point or config.target_face_point' in guards_src)
check('and selection prefers it over the largest face',
      'self.face_point or self.config.target_face_point' in detect_src)

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
      '_jitter_buffer.next_for_slot()' in _poll,
      'that buffer carries pipeline output; the raw webcam is a different buffer')
# `next_for_slot` may hand back the previously shown frame when nothing arrived
# in time. That is still pipeline output — the last *swapped* frame — which is
# the property this section exists to protect: a held frame is safe, the raw
# camera never is.
_before_pop = _poll[:_poll.find('next_for_slot')]
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
for path in (('vast', 'orchestrator.py'), ('vast', 'prewarm.py'),
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
