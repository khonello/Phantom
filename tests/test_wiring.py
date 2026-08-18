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

# ── Everything parses ──────────────────────────────────────────────────
print('\nSyntax')
for path in (('runpod', 'orchestrator.py'), ('runpod', 'prewarm.py')):
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
