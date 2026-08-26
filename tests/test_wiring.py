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

# A stopped pod keeps its host, so resting can end with that host's GPUs taken.
# The fallback to a new pod is the only thing that recovers it, and it has to
# stay gated: falling back on *any* resume failure would create a billing pod in
# response to a typo'd ID or a bad key.
resume_body = orch_src.split('def cmd_resume')[1].split(chr(10) + 'def ')[0]
check('resume falls back to start when the host is full',
      'cmd_start()' in resume_body)
check('that fallback is gated, not unconditional',
      '_is_capacity_error' in resume_body)
check('the phrase RunPod actually returned still matches',
      'not enough free gpus' in orch_src.lower(),
      'RunPod said "There are not enough free GPUs on the host machine"')

# insightface depends on the CPU `onnxruntime`, and both wheels write the same
# `onnxruntime/` directory, so the CPU one lands last and shadows the GPU build.
# The only symptom is an argparse error rejecting --execution-provider cuda, so
# the repair and its verification both have to survive.
startup_src = read('runpod', 'startup.sh')
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
      '/etc/profile.d/cudnn.sh' in orch_src.split('pipeline_cmd = (')[1][:400])


# The architecture filter is the only thing between a Blackwell card and a paid
# hour on an image whose torch and ONNX cannot use it — RunPod schedules one
# without complaint. An RTX PRO 4000 got through while 6000 and 4500 were listed
# by model, so the family keywords are what must not be lost.
for _kw in ('Blackwell', 'RTX PRO'):
    check('the GPU filter matches {} as a family'.format(_kw),
          "\"{}\"".format(_kw) in orch_src,
          'model-by-model entries go stale as NVIDIA ships more')

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

# Selection order. The price ceiling is applied before the sort, so everything
# that survives is affordable and there is nothing left for cost to decide —
# while the spread between fastest and slowest of them is the difference between
# a call and a slideshow.
check('GPU candidates are ordered by speed, not price',
      '_gpu_perf(c[1])' in orch_src)
check('AMD cards are excluded, since the pipeline needs CUDA',
      '_is_cuda_gpu' in orch_src and 'MI300' in orch_src)
# "A40" sits inside "RTX A4000". Shortest-match-wins scored an entry-level
# Ampere as a datacenter one, and put it above cards twice its speed.
for _fn in ('_gpu_perf', '_get_gpu_compute_cap'):
    _body = orch_src.split('def {}'.format(_fn))[1].split(chr(10) + 'def ')[0]
    check('{} matches the longest keyword first'.format(_fn),
          'key=len' in _body and 'reverse=True' in _body)

# ── Deploy guide describes the code that exists ────────────────────────

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
check('the extension check is not a mimetype lookup',
      'mimetypes' not in ffmpeg_src.split('def is_image')[1].split('def is_video')[0],
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
