#!/usr/bin/env python3
"""
Install and verify the GPU pipeline on your own machine.

    python tools/setup_local_gpu.py             # install, then verify
    python tools/setup_local_gpu.py --check     # verify only, install nothing
    python tools/setup_local_gpu.py --dry-run   # print the commands, run none

`requirements-pipeline-gpu.txt` alone is not enough on a local machine, and the
gaps are not obvious — each one ends with a working-looking install that
silently runs on CPU, which on a face-swap pipeline is seconds per frame rather
than a live call.

**Three things it cannot express, all handled here.**

1. **torch is not installed on Windows or Linux by that file.** It is pinned to
   macOS only, because the RunPod image ships torch already. Locally, nothing
   installs it until `gfpgan` pulls one in as a dependency — and the default
   PyPI wheel on Windows is the **CPU** build. A requirements file cannot name
   PyTorch's own index, so it is installed here explicitly.

2. **insightface depends on `onnxruntime`, the CPU wheel.** Both packages write
   the same `onnxruntime/` directory, so whichever pip resolves last wins and
   the GPU build disappears. `runpod/startup.sh` removes the CPU wheel after
   the fact; this does the same thing.

3. **onnxruntime-gpu needs cuDNN 9 on the loader path.** `nvidia-cudnn-cu12`
   puts it inside site-packages, where nothing looks for it by default —
   `nvidia/cudnn/lib` on Linux, `nvidia/cudnn/bin` on Windows.

Versions are pinned to what is actually running in production rather than to
the newest available: torch 2.2.0+cu121, numpy 1.x. Newer probably works and is
untested here. numpy in particular is not a free choice — torch 2.2.0 is built
against numpy 1.x, and numpy 2 breaks its bridge with
`_ARRAY_API not found`.

**Not run against a real GPU by its author.** The steps mirror
`runpod/startup.sh`, which is proven on a pod, but this path has not been
executed on a local NVIDIA machine. `--check` is safe everywhere and is what
tells you whether it worked.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional, Tuple

# The combination running in production. See the module docstring for why this
# is pinned rather than "latest".
_TORCH = 'torch==2.2.0'
_TORCH_INDEX = 'https://download.pytorch.org/whl/cu121'

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REQUIREMENTS = os.path.join(_REPO_ROOT, 'requirements-pipeline-gpu.txt')


def _run(cmd: List[str], dry_run: bool) -> int:
    """Run a command, or print it. Returns its exit status (0 when dry)."""
    print('  $ {}'.format(' '.join(cmd)))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _pip(args: List[str], dry_run: bool) -> int:
    return _run([sys.executable, '-m', 'pip'] + args, dry_run)


def _phase(title: str) -> None:
    print('\n== {} =='.format(title))


# ── Checks ─────────────────────────────────────────────────────────────


def check_driver() -> Tuple[bool, str]:
    """
    Whether an NVIDIA driver is present, and which GPU it reports.

    Checked first because everything below is pointless without it, and the
    failure is worth naming plainly: no driver means no amount of pip solves
    anything.
    """
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return (False, 'nvidia-smi not found - no NVIDIA driver on this machine')
    if out.returncode != 0:
        return (False, (out.stderr or 'nvidia-smi failed').strip().splitlines()[0])
    line = (out.stdout or '').strip().splitlines()
    return (True, line[0] if line else 'unknown GPU')


def cudnn_dir() -> Optional[str]:
    """
    Directory holding cuDNN inside the installed `nvidia-cudnn-cu12`.

    Platform-dependent, which is why `runpod/cudnn_path.py` cannot be reused:
    that one looks for `libcudnn.so.9` under `lib`, and on Windows the library
    is `cudnn64_9.dll` under `bin`.
    """
    subdir = 'bin' if sys.platform == 'win32' else 'lib'
    for root in sys.path:
        if not root:
            continue
        candidate = os.path.join(root, 'nvidia', 'cudnn', subdir)
        if os.path.isdir(candidate):
            names = os.listdir(candidate)
            if any('cudnn' in n for n in names):
                return candidate
    return None


def check_providers() -> Tuple[bool, str]:
    """
    Whether onnxruntime actually offers CUDA — the only check that matters.

    Every step can succeed and still leave this false. It is the difference
    between a live call and seconds per frame, and onnxruntime does not raise
    when a provider fails to initialise: it silently uses CPU.
    """
    try:
        import onnxruntime
    except ImportError:
        return (False, 'onnxruntime is not installed')

    providers = list(onnxruntime.get_available_providers())
    if 'CUDAExecutionProvider' in providers:
        return (True, ', '.join(providers))
    return (False, 'offers only: {}'.format(', '.join(providers) or 'nothing'))


def check_torch() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return (False, 'torch is not installed')
    if torch.cuda.is_available():
        return (True, '{} (CUDA {})'.format(torch.__version__, torch.version.cuda))
    return (False, '{} - built without CUDA, or no device visible'.format(
        torch.__version__))


def check_numpy() -> Tuple[bool, str]:
    try:
        import numpy
    except ImportError:
        return (False, 'numpy is not installed')
    major = int(numpy.__version__.split('.')[0])
    if major >= 2:
        return (False, '{} - torch 2.2 is built against numpy 1.x and its '
                       'bridge breaks with _ARRAY_API not found'.format(numpy.__version__))
    return (True, numpy.__version__)


def check_ffmpeg() -> Tuple[bool, str]:
    try:
        out = subprocess.run(['ffmpeg', '-version'], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return (False, 'not on PATH - RENDER mode cannot decode or encode')
    if out.returncode != 0:
        return (False, 'present but failed to run')
    return (True, (out.stdout or '').splitlines()[0])


def verify() -> int:
    """Report every check. Returns 0 only when the GPU path is truly usable."""
    _phase('Verifying')

    results = [
        ('NVIDIA driver', check_driver()),
        ('torch', check_torch()),
        ('numpy', check_numpy()),
        ('onnxruntime providers', check_providers()),
        ('ffmpeg', check_ffmpeg()),
    ]

    failed = []
    for name, (ok, detail) in results:
        print('  [{}] {:<24} {}'.format('ok' if ok else 'XX', name, detail))
        if not ok:
            failed.append(name)

    where = cudnn_dir()
    print('  [{}] {:<24} {}'.format(
        'ok' if where else '..', 'cuDNN',
        where or 'not found in site-packages (may be installed system-wide)'))

    if failed:
        print('\nNot usable on GPU yet: {}'.format(', '.join(failed)))
        if 'onnxruntime providers' in failed and where and sys.platform == 'win32':
            print('  cuDNN is installed but may not be on PATH. Try:')
            print('    set PATH={};%PATH%'.format(where))
        elif 'onnxruntime providers' in failed and where:
            print('  cuDNN is installed but may not be on the loader path. Try:')
            print('    export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH'.format(where))
        return 1

    print('\nGPU pipeline is ready. Start it with:')
    print('  python pipeline.py --execution-provider cuda')
    return 0


# ── Install ────────────────────────────────────────────────────────────


def install(dry_run: bool) -> int:
    ok, detail = check_driver()
    print('  GPU: {}'.format(detail))
    if not ok:
        print('\nNo NVIDIA driver. Install one from nvidia.com/drivers first - '
              'nothing below helps without it.')
        return 1

    _phase('1/4  PyTorch with CUDA')
    print('  The requirements file installs torch on macOS only, and the '
          'default\n  PyPI wheel on Windows is CPU-only, so this comes from '
          "PyTorch's index.")
    if _pip(['install', _TORCH, '--index-url', _TORCH_INDEX], dry_run) != 0:
        print('  torch install failed.')
        return 1

    _phase('2/4  Pipeline requirements')
    if _pip(['install', '-r', _REQUIREMENTS], dry_run) != 0:
        return 1

    _phase('3/4  One onnxruntime, and it has to be the GPU one')
    print('  insightface depends on the CPU wheel and both write the same\n'
          '  directory, so whichever pip resolved last wins.')
    listing = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
        capture_output=True, text=True,
    ).stdout or ''
    if any(line.startswith('onnxruntime==') for line in listing.splitlines()):
        print('  CPU onnxruntime present - removing and reinstating the GPU build.')
        _pip(['uninstall', '-y', 'onnxruntime'], dry_run)
        _pip(['install', '--force-reinstall', '--no-deps',
              'onnxruntime-gpu>=1.15.0'], dry_run)
    else:
        print('  Only the GPU build is installed.')

    _phase('4/4  cuDNN')
    if cudnn_dir():
        print('  Already present in site-packages.')
    else:
        print('  onnxruntime-gpu needs cuDNN 9; --no-deps so it cannot pull a '
              'different torch.')
        _pip(['install', '--no-deps', 'nvidia-cudnn-cu12>=9.0'], dry_run)

    if dry_run:
        print('\nDry run - nothing was installed.')
        return 0
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Install and verify the GPU pipeline on this machine.')
    parser.add_argument('--check', action='store_true',
                        help='verify only; install nothing')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the commands without running them')
    args = parser.parse_args(argv)

    print('Python {} - {}'.format(sys.version.split()[0], sys.executable))

    if not args.check:
        status = install(args.dry_run)
        if status != 0:
            return status
        if args.dry_run:
            return 0

    return verify()


if __name__ == '__main__':
    sys.exit(main())
