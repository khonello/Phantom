#!/usr/bin/env python3
"""
Build the desktop app into a standalone distribution with Nuitka.

**This is a distribution decision, not a performance one.** Qt and QML
rendering, JPEG decode and the WebSocket transport are all C already; what is
left for the interpreter is small against a 33ms display tick. The reason to
compile is that `desktop/` ships to customers and the access-code gate, the
session clock and the Firestore session plane are all enforced inside it.
Distributed as `.py` files, that enforcement comes out with a text editor.

A compiled binary is not a security boundary — nothing that runs on someone
else's machine is — but it moves tampering from "open the file" to "reverse a
binary", which is the difference that matters commercially.

`pipeline/` is deliberately **not** built this way. It runs inside a Docker
image on a rented pod where nobody reads the source, its startup is dominated by
model load, and compiling it would add twenty minutes to every image build while
making tracebacks worse on the layer under active development. See
docs/COMPILATION.md.

Usage:
    python tools/build_desktop.py --print-only    # show the command, build nothing
    python tools/build_desktop.py                 # build, console kept
    python tools/build_desktop.py --release       # build, console hidden
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from desktop.resources import bundled_data_arguments  # noqa: E402

_ENTRY = 'desktop.py'
_OUTPUT_DIR = 'build'
_BINARY_NAME = 'Phantom'

# Imported inside function bodies rather than at module scope, so they are easy
# for a static analyser to miss. Naming them costs nothing and the failure mode
# — an ImportError the first time an operator opens the virtual camera or the
# voice panel — would only show up in someone else's hands.
_LAZY_MODULES = (
    'pyvirtualcam',
    'sounddevice',
    'parselmouth',
)

# What `desktop/main.qml` imports. These are QML modules resolved by the engine
# at runtime, not Python packages — so they cannot be named with
# `--include-package`, and nothing in the Python source references them. The
# pyside6 plugin bundles them via `--include-qt-plugins`.
#
# Recorded here because they are what a broken build loses first, and because
# QtQuick.Effects is load-bearing rather than decorative: MultiEffect is what
# blurs the window behind the one-face notice, and it only arrived at Qt 6.5,
# which is why desktop/requirements.txt sets that floor.
#
# `tests/test_desktop_build.py` checks this stays in step with the QML file.
_QML_MODULES = (
    'QtQuick',
    'QtQuick.Window',
    'QtQuick.Layouts',
    'QtQuick.Effects',
)


def build_command(release: bool = False) -> List[str]:
    """
    The full Nuitka command line.

    Args:
        release: Hide the console window. Off by default — a first build is
                 something you watch, and a GUI app that fails silently at
                 startup is much harder to diagnose than one that prints.

    Returns:
        Argument list suitable for `subprocess.run`
    """
    command = [
        sys.executable, '-m', 'nuitka',

        # --standalone, never --onefile. A single-file build unpacks to a temp
        # directory on every launch, and unsigned single-file binaries are
        # routinely flagged by Windows antivirus — which reads as malware to a
        # customer rather than as a developer tool.
        '--standalone',

        '--enable-plugin=pyside6',
        '--assume-yes-for-downloads',

        '--output-dir={}'.format(_OUTPUT_DIR),
        '--output-filename={}'.format(_BINARY_NAME),

        '--company-name=Phantom',
        '--product-name=Phantom',
    ]

    command.extend(bundled_data_arguments())
    command.extend('--include-module={}'.format(m) for m in _LAZY_MODULES)

    # `all` rather than a curated list, for the first build. Trimming Qt
    # plugins is a size optimisation, and getting it wrong produces a binary
    # that starts and then fails to render — the expensive kind of wrong to
    # diagnose. Narrow it once there is a build known to work.
    command.append('--include-qt-plugins=all')

    if release and sys.platform == 'win32':
        # Nuitka 2.x spelling. On 1.x this is `--disable-console`.
        command.append('--windows-console-mode=disable')

    command.append(_ENTRY)
    return command


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Build the Phantom desktop app with Nuitka.',
    )
    parser.add_argument(
        '--release', action='store_true',
        help='hide the console window (do this only once the build works)',
    )
    parser.add_argument(
        '--print-only', action='store_true',
        help='print the command without running it',
    )
    args = parser.parse_args(argv)

    command = build_command(release=args.release)

    if args.print_only:
        print(' \\\n  '.join(command))
        return 0

    try:
        import nuitka  # noqa: F401
    except ImportError:
        print(
            'Nuitka is not installed. It is a build-time dependency only and is\n'
            'deliberately absent from the runtime requirements:\n\n'
            '    pip install nuitka\n\n'
            'Expect the first build to take 10-30 minutes.',
            file=sys.stderr,
        )
        return 1

    print('Building {} (this takes 10-30 minutes)...'.format(_BINARY_NAME))
    print(' \\\n  '.join(command))
    print()

    result = subprocess.run(command, cwd=_REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

    distribution = os.path.join(_REPO_ROOT, _OUTPUT_DIR, 'desktop.dist')
    print('\nBuilt: {}'.format(distribution))
    print(
        '\nSmoke-test it against a running pipeline before shipping:\n'
        '  1. python pipeline.py           (in another terminal)\n'
        '  2. run the binary in {}\n'
        '  3. confirm the window opens, connects, and shows a frame\n'
        '\nA build that lost main.qml exits -1; desktop/resources.py now says\n'
        'where it looked instead of failing silently.'.format(distribution)
    )
    if not args.release:
        print(
            '\nThis build keeps its console. Rebuild with --release once it '
            'works, then code-sign before any customer sees it — an unsigned '
            'binary is worse for them than a .py file.'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
