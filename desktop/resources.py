"""
Locating files that ship beside the desktop app.

Running from source, `main.qml` sits next to `main.py` and
`Path(__file__).parent` finds it. In a Nuitka standalone build it is a data
file copied into the distribution, and `__file__` on a compiled module does not
always point where a reader would expect — it depends on how the module was
included and on the Nuitka version.

A missing QML file is the worst shape of build failure: the compile succeeds,
the binary starts, `engine.rootObjects()` comes back empty, and the app exits
with status -1 and no message. So this checks the plausible locations rather
than asserting one, and says what it looked for when it finds nothing.

This is deliberately not a Qt resource (`.qrc`). Compiling QML into the binary
would remove the problem, but it also removes the ability to edit `main.qml`
and relaunch — which is most of how the UI actually gets worked on.
"""

import os
import sys
from pathlib import Path
from typing import List

# Every non-Python file the desktop needs at runtime, relative to the `desktop`
# package. The build script reads this to know what to bundle, so a new asset
# is added in one place and cannot be forgotten on the other side.
BUNDLED_FILES = ('main.qml',)


def _candidates(name: str) -> List[Path]:
    """
    Places `name` could be, in the order worth trying.

    Args:
        name: A path relative to the `desktop` package, e.g. `main.qml`

    Returns:
        Candidate paths, most likely first
    """
    here = Path(__file__).resolve().parent
    found = [here / name]

    # Nuitka standalone: data files land under the distribution root, which is
    # the directory holding the executable, keeping the package layout.
    executable = Path(sys.executable).resolve().parent
    found.append(executable / 'desktop' / name)
    found.append(executable / name)

    return found


def resource_path(name: str) -> Path:
    """
    Absolute path to a bundled file.

    Args:
        name: A path relative to the `desktop` package, e.g. `main.qml`

    Returns:
        The first candidate that exists

    Raises:
        FileNotFoundError: naming every place that was tried. A build missing
            its QML otherwise fails as a silent exit with status -1.
    """
    tried = _candidates(name)
    for candidate in tried:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        '{} is missing from this build. Looked in:\n  {}'.format(
            name, '\n  '.join(str(p) for p in tried),
        )
    )


def source_root() -> Path:
    """
    The repository root, when running from a source checkout.

    Returns:
        The directory containing `desktop/` and `pipeline/`
    """
    return Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """
    Whether this is running from a compiled build rather than source.

    Nuitka sets `__compiled__` on its modules; PyInstaller and py2exe set
    `sys.frozen`. Both are checked so the answer does not depend on which tool
    produced the build.

    Returns:
        True if compiled
    """
    return bool(globals().get('__compiled__')) or bool(getattr(sys, 'frozen', False))


def bundled_data_arguments() -> List[str]:
    """
    Nuitka `--include-data-files` arguments for everything in `BUNDLED_FILES`.

    Returns:
        One argument per bundled file, mapping source path to the same
        relative path inside the distribution
    """
    root = source_root()
    arguments = []
    for name in BUNDLED_FILES:
        source = root / 'desktop' / name
        arguments.append(
            '--include-data-files={}={}'.format(
                source, os.path.join('desktop', name),
            )
        )
    return arguments
