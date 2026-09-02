"""
Exercise the desktop standalone build definition.

A Nuitka build cannot be run here — it takes tens of minutes and needs Nuitka
installed — so what these check is the part that goes wrong quietly: whether the
build description still matches the app it claims to describe.

That failure mode is specific and nasty. A standalone build that lost `main.qml`
or a QML module **compiles successfully**, starts, produces no root object, and
exits with status -1 saying nothing. Nobody finds that in CI; they find it after
shipping. So the things worth asserting cheaply are that every bundled file
exists, that the QML imports and the recorded module list have not drifted apart,
and that the two decisions with a stated reason behind them — standalone over
onefile, console kept until asked otherwise — are still what the command does.
"""

import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import io
import re

from desktop import resources

sys.path.insert(0, _os.path.join(_REPO_ROOT, 'tools'))
import build_desktop  # noqa: E402

PASS, FAIL = [], []


def check(label: str, condition: bool, detail: str = '') -> None:
    """Record one assertion, printing it as it runs."""
    if condition:
        PASS.append(label)
        print('  ok   {}'.format(label))
    else:
        FAIL.append(label)
        print('  FAIL {}{}'.format(label, ' — ' + detail if detail else ''))


def read(*parts: str) -> str:
    """Read a repo file as text."""
    return io.open(_os.path.join(_REPO_ROOT, *parts), encoding='utf-8').read()


# ── Everything the app needs at runtime is actually there ─────────────
print('\nBundled files')

for name in resources.BUNDLED_FILES:
    path = _os.path.join(_REPO_ROOT, 'desktop', name)
    check('{} exists to be bundled'.format(name), _os.path.isfile(path), path)

check('main.qml is on the bundled list',
      'main.qml' in resources.BUNDLED_FILES,
      'the app exits -1 without it')

found = resources.resource_path('main.qml')
check('resource_path finds it from a source checkout', found.is_file(), str(found))

try:
    resources.resource_path('not-a-real-file.qml')
    check('a missing resource raises rather than returning a bad path', False)
except FileNotFoundError as e:
    check('a missing resource raises rather than returning a bad path', True)
    check('the error names every location tried',
          str(e).count('not-a-real-file.qml') >= 2,
          'a silent -1 exit is what this replaces')

arguments = resources.bundled_data_arguments()
check('one --include-data-files per bundled file',
      len(arguments) == len(resources.BUNDLED_FILES),
      str(arguments))
check('the data argument maps into the desktop package',
      all(a.startswith('--include-data-files=') and 'desktop' in a for a in arguments),
      str(arguments))

# ── The QML import list has not drifted ───────────────────────────────
print('\nQML modules')

qml = read('desktop', 'main.qml')
imported = set()
for line in qml.splitlines():
    match = re.match(r'\s*import\s+([A-Za-z][A-Za-z0-9_.]*)', line)
    if match:
        imported.add(match.group(1))

# `Phantom` is registered from Python with qmlRegisterType, not resolved from
# the Qt install, so it is not a module the build has to bundle.
imported.discard('Phantom')

recorded = set(build_desktop._QML_MODULES)
check('every QML module main.qml imports is recorded in the build script',
      not (imported - recorded),
      'undocumented: {}'.format(sorted(imported - recorded)))
check('the build script records no QML module main.qml does not import',
      not (recorded - imported),
      'stale: {}'.format(sorted(recorded - imported)))
check('QtQuick.Effects is among them',
      'QtQuick.Effects' in imported,
      'MultiEffect blurs the window behind the one-face notice')

# ── Nothing uses a module main.qml does not import ────────────────────
# An unresolved attached property is not a warning, it is a *load* error:
# `main.qml` produces no root object and the app exits -1 saying nothing —
# the same silent failure the bundled-file checks above exist for.
#
# This shipped once. `ToolTip.visible` was added to the output-path row with no
# `import QtQuick.Controls` anywhere in the project, and the desktop stopped
# starting at all — found on a machine other than the one it was written on,
# which is the shape of this bug: it depends on nothing but the import list, so
# it reproduces everywhere equally and gets noticed somewhere else.
_CONTROLS_ONLY = (
    'ToolTip', 'ScrollBar', 'ScrollView', 'Popup', 'Menu', 'MenuItem',
    'Button', 'ComboBox', 'CheckBox', 'RadioButton', 'Slider', 'SpinBox',
    'TextField', 'TextArea', 'Dialog', 'Label', 'Switch', 'TabBar',
)
_controls_used = [
    _type for _type in _CONTROLS_ONLY
    # Attached use (`ToolTip.visible:`) and declaration (`Button {`).
    if re.search(r'^\s*(?:{0}\.\w+\s*:|{0}\s*\{{)'.format(_type), qml, re.MULTILINE)
]
check('every QtQuick.Controls type main.qml uses is covered by its import',
      not _controls_used or 'QtQuick.Controls' in imported,
      'used without the import: {}'.format(_controls_used))

# ── The build command still says what the docstring says ──────────────
print('\nBuild command')

command = build_desktop.build_command()

check('standalone, not onefile',
      '--standalone' in command and not any('--onefile' in c for c in command),
      'unsigned onefile builds are routinely flagged as malware on Windows')
check('the pyside6 plugin is enabled', '--enable-plugin=pyside6' in command)
check('the entry point is desktop.py', command[-1] == 'desktop.py')
check('main.qml is bundled',
      any('main.qml' in c for c in command),
      str(command))

for module in build_desktop._LAZY_MODULES:
    check('lazily-imported {} is named explicitly'.format(module),
          '--include-module={}'.format(module) in command,
          'imported inside a function, so easy for analysis to miss')

# Nuitka refuses outright: "company name and file or product version need to be
# given when any version information is given". It fails in the first seconds
# rather than after twenty minutes, but only if someone runs the build — and the
# whole point of these checks is that the build is expensive to run.
named = any(c.startswith('--company-name') or c.startswith('--product-name')
            for c in command)
versioned = any(c.startswith('--product-version') or c.startswith('--file-version')
                for c in command)
check('naming the product also gives it a version',
      versioned if named else True,
      'Nuitka rejects one without the other, and an unversioned binary in a '
      "customer's hands cannot be matched back to a build")

check('the version has the four components Nuitka wants',
      re.fullmatch(r'\d+\.\d+\.\d+\.\d+', resources.APP_VERSION_FULL) is not None,
      resources.APP_VERSION_FULL)

check('the console is kept by default',
      not any('console' in c for c in command),
      'a GUI app that fails silently at startup is the hard kind to diagnose')

release = build_desktop.build_command(release=True)
if sys.platform == 'win32':
    check('--release hides the console on Windows',
          any('console' in c for c in release), str(release))

# ── The lazily-imported modules really are lazy ───────────────────────
print('\nLazy imports')

desktop_sources = ''
for name in _os.listdir(_os.path.join(_REPO_ROOT, 'desktop')):
    if name.endswith('.py'):
        desktop_sources += read('desktop', name)

for module in build_desktop._LAZY_MODULES:
    top_level = re.search(
        r'^import {}|^from {} import'.format(module, module),
        desktop_sources, re.M,
    )
    check('{} is not imported at module scope'.format(module),
          top_level is None,
          'if it became a hard import, the desktop would stop starting without it')

print('\n' + '=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
for f in FAIL:
    print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """
    Surface the checks above to pytest as one assertion.

    The bodies run at import: these are scripts first, so they stay runnable
    directly (`python tests/<file>.py`) when a failure needs poking at.
    """
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
