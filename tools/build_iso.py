#!/usr/bin/env python3
"""
Build a disc image of the project, for moving it to another machine.

    python tools/build_iso.py                 # -> phantom-<date>.iso
    python tools/build_iso.py --with-env      # include .env (a live API key)
    python tools/build_iso.py --list          # print what would go in, build nothing

**What is left out, and why.** The working tree is ~1.9 GB and almost none of
that is the project:

    pipeline/models/          912 MB   weights; re-downloaded on first use
    desktop/.qtcreator/       966 MB   a virtualenv
    environ-orchestrator/     114 MB   a virtualenv
    build/                    788 MB   Nuitka output, regenerable and tied to
                                       the Python that produced it
    __pycache__/                1 MB   bytecode, and stale copies of it on a
                                       machine with a different interpreter are
                                       worse than none

What is left is the repository — source, docs, tests, QML, requirements — plus
`.git` and the recorded measurement clips, which are gitignored but are the
reason a sweep on a new machine is comparable with an old one.

**`.env` is opt-in.** It holds `VAST_API_KEY`, and a disc image is a thing
that gets handed to people. `docs/SETUP_CHECKLIST.md` tells an operator to copy
it across precisely because it is gitignored, so for your own second machine it
is exactly what you want — which is why the flag exists rather than a refusal.

Backends, in the order they are tried: `xorriso`, `mkisofs`, `genisoimage`,
`hdiutil`, and on Windows the built-in IMAPI2 COM service, which needs nothing
installed. Joliet and Rock Ridge (or UDF) are requested in every case, because
plain ISO 9660 would truncate `requirements-pipeline-gpu.txt` to 8.3.
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VOLUME_ID = 'PHANTOM'

# Directories never copied, matched on the path relative to the repo root.
_EXCLUDE_DIRS = (
    'pipeline/models',
    'desktop/.qtcreator',
    'environ-orchestrator',
    'build',
)

# Directory *names* skipped wherever they appear.
_EXCLUDE_NAMES = ('__pycache__', '.mypy_cache', '.pytest_cache')

# Files never copied unless a flag says otherwise.
#
# Matched as a *prefix*, not a fixed name. The first version of this listed
# `.env` alone, and the tree also held `.env.backup-20260818-234644` — a real
# copy, with a real `VAST_API_KEY` in it — which sailed onto the image while
# `.env` itself was correctly held back. A gate that one spelling of the same
# secret walks around is not a gate.
#
# `.env.example` is the deliberate exception: it is the documented template and
# carries no values.
_SECRET_PREFIX = '.env'
_SECRET_ALLOWED = ('.env.example',)


def _is_secret(name: str) -> bool:
    """Whether a filename holds credentials and needs `--with-env`."""
    return name.startswith(_SECRET_PREFIX) and name not in _SECRET_ALLOWED


def _rel(path: str) -> str:
    return os.path.relpath(path, _REPO_ROOT).replace('\\', '/')


def _included(with_env: bool) -> Tuple[List[str], int]:
    """
    Every file that belongs in the image, and their total size.

    Returns:
        (relative paths, total bytes)
    """
    keep: List[str] = []
    total = 0

    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        rel_dir = _rel(dirpath)
        if rel_dir == '.':
            rel_dir = ''

        # Prune in place so os.walk does not descend into them at all — the
        # excluded trees are most of the bytes, and walking them is most of the
        # time this would otherwise take.
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDE_NAMES
            and ('{}/{}'.format(rel_dir, d).lstrip('/') not in _EXCLUDE_DIRS)
        ]

        for name in filenames:
            rel = '{}/{}'.format(rel_dir, name).lstrip('/')
            if name.endswith('.iso'):
                continue
            if _is_secret(name) and not with_env:
                continue
            full = os.path.join(dirpath, name)
            try:
                total += os.path.getsize(full)
            except OSError:
                continue
            keep.append(rel)

    return (sorted(keep), total)


def _stage(paths: List[str], staging: str) -> None:
    """Copy the included files into a clean tree the backend can read."""
    for i, rel in enumerate(paths, 1):
        dest = os.path.join(staging, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(os.path.join(_REPO_ROOT, rel), dest)
        if i % 500 == 0:
            print('  staged {} / {} files'.format(i, len(paths)))


# ── Backends ───────────────────────────────────────────────────────────


def _unix_backend() -> Optional[List[str]]:
    """The first available command-line ISO writer, as an argv template."""
    for name in ('xorriso', 'mkisofs', 'genisoimage'):
        if shutil.which(name):
            if name == 'xorriso':
                return [name, '-as', 'mkisofs', '-J', '-r', '-V', _VOLUME_ID,
                        '-o', '{iso}', '{dir}']
            return [name, '-J', '-r', '-V', _VOLUME_ID, '-o', '{iso}', '{dir}']
    if shutil.which('hdiutil'):
        return ['hdiutil', 'makehybrid', '-iso', '-joliet',
                '-default-volume-name', _VOLUME_ID, '-o', '{iso}', '{dir}']
    return None


# IMAPI2 is the service Windows itself uses to burn discs, so it is present on
# every Windows 10/11 machine and needs no install. Writing the resulting
# IStream to a file is the one part with no PowerShell equivalent, hence the
# small inline C#.
_IMAPI_PS = r'''
$ErrorActionPreference = 'Stop'
$stage = $args[0]
$iso   = $args[1]
$label = $args[2]

Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Runtime.InteropServices.ComTypes;
public static class IsoWriter {
    [System.Runtime.InteropServices.DllImport("shlwapi.dll", CharSet =
        System.Runtime.InteropServices.CharSet.Unicode,
        ExactSpelling = true, PreserveSig = false)]
    private static extern void SHCreateStreamOnFileEx(
        string f, uint m, uint d, bool c, IStream r, out IStream s);
    public static void Write(object stream, string path) {
        IStream inStream = (IStream)stream;
        IStream outStream;
        SHCreateStreamOnFileEx(path, 0x1001, 0x80, true, null, out outStream);
        try { inStream.CopyTo(outStream, Int64.MaxValue, IntPtr.Zero,
                              IntPtr.Zero); outStream.Commit(0); }
        finally {
            System.Runtime.InteropServices.Marshal.ReleaseComObject(outStream);
        }
    }
}
"@

$fsi = New-Object -ComObject IMAPI2FS.MsftFileSystemImage
# ISO9660 | Joliet | UDF. Joliet and UDF are what carry long names; ISO9660
# alone would truncate to 8.3.
$fsi.FileSystemsToCreate = 7
$fsi.VolumeName = $label
$fsi.Root.AddTree($stage, $false)
$result = $fsi.CreateResultImage()
[IsoWriter]::Write($result.ImageStream, $iso)
'''


def _build_windows(staging: str, iso: str) -> int:
    script = os.path.join(tempfile.gettempdir(), 'phantom_mkiso.ps1')
    with open(script, 'w', encoding='utf-8') as fh:
        fh.write(_IMAPI_PS)
    try:
        return subprocess.call([
            'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-File', script, staging, iso, _VOLUME_ID,
        ])
    finally:
        try:
            os.remove(script)
        except OSError:
            pass


def build(staging: str, iso: str) -> int:
    template = _unix_backend()
    if template is not None:
        cmd = [a.replace('{iso}', iso).replace('{dir}', staging)
               for a in template]
        print('  $ {}'.format(' '.join(cmd)))
        return subprocess.call(cmd)

    if sys.platform == 'win32':
        print('  using the built-in IMAPI2 service')
        return _build_windows(staging, iso)

    print('No ISO writer found. Install xorriso, mkisofs or genisoimage.')
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Build a disc image of the project.')
    parser.add_argument('--output', help='output path (default phantom-<date>.iso)')
    parser.add_argument('--with-env', action='store_true',
                        help='include .env, which holds a live API key')
    parser.add_argument('--list', action='store_true',
                        help='print what would be included, build nothing')
    args = parser.parse_args(argv)

    paths, total = _included(args.with_env)
    mb = total / (1024.0 * 1024.0)

    if args.list:
        for rel in paths:
            print(rel)
        print('\n{} files, {:.1f} MB'.format(len(paths), mb))
        return 0

    iso = args.output or 'phantom-{}.iso'.format(
        datetime.date.today().strftime('%Y%m%d'))
    iso = os.path.abspath(iso)

    print('Phantom -> {}'.format(iso))
    print('  {} files, {:.1f} MB'.format(len(paths), mb))
    if args.with_env:
        secrets = [p for p in paths if _is_secret(os.path.basename(p))]
        print('  WARNING: this image carries credentials - do not hand it to '
              'anyone else:')
        for rel in secrets:
            print('           {}'.format(rel))
    else:
        print('  .env excluded (pass --with-env if this image is for your own '
              'machine)')

    staging = tempfile.mkdtemp(prefix='phantom-iso-')
    try:
        print('\nStaging...')
        _stage(paths, staging)
        print('\nBuilding...')
        status = build(staging, iso)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if status != 0:
        print('\nFailed.')
        return status

    if not os.path.isfile(iso):
        print('\nThe backend reported success but wrote no file.')
        return 1

    print('\nWrote {} ({:.1f} MB)'.format(
        iso, os.path.getsize(iso) / (1024.0 * 1024.0)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
