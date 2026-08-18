"""
Shared setup for the Phantom test suite.

Two jobs, both about being able to run these at all.

**Path.** Tests live in `tests/` and import `pipeline.*`, so the repo root has
to be importable whether pytest is invoked from the root or from here.

**Stubs.** The pipeline's import chain reaches insightface, onnxruntime, torch
and tensorflow — several gigabytes of GPU-oriented dependencies. Almost nothing
worth testing needs them: the guards are pure predicates over detection data,
the FFmpeg plumbing is file I/O, the provider check inspects objects, and the
realism metrics are image statistics. Stubbing the ML layer is what lets this
suite run in seconds on any machine, including CI, instead of requiring a GPU
box nobody will keep green.

What that buys is coverage of the parts that break silently. What it does not
cover is the models themselves, which is a real limit and stated as such in
docs/TODO.md rather than papered over.
"""

import os
import sys
from unittest.mock import MagicMock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TOOLS = os.path.join(REPO_ROOT, 'tools')
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


class StubModule(MagicMock):
    """
    MagicMock that also satisfies `from x.y import z`.

    A plain MagicMock has no `__path__`, so Python refuses to treat it as a
    package and nested imports fail.
    """

    __path__: list = []


# Installed with setdefault so a machine that genuinely has these installed
# uses the real thing.
for _name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'onnxruntime', 'torch', 'torchvision', 'psutil',
    'tensorflow', 'opennsfw2', 'gfpgan', 'onnx',
):
    sys.modules.setdefault(_name, StubModule())
