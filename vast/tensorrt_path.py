#!/usr/bin/env python3
"""
Print the directory holding the TensorRT shared libraries, or exit non-zero.

Same job as `cudnn_path.py`, one package along: onnxruntime-gpu ships the
TensorRT execution provider but dlopens `libnvinfer.so` at runtime, and the pip
`tensorrt` packages install those libraries into site-packages rather than
anywhere the loader looks by default.

The search is done over the filesystem rather than by import, for the reason
`cudnn_path.py` records at length: the venv is created with
`--system-site-packages`, so a regular `nvidia` package anywhere on `sys.path`
hides namespace portions on every other entry, and `import` can resolve to the
image's copy while the venv's sits unreachable behind it. Walking `sys.path`
directly finds every candidate, whatever `import` would have chosen.

Two package layouts are in circulation and both are checked:

- `tensorrt_libs/` — what `tensorrt-cu12` and recent `tensorrt` wheels use
- `nvidia/tensorrt/lib/` — the older nvidia-namespace layout

**Unlike cuDNN, a failure here is not fatal.** Missing cuDNN means every ONNX
model silently runs on CPU, which is a paid GPU hour producing nothing usable.
Missing TensorRT means the models run on CUDA instead — still on the GPU, still
holding a live call, just not as fast as intended. The caller warns and carries
on. See docs/COMPILATION.md.
"""

import glob
import os
import sys

# What onnxruntime-gpu dlopens for the TensorRT provider. Unversioned here
# because the major version moves with the onnxruntime build, and the caller
# verifies by asking onnxruntime whether the provider actually registered —
# which is the only check that matters.
_SONAME = 'libnvinfer.so'

# Directories a TensorRT wheel puts its libraries in, relative to a sys.path
# entry. Order is preference, not likelihood.
_LAYOUTS = (
    ('tensorrt_libs',),
    ('nvidia', 'tensorrt', 'lib'),
    ('tensorrt',),
)


def tensorrt_lib_dir() -> str:
    """
    Locate the directory containing the TensorRT shared libraries.

    Returns:
        Absolute path to the directory holding libnvinfer.so*

    Raises:
        RuntimeError: if no candidate directory could be found
    """
    roots = []

    for entry in sys.path:
        if not entry:
            continue
        for layout in _LAYOUTS:
            candidate = os.path.join(entry, *layout)
            if os.path.isdir(candidate):
                roots.append(candidate)

    seen = set()
    roots = [r for r in roots if not (r in seen or seen.add(r))]

    if not roots:
        raise RuntimeError(
            'no TensorRT library directory found on sys.path; '
            'is the `tensorrt` wheel installed?'
        )

    fallback = None
    for root in roots:
        if glob.glob(os.path.join(root, _SONAME + '*')):
            return root
        if fallback is None:
            fallback = root

    # No directory advertised the soname. Returning one is still better than
    # nothing — the caller verifies by asking onnxruntime for the provider, so a
    # differently named build should not become a hard failure here.
    if fallback is not None:
        print(
            'warning: no {} under any of: {}; using {}'.format(
                _SONAME, ', '.join(roots), fallback),
            file=sys.stderr,
        )
        return fallback

    raise RuntimeError('no TensorRT lib directory under: {}'.format(', '.join(roots)))


def main() -> int:
    """
    Print the directory, or report why it could not be found.

    Returns:
        0 on success, 1 otherwise
    """
    try:
        print(tensorrt_lib_dir())
    except RuntimeError as e:
        print('error: {}'.format(e), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
