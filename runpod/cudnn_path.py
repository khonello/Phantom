#!/usr/bin/env python3
"""
Print the directory holding the cuDNN shared libraries, or exit non-zero.

Used by both deploy paths — the Docker build and runpod/startup.sh — to point
the dynamic linker at the cuDNN that `nvidia-cudnn-cu12` installs into
site-packages. onnxruntime-gpu needs `libcudnn.so.9` on the loader path;
without it every ONNX model falls back to CPU, silently.

Exists as a file rather than an inline `python -c` because both paths had the
same bug and fixed it in neither: they used `nvidia.cudnn.__file__`, which is
**None** for a namespace package, so `os.path.dirname` raised TypeError. In the
Dockerfile that failed the build loudly; in startup.sh it was swallowed by
`2>/dev/null` and turned into a warning, leaving LD_LIBRARY_PATH unset and ONNX
on CPU.

`__path__` is the attribute that works for both regular and namespace packages.
"""

import os
import sys


def cudnn_lib_dir() -> str:
    """
    Locate the cuDNN `lib` directory inside the installed nvidia package.

    Returns:
        Absolute path to the directory containing libcudnn.so.*

    Raises:
        RuntimeError: if the package is missing or has no lib directory
    """
    try:
        import nvidia.cudnn
    except ImportError as exc:
        raise RuntimeError('nvidia-cudnn-cu12 is not installed') from exc

    # Namespace packages report __file__ as None but always carry __path__.
    roots = list(getattr(nvidia.cudnn, '__path__', None) or [])
    if not roots:
        module_file = getattr(nvidia.cudnn, '__file__', None)
        if module_file:
            roots = [os.path.dirname(module_file)]

    if not roots:
        raise RuntimeError('nvidia.cudnn exposes neither __path__ nor __file__')

    for root in roots:
        candidate = os.path.join(root, 'lib')
        if os.path.isdir(candidate):
            return candidate

    raise RuntimeError(
        'no lib directory under: {}'.format(', '.join(roots)),
    )


def main() -> int:
    """
    Print the directory, or report why it could not be found.

    Returns:
        0 on success, 1 otherwise
    """
    try:
        print(cudnn_lib_dir())
        return 0
    except RuntimeError as exc:
        print('cuDNN lib directory not found: {}'.format(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
