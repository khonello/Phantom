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

It can also hold **more than one root**. The pod's venv is created with
`--system-site-packages` so the image's PyTorch is inherited, and `nvidia` is a
namespace package, so `nvidia.cudnn.__path__` spans both the venv's
site-packages and the image's dist-packages. `startup.sh` installs
`nvidia-cudnn-cu12>=9` into the venv while the image ships cuDNN 8 in
dist-packages — so returning the first root with a `lib` directory returned the
cuDNN 8 one, and `libcudnn.so.9` was not in it. Which root is right is decided
by which one actually holds the library, not by which is listed first.
"""

import glob
import os
import sys

# What onnxruntime-gpu dlopens. Only used to choose between roots.
_SONAME = 'libcudnn.so.9'


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

    fallback = None
    for root in roots:
        candidate = os.path.join(root, 'lib')
        if not os.path.isdir(candidate):
            continue
        if glob.glob(os.path.join(candidate, _SONAME + '*')):
            return candidate
        if fallback is None:
            fallback = candidate

    # No root advertised the soname. One lib directory is still better than
    # none — the caller verifies with a real dlopen either way, and a differently
    # named build should not be turned into a hard failure here.
    if fallback is not None:
        return fallback

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
