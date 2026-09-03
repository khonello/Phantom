#!/usr/bin/env python3
"""
Print the directory holding the cuDNN shared libraries, or exit non-zero.

Used by vast/startup.sh to point
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

But `import` is not a reliable way to find the *right* install here, and on the
pod it actively points at the wrong one.

The venv is created with `--system-site-packages` so the image's PyTorch is
inherited, and `startup.sh` then installs `nvidia-cudnn-cu12>=9` into the venv
while the image ships cuDNN 8 in its dist-packages. Two copies, and the venv's
own interpreter resolved `nvidia.cudnn` to the image's: if any `sys.path` entry
holds a **regular** `nvidia` package — one with `__init__.py` — it wins outright
and every namespace portion found on earlier entries is discarded. The venv's
cuDNN 9 becomes unreachable by import even though it is first on `sys.path`.

So the search is done over the filesystem instead. `sys.path` says where
packages live; walking it directly finds every `nvidia/cudnn` present, whatever
`import` would have chosen. The right one is then the one that actually holds
the library, not the one listed first.
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
    roots = []

    try:
        import nvidia.cudnn
    except ImportError:
        imported = None
    else:
        # Namespace packages report __file__ as None but always carry __path__.
        roots.extend(getattr(nvidia.cudnn, '__path__', None) or [])
        module_file = getattr(nvidia.cudnn, '__file__', None)
        if module_file:
            roots.append(os.path.dirname(module_file))
        imported = nvidia.cudnn

    # Whatever import chose, walk sys.path for the copies it did not. A regular
    # `nvidia` package on any entry hides the namespace portions on all the
    # others, so this is the only way to see a venv-local install sitting behind
    # one from the image.
    for entry in sys.path:
        if not entry:
            continue
        candidate = os.path.join(entry, 'nvidia', 'cudnn')
        if os.path.isdir(candidate):
            roots.append(candidate)

    seen = set()
    roots = [r for r in roots if not (r in seen or seen.add(r))]

    if not roots:
        if imported is None:
            raise RuntimeError('nvidia-cudnn-cu12 is not installed')
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
        print(
            'warning: no {} under any of: {}; using {}'.format(
                _SONAME, ', '.join(roots), fallback),
            file=sys.stderr,
        )
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
