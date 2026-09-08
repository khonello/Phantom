"""
Fetch the background-segmentation model the desktop's FILTERS panel needs.

The desktop deliberately installs no ML runtime — `requirements-desktop.txt`
says so, and the background layer runs through `cv2.dnn`, which opencv-python
already provides. What it cannot provide is the weights, and `models` is
gitignored for the same reason the face weights are, so the file has to be
fetched rather than cloned.

Run it once per machine:

    python tools/fetch_segmentation_model.py

Without it the background rail still appears and selecting a background does
nothing, on purpose: this is a decorative stage on the display path and it must
not be able to fail a live call. That silence is exactly why this script also
*verifies* rather than just downloading — a truncated file, an HTML error page
saved under an .onnx name, or a Git LFS pointer instead of the payload would all
leave the feature quietly dead, and the last of those is not hypothetical, since
GitHub serves the pointer from `raw.githubusercontent.com` and the real file
only from `media.githubusercontent.com`.
"""

import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from desktop import backgrounds                                    # noqa: E402

# Anything smaller than this is not a segmentation model. An LFS pointer is
# ~130 bytes and a GitHub error page a few kilobytes, so the bar only has to be
# low and unambiguous rather than exact.
_MIN_BYTES = 100_000

# Every ONNX file is a protobuf whose first field is `ir_version`, varint-tagged
# as 0x08. Cheaper and more honest than a hash: it says "this is the kind of
# file we asked for", which is the failure actually being guarded against, and
# it does not go stale when upstream re-exports the model.
_ONNX_MAGIC = b'\x08'


def _destination(model: 'backgrounds.SegmentationModel') -> str:
    """Where the weights should land."""
    return os.path.join(_REPO_ROOT, 'desktop', 'models', model.filename)


def _download(url: str, target: str) -> None:
    """
    Fetch `url` to `target`, atomically.

    Written to a temporary file in the same directory and moved into place only
    once it has been verified, so an interrupted run cannot leave a half-file
    that loads as a corrupt model rather than as a missing one.

    Args:
        url: Source URL
        target: Final path

    Raises:
        SystemExit: on any transport or validation failure
    """
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)

    handle, staging = tempfile.mkstemp(suffix='.part', dir=directory)
    os.close(handle)
    try:
        print('  fetching {}'.format(url))
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()

        if len(payload) < _MIN_BYTES:
            raise ValueError(
                'got {} bytes, which is too small to be the model. A Git LFS '
                'pointer or an error page is the usual cause.'.format(
                    len(payload)))
        if not payload.startswith(_ONNX_MAGIC):
            raise ValueError(
                'the payload does not begin like an ONNX protobuf, so this is '
                'not the file it claims to be.')

        with open(staging, 'wb') as out:
            out.write(payload)
        os.replace(staging, target)
        print('  wrote {} ({:.1f} MB)'.format(target, len(payload) / 1e6))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        if os.path.exists(staging):
            os.remove(staging)
        print('ERROR: {}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)


def _verify(path: str, model: 'backgrounds.SegmentationModel') -> None:
    """
    Load the file the way the desktop will, and run one frame through it.

    A file that downloads cleanly and then fails to load is the same outcome as
    no file at all, except that it looks fixed. Checking here means the failure
    lands on the person running the setup step rather than mid-call.

    Args:
        path: Downloaded weights
        model: Its registry profile

    Raises:
        SystemExit: if the model does not load or does not produce a matte
    """
    import numpy as np

    segmenter = backgrounds.Segmenter(path=path, model=model)
    if not segmenter.available:
        print('ERROR: downloaded but will not load: {}'.format(
            segmenter.failure), file=sys.stderr)
        raise SystemExit(1)

    probe = np.zeros((180, 320, 3), dtype=np.uint8)
    matte = segmenter.matte(probe)
    if matte is None or matte.shape != (180, 320):
        print('ERROR: the model loaded but produced no usable matte.',
              file=sys.stderr)
        raise SystemExit(1)

    print('  loads, and returns a {}x{} matte'.format(
        matte.shape[1], matte.shape[0]))


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch the desktop background-segmentation model.')
    parser.add_argument(
        '--model', default=backgrounds.MODELS[0].key,
        choices=[m.key for m in backgrounds.MODELS if m.url],
        help='Which registered model to fetch (default: %(default)s)')
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download even if the file is already present')
    parser.add_argument(
        '--check', action='store_true',
        help='Only report what is installed; download nothing')
    args = parser.parse_args()

    model = next(m for m in backgrounds.MODELS if m.key == args.model)
    target = _destination(model)

    if args.check:
        found_model, found_path = backgrounds.resolve_model()
        if found_path:
            print('installed: {} ({})'.format(
                found_path, found_model.key if found_model else 'unknown'))
            return 0
        print('not installed. Run this script without --check.')
        return 1

    print('{}  ({})'.format(model.key, model.licence))
    if model.notes:
        print('  {}'.format(model.notes))

    if os.path.isfile(target) and not args.force:
        print('  already present at {}'.format(target))
    else:
        _download(model.url, target)

    _verify(target, model)
    print('\nDone. The background rail in the FILTERS panel will work now.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
