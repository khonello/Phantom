#!/usr/bin/env python3
"""
Record a webcam clip to measure the pipeline against.

A speed sweep needs input that is *identical* across configurations, which a
live camera cannot give — so the clip stands in for the webcam feed and every
lever sees the same frames.

**Record it on the camera you would actually use.** The design target is what a
real video call looks like: sensor noise, compression, ordinary imperfection.
A clean studio video measures a workload the product never sees, and so does
the 1080p sample in `.github/examples`.

The resolution comes from the quality preset rather than from a flag with a
default, because getting it wrong is silent and ruins the measurement. With
`--input-url` the pipeline does **not** set capture dimensions — those apply
only to a real webcam — so the file plays at whatever it was encoded at, and
that decides how much frame-space work the compositor does. A 1080p clip has
about nine times the pixels of 640x360 in `_paste`, `_add_grain` and the JPEG
encode, which inflates the CPU share and deflates the inference share. The
ratio between those two is the one thing the whole measurement is for.

Usage:
    python tools/record_clip.py                     # optimal preset, 90s
    python tools/record_clip.py --preset production
    python tools/record_clip.py --seconds 120 --device 1
"""

import argparse
import os
import sys
import time
from typing import List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.api.schema import PRESETS  # noqa: E402

# Long enough that p95 and p99 mean something. The stream loops a file at EOF,
# so a short clip still runs — but every frame would be one of a handful, and a
# percentile over twenty-four repeated moments describes the clip rather than
# the pipeline.
_DEFAULT_SECONDS = 90

# The camera's first frames are auto-exposure and auto-white-balance settling,
# which look nothing like the rest. `warmup_frames` on FaceSwapConfig drops
# these for the same reason.
_WARMUP_FRAMES = 15


def _open_writer(path: str, fps: int, size: tuple, cv2: object) -> object:
    """
    Open a VideoWriter, preferring H.264 and falling back to MPEG-4.

    H.264 is what a real call carries, so its compression artefacts are the
    ones the swap has to sit convincingly inside. Not every OpenCV build ships
    an H.264 encoder, though, and a silent failure here produces a zero-byte
    file — so the fallback is explicit and reported.

    Args:
        path: Output file
        fps: Frames per second to declare
        size: (width, height)
        cv2: The imported module

    Returns:
        An opened VideoWriter

    Raises:
        RuntimeError: if neither codec could be opened
    """
    for fourcc_name in ('avc1', 'mp4v'):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(path, fourcc, fps, size)  # type: ignore[attr-defined]
        if writer.isOpened():
            if fourcc_name != 'avc1':
                print('  codec: {} (H.264 unavailable in this OpenCV build)'
                      .format(fourcc_name))
            else:
                print('  codec: H.264')
            return writer
        writer.release()

    raise RuntimeError('could not open a video writer for {}'.format(path))


def record(args: argparse.Namespace) -> int:
    """
    Capture from the webcam until the duration is reached.

    Returns:
        0 on success
    """
    import cv2

    preset = PRESETS.get(args.preset)
    if preset is None:
        print('Unknown preset {}. Choose from: {}'.format(
            args.preset, ', '.join(sorted(PRESETS))), file=sys.stderr)
        return 1

    width = int(args.width or preset['capture_width'])
    height = int(args.height or preset['capture_height'])
    fps = int(args.fps or preset['capture_fps'])

    out_name = args.out or 'clip_{}x{}.mp4'.format(width, height)
    out_path = out_name if os.path.isabs(out_name) else os.path.join(_REPO_ROOT, out_name)

    print('Opening camera {}...'.format(args.device))
    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        print('ERROR: could not open camera {}.'.format(args.device), file=sys.stderr)
        print('       Try --device 1 if you have more than one.', file=sys.stderr)
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    # Cameras accept a request and then deliver whatever they support. Reading
    # it back matters: a clip silently recorded at 640x480 measures a workload
    # no preset runs, and nothing downstream would say so.
    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (width, height):
        print('  NOTE: camera gave {}x{}, not {}x{}. Frames will be resized.'
              .format(actual_w, actual_h, width, height))

    try:
        writer = _open_writer(out_path, fps, (width, height), cv2)
    except RuntimeError as e:
        print('ERROR: {}'.format(e), file=sys.stderr)
        capture.release()
        return 1

    print('')
    print('Recording {}s at {}x{} @{}fps -> {}'.format(
        args.seconds, width, height, fps, out_path))
    print('Look at the camera and behave as you would on a call — talk, move a')
    print('little, look away and back. Press q to stop early.')
    print('')

    frames = 0
    warmed = 0
    started: Optional[float] = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print('ERROR: camera stopped delivering frames.', file=sys.stderr)
                break

            if warmed < _WARMUP_FRAMES:
                warmed += 1
                continue

            if started is None:
                started = time.time()

            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            frames += 1

            elapsed = time.time() - started
            if frames % max(1, fps) == 0:
                remaining = max(0.0, args.seconds - elapsed)
                print('\r  {:.0f}s recorded, {:.0f}s left ({} frames) '.format(
                    elapsed, remaining, frames), end='', flush=True)

            if not args.no_preview:
                cv2.imshow('recording - press q to stop', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('\n  stopped early.')
                    break

            if elapsed >= args.seconds:
                break
    except KeyboardInterrupt:
        print('\n  interrupted.')
    finally:
        capture.release()
        writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    if frames == 0:
        print('\nERROR: no frames recorded.', file=sys.stderr)
        return 1

    size_mb = os.path.getsize(out_path) / (1024 * 1024) if os.path.isfile(out_path) else 0.0
    print('\n')
    print('Wrote {} — {} frames, {:.1f} MB'.format(out_path, frames, size_mb))
    print('')
    print('Next:')
    print('  python runpod/orchestrator.py push {}'.format(os.path.basename(out_path)))
    print('  (push prints the sweep_levers command to run afterwards)')
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Record a webcam clip for pipeline measurement.',
    )
    parser.add_argument('--preset', default='optimal', choices=sorted(PRESETS),
                        help='take resolution and frame rate from this quality preset')
    parser.add_argument('--seconds', type=float, default=_DEFAULT_SECONDS,
                        help='how long to record (default {})'.format(_DEFAULT_SECONDS))
    parser.add_argument('--device', type=int, default=0, help='camera index')
    parser.add_argument('--width', type=int, help='override the preset width')
    parser.add_argument('--height', type=int, help='override the preset height')
    parser.add_argument('--fps', type=int, help='override the preset frame rate')
    parser.add_argument('--out', help='output file (default clip_<w>x<h>.mp4 at the repo root)')
    parser.add_argument('--no-preview', action='store_true',
                        help='do not open a preview window')
    args = parser.parse_args(argv)

    try:
        return record(args)
    except ImportError:
        print('opencv-python is required: pip install opencv-python', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
