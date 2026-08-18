"""
Exercise ProcessingPipeline._process_video_batch end to end.

The ML services are stubbed - this is testing the FFmpeg plumbing, frame
iteration, ordering, audio restoration, cancellation and temp cleanup, not
the swap itself. A tint stands in for the swap so processing is provable.
"""

import os
import subprocess
import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import tempfile
from unittest.mock import MagicMock


class StubModule(MagicMock):
    """MagicMock that also satisfies `from x.y import z` for nested paths."""

    __path__: list = []


# Stub the ML dependencies the import chain pulls in.
for name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'onnxruntime', 'torch', 'torchvision', 'psutil',
    'tensorflow', 'opennsfw2', 'gfpgan', 'onnx',
):
    sys.modules.setdefault(name, StubModule())

import logging

import cv2
import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.events import EventBus, BUS as GLOBAL_BUS, BATCH_PROGRESS, STATUS_CHANGED
from pipeline.io import ffmpeg as ff
from pipeline.processing.pipeline import ProcessingPipeline

# The per-frame progress log would bury the results.
logging.disable(logging.INFO)

WORK = tempfile.mkdtemp(prefix='phantom-batch-test-')
PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def make_video(path: str, seconds: int = 4, fps: int = 24, audio: bool = True) -> None:
    """Generate a test clip: moving box on a gradient, optional audio tone."""
    args = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'lavfi', '-i', f'testsrc=size=320x240:rate={fps}:duration={seconds}',
    ]
    if audio:
        args += ['-f', 'lavfi', '-i', f'sine=frequency=440:duration={seconds}']
    args += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    if audio:
        args += ['-c:a', 'aac', '-shortest']
    args += [path]
    subprocess.check_output(args, stderr=subprocess.STDOUT)


def probe(path: str, stream: str, entries: str) -> str:
    out = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', stream,
        '-show_entries', entries, '-of', 'default=noprint_wrappers=1:nokey=1', path,
    ], stderr=subprocess.DEVNULL)
    return out.decode().strip()


class StubPipeline(ProcessingPipeline):
    """ProcessingPipeline with the ML stages replaced by a provable tint."""

    def __init__(self, config, bus, stop_after=None):
        super().__init__(config, bus)
        self.frames_seen = 0
        self._stop_after = stop_after

    def _build_processors(self) -> None:
        pass

    def _reset_temporal_state(self) -> None:
        pass

    def _swap_frame_faces(self, frame, stabilize):
        self.frames_seen += 1
        if self._stop_after and self.frames_seen >= self._stop_after:
            self._stop_event.set()
        # Force the red channel high - nothing in testsrc is uniformly red, so
        # its presence in the output proves this ran on every frame.
        frame = frame.copy()
        frame[:, :, 2] = 255
        return frame


def fresh(session: str) -> tuple:
    config = FaceSwapConfig()
    config.log_level = 'error'
    bus = EventBus()
    ff.set_temp_scope(session)
    return config, bus


def run_case(label: str, session: str, audio: bool, keep_fps: bool,
             stop_after=None, keep_frames: bool = False) -> dict:
    print(f'\n{label}')
    target = os.path.join(WORK, f'{session}_target.mp4')
    output = os.path.join(WORK, f'{session}_out.mp4')
    make_video(target, audio=audio)

    config, bus = fresh(session)
    config.keep_fps = keep_fps
    config.keep_frames = keep_frames

    progress: list = []
    statuses: list = []
    bus.on(BATCH_PROGRESS, lambda **kw: progress.append(kw))
    # emit_status publishes to the global BUS, not the injected one.
    handler = lambda **kw: statuses.append(kw.get('message', ''))  # noqa: E731
    GLOBAL_BUS.on(STATUS_CHANGED, handler)

    pipe = StubPipeline(config, bus, stop_after=stop_after)
    pipe._stop_event.clear()
    temp_dir = ff.get_temp_directory_path(target)
    pipe._process_video_batch(target, output)

    import time
    time.sleep(0.4)  # EventBus dispatches on a thread pool

    return {
        'target': target, 'output': output, 'pipe': pipe,
        'progress': progress, 'statuses': statuses, 'temp_dir': temp_dir,
    }


print('=' * 70)
print('ProcessingPipeline._process_video_batch')
print('=' * 70)

# -- Case 1: video with audio, source fps preserved ----------------------
r = run_case('Case 1 - clip with audio, keep_fps=True', 'sess-1',
             audio=True, keep_fps=True)
check('output file written', os.path.isfile(r['output']))
if os.path.isfile(r['output']):
    in_frames = int(probe(r['target'], 'v:0', 'stream=nb_frames') or 0)
    out_frames = int(probe(r['output'], 'v:0', 'stream=nb_frames') or 0)
    check('every input frame processed', r['pipe'].frames_seen == in_frames,
          f'seen={r["pipe"].frames_seen} input={in_frames}')
    check('frame count preserved', out_frames == in_frames,
          f'in={in_frames} out={out_frames}')
    check('source fps preserved', probe(r['output'], 'v:0', 'stream=r_frame_rate') == '24/1',
          probe(r['output'], 'v:0', 'stream=r_frame_rate'))
    check('audio stream restored', probe(r['output'], 'a:0', 'stream=codec_type') == 'audio')
    v_dur = float(probe(r['output'], 'v:0', 'stream=duration') or 0)
    a_dur = float(probe(r['output'], 'a:0', 'stream=duration') or 0)
    check('video and audio same length', abs(v_dur - a_dur) < 0.15,
          f'video={v_dur:.2f}s audio={a_dur:.2f}s')
    frame = cv2.VideoCapture(r['output']).read()[1]
    check('swap applied to output frames', frame is not None and frame[:, :, 2].mean() > 200,
          f'red mean={frame[:, :, 2].mean():.0f}' if frame is not None else 'no frame')
check('temp directory cleaned up', not os.path.isdir(r['temp_dir']))
check('progress events emitted', len(r['progress']) > 0, f'{len(r["progress"])} events')
check('progress ends at 100%', bool(r['progress']) and r['progress'][-1]['percent'] == 100.0)
check('progress is monotonic',
      all(b['done'] >= a['done'] for a, b in zip(r['progress'], r['progress'][1:])))
check('progress throttled, not one per frame', len(r['progress']) < 20,
      f'{len(r["progress"])} events for {r["pipe"].frames_seen} frames')

# -- Case 2: silent clip ------------------------------------------------
r = run_case('Case 2 - silent clip', 'sess-2', audio=False, keep_fps=True)
check('output file written', os.path.isfile(r['output']))
check('no audio stream invented',
      probe(r['output'], 'a:0', 'stream=codec_type') == '' if os.path.isfile(r['output']) else False)
check('temp directory cleaned up', not os.path.isdir(r['temp_dir']))

# -- Case 3: keep_fps off -> retimed to 30fps, duration preserved --------
r = run_case('Case 3 - keep_fps=False, 24fps source', 'sess-3', audio=True, keep_fps=False)
check('output file written', os.path.isfile(r['output']))
if os.path.isfile(r['output']):
    rate = probe(r['output'], 'v:0', 'stream=r_frame_rate')
    check('re-timed to 30fps', rate == '30/1', rate)
    v_dur = float(probe(r['output'], 'v:0', 'stream=duration') or 0)
    a_dur = float(probe(r['output'], 'a:0', 'stream=duration') or 0)
    src_dur = float(probe(r['target'], 'v:0', 'stream=duration') or 0)
    check('duration preserved, not rescaled', abs(v_dur - src_dur) < 0.15,
          f'source={src_dur:.2f}s output={v_dur:.2f}s')
    check('video stays in sync with restored audio', abs(v_dur - a_dur) < 0.15,
          f'video={v_dur:.2f}s audio={a_dur:.2f}s')
    out_frames = int(probe(r['output'], 'v:0', 'stream=nb_frames') or 0)
    check('frames resampled to the new rate', out_frames > r['pipe'].frames_seen,
          f'{r["pipe"].frames_seen} source frames -> {out_frames} at 30fps')

# -- Case 4: cancellation mid-job ---------------------------------------
r = run_case('Case 4 - stopped after 10 frames', 'sess-4', audio=True,
             keep_fps=True, stop_after=10)
check('no output written on cancel', not os.path.isfile(r['output']))
check('stopped early', r['pipe'].frames_seen == 10, f'seen={r["pipe"].frames_seen}')
check('temp directory cleaned up on cancel', not os.path.isdir(r['temp_dir']))
check('cancellation reported',
      any('cancel' in s.lower() for s in r['statuses']))

# -- Case 5: keep_frames retains scratch --------------------------------
r = run_case('Case 5 - keep_frames=True', 'sess-5', audio=True, keep_fps=True,
             keep_frames=True)
check('output file written', os.path.isfile(r['output']))
check('frames retained', os.path.isdir(r['temp_dir']))
if os.path.isdir(r['temp_dir']):
    pngs = [f for f in os.listdir(r['temp_dir']) if f.endswith('.png')]
    check('frames are 6-digit named', bool(pngs) and all(len(p) == 10 for p in pngs),
          f'e.g. {sorted(pngs)[:2]}')

# -- Case 5b: stale frames from an aborted run do not leak in -----------
print('\nCase 5b - stale frames present, keep_frames=True')
ff.set_temp_scope('sess-stale')
stale_target = os.path.join(WORK, 'sess-stale_target.mp4')
make_video(stale_target, seconds=2, audio=True)
stale_dir = ff.get_temp_directory_path(stale_target)
os.makedirs(stale_dir, exist_ok=True)
# Frames numbered beyond what this clip will produce: if they survive, they get
# encoded into the output and the clip is longer than its source.
for n in range(900, 940):
    cv2.imwrite(os.path.join(stale_dir, ff.FRAME_PATTERN % n),
                np.zeros((240, 320, 3), np.uint8))
stale_config, stale_bus = fresh('sess-stale')
stale_config.keep_fps = True
stale_config.keep_frames = True
stale_pipe = StubPipeline(stale_config, stale_bus)
stale_pipe._stop_event.clear()
stale_out = os.path.join(WORK, 'sess-stale_out.mp4')
stale_pipe._process_video_batch(stale_target, stale_out)
src_n = int(probe(stale_target, 'v:0', 'stream=nb_frames') or 0)
check('stale frames not processed', stale_pipe.frames_seen == src_n,
      f'seen={stale_pipe.frames_seen} source={src_n}')
if os.path.isfile(stale_out):
    check('stale frames not encoded into output',
          int(probe(stale_out, 'v:0', 'stream=nb_frames') or 0) == src_n,
          f'output={probe(stale_out, "v:0", "stream=nb_frames")} source={src_n}')

# -- Case 6: sessions do not collide on identical target names ----------
print('\nCase 6 - two sessions, same target filename')
ff.set_temp_scope('sess-A')
a = ff.get_temp_directory_path('/uploads/target.mp4')
ff.set_temp_scope('sess-B')
b = ff.get_temp_directory_path('/uploads/target.mp4')
check('scratch dirs differ per session', a != b, f'{os.path.basename(os.path.dirname(os.path.dirname(a)))} vs {os.path.basename(os.path.dirname(os.path.dirname(b)))}')
check('scratch is outside the target directory', '/uploads' not in a.replace('\\', '/'))

# -- Case 7: frame ordering past 9999 -----------------------------------
print('\nCase 7 - frame ordering past 9999')
ff.set_temp_scope('sess-order')
order_target = os.path.join(WORK, 'order.mp4')
order_dir = ff.get_temp_directory_path(order_target)
os.makedirs(order_dir, exist_ok=True)
for n in (1, 2, 9999, 10000, 10001, 123456):
    open(os.path.join(order_dir, ff.FRAME_PATTERN % n), 'wb').close()
paths = ff.get_temp_frame_paths(order_target)
numbers = [int(os.path.splitext(os.path.basename(p))[0]) for p in paths]
check('sorted order matches numeric order', numbers == sorted(numbers), str(numbers))

# -- Case 8: unsupported target routed to an error ----------------------
print('\nCase 8 - unsupported target')
config, bus = fresh('sess-8')
errors: list = []
GLOBAL_BUS.on('error', lambda **kw: errors.append(kw.get('message', '')))
pipe = StubPipeline(config, bus)
bogus = os.path.join(WORK, 'notes.txt')
open(bogus, 'w').write('hello')
pipe._process_target_batch(bogus, os.path.join(WORK, 'out.mp4'))
import time
time.sleep(0.4)
check('unsupported target reports an error',
      any('nsupported' in e for e in errors), str(errors[:2]))

print('\n' + '=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    for f in FAIL:
        print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """
    Surface the checks above to pytest as one assertion.

    The bodies run at import: these are scripts first, so they stay runnable
    directly (`python tests/test_x.py`) when a failure needs poking at, and the
    per-check output is the diagnostic. This function is what makes the same
    file a pytest test without duplicating any of it.
    """
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
