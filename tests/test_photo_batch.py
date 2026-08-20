"""
Exercise photo mode: several image targets, each swapped independently.

Two things are being proved here, and neither is the swap itself (the ML layer
is stubbed, as everywhere in this suite):

1. **A photo that cannot be swapped produces no output file.** The image path
   used to write unconditionally, so a guarded or faceless target left a file
   that was byte-for-byte its input but named like a result. That is the
   "confidently wrong output" the guards exist to prevent, and it is invisible
   to whoever opens the folder afterwards.
2. **One bad photo costs only itself.** Independence is the whole contract of
   the mode, so every failure shape - no face, a guard, an unreadable file, a
   missing file, an exception out of the swap - is checked to leave the other
   photos alone.
"""

import os
import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import base64
import tempfile
from unittest.mock import MagicMock


class StubModule(MagicMock):
    """MagicMock that also satisfies `from x.y import z` for nested paths."""

    __path__: list = []


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

from pipeline.api import handlers
from pipeline.api.schema import MAX_PHOTO_BYTES, MAX_PHOTO_TARGETS
from pipeline.config import FaceSwapConfig
from pipeline.events import EventBus, PHOTO_RESULT
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.types import PhotoResult

logging.disable(logging.INFO)

WORK = tempfile.mkdtemp(prefix='phantom-photo-test-')
PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def write_photo(name: str, size: int = 400) -> str:
    """A photo on disk. Content is irrelevant; the stub decides the outcome."""
    path = os.path.join(WORK, name)
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    cv2.imwrite(path, rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    return path


class StubPipeline(ProcessingPipeline):
    """
    Pipeline with the swap replaced by a scripted outcome per photo.

    `outcomes` maps a target's basename to the reason it fails, or to '' for a
    photo that swaps. A basename mapped to an Exception raises it, standing in
    for a swap that blows up rather than declining.
    """

    def __init__(self, config, bus, outcomes=None):
        super().__init__(config, bus)
        self.outcomes = outcomes or {}
        self.seen: list = []

    def _build_processors(self) -> None:
        pass

    def _reset_temporal_state(self) -> None:
        pass

    def _swap_frame_detail(self, frame, stabilize):
        name = os.path.basename(self.seen[-1]) if self.seen else ''
        outcome = self.outcomes.get(name, '')
        if isinstance(outcome, Exception):
            raise outcome
        if outcome:
            return frame, outcome, 0
        # Force the red channel high, so a swapped output is provable.
        frame = frame.copy()
        frame[:, :, 2] = 255
        return frame, '', 1

    def _process_image_batch(self, target_path, output_path):
        self.seen.append(target_path)
        return super()._process_image_batch(target_path, output_path)


def fresh():
    config = FaceSwapConfig()
    config.log_level = 'error'
    return config, EventBus()


print('=' * 70)
print('Photo mode - independent targets, skip on failure')
print('=' * 70)


# ── A photo that swaps ────────────────────────────────────────────────────
print('\nA photo that swaps')

good = write_photo('good.png')
config, bus = fresh()
pipe = StubPipeline(config, bus)
result = pipe._process_image_batch(good, None)

expected_out = os.path.join(WORK, 'good_swapped.png')
check('result reports success', result.ok, result.reason)
check('output path is beside the target', result.output_path == expected_out,
      str(result.output_path))
check('output file exists', os.path.isfile(expected_out))
check('faces counted', result.faces == 1, str(result.faces))

written = cv2.imread(expected_out)
check('output carries the swap', written is not None and bool((written[:, :, 2] == 255).all()))


# ── A photo that cannot be swapped writes nothing ─────────────────────────
print('\nA photo that cannot be swapped writes nothing')

for label, reason in (
    ('no face', 'no face detected'),
    ('guarded', 'two faces in frame'),
    ('compositor declined', 'the compositor produced no swap'),
):
    target = write_photo(f'{label.replace(" ", "_")}.png')
    config, bus = fresh()
    pipe = StubPipeline(config, bus, {os.path.basename(target): reason})
    result = pipe._process_image_batch(target, None)
    out = os.path.join(WORK, os.path.basename(target).replace('.png', '_swapped.png'))

    check(f'{label}: reported as skipped', not result.ok)
    check(f'{label}: reason is carried through', result.reason == reason, result.reason)
    check(f'{label}: no output file written', not os.path.exists(out))


# ── An unreadable file ────────────────────────────────────────────────────
print('\nAn unreadable file')

not_an_image = os.path.join(WORK, 'broken.png')
with open(not_an_image, 'wb') as fh:
    fh.write(b'this is not a PNG')

config, bus = fresh()
pipe = StubPipeline(config, bus)
result = pipe._process_image_batch(not_an_image, None)
check('unreadable file is skipped', not result.ok)
check('unreadable file says why', 'could not be read' in result.reason, result.reason)
check('unreadable file writes nothing',
      not os.path.exists(os.path.join(WORK, 'broken_swapped.png')))


# ── Independence across a job ─────────────────────────────────────────────
print('\nOne bad photo costs only itself')

targets = [write_photo(f'batch_{i}.png') for i in range(4)]
config, bus = fresh()
events: list = []
bus.on(PHOTO_RESULT, lambda **kw: events.append(kw))

pipe = StubPipeline(config, bus, {
    'batch_1.png': 'no face detected',
    'batch_2.png': RuntimeError('swap exploded'),
})
results = pipe._process_photos_batch(targets)

import time
time.sleep(0.4)  # EventBus dispatches on a thread pool

check('every target produced a result', len(results) == 4, str(len(results)))
check('the good photos swapped', results[0].ok and results[3].ok)
check('the faceless photo was skipped', not results[1].ok)
check('the exploding photo was skipped', not results[2].ok)
check('the exception is reported, not raised',
      'RuntimeError' in results[2].reason, results[2].reason)
check('results stay in target order',
      [r.target_path for r in results] == targets)
check('outputs exist only for the photos that swapped',
      os.path.isfile(os.path.join(WORK, 'batch_0_swapped.png'))
      and not os.path.exists(os.path.join(WORK, 'batch_1_swapped.png'))
      and not os.path.exists(os.path.join(WORK, 'batch_2_swapped.png'))
      and os.path.isfile(os.path.join(WORK, 'batch_3_swapped.png')))

check('one PHOTO_RESULT per photo', len(events) == 4, str(len(events)))
check('PHOTO_RESULT carries index and total',
      sorted(e['index'] for e in events) == [0, 1, 2, 3]
      and all(e['total'] == 4 for e in events))

check('pipeline exposes the results', len(pipe.photo_results) == 4)


# ── A missing file ────────────────────────────────────────────────────────
print('\nA missing file')

config, bus = fresh()
pipe = StubPipeline(config, bus)
results = pipe._process_photos_batch([
    os.path.join(WORK, 'does_not_exist.png'),
    write_photo('after_missing.png'),
])
check('missing file is skipped', not results[0].ok)
check('missing file says why', results[0].reason == 'file not found', results[0].reason)
check('the job continues past it', results[1].ok)


# ── Cancellation ──────────────────────────────────────────────────────────
print('\nCancellation stops the loop')

targets = [write_photo(f'cancel_{i}.png') for i in range(4)]
config, bus = fresh()


class CancellingPipeline(StubPipeline):
    def _process_image_batch(self, target_path, output_path):
        result = super()._process_image_batch(target_path, output_path)
        self._stop_event.set()  # cancel after the first photo
        return result


pipe = CancellingPipeline(config, bus)
pipe._stop_event.clear()
results = pipe._process_photos_batch(targets)
check('cancellation stops after the photo in flight', len(results) == 1, str(len(results)))
check('the remaining photos were not written',
      not os.path.exists(os.path.join(WORK, 'cancel_1_swapped.png')))


# ── Routing: target_paths selects photo mode ──────────────────────────────
print('\nA job with target_paths runs as photos')

targets = [write_photo(f'route_{i}.png') for i in range(2)]
config, bus = fresh()
config.target_paths = targets
config.source_path = write_photo('route_source.png')

pipe = StubPipeline(config, bus)
pipe._swapping_proc = MagicMock()
pipe._swapping_proc.set_source.return_value = True
pipe._run_batch_impl()

check('both photos ran', len(pipe.photo_results) == 2, str(len(pipe.photo_results)))
check('the single-file path was not used', pipe.seen == targets)


# ── upload_target ─────────────────────────────────────────────────────────
print('\nupload_target')

handlers._UPLOAD_DIR = os.path.join(WORK, 'uploads')


def payload(name: str, blob: bytes) -> dict:
    return {'name': name, 'data': base64.b64encode(blob).decode('ascii')}


config = FaceSwapConfig()
config.target_path = '/previous/target.mp4'
response = handlers.handle_upload_target(config, [
    payload('a.jpg', b'aaaa'),
    payload('b.jpg', b'bbbb'),
])
check('upload succeeds', response.success, response.error or '')
check('both photos staged', len(response.data['paths']) == 2)
check('config points at the uploads', config.target_paths == response.data['paths'])
check('the previous single target is cleared', config.target_path is None)
check('staged files exist on disk',
      all(os.path.isfile(p) for p in response.data['paths']))

# Same filename from two folders must not collide.
first = handlers.handle_upload_target(config, [payload('IMG_1.jpg', b'first')])
second = handlers.handle_upload_target(config, [payload('IMG_1.jpg', b'second')])
check('same-named photos land in separate jobs',
      first.data['paths'][0] != second.data['paths'][0])
check('the earlier upload survives the later one',
      open(first.data['paths'][0], 'rb').read() == b'first')

over_cap = handlers.handle_upload_target(
    config, [payload(f'{i}.jpg', b'x') for i in range(MAX_PHOTO_TARGETS + 1)]
)
check('more than the cap is refused', not over_cap.success)
check('the refusal names the cap', str(MAX_PHOTO_TARGETS) in (over_cap.error or ''))

mixed = handlers.handle_upload_target(config, [
    payload('fine.jpg', b'ok'),
    payload('huge.jpg', b'x' * (MAX_PHOTO_BYTES + 1)),
    {'name': 'garbage.jpg', 'data': '!!!not base64!!!'},
    {'name': 'empty.jpg', 'data': ''},
])
check('a mixed upload keeps the usable photo', mixed.success and len(mixed.data['paths']) == 1)
check('the oversized photo is refused individually',
      any(r['name'] == 'huge.jpg' for r in mixed.data['rejected']))
check('the oversize reason names the limit',
      any('MB' in r['reason'] for r in mixed.data['rejected'] if r['name'] == 'huge.jpg'))
check('undecodable and empty photos are refused too',
      {'garbage.jpg', 'empty.jpg'} <= {r['name'] for r in mixed.data['rejected']})

all_bad = handlers.handle_upload_target(config, [{'name': 'x.jpg', 'data': ''}])
check('an upload with nothing usable fails', not all_bad.success)
check('the failure names the photo', 'x.jpg' in (all_bad.error or ''))

check('an empty upload fails', not handlers.handle_upload_target(config, []).success)


# ── get_photo_results ─────────────────────────────────────────────────────
print('\nget_photo_results')

swapped_path = write_photo('returned.png')
pipeline_stub = MagicMock()
pipeline_stub.photo_results = [
    PhotoResult.swapped(os.path.join(WORK, 'returned_src.png'), swapped_path, 1),
    PhotoResult.skipped(os.path.join(WORK, 'refused.png'), 'no face detected'),
]

response = handlers.handle_get_photo_results(pipeline_stub)
check('results are returned', response.success)
check('one entry per photo', response.data['total'] == 2)
check('the counts split correctly',
      response.data['swapped'] == 1 and response.data['skipped'] == 1)

entries = response.data['results']
check('the swapped photo carries its bytes', bool(entries[0].get('data')))
check('the returned bytes are the output file',
      base64.b64decode(entries[0]['data']) == open(swapped_path, 'rb').read())
check('the skipped photo carries no bytes and keeps its reason',
      not entries[1].get('data') and entries[1]['reason'] == 'no face detected')

response = handlers.handle_get_photo_results(pipeline_stub, include_images=False)
check('images can be left out', not response.data['results'][0].get('data'))

pipeline_stub.photo_results = [
    PhotoResult.swapped(os.path.join(WORK, 'gone_src.png'),
                        os.path.join(WORK, 'gone_missing.png'), 1),
]
response = handlers.handle_get_photo_results(pipeline_stub)
check('an output that cannot be read back is reported as skipped',
      not response.data['results'][0]['ok'])

check('no pipeline is an error, not a crash',
      not handlers.handle_get_photo_results(None).success)


print('=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    for f in FAIL:
        print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """
    Surface the checks above to pytest as one assertion.

    Same shape as the rest of the suite: the body runs at import so the file
    stays runnable directly when a failure needs poking at, and this function
    is what makes it a pytest test without duplicating any of it.
    """
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
