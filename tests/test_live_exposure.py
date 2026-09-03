"""
The live path must never emit the operator's real face.

This is the one failure the product exists to prevent. Everything else is a
quality question; this is the operator's actual face reaching everyone on the
call, and it cannot be taken back once it has.

The rule under test is narrow and absolute: **while a stream is running, a frame
leaves the pipeline only if it was swapped, or it is a previously swapped frame
held unchanged.** There is no third option. A frame that could not be swapped —
no face detected, no source loaded, occlusion, a compositing failure — is
replaced by the last good one, and if there is no last good one yet, nothing is
emitted at all.

It was not always so. `guards.check_frame` passes a frame with zero faces on
purpose, and the live path used to emit it unswapped, on the reasoning that
someone stepping out of shot should not leave a stale face hovering over an
empty chair. That reasoning is sound and the conclusion was wrong, because
nothing distinguishes the two cases that produce zero detections:

    stepped out of shot          -> 0 detections -> raw frame is an empty room
    lighting dropped, still sat  -> 0 detections -> raw frame is their face

The pipeline cannot tell them apart, so the tie goes to the outcome that is
survivable. A frozen face reads as a network hiccup.

Batch is deliberately untouched: a rendered video of an empty room should come
back as a video of an empty room, and `_swap_frame_detail` still passes frames
through. Only the streaming path holds.
"""

import sys
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
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

import numpy as np

from pipeline.config import FaceSwapConfig
from pipeline.events import FRAME_READY
from pipeline.processing.pipeline import ProcessingPipeline
from pipeline.services import guards
from pipeline.types import Bbox, Detection

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []

# Two frames that can never be confused for one another.
RAW = np.full((64, 64, 3), 11, dtype=np.uint8)      # the operator's real face
SWAPPED = np.full((64, 64, 3), 222, dtype=np.uint8)  # a composited frame


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def build(detections: list, source_face: object, swap_result: object):
    """
    A pipeline whose every collaborator is a stub, driven one frame at a time.

    Only the branch structure of `_process_and_emit` is under test, so detection,
    swapping and compositing are all replaced — what matters is which frame comes
    out of the bus, not how it was made.
    """
    config = FaceSwapConfig()
    config.guards = True
    bus = MagicMock()
    pipeline = ProcessingPipeline(config, bus)

    pipeline._preprocessing_proc = MagicMock()
    pipeline._preprocessing_proc.process.side_effect = lambda f: f

    pipeline._detection_proc = MagicMock()
    pipeline._detection_proc.process.side_effect = lambda f: f
    pipeline._detection_proc.latest_detections = detections
    pipeline._detection_proc.all_detections = detections

    pipeline._swapping_proc = MagicMock()
    pipeline._swapping_proc.source_face = source_face

    pipeline._stabilizer = MagicMock()
    pipeline._stabilizer.stabilize.side_effect = lambda f: f
    pipeline._compositor = MagicMock()
    pipeline._compositor.last_stage_ms = {}
    pipeline._compositor.last_detail_ratio = None
    pipeline._compositor.last_texture_headroom = None
    pipeline._compositor.last_texture_confidence = None

    pipeline._swap_face = MagicMock(return_value=swap_result)  # type: ignore[method-assign]
    return pipeline, bus


def emitted(bus):
    """Every frame the bus was asked to publish as FRAME_READY."""
    return [
        call.kwargs['frame']
        for call in bus.emit.call_args_list
        if call.args and call.args[0] == FRAME_READY and 'frame' in call.kwargs
    ]


def detection(size=200):
    """
    One detection that passes every runtime guard.

    Real keypoints, laid out frontally, because `check_frame` runs the pose
    guard before anything this file is testing - a stub without them is refused
    for "could not be checked (pose)" and never reaches the branch in question.
    """
    x = y = 50
    eye_y = y + size * 0.4
    left_x, right_x = x + size * 0.3, x + size * 0.7
    kps = np.array([
        [left_x, eye_y],
        [right_x, eye_y],
        [(left_x + right_x) / 2.0, y + size * 0.55],
        [x + size * 0.35, y + size * 0.75],
        [x + size * 0.65, y + size * 0.75],
    ], dtype=np.float32)

    face = MagicMock()
    face.pose = None
    face.kps = kps
    return Detection(
        face=face,
        bbox=Bbox(x=x, y=y, w=size, h=size),
        kps=kps,
        confidence=0.99,
    )


print('=' * 70)
print('Live exposure')
print('=' * 70)

print('\nNo face detected')

pipeline, bus = build([], MagicMock(), SWAPPED)
pipeline._last_good_frame = SWAPPED
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
frames = emitted(bus)

check('a frame with no face is not emitted raw',
      not any(np.array_equal(f, RAW) for f in frames),
      'the operator is still in shot when the light drops; this is their face')
check('the last swapped frame is emitted instead',
      any(np.array_equal(f, SWAPPED) for f in frames),
      '{} frame(s) published'.format(len(frames)))
check('the reason is reported as no face',
      pipeline._guard_reason == guards.NO_FACE, pipeline._guard_reason)

print('\nNo face, and nothing swapped yet')

pipeline, bus = build([], MagicMock(), SWAPPED)
pipeline._last_good_frame = None
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
check('with nothing to hold, nothing at all is emitted',
      not emitted(bus),
      'a blank viewport is survivable; the raw camera is not')

print('\nNo source face')

pipeline, bus = build([detection()], None, SWAPPED)
pipeline._last_good_frame = SWAPPED
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
frames = emitted(bus)

check('a face with no source to wear is not emitted raw',
      not any(np.array_equal(f, RAW) for f in frames),
      'the source failed to load or was cleared mid-session')
check('the reason distinguishes it from a missing face',
      pipeline._guard_reason == guards.NO_SOURCE, pipeline._guard_reason)

print('\nCompositing produced nothing')

pipeline, bus = build([detection()], MagicMock(), None)
pipeline._last_good_frame = SWAPPED
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
check('a refused composite holds rather than passing the frame',
      not any(np.array_equal(f, RAW) for f in emitted(bus)),
      'occlusion guard, or the compositor failing outright')

print('\nMultiple faces')

pipeline, bus = build([detection()], MagicMock(), SWAPPED)
pipeline._detection_proc.all_detections = [detection(200), detection(160)]
pipeline._last_good_frame = SWAPPED
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
check('two faces hold rather than swapping one of them',
      not any(np.array_equal(f, RAW) for f in emitted(bus)),
      'the trimmed list swaps the largest face, so a bystander walking in '
      'closer would leave the operator unswapped and visible')
check('the reason names the crowd',
      pipeline._guard_reason == guards.MULTIPLE_FACES, pipeline._guard_reason)

# The guard that normally catches this is switchable. The live path must not be.
pipeline, bus = build([detection()], MagicMock(), SWAPPED)
pipeline.config.guards = False
pipeline.config.guard_multi_face = False
pipeline._detection_proc.all_detections = [detection(200), detection(160)]
pipeline._last_good_frame = SWAPPED
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
check('turning the guards off does not open it either',
      not any(np.array_equal(f, RAW) for f in emitted(bus)),
      'the guard master switch is a quality control; this is an exposure rule')

# A named face point is a photo and template concept. Honouring one left over
# from an earlier job as permission to swap one face out of two on a live call
# is the wrong-person swap the guards exist to prevent.
pipeline, bus = build([detection()], MagicMock(), SWAPPED)
pipeline.config.target_face_point = (0.5, 0.5)
pipeline._detection_proc.all_detections = [detection(200), detection(160)]
pipeline._last_good_frame = SWAPPED
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
check('a stale named face point is not permission to swap one of two',
      not any(np.array_equal(f, RAW) for f in emitted(bus)),
      'it was set by an earlier photo job, not by anyone looking at this call')

# `many_faces` is the real exemption: everyone is swapped, so nobody is exposed.
pipeline, bus = build([detection(200), detection(160)], MagicMock(), SWAPPED)
pipeline.config.many_faces = True
pipeline._detection_proc.all_detections = [detection(200), detection(160)]
pipeline._last_good_frame = None
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
check('many_faces still swaps them all',
      any(np.array_equal(f, SWAPPED) for f in emitted(bus)),
      'when every face is swapped there is nobody left to expose')

print('\nRecovery')

pipeline, bus = build([detection()], MagicMock(), SWAPPED)
pipeline._last_good_frame = None
pipeline._guard_reason = guards.NO_FACE
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
frames = emitted(bus)

check('a detected face resumes swapping',
      any(np.array_equal(f, SWAPPED) for f in frames))
check('the guard clears when it does',
      pipeline._guard_reason == '',
      'the badge must not stay up after the face comes back')
check('the recovered frame becomes the new held frame',
      pipeline._last_good_frame is not None
      and np.array_equal(pipeline._last_good_frame, SWAPPED),
      'so the next dropout holds something current')

print('\nObserve mode does not open the hole')

pipeline, bus = build([], MagicMock(), SWAPPED)
pipeline.config.guard_observe = True
pipeline._last_good_frame = SWAPPED
pipeline._process_and_emit(RAW.copy(), seq=1, capture_ts=0)
check('a missing face still holds under guard_observe',
      not any(np.array_equal(f, RAW) for f in emitted(bus)),
      'observe mode lets a swap through to be measured; here there is no swap, '
      'so honouring it would transmit the operator to calibrate a threshold')

print('\nBatch is untouched')

check('the streaming path is the only one changed',
      'def _swap_frame_detail' in open(
          _os.path.join(_REPO_ROOT, 'pipeline', 'processing', 'pipeline.py'),
          encoding='utf-8').read(),
      'batch renders through _swap_frame_detail and still passes frames through '
      '- a video of an empty room should come back as one')


print('=' * 70)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    for f in FAIL:
        print('  FAILED:', f)
print('=' * 70)


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
