"""
Exercise the bundled template library and the face it names.

A template is a target *we* ship, so the failures worth testing are the ones a
user would have no way to work around:

1. **The named face is the one that gets swapped.** `face_point` is stored as a
   normalised point rather than an index precisely because detection order is
   not a stable contract. If proximity matching regressed, a template would
   quietly swap the wrong person in a group scene — no error, no crash, just
   somebody else's face.
2. **A named face answers the multi-face guard.** The guard refuses a crowd
   because the question has no safe default. Once the template's author has
   answered it, a scene we chose on purpose must not be refused.
3. **Selecting a template cannot leak into the next job**, and nothing writes
   into the shared library directory.
"""

import os
import sys
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import json
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
from pipeline.config import FaceSwapConfig
from pipeline.services import guards, templates
from pipeline.types import Bbox, Detection

logging.disable(logging.INFO)

WORK = tempfile.mkdtemp(prefix='phantom-template-test-')
LIB = os.path.join(WORK, 'library')
os.makedirs(LIB, exist_ok=True)

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print(f'  [{mark}] {label}' + (f' - {detail}' if detail else ''))


def write_image(name: str, width: int = 800, height: int = 600) -> str:
    path = os.path.join(LIB, name)
    rng = np.random.default_rng(7)
    cv2.imwrite(path, rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    return path


def write_rgba(name: str, width: int = 800, height: int = 600) -> str:
    path = os.path.join(LIB, name)
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[:, :, 2] = 255           # red, so its presence is provable
    layer[100:200, 100:200, 3] = 255   # opaque only in that square
    cv2.imwrite(path, layer)
    return path


def detection(x: int, y: int, size: int = 100) -> Detection:
    """
    A Detection with geometry the runtime guards will accept.

    Keypoints are laid out the way `test_guards.make_detection` does — a
    frontal face with the nose centred between the eyes — because the guard
    chain runs past the multi-face check to confidence, size and pose, and a
    stub without real keypoints fails on pose for reasons unrelated to what is
    being tested here.
    """
    eye_y = y + size * 0.4
    left_x = x + size * 0.3
    right_x = x + size * 0.7
    kps = np.array([
        [left_x, eye_y],
        [right_x, eye_y],
        [(left_x + right_x) / 2.0, y + size * 0.55],
        [x + size * 0.35, y + size * 0.75],
        [x + size * 0.65, y + size * 0.75],
    ], dtype=np.float32)

    face = MagicMock()
    face.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
    face.kps = kps
    return Detection(
        face=face,
        bbox=Bbox(x=x, y=y, w=size, h=size),
        kps=kps,
        confidence=0.9,
    )


def write_manifest(entries: list) -> None:
    with open(os.path.join(LIB, templates.MANIFEST_NAME), 'w', encoding='utf-8') as fh:
        json.dump({'version': 1, 'templates': entries}, fh)


print('=' * 70)
print('Bundled templates')
print('=' * 70)


# ── The library ───────────────────────────────────────────────────────────
print('\nLoading the library')

write_image('scene.jpg')
write_image('crowd.jpg')
write_rgba('scene_fg.png')
write_manifest([
    {'id': 'scene', 'name': 'A scene', 'image': 'scene.jpg',
     'face_point': [0.25, 0.5], 'foreground': 'scene_fg.png', 'credit': 'test'},
    {'id': 'crowd', 'name': 'A crowd', 'image': 'crowd.jpg'},
    {'id': 'ghost', 'name': 'Missing file', 'image': 'nope.jpg'},
    {'id': '', 'name': 'No id', 'image': 'scene.jpg'},
])

library = templates.TemplateLibrary(LIB)
entries = library.all()

check('usable templates load', len(entries) == 2, str(len(entries)))
check('a template naming a missing image is skipped',
      library.get('ghost') is None)
check('an entry with no id is skipped',
      all(t.id for t in entries))
check('face_point is parsed', library.get('scene').face_point == (0.25, 0.5))
check('foreground resolves to a path',
      library.get('scene').foreground.endswith('scene_fg.png'))
check('thumbnail falls back to the image',
      library.get('crowd').thumbnail == library.get('crowd').image)
check('a template with no face_point keeps None',
      library.get('crowd').face_point is None)
check('an unknown id returns None', library.get('nope') is None)

check('a directory with no manifest is an empty library, not an error',
      templates.TemplateLibrary(os.path.join(WORK, 'nothing')).all() == [])

broken = os.path.join(WORK, 'broken')
os.makedirs(broken, exist_ok=True)
with open(os.path.join(broken, templates.MANIFEST_NAME), 'w', encoding='utf-8') as fh:
    fh.write('{not json')
check('an unreadable manifest is an empty library, not a crash',
      templates.TemplateLibrary(broken).all() == [])


# ── Picking the named face ────────────────────────────────────────────────
print('\nThe named face is the one chosen')

shape = (600, 800)  # height, width
left = detection(100, 250)     # centre ~(150, 300) -> normalised (0.19, 0.5)
right = detection(600, 250)    # centre ~(650, 300) -> normalised (0.81, 0.5)
faces = [left, right]

chosen = templates.select_by_point(faces, (0.19, 0.5), shape)
check('a point inside a box picks that face', chosen is left)

chosen = templates.select_by_point(faces, (0.81, 0.5), shape)
check('the other point picks the other face', chosen is right)

check('order does not decide it',
      templates.select_by_point(list(reversed(faces)), (0.19, 0.5), shape) is left)

chosen = templates.select_by_point(faces, (0.10, 0.1), shape)
check('a point outside every box falls back to the nearest', chosen is left)

check('no point means no opinion',
      templates.select_by_point(faces, None, shape) is None)
check('no detections means no choice',
      templates.select_by_point([], (0.5, 0.5), shape) is None)


# ── The guard ─────────────────────────────────────────────────────────────
print('\nA named face answers the multi-face guard')

config = FaceSwapConfig()
config.target_face_point = None
verdict = guards.check_frame(config, faces)
check('a crowd with no named face is refused', not verdict.ok)
check('and says why', verdict.reason == guards.MULTIPLE_FACES, verdict.reason)

config.target_face_point = (0.19, 0.5)
verdict = guards.check_frame(config, faces)
check('a crowd with a named face is allowed', verdict.ok,
      verdict.message if not verdict.ok else '')

config.target_face_point = None
check('one face is still fine either way', guards.check_frame(config, [left]).ok)


# ── The foreground layer ──────────────────────────────────────────────────
print('\nThe foreground layer')

frame = np.zeros((600, 800, 3), dtype=np.uint8)
out = templates.composite_foreground(frame, os.path.join(LIB, 'scene_fg.png'))
check('the opaque region is drawn', bool((out[150, 150] == [0, 0, 255]).all()),
      str(out[150, 150]))
check('the transparent region is untouched', bool((out[400, 400] == [0, 0, 0]).all()),
      str(out[400, 400]))

check('an unreadable layer leaves the frame alone',
      bool((templates.composite_foreground(frame, os.path.join(WORK, 'nope.png'))
            == frame).all()))

no_alpha = os.path.join(WORK, 'no_alpha.png')
cv2.imwrite(no_alpha, np.zeros((600, 800, 3), dtype=np.uint8))
check('a layer with no alpha leaves the frame alone',
      bool((templates.composite_foreground(frame, no_alpha) == frame).all()))

small = os.path.join(WORK, 'small.png')
cv2.imwrite(small, np.dstack([
    np.full((300, 400, 3), 255, dtype=np.uint8),
    np.full((300, 400), 255, dtype=np.uint8),
]))
check('a mismatched layer is resized rather than refused',
      templates.composite_foreground(frame, small).shape == frame.shape)


# ── The API ───────────────────────────────────────────────────────────────
print('\nset_template / list_templates')

handlers._UPLOAD_DIR = os.path.join(WORK, 'uploads')
_real_library = templates.TemplateLibrary


def fake_library(directory=None):
    return _real_library(LIB)


handlers.TemplateLibrary = fake_library

listed = handlers.handle_list_templates()
check('templates are listed', listed.success)
check('the count matches the library', listed.data['count'] == 2)
check('thumbnails travel inline',
      all(e.get('thumbnail') for e in listed.data['templates']))
check('server paths are not exposed',
      all('image' not in e for e in listed.data['templates']))

config = FaceSwapConfig()
response = handlers.handle_set_template(config, 'scene')
check('a template can be selected', response.success, response.error or '')
check('the template becomes the target', config.target_paths == [library.get('scene').image])
check('it runs as a photo job of one', config.target_path is None)
check('the named face is applied', config.target_face_point == (0.25, 0.5))
check('the foreground is applied', config.target_foreground is not None)
check('outputs go outside the library',
      config.output_dir is not None and not config.output_dir.startswith(LIB))

check('an unknown template is refused',
      not handlers.handle_set_template(config, 'nope').success)
check('an empty id is refused',
      not handlers.handle_set_template(config, '').success)

# The leak that would be invisible: a template's face point steering an
# unrelated target, and its foreground being drawn over someone else's photo.
plain = write_image('plain_target.jpg')
handlers.handle_set_template(config, 'scene')
handlers.handle_set_target(config, plain)
check('choosing a file clears the template', config.template_id is None)
check('choosing a file clears the named face', config.target_face_point is None)
check('choosing a file clears the foreground', config.target_foreground is None)
check('choosing a file clears the output directory', config.output_dir is None)

handlers.handle_set_template(config, 'scene')
import base64
handlers.handle_upload_target(config, [
    {'name': 'p.jpg', 'data': base64.b64encode(b'bytes').decode('ascii')},
])
check('uploading photos clears the template', config.template_id is None)
check('uploading photos clears the named face', config.target_face_point is None)
check('uploading photos clears the foreground', config.target_foreground is None)


# ── A template job still starts ───────────────────────────────────────────
print('\nA template job needs no output path')

config = FaceSwapConfig()
config.source_path = write_image('src.jpg')
handlers.handle_set_template(config, 'scene')

pipeline_stub = MagicMock()
pipeline_stub.is_running.return_value = False
response = handlers.handle_start(config, pipeline_stub)
check('start accepts a template job with no output path', response.success,
      response.error or '')


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
