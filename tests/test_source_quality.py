"""
Two ways a good source photo used to be refused, or a bad one accepted.

**Focus was measured on the whole image.** A portrait shot wide open has a
deliberately blurred background, and averaged over the frame that reads as an
out-of-focus photograph - so the composition that makes a portrait good was the
thing getting it refused. The same average carried the opposite error: a sharp
busy background floats a soft face over the floor, which is the case the guard
exists for. It is measured on the face now, at a canonical size, so the reading
no longer moves with how many pixels the camera spent on it either.

**Uploads were addressed by basename.** Phones produce `IMG_0001.jpg` by the
thousand, so two photos chosen from two folders collided: the second overwrote
the first and the review then saw the *same* image twice. Two identical images
agree perfectly, so the identity check waved them through, and the averaged
embedding was one photo counted twice with the other silently gone.

The threshold is checked here rather than asserted in a comment. `40.0` was a
guess, and a guess about a number that decides whether a customer's photo is
accepted deserves to be pinned against measured populations - so if the crop or
the normalisation is ever retuned, the failure lands here rather than on
somebody's upload.
"""

import os
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


for name in (
    'insightface', 'insightface.app', 'insightface.app.common',
    'insightface.model_zoo', 'insightface.utils', 'insightface.utils.face_align',
    'onnxruntime', 'torch', 'torchvision', 'psutil',
    'tensorflow', 'opennsfw2', 'gfpgan', 'onnx',
):
    sys.modules.setdefault(name, StubModule())

import logging

WORK = tempfile.mkdtemp(prefix='phantom-quality-test-')
os.environ['PHANTOM_TEMP_DIR'] = WORK

import cv2                                             # noqa: E402
import numpy as np                                     # noqa: E402

from pipeline.api import handlers                      # noqa: E402
from pipeline.config import FaceSwapConfig             # noqa: E402
from pipeline.services import guards                   # noqa: E402
from pipeline.types import Bbox, Detection             # noqa: E402

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print('  [{}] {}'.format(mark, label) + (' - {}'.format(detail) if detail else ''))


def photo_like(width: int, height: int, seed: int = 7):
    """
    Content that behaves like a photograph under blur.

    Deliberately not white noise, which was the first attempt: noise keeps an
    enormous Laplacian variance however much it is blurred, so it cannot tell a
    sharp picture from a soft one and every threshold looks like it passes. A
    photograph is mostly smooth ground with a limited number of real edges, and
    that is what makes blurring it actually remove high-frequency energy.
    """
    rng = np.random.default_rng(seed)
    scale = min(width, height)

    base = rng.integers(60, 200, (8, 8, 3), dtype=np.uint8)
    image = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)

    for _ in range(14):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(scale // 20, scale // 5)),
                int(rng.integers(scale // 20, scale // 5)))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.ellipse(image, centre, axes, float(rng.integers(0, 180)),
                    0, 360, colour, -1)
    for _ in range(10):
        start_point = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        end_point = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(image, start_point, end_point, colour, max(1, scale // 200))

    grain = rng.normal(0, 4.0, image.shape)
    return np.clip(image.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def portrait(width: int, height: int, box: Bbox,
             face_blur: float = 0.0, background_blur: float = 0.0):
    """A frame whose face region and background are blurred independently."""
    image = photo_like(width, height)
    face = image[box.y:box.y + box.h, box.x:box.x + box.w].copy()
    if background_blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), background_blur)
    if face_blur > 0:
        face = cv2.GaussianBlur(face, (0, 0), face_blur)
    image[box.y:box.y + box.h, box.x:box.x + box.w] = face
    return image


def detection_for(box: Bbox) -> Detection:
    """A detection whose pose is readable, so the yaw guard does not short-circuit."""
    face = MagicMock()
    face.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # pitch, yaw, roll
    return Detection(
        face=face, bbox=box,
        kps=np.zeros((5, 2), dtype=np.float32), confidence=0.9,
    )


print('=' * 70)
print('Source photo quality')
print('=' * 70)

# ── Focus is a property of the face ───────────────────────
print('\nWhere focus is measured')

config = FaceSwapConfig()
box = Bbox(x=300, y=200, w=400, h=400)

# The portrait: sharp face, background thrown out of focus on purpose.
good = portrait(1000, 800, box, background_blur=9.0)
# The photo the guard exists to catch: soft face, sharp busy background.
bad = portrait(1000, 800, box, face_blur=3.0)

face_good = guards.sharpness(good, box)
face_bad = guards.sharpness(bad, box)
frame_good = guards.sharpness(good)
frame_bad = guards.sharpness(bad)

check('a sharp face reads as sharp',
      face_good > config.guard_min_sharpness,
      'face {:.0f} vs floor {:.0f}'.format(face_good, config.guard_min_sharpness))
check('a soft face reads as soft',
      face_bad < config.guard_min_sharpness,
      'face {:.0f} vs floor {:.0f}'.format(face_bad, config.guard_min_sharpness))
check('the default floor sits between the two populations',
      face_bad < config.guard_min_sharpness < face_good,
      '{:.0f} < {:.0f} < {:.0f}'.format(
          face_bad, config.guard_min_sharpness, face_good))

check('the portrait passes the source guards',
      guards.check_source(config, good, [detection_for(box)]).ok)
refused = guards.check_source(config, bad, [detection_for(box)])
check('the soft face is refused', not refused.ok, refused.reason)
check('and refused for being blurred', refused.reason == guards.BLURRED,
      refused.reason)

# The frame reading is not simply wrong, it is wrong in *both* directions, and
# each one is a real complaint: a portrait is dragged down by the background it
# was composed to have, and a soft face is floated up by a background that has
# nothing to do with whether the person is in focus.
check('measuring the frame drags a sharp portrait down',
      frame_good < face_good,
      'frame {:.0f} against face {:.0f} - the blurred background is the point '
      'of the picture'.format(frame_good, face_good))
check('measuring the frame floats a soft face up',
      frame_bad > face_bad,
      'frame {:.0f} against face {:.0f} - a sharp background says nothing '
      'about the person'.format(frame_bad, face_bad))
check('and it inverts the two photos entirely',
      frame_bad > frame_good and face_bad < face_good,
      'by frame the soft photo ({:.0f}) beats the portrait ({:.0f})'.format(
          frame_bad, frame_good))

# ── And no longer moves with the photo's size ─────────────────
print('\nScale')

# One face, resampled - so the only variable is how many pixels it covers.
master = photo_like(1400, 1400)
readings = {}
for edge in (140, 256, 512, 1120):
    crop = cv2.resize(master, (edge, edge), interpolation=cv2.INTER_AREA)
    frame = np.zeros((edge + 40, edge + 40, 3), dtype=np.uint8)
    frame[20:20 + edge, 20:20 + edge] = crop
    # Against the raw Laplacian variance, which is what this measured before —
    # `guards.sharpness` normalises whether or not a box is passed, so
    # comparing it with itself would show nothing.
    raw = float(cv2.Laplacian(
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    readings[edge] = (guards.sharpness(frame, Bbox(x=20, y=20, w=edge, h=edge)), raw)

face_spread = (max(r[0] for r in readings.values())
               / max(1e-6, min(r[0] for r in readings.values())))
raw_spread = (max(r[1] for r in readings.values())
              / max(1e-6, min(r[1] for r in readings.values())))

check('every size clears the floor',
      min(r[0] for r in readings.values()) > config.guard_min_sharpness,
      ', '.join('{}px={:.0f}'.format(k, v[0]) for k, v in readings.items()))
check('normalising narrows the spread across an 8x size change',
      face_spread < raw_spread,
      'normalised {:.1f}x vs unnormalised {:.1f}x'.format(face_spread, raw_spread))

# ── Two photos with one name ──────────────────────────────────────────
print('\nColliding filenames')

job = os.path.join(WORK, 'names')
os.makedirs(job, exist_ok=True)

first = handlers._unique_name(job, 'IMG_0001.jpg')
open(os.path.join(job, first), 'wb').close()
second = handlers._unique_name(job, 'IMG_0001.jpg')
open(os.path.join(job, second), 'wb').close()
third = handlers._unique_name(job, 'IMG_0001.jpg')

check('the first keeps its name', first == 'IMG_0001.jpg', first)
check('the second is given a distinct one', second != first, second)
check('the third too', third not in (first, second), third)
check('the name a person would recognise survives',
      all(n.startswith('IMG_0001') and n.endswith('.jpg')
          for n in (first, second, third)),
      'shown back to them when a photo is refused')

# End to end: two different images, one filename, both uploaded.
import base64  # noqa: E402


def encoded(seed: int) -> str:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode('.png', image)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode('ascii')


upload_config = FaceSwapConfig()
reply = handlers.handle_upload_source(
    upload_config,
    [{'name': 'IMG_0001.png', 'data': encoded(1)},
     {'name': 'IMG_0001.png', 'data': encoded(2)}],
    None,
)
paths = reply.data.get('paths', [])
check('both images are saved', len(paths) == 2, str(paths))
check('to two different files', len(set(paths)) == 2, str(paths))
check('with different content',
      len({open(p, 'rb').read() for p in paths}) == 2,
      'the second used to overwrite the first and be averaged twice')

# A replacement upload does not leave the previous face on the pod.
old_dirs = {os.path.dirname(p) for p in paths}
handlers.handle_upload_source(
    upload_config, [{'name': 'other.png', 'data': encoded(3)}], None,
)
check('the superseded upload is removed',
      not any(os.path.isdir(d) for d in old_dirs),
      'replacing a source should not leave the old one on a rented machine')

print('=' * 70)
print('{} passed, {} failed'.format(len(PASS), len(FAIL)))
if FAIL:
    for failure in FAIL:
        print('  FAILED:', failure)
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
