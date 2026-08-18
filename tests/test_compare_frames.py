"""
Exercise the realism comparison script against frames with known properties.

The point is that each metric detects the thing it claims to. A metric that
merely produces a number is worse than none: it would be quoted in decisions.
So each case here builds a frame with one defect deliberately present and
checks the corresponding measure moves in the right direction.
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# conftest.py puts tools/ on the path under pytest; this covers a direct run.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _extra in (_REPO, os.path.join(_REPO, 'tools')):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import compare_frames as cf  # noqa: E402

PASS, FAIL = [], []


def check(label, condition, detail=''):
    (PASS if condition else FAIL).append(label)
    print('  [{}] {}'.format('PASS' if condition else 'FAIL', label)
          + (' - {}'.format(detail) if detail else ''))


RNG = np.random.default_rng(11)
H, W = 240, 320
CENTRE = (W // 2, H // 2)
RADIUS = 55


def base_frame(noise_sigma=6.0):
    """A textured frame with sensor-like noise, standing in for a webcam."""
    frame = np.full((H, W, 3), 118, np.uint8)
    # Low-frequency structure so the frame is not featureless.
    for _ in range(40):
        x, y = RNG.integers(0, W), RNG.integers(0, H)
        cv2.circle(frame, (int(x), int(y)), int(RNG.integers(12, 40)),
                   tuple(int(v) for v in RNG.integers(80, 170, 3)), -1)
    frame = cv2.GaussianBlur(frame, (0, 0), 2.0)
    noise = RNG.normal(0, noise_sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def circle_mask():
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, CENTRE, RADIUS, (255,), -1)
    return mask


def composite(frame, face_bgr, feather=9):
    """Paste `face_bgr` into `frame` through a feathered circular mask."""
    mask = circle_mask()
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather)
    alpha = np.clip(alpha, 0, 1)[:, :, None]
    out = frame.astype(np.float32) * (1 - alpha) + face_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def write_clip(directory, builder, count=24, motion=False):
    """Write `count` in/out pairs built by `builder(index) -> (src, out)`."""
    os.makedirs(directory, exist_ok=True)
    for index in range(count):
        src, out = builder(index, motion)
        cv2.imwrite(os.path.join(directory, '{:06d}_in.png'.format(index)), src)
        cv2.imwrite(os.path.join(directory, '{:06d}_out.png'.format(index)), out)
    return directory


WORK = tempfile.mkdtemp(prefix='phantom-compare-')

print('=' * 70)
print('Realism comparison script')
print('=' * 70)

# -- Face region derivation ---------------------------------------------
print('\nFace region')
src = base_frame()
face = cv2.GaussianBlur(src, (0, 0), 6.0)
out = composite(src, face)
mask = cf.face_mask(src, out)
check('the changed region is found without a detector', mask is not None)
if mask is not None:
    area = cv2.countNonZero(mask)
    expected = np.pi * RADIUS ** 2
    check('it is about the size of the pasted region',
          0.5 * expected < area < 1.8 * expected,
          '{} px vs {:.0f} expected'.format(area, expected))

check('identical frames yield no region',
      cf.face_mask(src, src.copy()) is None,
      'a frame with no swap must not be measured')

# -- Too clean ----------------------------------------------------------
print('\nToo clean (noise and detail)')


def smooth_face(index, motion):
    frame = base_frame(noise_sigma=7.0)
    # A denoised, slightly blurred face: exactly the "poreless" failure.
    quiet = cv2.GaussianBlur(cv2.medianBlur(frame, 7), (0, 0), 2.0)
    return frame, composite(frame, quiet)


def matched_face(index, motion):
    frame = base_frame(noise_sigma=7.0)
    # Same statistics as the frame: re-noised after smoothing.
    quiet = cv2.GaussianBlur(cv2.medianBlur(frame, 7), (0, 0), 2.0)
    grain = RNG.normal(0, 7.0, quiet.shape)
    noisy = np.clip(quiet.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return frame, composite(frame, noisy)


clean = cf.analyse(write_clip(os.path.join(WORK, 'clean'), smooth_face))
matched = cf.analyse(write_clip(os.path.join(WORK, 'matched'), matched_face))

check('a denoised face reads below 1.0 on noise',
      clean['noise_ratio']['p50'] < 0.8,
      'ratio {:.2f}'.format(clean['noise_ratio']['p50']))
check('a grain-matched face reads higher than a denoised one',
      matched['noise_ratio']['p50'] > clean['noise_ratio']['p50'],
      '{:.2f} vs {:.2f}'.format(matched['noise_ratio']['p50'],
                                clean['noise_ratio']['p50']))
check('the verdict names the too-clean failure',
      any('TOO CLEAN' in n for n in cf.verdict(clean)),
      [n for n in cf.verdict(clean) if 'TOO CLEAN' in n][:1])
check('a matched face is not flagged too clean',
      not any('TOO CLEAN' in n for n in cf.verdict(matched)))

check('high-frequency detail also drops for a blurred face',
      clean['hf_ratio']['p50'] < 1.0,
      'hf ratio {:.2f}'.format(clean['hf_ratio']['p50']))

# -- Seam ---------------------------------------------------------------
print('\nVisible seam')


def hard_edge(index, motion):
    frame = base_frame(noise_sigma=4.0)
    face = np.full_like(frame, 200)
    return frame, composite(frame, face, feather=0.5)


def soft_edge(index, motion):
    """Texture-matched face, feathered in - the genuine no-seam case.

    An earlier version pasted a *blurrier* face, which produces a real gradient
    step at the boundary: the script was right to flag it and the test was
    wrong. Matching the texture and re-rolling only the noise leaves a region
    that is detectably different pixel-by-pixel but statistically identical.
    """
    frame = base_frame(noise_sigma=4.0)
    regrained = np.clip(
        frame.astype(np.float32) + RNG.normal(0, 6.0, frame.shape), 0, 255,
    ).astype(np.uint8)
    return frame, composite(frame, regrained, feather=11)


seamed = cf.analyse(write_clip(os.path.join(WORK, 'seam'), hard_edge))
smooth = cf.analyse(write_clip(os.path.join(WORK, 'noseam'), soft_edge))

check('a hard edge reads above a feathered one',
      seamed['seam_ratio']['p95'] > smooth['seam_ratio']['p95'],
      '{:.2f} vs {:.2f}'.format(seamed['seam_ratio']['p95'],
                                smooth['seam_ratio']['p95']))
check('the verdict names a visible seam',
      any('VISIBLE SEAM' in n for n in cf.verdict(seamed)),
      'p95 {:.2f}'.format(seamed['seam_ratio']['p95']))
check('a feathered composite is not flagged',
      not any('VISIBLE SEAM' in n for n in cf.verdict(smooth)),
      'p95 {:.2f}'.format(smooth['seam_ratio']['p95']))

# -- Motion mismatch ----------------------------------------------------
print('\nMotion mismatch')


def smeared_frame_sharp_face(index, motion):
    """The frame is motion-blurred horizontally; the face is not."""
    frame = base_frame(noise_sigma=5.0)
    kernel = np.zeros((15, 15), np.float32)
    kernel[7, :] = 1.0 / 15.0
    smeared = cv2.filter2D(frame, -1, kernel)
    # Shift so consecutive frames differ enough to count as motion.
    matrix = np.float32([[1, 0, (index % 5) * 6], [0, 1, 0]])
    smeared = cv2.warpAffine(smeared, matrix, (W, H), borderMode=cv2.BORDER_REFLECT)
    sharp_face = cv2.warpAffine(frame, matrix, (W, H), borderMode=cv2.BORDER_REFLECT)
    return smeared, composite(smeared, sharp_face)


motion_clip = cf.analyse(
    write_clip(os.path.join(WORK, 'motion'), smeared_frame_sharp_face, count=30))

check('motion frames are identified',
      motion_clip['moving_frames'] >= 10,
      '{} of {}'.format(motion_clip['moving_frames'], motion_clip['frames']))
check('the frame reads more anisotropic than the face',
      motion_clip['anisotropy_out_motion']['p50']
      > motion_clip['anisotropy_in_motion']['p50'],
      'frame {:.2f} vs face {:.2f}'.format(
          motion_clip['anisotropy_out_motion']['p50'],
          motion_clip['anisotropy_in_motion']['p50']))
check('the verdict names the motion mismatch',
      any('MOTION MISMATCH' in n for n in cf.verdict(motion_clip)),
      [n for n in cf.verdict(motion_clip) if 'MOTION' in n][:1])

_STILL_BASE = base_frame(noise_sigma=7.0)


def still_face(index, motion):
    """One fixed scene, only the sensor noise changing between frames."""
    frame = np.clip(
        _STILL_BASE.astype(np.float32) + RNG.normal(0, 1.5, _STILL_BASE.shape),
        0, 255,
    ).astype(np.uint8)
    quiet = cv2.GaussianBlur(cv2.medianBlur(frame, 7), (0, 0), 2.0)
    grain = RNG.normal(0, 7.0, quiet.shape)
    noisy = np.clip(quiet.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return frame, composite(frame, noisy)


still = cf.analyse(write_clip(os.path.join(WORK, 'still'), still_face))
check('a still clip says so rather than guessing',
      any('Not enough motion' in n for n in cf.verdict(still)),
      '{} moving frames'.format(still['moving_frames']))

# -- Colour -------------------------------------------------------------
print('\nColour')


def tinted_face(index, motion):
    frame = base_frame(noise_sigma=5.0)
    tinted = frame.astype(np.int16).copy()
    tinted[:, :, 2] = np.clip(tinted[:, :, 2] + 40, 0, 255)   # push red
    return frame, composite(frame, tinted.astype(np.uint8))


tinted = cf.analyse(write_clip(os.path.join(WORK, 'tint'), tinted_face))
check('a colour-shifted face shows a LAB delta',
      abs(tinted['lab_delta']['a_mean']) > 3.0,
      'a_mean {:+.1f}'.format(tinted['lab_delta']['a_mean']))
check('a matched face shows a small LAB delta',
      abs(matched['lab_delta']['a_mean']) < 3.0,
      'a_mean {:+.1f}'.format(matched['lab_delta']['a_mean']))

# -- Plumbing -----------------------------------------------------------
print('\nPlumbing')
check('pairs are discovered', len(cf.load_pairs(os.path.join(WORK, 'clean'))) == 24)
check('limit is honoured', len(cf.load_pairs(os.path.join(WORK, 'clean'), 5)) == 5)

empty = os.path.join(WORK, 'empty')
os.makedirs(empty, exist_ok=True)
raised = False
try:
    cf.analyse(empty)
except SystemExit:
    raised = True
check('an empty directory fails clearly', raised)

no_swap = os.path.join(WORK, 'noswap')
os.makedirs(no_swap, exist_ok=True)
plain = base_frame()
cv2.imwrite(os.path.join(no_swap, '000001_in.png'), plain)
cv2.imwrite(os.path.join(no_swap, '000001_out.png'), plain)
raised = False
try:
    cf.analyse(no_swap)
except SystemExit as exc:
    raised = 'source face' in str(exc)
check('a clip with no swap says why', raised)

print('\n--- sample output ---')
cf.print_summary(clean)

print('\n' + '=' * 70)
print('{} passed, {} failed'.format(len(PASS), len(FAIL)))
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
