"""
Sending upstream at the rate the preset asked for.

The desktop sent every frame the camera produced, and on Windows the camera
ignores the rate it is asked for — 20 requested, 30 delivered — so the uplink
carried half again as many JPEGs as the preset's own budget assumes. That
direction is the asymmetric one on a home connection, and a saturated uplink
does not present as dropped frames, it presents as latency: frames queue in the
OS send buffer while throughput still looks healthy.

The whole reason this is a module with a test rather than two lines in the
capture loop is the aliasing trap. The obvious form — `now - last_sent >=
interval` — silently delivers *less* than asked for whenever the source rate is
not a multiple of the target: a 30fps camera against a 50ms interval sends at
0, 67, 133 and lands on 15fps, not 20. That is the same failure that ruled out
setting the rate on the device, arrived at from the other direction, and it
would have been invisible — the number on screen would just have been lower
than the preset claimed, which is what it already was.
"""

import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import random   # noqa: E402

from desktop.pacing import FramePacer   # noqa: E402

_NS = 1_000_000_000

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print('  [{}] {}'.format(mark, label) + (' - {}'.format(detail) if detail else ''))


def paced_rate(source_fps: float, target_fps: float, seconds: float = 10.0) -> float:
    """Frames per second actually sent, driven by an ideal camera."""
    pacer = FramePacer(target_fps)
    interval = _NS / source_fps
    sent = sum(
        1 for i in range(int(source_fps * seconds))
        if pacer.due(int(i * interval))
    )
    return sent / seconds


def naive_rate(source_fps: float, target_fps: float, seconds: float = 10.0) -> float:
    """What `now - last_sent >= interval` would have delivered."""
    interval = _NS / source_fps
    target_ns = _NS / target_fps
    last = None
    sent = 0
    for i in range(int(source_fps * seconds)):
        ts = int(i * interval)
        if last is None or ts - last >= target_ns:
            sent += 1
            last = ts
    return sent / seconds


print('=' * 70)
print('Uplink frame pacing')
print('=' * 70)

print('\nThe rate asked for is the rate sent')

for source, target in ((30, 20), (30, 15), (30, 30), (24, 20), (60, 20)):
    rate = paced_rate(source, target)
    check('{}fps camera, {}fps preset -> {:.1f}fps sent'.format(source, target, rate),
          abs(rate - target) <= 0.5,
          'wanted {}'.format(target))

print('\nA camera that already meets the target loses nothing')

# The target is a target, not a ratio. Nothing here assumes the camera runs at
# 30 - the decision is made against the rate the camera is measured to be
# running at, so one that honours its configuration is left alone entirely.
for source in (20, 19.5, 20.5, 21, 18):
    pacer = FramePacer(20)
    interval = _NS / source
    frames = int(source * 10)
    sent = sum(1 for i in range(frames) if pacer.due(int(i * interval)))
    check('a {}fps camera against a 20fps preset drops nothing'.format(source),
          sent == frames,
          'sent {} of {}, pacing={}'.format(sent, frames, pacer.pacing))

# Jitter is what a bare schedule gets wrong: a frame arriving a millisecond
# early is one the schedule has not reached yet, so a nominal 20fps camera
# would bleed frames for no gain at all.
random.seed(11)
pacer = FramePacer(20)
timestamp = 0
frames = 200
sent = 0
for _ in range(frames):
    if pacer.due(timestamp):
        sent += 1
    timestamp += int(_NS / 20 * random.uniform(0.88, 1.12))
check('a jittery 20fps camera still drops nothing',
      sent == frames,
      'sent {} of {} with +/-12% jitter'.format(sent, frames))

print('\nDropping starts only once the camera really is faster')

for source, expected in ((20, False), (21, False), (24, True), (30, True)):
    pacer = FramePacer(20)
    interval = _NS / source
    for i in range(60):
        pacer.due(int(i * interval))
    check('{}fps camera -> dropping {}'.format(
              source, 'on' if expected else 'off'),
          pacer.pacing is expected,
          'measured {:.1f}fps'.format(pacer.observed_fps))

print('\nWhat the obvious version would have done')

for source, target in ((30, 20), (24, 20)):
    naive = naive_rate(source, target)
    check('the naive form aliases {}fps down to {:.1f} instead of {}'.format(
              source, naive, target),
          naive < target - 1.0,
          'asking for less would have got less than asked')

print('\nEdges')

slow = paced_rate(15, 20)
check('a camera slower than the preset sends everything it has',
      abs(slow - 15) <= 0.5,
      '{:.1f}fps - frames cannot be invented'.format(slow))

pacer = FramePacer(0)
check('a rate of zero disables pacing',
      all(pacer.due(i * 1000) for i in range(10)),
      'the previous behaviour stays reachable rather than being removed')

# A stall must not be repaid as a burst into the link that just stalled. The
# schedule is five seconds behind at this point, and paying that off literally
# means sending every frame that arrives until it catches up.
pacer = FramePacer(20)
for i in range(10):
    pacer.due(int(i * _NS / 30))
resume = int(5 * _NS)                      # five seconds of nothing
after = [pacer.due(resume + int(i * _NS / 30)) for i in range(30)]

check('a stall is resynchronised, not caught up on',
      sum(after) <= 21,
      '{} of 30 frames sent after a 5s gap - catching up would send all 30, '
      'into the link that just failed'.format(sum(after)))
check('and the normal rate resumes immediately',
      abs(sum(after) / 1.0 - 20) <= 1.0,
      '{:.0f}fps in the first second back'.format(sum(after)))

pacer = FramePacer(20)
pacer.due(0)
pacer.reset()
check('reset makes the next frame due immediately', pacer.due(0))

print('\nAccounting')

pacer = FramePacer(20)
for i in range(300):
    pacer.due(int(i * _NS / 30))
check('sent and skipped are counted',
      pacer.sent + pacer.skipped == 300,
      'sent={} skipped={}'.format(pacer.sent, pacer.skipped))
check('and settle on the target once the camera has been measured',
      abs(pacer.sent - 200) <= 10,
      'sent {} of 300 frames from a 30fps camera over 10s; the first few are '
      'the observation window, where nothing is dropped'.format(pacer.sent))

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
