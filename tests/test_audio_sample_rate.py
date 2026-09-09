"""
One sample rate, taken from the device that cannot bend.

The path is mic -> ring buffer -> virtual cable, and every stage has to agree,
because a ring buffer filled at one rate and drained at another is a pitch
shift plus a steady drift. It used to agree by all three reading the same
constant, and the constant was wrong: 44.1 kHz against a VB-CABLE endpoint
configured at 48, so playback could not open at all.

What makes that worth a test rather than a new constant is *why* it broke.
`find_virtual_output` picks the lowest-latency instance of the cable, which on
Windows is the WASAPI one, and WASAPI in shared mode will not resample - ask it
for a rate its endpoint is not set to and the stream refuses. MME and
DirectSound resample silently, so before that selection landed the mismatch was
hidden by the same slow path that cost 88ms. The faster device did not
introduce this; it stopped covering it up.

So changing 44100 to 48000 fixes one machine and breaks the next one whose
cable is set to 44.1 - the endpoint's rate is a dropdown in the Windows sound
control panel and none of it is ours to assume. These pin the asking.

They also pin the third rate the first diagnosis missed: `VoiceTransformer`
carried its own literal 44100 and `Bridge` built it with no argument, so it was
the one place the rate was not shared. Praat is told that number, and getting
it wrong does not fail - it biases pitch detection by the ratio and quietly
degrades the transform.
"""

import os
import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import io
import contextlib
from unittest.mock import MagicMock


class StubModule(MagicMock):
    """MagicMock that also satisfies `from x.y import z` for nested paths."""

    __path__: list = []


for name in ('parselmouth', 'parselmouth.praat'):
    sys.modules.setdefault(name, StubModule())

import logging  # noqa: E402

from desktop import audio                                  # noqa: E402
from desktop.audio import (                                # noqa: E402
    DEFAULT_SAMPLE_RATE, AudioCapture, AudioPlayback,
    device_sample_rate, resolve_sample_rate,
)
from desktop.voice import VoiceTransformer                 # noqa: E402

logging.disable(logging.INFO)

PASS: list = []
FAIL: list = []


def check(label: str, condition: bool, detail: str = '') -> None:
    (PASS if condition else FAIL).append(label)
    mark = 'PASS' if condition else 'FAIL'
    print('  [{}] {}'.format(mark, label) + (' - {}'.format(detail) if detail else ''))


class FakeSoundDevice:
    """
    Just enough of `sounddevice` to answer the two questions this asks.

    `input_ok` is what a microphone that refuses the cable's rate looks like:
    `check_input_settings` raises, which is exactly how the real module reports
    it without opening a stream.
    """

    def __init__(self, devices, input_ok=True):
        self._devices = devices
        self._input_ok = input_ok

    def query_devices(self, index=None):
        if index is None:
            return self._devices
        return self._devices[index]

    def check_input_settings(self, **kwargs):
        if not self._input_ok:
            raise ValueError('Invalid sample rate')


def install(devices, input_ok=True):
    sys.modules['sounddevice'] = FakeSoundDevice(devices, input_ok)


def capture_stderr(fn):
    """Run `fn`, returning (result, whatever it printed to stderr)."""
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        result = fn()
    return result, buffer.getvalue()


CABLE_48 = {
    'name': 'CABLE Input (VB-Audio Virtual Cable)',
    'max_output_channels': 2, 'max_input_channels': 0,
    'default_low_output_latency': 0.002, 'default_samplerate': 48000.0,
}
CABLE_44 = dict(CABLE_48, default_samplerate=44100.0)
CABLE_SILENT = dict(CABLE_48)
CABLE_SILENT.pop('default_samplerate')

print('=' * 70)
print('Audio sample rate')
print('=' * 70)

# ── The device is asked, not assumed ──────────────────────────────────
print('\nWhere the rate comes from')

install([CABLE_48])
check('a 48 kHz cable gives 48 kHz',
      resolve_sample_rate(0) == 48000)

install([CABLE_44])
rate_44 = resolve_sample_rate(0)
check('a 44.1 kHz cable gives 44.1 kHz, not the default',
      rate_44 == 44100,
      'got {} - this is the machine a hardcoded 48000 would break'.format(rate_44))

install([CABLE_SILENT])
result, said = capture_stderr(lambda: resolve_sample_rate(0))
check('a device that reports nothing falls back to the default',
      result == DEFAULT_SAMPLE_RATE, str(result))
check('and says so rather than failing silently',
      'did not report' in said, said.strip())

install([CABLE_48])
check('device_sample_rate reads the raw field',
      device_sample_rate(0) == 48000)
check('and returns None when there is nothing to read',
      device_sample_rate(99) is None,
      'an out-of-range index must not raise into the caller')

# ── A microphone that cannot follow is named ──────────────────────────
print('\nWhen the microphone disagrees')

install([CABLE_48], input_ok=False)
result, said = capture_stderr(lambda: resolve_sample_rate(0))

check('the output device still decides',
      result == 48000,
      'it is the one with no alternative - WASAPI shared mode will not resample')
check('the disagreement is reported', said.strip() != '')
check('and names the rate that is wanted', '48000' in said, said.strip())
check('and names the fix', 'sound settings' in said.lower(), said.strip())

# ── The default is a fallback, and it is 48 kHz ───────────────────────
print('\nThe fallback')

check('the default is 48000',
      DEFAULT_SAMPLE_RATE == 48000,
      'virtual cables overwhelmingly want 48; 44.1 was a legacy assumption')

# ── One number, not three ─────────────────────────────────────────────
print('\nOne number')

capture_default = AudioCapture.__init__.__defaults__
playback_default = AudioPlayback.__init__.__defaults__
voice_default = VoiceTransformer.__init__.__defaults__

check('capture defaults to the shared constant',
      DEFAULT_SAMPLE_RATE in capture_default, str(capture_default))
check('playback defaults to the shared constant',
      DEFAULT_SAMPLE_RATE in playback_default, str(playback_default))
check('the voice transformer defaults to it too',
      voice_default == (DEFAULT_SAMPLE_RATE,),
      'it carried its own literal 44100, which nothing updated when the '
      'others changed: {}'.format(voice_default))

# The rate the transformer is told is the rate Praat is told.
transformer = VoiceTransformer(sample_rate=48000)
check('and it keeps the rate it was given',
      transformer._sample_rate == 48000,
      'parselmouth.Sound is handed this; a wrong value biases every pitch '
      'reading by the ratio rather than raising')

# ── A refused stream says what the device wanted ──────────────────────
print('\nWhen a stream will not open')


class RefusingSoundDevice(FakeSoundDevice):
    """A device that reports 48 kHz and refuses anything else."""

    def InputStream(self, **kwargs):
        if kwargs.get('samplerate') != 48000:
            raise ValueError('Invalid sample rate')
        return MagicMock()

    def OutputStream(self, **kwargs):
        if kwargs.get('samplerate') != 48000:
            raise ValueError('Invalid sample rate')
        return MagicMock()


sys.modules['sounddevice'] = RefusingSoundDevice([CABLE_48])

wrong = AudioCapture(device=0, sample_rate=44100)
_, said = capture_stderr(wrong.start)
check('a refused capture names the rate asked for',
      '44100' in said, said.strip())
check('and the rate the device actually wants',
      '48000' in said,
      'without both, this is "audio does not work" and the next step is '
      'guessing at devices')

playback = AudioPlayback(
    audio.AudioRingBuffer(10, 48000), audio.JitterBuffer(),
    sample_rate=44100, device=0,
)
_, said = capture_stderr(playback.start)
check('a refused playback does the same',
      '44100' in said and '48000' in said, said.strip())

# ── The input device the cable's own output steals ─────────────────────
# The failure this section exists for is silent and total. The delayed audio is
# written into a virtual cable so the conferencing app can pick that cable as
# its microphone; the natural way to set that up is to *also* make the cable's
# output the system default recording device, at which point capture reads from
# the same cable playback writes into. A loop with no microphone in it, a stream
# that looks entirely healthy, and a call that receives silence. Measured on the
# development machine: the default input was `CABLE Output (VB-Audio Virtual
# Cable)` at an RMS of 0.000015 against the real microphone's 0.0093.
print('\nThe input the cable steals')

from desktop.audio import (                                        # noqa: E402
    find_real_input, is_virtual_input, resolve_input_device,
)

MIC = {
    'name': 'Microphone Array (Realtek Audio)',
    'max_input_channels': 2, 'max_output_channels': 0,
    'default_low_input_latency': 0.01, 'default_samplerate': 48000.0,
}
MIC_SLOW = dict(MIC, default_low_input_latency=0.09)
CABLE_OUT = {
    'name': 'CABLE Output (VB-Audio Virtual Cable)',
    'max_input_channels': 2, 'max_output_channels': 0,
    'default_low_input_latency': 0.01, 'default_samplerate': 44100.0,
}
MAPPER = {
    'name': 'Microsoft Sound Mapper - Input',
    'max_input_channels': 2, 'max_output_channels': 0,
    'default_low_input_latency': 0.0, 'default_samplerate': 44100.0,
}


class DeviceStub(FakeSoundDevice):
    """`FakeSoundDevice` plus the default device pair this asks about."""

    def __init__(self, devices, default_input):
        super().__init__(devices)
        self.default = type('D', (), {'device': (default_input, 0)})()


def with_devices(devices, default_input):
    sys.modules['sounddevice'] = DeviceStub(devices, default_input)


check('a cable output is recognised as not a microphone',
      is_virtual_input('CABLE Output (VB-Audio Virtual Cable)'))
check('and so are the other cables people install',
      all(is_virtual_input(n) for n in
          ('VoiceMeeter Output (VB-Audio)', 'BlackHole 2ch', 'Soundflower (2ch)')))
check('a real microphone is not',
      not is_virtual_input('Microphone Array (Realtek Audio)'))
check("nor is anything merely called virtual, nor Linux's pulse",
      not is_virtual_input('Virtual Desktop Audio')
      and not is_virtual_input('pulse'),
      'a false positive here refuses a real microphone, which is the very '
      'failure this is trying to prevent')

with_devices([CABLE_OUT, MIC], default_input=0)
device, note = resolve_input_device(None)
check('a default input that is the cable is not used',
      device == 1, 'picked index {}'.format(device))
check('and the swap is explained rather than done quietly',
      note is not None and 'silence' in (note or ''), (note or '')[:60])
check('the message names both devices',
      'CABLE Output' in (note or '') and 'Microphone Array' in (note or ''),
      'without both, this is "audio does not work" and the next step is '
      'guessing at devices')

with_devices([MIC, CABLE_OUT], default_input=0)
device, note = resolve_input_device(None)
check('a default input that is a real microphone is left alone',
      device is None and note is None,
      'None means the system default, which is what it already was')

with_devices([CABLE_OUT], default_input=0)
device, note = resolve_input_device(None)
check('with no real microphone at all it says so and does not invent one',
      device is None and note is not None and 'silence' in note)

with_devices([MAPPER, CABLE_OUT, MIC_SLOW, MIC], default_input=1)
check('the lowest-latency real input wins',
      find_real_input() == 3,
      'Windows exposes each device once per host API and the spread is tens '
      'of milliseconds')
check('and the aggregate endpoints are skipped',
      find_real_input() != 0,
      'Sound Mapper is a redirection to the default, which is the thing '
      'under suspicion')


# The other half of the same mis-setup, found on the same machine: both ends of
# the cable were the system defaults at once. This one the app does not route
# around -- its own audio reaches the cable either way -- but it silences the
# operator and can feed the call back to the other party.
from desktop.audio import check_default_playback                   # noqa: E402

SPEAKERS = {
    'name': 'Speakers / Headphones (Realtek Audio)',
    'max_input_channels': 0, 'max_output_channels': 2,
    'default_low_output_latency': 0.003, 'default_samplerate': 48000.0,
}


class OutStub(FakeSoundDevice):
    """`FakeSoundDevice` with a settable default *output*."""

    def __init__(self, devices, default_output):
        super().__init__(devices)
        self.default = type('D', (), {'device': (0, default_output)})()


def with_output(devices, default_output):
    sys.modules['sounddevice'] = OutStub(devices, default_output)


with_output([CABLE_48, SPEAKERS], default_output=0)
said = check_default_playback()
check('a default playback that is the cable is reported',
      said is not None and 'CABLE Input' in said, (said or '')[:60])
check('and the message says what it does to the call',
      'hear themselves' in (said or ''),
      'the operator hears nothing and the other party hears themselves — '
      'harder to attribute than silence, so it has to be named')

with_output([CABLE_48, SPEAKERS], default_output=1)
check('real speakers are left alone',
      check_default_playback() is None)

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
