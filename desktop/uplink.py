"""
Choosing how much uplink to spend, from what the link is actually doing.

The preset ladder was already a rate ladder — `fast` 1.58 Mbps, `optimal` 2.45,
`production` 3.96 — and the operator drove it by hand. Measured 2026-09-05 from
West Africa, that hand movement was worth more than any compute lever in the
project: dropping a gear took delivery from 61% to 94% and p50 latency from
1222ms to 366ms. This does the same thing on evidence rather than on a hunch,
and within seconds rather than whenever someone notices.

**The dropdown is a ceiling, not a target.** The governor only ever moves
downward of what the operator chose, and back up to it. That is what stops
automation overriding intent: a gear change can protect them from a link that
cannot carry their choice, and can never hand them what they did not ask for.

**The signal is send-buffer backpressure, not bandwidth.** `websockets.sync`
blocks inside `send()` until the kernel accepts the bytes, so the time the
uplink thread spends in there measures the one thing that matters directly:
whether the link is absorbing frames as fast as they are produced. On a healthy
link it is microseconds. On a saturated one it is tens of milliseconds, and it
says so *before* the consequence — frames queueing in the OS send buffer,
arriving in bursts, and being evicted at the pod — has had time to show up as
latency.

Two reasons to prefer it to RTT. It is local, so it costs no round trip and
reports at the moment of the event rather than one round trip later. And it
cannot confuse congestion with distance: a 350ms RTT to Europe is a fact of
geography that no gear change will alter, and a controller steering on RTT
alone would read that floor as a fault and shift down forever.

Delivery ratio corroborates it, as a **ratio of rates** rather than of counts.
Frames sent in the last round trip have not come back yet, so comparing
cumulative counts under-reports delivery by the whole RTT; comparing the send
rate against the return rate is exact in steady state and merely noisy during a
transition, which is what the cooldown is for.

Asymmetric, for the same reason `RTTTracker` is: shifting down late costs a
visibly broken call, shifting up early costs a gear change nobody needed. So
down is one bad window, and up is twenty seconds of clean ones.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pipeline.api.schema import PRESETS

# Ascending uplink cost. Written out rather than derived by sorting, because the
# ordering is a claim about bitrate that PRESETS does not itself record — `fast`
# is cheaper than `optimal` on resolution while `production` is dearer on rate
# and quality, so no single field orders them.
LADDER: Tuple[str, ...] = ('fast', 'optimal', 'production')


@dataclass(frozen=True)
class Gear:
    """
    The uplink axes of one preset, and nothing else.

    Deliberately excludes `det_size`, `aligned_size` and `occluder`: those decide
    how the face looks, and an automatic gear change must not alter the swap's
    appearance mid-call. A gear is what the desktop encodes and sends, which is
    the only part the link cares about.
    """

    name: str
    width: int
    height: int
    fps: int
    jpeg_quality: int

    @property
    def interval_ms(self) -> float:
        """Milliseconds between sends at this gear's rate."""
        return 1000.0 / self.fps if self.fps > 0 else 0.0


def gear_for(preset: str) -> Gear:
    """The uplink settings of a named preset, falling back to `optimal`."""
    values = PRESETS.get(preset) or PRESETS['optimal']
    return Gear(
        name=preset if preset in PRESETS else 'optimal',
        width=int(values['capture_width']),
        height=int(values['capture_height']),
        fps=int(values['capture_fps']),
        jpeg_quality=int(values['jpeg_quality']),
    )


def capture_ceiling() -> Tuple[int, int]:
    """
    The one capture resolution that serves every gear on the ladder.

    Applying a preset to the device costs about 3.9 seconds to first frame on
    Windows MSMF — see `Bridge._configure_capture` — so a governor that
    reconfigured the camera per gear change would black the call out for four
    seconds each time it acted, which is worse than not adapting at all.
    Capturing once at the largest gear and resizing down per frame costs one
    `cv2.resize` and nothing else.

    Downscaling in software is also the better picture: 640x360 resampled to
    480x270 supersamples, where asking the camera for 270p does not.

    Returns:
        (width, height) covering every entry in LADDER
    """
    gears = [gear_for(name) for name in LADDER]
    return (max(g.width for g in gears), max(g.height for g in gears))


def adaptation_enabled() -> bool:
    """
    Whether `PHANTOM_UPLINK_ADAPT` leaves the governor steering.

    On by default. It is turned off for a measurement run, for the reason
    `PHANTOM_PLAYOUT_DELAY_MS` exists: two sessions cannot be compared if the
    thing under test chose itself differently in each.
    """
    raw = os.environ.get('PHANTOM_UPLINK_ADAPT', '').strip().lower()
    return raw not in ('0', 'false', 'no', 'off')


class UplinkGovernor:
    """
    Picks an uplink gear from the last window of link behaviour.

    Example:
        gov = UplinkGovernor('optimal')
        changed = gov.observe(sent=30, delivered_ratio=0.62,
                              block_ms_per_frame=41.0, now=t)
        if changed:
            apply(gov.gear)
    """

    # Delivery below this, or blocking above it, is a link that cannot carry the
    # current gear. 0.85 sits between the two measured points — 94% on a gear
    # that worked and 61% on one that did not — rather than close to either,
    # since the gap between them is where the useful threshold lies and nothing
    # has yet measured its shape.
    DOWN_DELIVERED = 0.85

    # Mean time blocked in `send()` per frame, as a fraction of the interval
    # between frames. Above a quarter of the interval the sender is spending
    # real time waiting on the socket, which on a link with headroom it never
    # does. A fraction rather than a millisecond count, so the threshold means
    # the same thing at 15fps and at 20.
    DOWN_BLOCK_FRACTION = 0.25

    # What a link with headroom looks like. Both have to hold, and hold for
    # `UP_AFTER_S`, before anything climbs.
    UP_DELIVERED = 0.97
    UP_BLOCK_FRACTION = 0.05
    UP_AFTER_S = 20.0

    # After any change, wait before considering another. A gear change alters
    # the very thing being measured, so acting again before its effect is
    # visible is how a controller oscillates.
    COOLDOWN_S = 8.0

    # Frames in the window below which the sample says nothing. A stream that is
    # starting, stopping or paused produces small windows whose ratios are
    # arithmetic on two or three frames.
    MIN_FRAMES = 10

    def __init__(self, ceiling: str = 'optimal', enabled: Optional[bool] = None) -> None:
        """
        Args:
            ceiling: The operator's chosen preset. The governor never exceeds it
            enabled: Steer, or only observe. Defaults to `PHANTOM_UPLINK_ADAPT`
        """
        self.enabled = adaptation_enabled() if enabled is None else bool(enabled)
        self._ceiling = self._index(ceiling)
        self._index_now = self._ceiling
        self._changed_at = 0.0
        self._clean_since: Optional[float] = None
        self.reason = ''
        self.shifts_down = 0
        self.shifts_up = 0

    @staticmethod
    def _index(preset: str) -> int:
        try:
            return LADDER.index(preset)
        except ValueError:
            return LADDER.index('optimal')

    @property
    def gear(self) -> Gear:
        """The gear frames should currently be sent at."""
        return gear_for(LADDER[self._index_now])

    @property
    def ceiling(self) -> str:
        """The operator's chosen preset — the most this will ever send."""
        return LADDER[self._ceiling]

    @property
    def adapted(self) -> bool:
        """Whether the governor is currently below what the operator picked."""
        return self._index_now < self._ceiling

    def set_ceiling(self, preset: str, now: float = 0.0) -> bool:
        """
        Apply the operator's choice, immediately and in both directions.

        An explicit choice takes effect now rather than being climbed towards:
        someone who selects a gear expects to see it. Adaptation then continues
        onward of that gear, so choosing a higher one on a link that cannot hold
        it costs a single window before the governor takes it back.

        Args:
            preset: The chosen preset name
            now: Monotonic seconds, for the cooldown

        Returns:
            True if the gear actually moved
        """
        self._ceiling = self._index(preset)
        if self._index_now == self._ceiling:
            return False
        self._index_now = self._ceiling
        self._changed_at = now
        self._clean_since = None
        self.reason = ''
        return True

    def reset(self) -> None:
        """Return to the ceiling and forget the window. A new link, or a new stream."""
        self._index_now = self._ceiling
        self._changed_at = 0.0
        self._clean_since = None
        self.reason = ''

    def observe(
        self,
        *,
        sent: int,
        delivered_ratio: float,
        block_ms_per_frame: float,
        now: float,
    ) -> Optional[str]:
        """
        Fold one window of link behaviour in, and shift gear if it warrants one.

        Args:
            sent: Frames handed to the socket during the window
            delivered_ratio: Return rate over send rate, 0..1+
            block_ms_per_frame: Mean time inside `send()` per frame sent
            now: Monotonic seconds

        Returns:
            The new preset name if the gear changed, else None
        """
        if sent < self.MIN_FRAMES:
            # Not evidence either way, and specifically not evidence of health:
            # leave the clean run alone rather than crediting an idle window
            # towards a climb.
            return None

        interval = self.gear.interval_ms or 1.0
        block_fraction = block_ms_per_frame / interval

        saturated = (delivered_ratio < self.DOWN_DELIVERED
                     or block_fraction > self.DOWN_BLOCK_FRACTION)
        clean = (delivered_ratio >= self.UP_DELIVERED
                 and block_fraction <= self.UP_BLOCK_FRACTION)

        if clean:
            if self._clean_since is None:
                self._clean_since = now
        else:
            self._clean_since = None

        if not self.enabled:
            return None
        if now - self._changed_at < self.COOLDOWN_S:
            return None

        if saturated and self._index_now > 0:
            self._index_now -= 1
            self._changed_at = now
            self._clean_since = None
            self.shifts_down += 1
            self.reason = (
                'link saturated — {:.0f}% delivered, {:.0f}ms blocked/frame'.format(
                    delivered_ratio * 100.0, block_ms_per_frame))
            return LADDER[self._index_now]

        if (self._index_now < self._ceiling
                and self._clean_since is not None
                and now - self._clean_since >= self.UP_AFTER_S):
            self._index_now += 1
            self._changed_at = now
            self._clean_since = None
            self.shifts_up += 1
            self.reason = 'link has headroom — {:.0f}% delivered'.format(
                delivered_ratio * 100.0)
            return LADDER[self._index_now]

        return None

    def stats(self) -> Dict[str, Any]:
        """What the governor has done, for the readout and the log."""
        return {
            'gear': LADDER[self._index_now],
            'ceiling': self.ceiling,
            'adapted': self.adapted,
            'enabled': self.enabled,
            'shifts_down': self.shifts_down,
            'shifts_up': self.shifts_up,
            'reason': self.reason,
        }
