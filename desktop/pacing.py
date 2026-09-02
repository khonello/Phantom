"""
Sending frames upstream at the rate the preset asked for.

The desktop used to send every frame the camera produced. That sounds like the
generous choice and is not: the preset's `capture_fps` is a *budget*, and on
Windows the camera ignores it — asked for 20 it delivers 30 — so the uplink
carried half again as many JPEGs as anything downstream was sized for.

That direction is the one that hurts. A home connection is asymmetric, and a
saturated uplink does not present as dropped frames; it presents as **latency**,
because frames queue in the OS send buffer while throughput still looks
healthy. So the extra frames buy nothing and cost the one thing the whole
pipeline is being tuned for.

Pacing happens here rather than at the camera on purpose. Forcing the device to
20fps was measured and rejected: DirectShow honours the request by snapping to
its nearest mode, which on the test camera was 15fps, so asking for less
delivered less than asked. Capturing at whatever the camera gives and choosing
what to send keeps the local preview smooth and costs one comparison per frame.

**The target is a target, not a ratio.** Nothing here assumes 30fps in, and no
fixed fraction is discarded. The camera's actual rate is measured, and frames
are only dropped while it is genuinely overshooting — a camera already
delivering 20 has no spare frames, so discarding one would cost picture and
save nothing. That margin is what makes this safe against ordinary jitter,
where a frame arriving a millisecond early is one a bare schedule has simply
not reached yet.
"""

from typing import Optional

_NS_PER_SECOND = 1_000_000_000


class FramePacer:
    """
    Decides which captured frames are sent upstream.

    Example:
        pacer = FramePacer(20)
        if pacer.due(capture_ts_ns):
            send(frame)
    """

    # How far behind schedule the clock may fall before it is resynchronised
    # rather than caught up on, as a multiple of the interval. Catching up means
    # sending consecutive frames as fast as they arrive, which is a burst into
    # exactly the link that just stalled.
    _RESYNC_AFTER = 2.0

    # How much faster than the target the camera has to actually be running
    # before any frame is dropped.
    #
    # Below this, pacing is off entirely and every frame is sent. Dropping is
    # only ever worth doing against a camera that is genuinely overshooting —
    # a camera already delivering the target rate has no spare frames, and
    # discarding one costs a real frame and gains nothing. Without this a
    # nominal 20fps camera would still lose frames to ordinary jitter: any
    # frame arriving a millisecond early is one the schedule has not reached
    # yet.
    _PACE_ABOVE = 1.1

    # Frames observed before the measured rate is trusted. Until then nothing
    # is dropped, which is the safe direction — a few extra frames at startup
    # cost bandwidth, and a few missing ones cost picture.
    _OBSERVE_FRAMES = 8

    # EMA weight on the observed frame interval. Slow, because this decides
    # whether to drop frames at all and should not follow a hiccup.
    _OBSERVE_ALPHA = 0.15

    def __init__(self, fps: float) -> None:
        """
        Args:
            fps: Target sends per second. Zero or negative disables pacing and
                 every frame is due, which is what a preset without a rate
                 should do — the old behaviour, reachable rather than removed.
        """
        self.interval_ns = int(_NS_PER_SECOND / fps) if fps > 0 else 0
        self._next_due_ns: Optional[int] = None
        self._last_seen_ns: Optional[int] = None
        self._observed_ns: Optional[float] = None
        self._observed_count = 0
        self.sent = 0
        self.skipped = 0

    @property
    def observed_fps(self) -> float:
        """Measured capture rate, or 0.0 before enough frames have been seen."""
        if self._observed_ns is None or self._observed_ns <= 0:
            return 0.0
        return _NS_PER_SECOND / self._observed_ns

    @property
    def pacing(self) -> bool:
        """
        Whether frames are currently being dropped at all.

        False when the camera is not meaningfully faster than the target, which
        is the ordinary case on a camera that honours its configuration. The
        distinction is worth being able to read: "20fps out" means something
        different when nothing was dropped to get there.
        """
        if self.interval_ns <= 0:
            return False
        if self._observed_count < self._OBSERVE_FRAMES or self._observed_ns is None:
            return False
        return self._observed_ns * self._PACE_ABOVE < self.interval_ns

    def _observe(self, capture_ts_ns: int) -> None:
        """Fold this frame's arrival into the measured capture interval."""
        previous, self._last_seen_ns = self._last_seen_ns, capture_ts_ns
        if previous is None:
            return
        gap = capture_ts_ns - previous
        # A gap longer than a stall is not a frame rate, it is an outage, and
        # letting it into the average would turn pacing off for a while
        # afterwards.
        if gap <= 0 or gap > self.interval_ns * self._RESYNC_AFTER * 2:
            return
        self._observed_count += 1
        if self._observed_ns is None:
            self._observed_ns = float(gap)
        else:
            self._observed_ns += (gap - self._observed_ns) * self._OBSERVE_ALPHA

    def due(self, capture_ts_ns: int) -> bool:
        """
        Whether the frame captured at `capture_ts_ns` should be sent.

        The schedule advances by whole intervals rather than being reset to the
        moment of each send, and that is the part worth not simplifying. A
        naive `now - last_sent >= interval` **aliases**: a 30fps camera against
        a 50ms interval gives frames at 0, 33, 67, 100 — 33 is too early, 67
        passes, and the next comparison starts from 67, so it sends at 0, 67,
        133 and delivers 15fps instead of the 20 that was asked for. Asking for
        less would have got less than asked, which is the same failure that
        ruled out setting the rate on the device.

        Advancing the due time by exactly one interval keeps the fractional
        remainder, so the same camera sends two frames out of every three and
        lands on 20.

        Args:
            capture_ts_ns: When the frame was captured (`perf_counter_ns`)

        Returns:
            True to send it
        """
        if self.interval_ns <= 0:
            self.sent += 1
            return True

        self._observe(capture_ts_ns)

        # The camera is not overshooting, so there is nothing to trim. Keep the
        # schedule alongside the current frame so that if the rate does climb
        # later, pacing starts from now rather than from a due time left behind
        # minutes ago.
        if not self.pacing:
            self._next_due_ns = capture_ts_ns + self.interval_ns
            self.sent += 1
            return True

        if self._next_due_ns is None:
            self._next_due_ns = capture_ts_ns + self.interval_ns
            self.sent += 1
            return True

        if capture_ts_ns < self._next_due_ns:
            self.skipped += 1
            return False

        behind = capture_ts_ns - self._next_due_ns
        if behind > self.interval_ns * self._RESYNC_AFTER:
            # A stall, a suspended laptop, a camera that stopped delivering.
            # Start the schedule again from now instead of firing every frame
            # until the backlog is paid off.
            self._next_due_ns = capture_ts_ns + self.interval_ns
        else:
            self._next_due_ns += self.interval_ns

        self.sent += 1
        return True

    def reset(self) -> None:
        """Forget the schedule and the measured rate."""
        self._next_due_ns = None
        self._last_seen_ns = None
        self._observed_ns = None
        self._observed_count = 0
        self.sent = 0
        self.skipped = 0
