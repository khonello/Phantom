# Phantom — playout delay and A/V sync

Why both streams are presented `D` after capture, how `D` is measured, what
escalates it, and the timebase errors that all pushed the sound later than
the picture. Summarised in CLAUDE.md; this is the full record.

## Playout is measured once, then fixed

Both streams are presented **D after capture**, whatever the network did in
between. D is measured from the link over the first full RTT window (~3s),
committed once, and then held (`_maybe_calibrate`; `PHANTOM_PLAYOUT_DELAY_MS`
pins a value instead, and `0` there restores the adaptive behaviour).

Adaptive playout is right for video alone — it chases the network and the
viewer sees nothing. It becomes wrong the moment audio is played against the
same number, because every adjustment is a discontinuity: move the read point
forward and samples are skipped, back and silence is inserted. A measured
session had the target swinging 380 -> 500 -> 420 -> 490ms every two seconds,
which is what an operator hears as speech breaking up. **Jitter is far more
damaging than delay** — people adapt to a constant delay and never to one that
moves.

Calibration keeps that property and drops the guessed number. D was a constant
550ms, from one session against EU-RO-1: RTT p50 ~350ms, p95 ~450ms, 700ms
outliers. **D cannot be smaller than what video costs** — a frame cannot be
shown before it arrives, so audio necessarily waits as long as video does. What
it must not be is *larger*, and a constant is larger everywhere the link is
better than the one it was measured on. On a 200ms link, 550ms held audio a
third of a second past the picture. Calibration takes `p95 + spread + 80ms`,
quantised to 25ms and floored at 100ms — the floor is not `RTTTracker.FLOOR_NS`
because D also absorbs the 33ms display tick and the vcam queue, neither of
which is RTT.

**Both streams read one accessor, and this is not decoration.** `pop_eligible`
read `self._rtt.target_delay_ns` — the adaptive estimate — while
`AudioPlayback` read `self.target_delay_ns`, the fixed value. Video was
therefore released on arrival while audio waited D, and on a call the sound
trailed the lips by the difference. Escalation made it worse rather than
better: it raised the number **only audio read**, pushing the sound further
back while doing nothing about the repeats it was raised to steady.

Nothing disagreed, because nothing compared them. `sync_stats` published the
adaptive estimate as `target_delay_ms`, so the `[SYNC] delay=` line and the
viewport badge had never shown the delay either stream was actually held to.
There is now a second line — `[SYNC] av video=+Xms audio=+Yms skew=±Zms` — from
`JitterBuffer.video_age_ns` and `AudioPlayback.stats()['audio_age_ms']`, both
measured on the material each stream **actually presented**. A negative skew is
audio presenting older material than the picture: the sound behind the lips.
Video is allowed to be up to one frame interval older, because frames are
discrete; past that it is a fault, and the badge says so.

**Audio's cursor follows the epoch, never the delay.** It is deliberately
continuous — re-deriving position per block against a moving target is what
made speech break up — so `JitterBuffer.delay_epoch` increments when D is
*decided* (calibration committing, escalation stepping, `clear()` on a new
link) and audio repositions exactly then. One discontinuity at second three
buys a delay that is right for the rest of the session.

Three one-directional errors in the audio timebase were fixed with it, all of
which pushed the sound later:

- **The drift baseline started before the device did.** `_drift_start_ns` was
  taken in `start()`, so the tens-to-hundreds of milliseconds a Windows device
  takes to open counted as samples that failed to arrive, and `drift_ns` sat
  permanently negative by that amount — which was then subtracted from the
  playback position for the whole session. It is taken at the first delivered
  block now, and is no longer fed into the position at all: the cursor is
  continuous, so a drift term only moves the seek point, and a rate mismatch
  cannot be repaired by seeking somewhere the samples are not. It is measured
  and reported.
- **A chunk's timestamp was its delivery, used as its first sample.** `_seek`
  computes its offset as `playback_point - chunk_ts`, so the stamp must be when
  the first sample was recorded — a block duration plus the input latency
  earlier than the callback runs.
- **The output device's own buffer was unaccounted.** A block written now is
  audible once the device has played what it holds, so the real delay was D
  plus the device — 2ms on WASAPI and 90ms on MME for the same cable.

**Only video crosses the network.** Audio is captured into a local ring buffer
and is available in ~23ms; video does a round trip to the pod and takes ~350ms.
So audio spends most of the budget waiting, jitter is entirely video's, and D
is set by the video distribution alone. That changes if voice processing ever
moves server-side — there is a `set_voice_transformer` hook suggesting someone
considered it — at which point a fixed buffer matters more, not less.

`JitterBuffer.next_for_slot` holds the schedule, and the second rule is the one
that is easy to get wrong:

1. **The slot fires on time, always.** Stalling for a late frame slips the
   schedule, and audio is locked to the same clock, so a stall is either a gap
   in speech or a drift out of sync.
2. **A frame that missed its slot is discarded, not shown late.** Playing the
   straggler shifts everything one slot later and the pattern never recovers.
3. **An empty slot repeats the last shown frame.** 33-50ms of an already
   mostly-still face is invisible, which is exactly why video is the cheap
   place to absorb jitter. Always the last *swapped* frame — never the raw
   camera, never black, which is the same invariant `_run_vcam` holds.

Repeats are counted and shown in the badge as `N held`. One is invisible; a
sustained rate is a frozen face while audio continues, which reads as a broken
swap rather than a slow link.

**But a repeat is not the evidence to step D on, and using it was a runaway.**
The display ticks at 30/s while `optimal` streams at 20fps, so a third of all
slots have no new frame due — on a perfect link, forever — and half of them at
`fast`. The escalation condition was therefore permanently true. That was
invisible for as long as `_fixed_delay_ns` was a number only audio read;
pointing video at it sent D to the 2s ceiling in a couple of minutes and froze
the picture.

The signal is **starvation**: a slot with nothing eligible *and nothing
waiting*, meaning frames are arriving already past their deadline, which is the
one thing more headroom fixes. A slot with frames waiting is D being generous,
and raising it is precisely backwards. So **>20% of slots starving over ~10s,
while frames are actually arriving, steps the delay up by 100ms, once, and says
so**. That is the only place playout adapts, and it adapts on evidence rather
than per frame.

Two capacity facts belong with it. D holds `(D - rtt) * fps` frames in flight,
so `MAX_FRAMES` has to exceed the escalation ceiling times the frame rate — 60
was exactly 2s at 30fps and therefore no headroom at all; it is 120. And when
the buffer is full anyway, `pop_eligible` releases the frame at the front
rather than letting the next push evict it unseen, because that eviction
repeats and the picture never moves again. Early by a fraction of D beats
frozen, and `forced` counts it so the cause is not read as a network fault.

**Calibration does not believe the warm-up.** The first stream after a pipeline
start pays model load — tens of seconds — and those frames come back carrying
round trips of exactly that size. `_CALIBRATE_SKIP` discards them, and the
window that follows has to look settled (p95 within 2x p50, and under a second)
before it is committed. Five unsettled windows and it keeps the conservative
provisional and says so, leaving the correction to escalation.
