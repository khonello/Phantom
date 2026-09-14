# Testing the realism work

Two different sessions with two different purposes, and conflating them is how
a tuning decision gets made on evidence that could not support it.

**Pass A — pre-flight.** Does it still work? Run in whatever conditions are to
hand. Nothing is tuned on the result.

**Pass B — tuning.** What should the knobs be? Run only in conditions that can
answer that, which is a much shorter list.

Both cost about $0.74/hr on a RunPod 4090, so neither is expensive. What is
expensive is a number from Pass A used as though it came from Pass B.

---

## Pass A — pre-flight (regression)

### The question

*Did the texture and seam work break anything that used to work?*

Not "is it good". Conditions do not have to be favourable, the source does not
have to match the target, and a result that looks poor is not a failure.

### Why it is a real test even in bad conditions

`texture_strength` and `diffuse_strength` both default to `0.0`, so **the new
layers do not run**. What did change on the default path, and is therefore what
Pass A actually exercises:

- **The seam.** `mask_feather` measured against the face's own extent (5px to
  10px on a ~92px face), and `mask_erode` pulling the 50%-alpha line onto skin
  rather than straddling onto neck and hair.
- **The colour deadband.** `_COLOR_FLOOR` 4.0 to 1.5, range 12.0 to 8.0.
  Sub-4-unit LAB differences now get corrected where they previously got none.
- **The CPU work.** One shared LAB conversion across both shading stages, grain
  reusing a cached noise tile, `_estimate_noise` bounding its sample earlier.
  These should be invisible; if they are not, that is the bug.

### What counts as broken

1. Pipeline never becomes healthy, or dies mid-stream
2. `tools/stats.py` exits non-zero — a requested accelerator is not loaded
3. No frames reach the desktop, or the swap never appears
4. Guard badge stuck on
5. Frame time far from expectation for the configured models
6. A seam or halo **worse than before** — the feather change is the suspect
7. Shimmer, or the face lagging the head

### What does NOT count as broken

- Colour looking wrong on a mismatched source and target
- The swap looking unconvincing in poor light
- Texture appearing to do nothing — it is off

### Roles

**Operator:** runs the orchestrator, uploads the source, streams, narrates.
Report what is *different*, not what is *good*. Verbatim error text.

**Assistant:** does not tune. Reads logs and reports, identifies whether a
symptom is a fault or designed behaviour, and says which. Records findings.
Proposes no knob changes off this pass.

---

## Pass B — tuning

### The question

*What should `texture_strength` and `diffuse_strength` be?*

### Preconditions — none of these are optional

The layers are bounded by measurement, so the measurement has to be worth
trusting.

1. **A sharp, frontal source photo.** The texture layer lifts high-frequency
   detail *from the source*. A soft source has nothing to lift, so the layer
   correctly adds nothing — and that looks identical to it being broken.
   `guard_min_sharpness` (40) is protecting exactly this; **do not lower it to
   get a photo accepted for this pass.**
2. **Reasonable, even lighting on the operator.** `_texture_headroom` measures
   the real face's high-frequency energy in the same pixels to decide what
   there is to add. Under-lit and noisy, that measurement is not describing
   skin.
3. **A source reasonably matched to the target** in complexion and lighting.
   A fair, well-lit source against a dark, under-lit target is the hardest case
   for colour matching and will dominate anything the texture layer does.
4. **Face the camera.** `texture_confidence` scales the layer from full at 12
   degrees of pose disagreement with the source photograph to nothing at 45.

If any of these fail, the session is Pass A again whether it was meant to be or
not.

### Protocol

`set_realism` applies live, so knobs can be changed mid-stream and watched. The
**readings** only emit when a stream stops, so numbers need a stop per setting.

1. Stream with both layers off. Stop. Capture the baseline report.
2. `texture_strength=1.0` — parity with the measured headroom, and the most
   visible the layer can be. Overshoot is arithmetically impossible, so this is
   safe as well as legible. Watch, then walk down: 0.5, 0.3.
3. Texture off, `diffuse_strength=0.6`, then 0.3. A different complaint in a
   different band: "the skin reads hard", not "the skin reads plastic".
4. Both, at whatever the two passes above suggested.
5. Stop after each setting whose numbers matter.

`tools/sweep_levers.py --sweep realism` runs the same ladder against a fixed
clip, which removes the operator as a variable at the cost of removing the
operator's judgement.

### The reading that decides it

`detail_ratio`, and specifically its `at limit N% of frames`. It is the
correction `_match_detail` **wanted** before its clamp.

If the clamp binds on most baseline frames, then part of the 0.42 face/frame
detail gap is the clamp rather than the swap — and raising one constant is a
cheaper fix than the entire texture layer. That single number can retire a
feature.

`texture_headroom` says whether the layer had anything to spend.
`texture_confidence` says whether pose let it spend it. A layer that appears to
do nothing is explained by one of those two before it is called broken.

### Roles

**Operator:** provides the conditions above, streams, and judges. The verdict
on realism is the operator's — the metrics are a cross-check, not a jury.

**Assistant:** drives the knobs so the operator never touches config, keeps the
ladder in order, captures a report per setting, and reports what the numbers
say **including when they contradict the eye**. Does not argue the operator out
of what they can see; records the disagreement.

---

## Pass C — resemblance (RESEMBLANCE.md)

### The question

*Does the output carry the source's complexion, texture and shape — and at
what cost to the seam?*

### The cross-tone pair

Pass B's third precondition — *a source reasonably matched to the target in
complexion* — is **inverted here on purpose.** Every measurement so far paired
`source/one` with `source/two`, the two closest complexions in the fixtures;
a route that only works on that pair has not been tested. The tracked
fixtures hold three people, and the pair with the largest complexion gap is
pinned as the first pair every route in RESEMBLANCE.md runs on, **in both
directions**, because fair-onto-dark and dark-onto-fair fail differently:
luminance has to come down in one and up in the other, and a chroma-only
transfer lands on the wrong brightness in the right hue in both.

| Pair | Source set | Target | Exercises |
|---|---|---|---|
| **C1 fair → dark** | `source/two/*` (six photographs) | `target/face-3.jpeg` (still), `target/face-6.mp4` (stream) | L down, chroma toward warm-neutral; the neck and hands are in shot on the video |
| **C2 dark → fair** | `target/face-1..5.jpeg` (five photographs) | `source/two/IMG_3623.jpg` | L up; and the still every earlier measurement used, so `id_*` and `shape_*` stay comparable |

**Chosen by eye, 2026-09-13, and provisional.** No detector runs on the
development machine, so `complexion_gap` and `shape_mismatch` for these pairs
are unmeasured. The first probe run prints both:

```bash
python tools/identity_probe.py -s source/two -t target/face-3.jpeg --save-frames c1/
python tools/identity_probe.py -s target/face-1.jpeg target/face-2.jpeg target/face-3.jpeg target/face-4.jpeg target/face-5.jpeg -t source/two/IMG_3623.jpg --save-frames c2/
```

Record the two `complexion_gap` values and the two `shape_mismatch` values in
RESEMBLANCE.md §3.3 when they exist. **C1 measured 2026-09-14** (inswapper,
`identity_probe.py` on the pod): `shape_mismatch` **0.309** (0.282 at the
outline), `complexion_gap` **6.0** in chroma with **31 L units** of lightness
difference — on this pairing the complexion gap is almost entirely lightness,
which is the axis `complexion_target_base` exists for. C2 still unmeasured. If a different pairing in the fixtures
measures larger on either, swap it in — the point is the hardest case, not
these files.

### The readings that decide it

`complexion_face` toward zero and `complexion_seam` held under 4 — the
second is the one Route A exists for, and a route that wins the first while
losing the second has moved the seam, not removed it. `id_target` falling.
`detail_ratio` and `texture_delivered` as in Pass B. `id_out` with the caveat
RESEMBLANCE.md §3.1 gives: ArcFace carries a little skin tone, so it rises when
complexion transfers and cannot say that is why.

### Roles

As Pass B. The operator's verdict on *resemblance* outranks every number
above; the numbers say which stage to look at.

---

## What Pass A established, 2026-09-04

RunPod 4090, `hyperswap_1a_256` + `gpen_bfr_256`, aligned pinned 256/256,
texture and scatter both off, `optimal` at 20fps.

    [HOLDS] deadline 50.0ms, 166 frames
      detect          p95=  8.5ms
      swap+composite  p95= 35.1ms
        mask          p95= 15.8ms
        restore       p95=  9.0ms
        colour        p95=  1.6ms
        detail        p95=  1.0ms
        paste         p95=  1.1ms
        smooth        p95=  0.6ms
      total           p95= 43.4ms   6.57ms headroom

**XSeg masking is now the largest stage, at nearly double restoration.** It had
never been broken out — it lived inside `swap+composite` — and with
`gpen_bfr_256` cutting restore to 9.0ms, the dominant term moved without anyone
noticing. Any future work aimed at frame time should start here, not at
restoration.

**24fps would miss.** Its deadline is 41.7ms against a measured 43.4ms. The
estimate that said otherwise assumed a ~10.3ms compositor, which was measured
with `inswapper_128` clamping the aligned size down; `hyperswap_1a_256` forces
256/256, so every compositor stage runs at full size every frame. 20fps as a
hard target is now measured rather than assumed.

Guards over 6102 frames: **0 would guard**. Confidence p50 0.86, coverage 0.85,
identity 0.91, face 92px, yaw p99 20 degrees.

### The stall, which was not a fault

Every source photo was rejected on a re-review, leaving `No usable source face
in 0 image(s)`. With no source, the live path holds the last swapped frame and
emits nothing new, because the alternative is transmitting the operator's real
face. From outside that is indistinguishable from a hang, and the restart then
could not start because there was still no source.

Worth improving: the desktop should say *why* it is stuck. "Starting stream"
forever is the wrong message for "your source was refused".

One rejection read `sharpness 40, need 40`, which looks self-contradictory. The
check is `variance < threshold`, so the real value was between 39.5 and 39.9
and `:.0f` rounded it up in the message. The logic is right; the message is
not.
