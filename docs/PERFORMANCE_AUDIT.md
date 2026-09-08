# Phantom — compositor performance audit

A pass over the CPU side of the pipeline for waste: work done twice, work done
at a resolution it does not need, statistics computed over more pixels than a
statistic requires, and conversions paid for twice over. **Not a realism
change.** Everything acted on here was verified to leave the output the same, or
is flagged in §6 as something that would not.

Measured on a development laptop's CPU. See [§7](#7-what-the-pod-will-say) for
why that is the right machine to *find* waste on and the wrong one to quote
absolutes from.

---

## 1. The headline

**`optimal` is comfortable. `production` does not fit, and did not before any of
this work.**

| compositor total | baseline | with texture + scatter | deadline |
|---|---|---|---|
| `optimal` — 640x360, 101px face, aligned 256 | **5.7ms** | **7.4ms** | 50ms |
| `production` — 960x540, 400px face, aligned 320 | **39.2ms** | **48.2ms** | 33ms |

At `optimal` the whole compositor is a ninth of the budget and the two new
realism layers cost 1.7ms between them. Nothing here needs attention.

At `production` the compositor alone exceeds the frame deadline **before**
detection, the swap, restoration or JPEG encode have run. That is not a
regression introduced by the realism work — the baseline column is the same
pipeline without it. It is what compositing a 400px face at 320 aligned costs
when every stage is a full-size elementwise pass on the CPU.

The honest conclusion is in [§8](#8-the-gpu-question-answered-with-this-data):
`production` is not a live preset on CPU compositing, and saying so is more use
than shaving another millisecond off it.

---

## 2. Per stage, after the fixes in §3

`optimal`, 101px face:

| stage | baseline | + texture/scatter |
|---|---|---|
| paste | 1.93 | 3.31 |
| texture | — | 1.66 |
| detail | 1.02 | 1.00 |
| colour | 0.85 | 0.78 |
| smooth | 0.80 | 0.81 |
| mask | 0.72 | 0.70 |
| scatter | 0.16 | 0.47 |
| **total** | **5.73** | **7.40** |

`production`, 400px face:

| stage | baseline | + texture/scatter |
|---|---|---|
| paste | 17.27 | 24.61 |
| detail | 7.54 | 7.44 |
| texture | — | 6.99 |
| smooth | 5.21 | 5.23 |
| colour | 5.20 | 5.08 |
| mask | 2.21 | 2.23 |
| scatter | 0.89 | 2.60 |
| **total** | **39.23** | **48.17** |

Restoration is excluded — it is GPU work and is not what this audit is about.

---

## 3. Acted on — measured end to end

Same profiler against the pre-audit commit and against now:

| | before | after | saved |
|---|---|---|---|
| `optimal` baseline | 7.21ms | 6.47ms | **0.74ms (10%)** |
| `optimal` + texture/scatter | 8.59ms | 7.78ms | **0.81ms (9%)** |
| `production` baseline | 47.24ms | 40.47ms | **6.77ms (14%)** |
| `production` + texture/scatter | 56.58ms | 48.75ms | **7.83ms (14%)** |

The "after" column is paying for §3.7 while it does that. The extent fix made the
feather three times wider and the texture map four times larger — about 3.3ms at
`production` — so the optimisations were worth roughly **11ms gross** there and
absorbed a correctness fix out of it.



Each was measured before and after, and each leaves the output either identical
or different by an amount stated below.

### 3.1 One colour conversion for two LAB stages

`_scatter` and `_match_color` both operate in LAB and each converted for itself.
A BGR->LAB->BGR round trip is **1.9ms at 256** — more than everything the scatter
pass *does* with it. The conversion moved up to `_composite_impl`; both stages
now take and return LAB, which is what they always operated on.

**Scatter's marginal cost: 3.18ms -> 0.07ms at aligned 256.** Output identical
(one fewer uint8 quantisation, so marginally *more* accurate).

### 3.2 The seam feather is blurred at a resolution matched to its radius

A feathered mask is a smooth field by construction. At a 400px face the
frame-space feather is 16px, and that Gaussian over a ~500px region cost
**6.16ms** — the single largest operation in the compositor. Done at a quarter
and scaled back, **1.39ms**, and the two agree to within **0.006** on a mask that
runs 0 to 1.

The reduction follows the radius rather than being fixed: below 4px sigma the
blur is cheap anyway and too small to survive a resample, and losing a tight
feather would restore the hard edge the feather exists to remove.

### 3.3 Grain no longer generates fresh noise every frame

`np.random.normal` at region size cost **1.5ms at 256 and ~5.9ms at 500**, for a
field whose only requirement is to look like sensor noise. It now cuts a window
from a cached unit-variance tile at a random per-frame offset. The offset is
load-bearing: a fixed pattern would read as fixed-pattern noise, which is a worse
artefact than no grain.

### 3.4 Statistics are bounded before their transform, not after

`_estimate_noise` already knew a noise sigma converges on far fewer pixels than a
face region carries — and applied that to the *result*, striding the Laplacian
after computing it over every pixel. The colour conversion and the Laplacian were
still full size: **4.6ms at a 500px region**, now flat in region size.

`_texture_headroom` got the same treatment when it was written, and both now
share `_stat_window`.

### 3.5 `_add_grain` uses cv2 ops rather than numpy broadcasting

`_paste` already spells its composite out in `cv2.add`/`multiply`/`subtract` with
a comment explaining that `mask[:, :, None]` broadcasting expands into a
full-size temporary every frame. `_add_grain` did the broadcast anyway.
**4.2ms -> 2.4ms at a 500px region.** The inconsistency was the tell.

### 3.6 The scatter feature weight is built at a quarter resolution

Five soft holes in a soft mask. **0.43ms -> 0.08ms at 256**, agreeing to within
0.17 on a [0, 1] weight, at the circle edges only.

### 3.7 A correctness bug the audit found

`_paste` was passed `scale` meaning "the ratio between the swapper's crop and the
working resolution" (2.5) and used it as if it were the affine's geometric scale.
For a 400px face it computed the face's extent as **128**.

Two consequences, both invisible without measuring:

- the seam feather was 5px where it should have been 16 — a third of the
  intended transition, on the failure that was reported from footage;
- `texture.map_size` built the detail map at 128 for a 400px face, which is
  precisely the decimation the whole texture layer exists to avoid.

Fixed by computing the extent from the affine's determinant in
`_composite_impl` and passing that. **This made `production` slower** — paste
14.0 -> 17.3ms — because the feather is now the width it was supposed to be and
the texture map four times the area. Correct before fast.

---

## 4. Not acted on, and now closed

Three real items, measured at every preset, and the measurement closes them.

| | `fast` | `optimal` | `production` |
|---|---|---|---|
| `real` -> LAB at half size | 0.09ms | 0.63ms | 0.63ms |
| region padding 3σ -> 2σ | 0.37ms | ~0.6ms | ~3.5ms |
| drop `frame.copy()` | 0.01ms | 0.09ms | 0.48ms |
| **total** | **0.47ms** | **~1.3ms** | **~4.6ms** |
| **frame deadline** | **66.7ms** | **50.0ms** | **33.3ms** |
| **share of budget** | **0.7%** | **2.6%** | — |

At `fast` and `optimal` this is noise. At `production` it is 4.6ms against a
16ms overrun, so it does not fix the one case that needs fixing either.

**The reason `fast` is not cheaper than `optimal` is worth keeping.** At a normal
seated distance both clamp to the *same* aligned size:

    fast        ceiling 192, face  76px -> aligned 128
    optimal     ceiling 256, face 101px -> aligned 128
    production  ceiling 320, face 400px -> aligned 320

`_ALIGNED_MIN` is 128 and both faces are under it, so the aligned-space work —
smoothing, colour, detail, scatter — is byte-for-byte identical between the two
presets. Only the frame-space region differs, 104px against 135px. **The
compositor is not where `fast` and `optimal` differ at live face sizes.** What
`fast` actually buys is uplink, a smaller detector input, and the XSeg pass
turned off, and none of those is compositor work.

**Revive this section if** `production` becomes a live target, or if the pod's
CPU turns out far slower than the one these were taken on — in which case
everything here scales together and the *share of budget* column, which is what
decides it, stays where it is.

### The items, for whoever revives them

- **`_match_color` converts `real` to LAB at full size** and uses it for two
  things: masked mean/std, and a residual that `_match_illumination` immediately
  downscales to 1/8. Neither needs full resolution. Measured: **0.63ms** saved at
  either 256 or 320, by converting a half-size copy instead.
- **The region of interest is padded by 3 sigma** of the feather, which at a
  400px face is 52px on each side — making every frame-space operation run over
  1.5x the pixels. 2 sigma would take the region from 504px to 472px, 88% of the
  area: **~3.5ms at `production`**, ~0.6ms at `optimal`.

  **Filed here wrongly, and corrected:** this is not free. At 2 sigma the
  feather's tail truncates at ~5% of peak rather than ~0.1%, so the mask does not
  quite reach zero at the region border. Probably invisible — but it is the seam
  again, and the seam is what footage just complained about. It belongs in §6
  with the others until somebody has looked at it.
- **`frame.copy()` in `_paste`** duplicates the whole frame (1.5MB at 960x540) to
  write back a region. Writing into a view of the input would avoid it, at the
  cost of mutating a caller's array — which the current signature promises not
  to do. Measured: **0.48ms** at 960x540, 0.09ms at 640x360.

**Total for this section: ~1.3ms at `optimal`, ~4.6ms at `production`** — which
is why none of it has been done. At `optimal` it turns 7.8ms into 6.5ms of a 50ms
budget; at `production` it takes 48.8ms to 44ms against a 33ms deadline. Neither
changes an outcome.

---

## 5. Not acted on: structural

- **`_smooth` and the elementwise chain are near-inherent.** At aligned 320 a
  three-channel float array is 1.2MB, and the compositor allocates roughly
  **28 of them per frame** across smoothing, colour, detail and paste. The cost
  is memory bandwidth, not arithmetic, and no single op is wasteful. This is the
  thing a GPU port actually addresses.
- **JPEG per frame, twice each way.** The desktop encodes, the pod decodes,
  processes, encodes, and the desktop decodes. At 640x360 q70 that is ~30KB a
  frame, ~4.8Mbps up — and CLAUDE.md already records the uplink as the untested
  hypothesis for the latency that did not go away when compute halved. It is the
  largest single item in this document and it is not a compositor change. See
  [§9](#9-why-the-wire-format-is-jpeg-and-what-should-replace-it) for why the
  answer is a video codec rather than sending frames raw, which is 23x the
  bytes.

---

## 6. Realism risk register

The things that would buy time and would **not** be free. None has been done.
Each is listed with what it would change, because "optimisation" that quietly
moves the output is not optimisation.

| Change | Would buy | What it risks |
|---|---|---|
| `_match_detail` statistics on a sampled window rather than the whole mask | ~1.5ms at 320 | The detail ratio is the number that decides how much high-frequency energy the face carries. Sampling the centre of the region biases it toward nose and mouth — *structure*, not skin — exactly the bias found and fixed in `_texture_headroom`. Would need the same feature exclusion, and then verification that the ratio is stable against the current whole-mask figure |
| Lowering `production`'s `aligned_size` ceiling from 320 | several ms | This is a realism knob, not a performance one. It sets how much of the swap's and restorer's output survives. `_aligned_size` already follows the face's real size; lowering the ceiling would discard detail that is present |
| Reducing the feather again | ~1ms | It was just widened, deliberately, to fix a seam reported on real footage. Narrowing it to save a millisecond reverses the fix |
| Dropping the second (frame-space) feather | ~1.4ms | The aligned-space feather is a fraction of a *crop*, so it shrinks by the warp's scale on the way to the frame. That product being unwatched is what produced the 1.4px transition in the first place |
| Grain at reduced resolution | ~1ms | **No.** `_add_grain`'s own comment explains why grain is added in frame space at full resolution: resampling noise filters it into blobs. It is the one field in the compositor that must not be built small |
| Skipping the noise estimate on some frames | ~0.4ms | The estimate is what matches grain to the plate. Holding it across frames is defensible; guessing it is not |

---

## 7. What the pod will say

Every number here is a laptop CPU. That is the right machine to *find* waste on —
a redundant colour conversion is redundant everywhere — and the wrong one to
quote absolutes from.

**A GPU does not touch any of this.** The models are ONNX on the card; the
compositor is NumPy and OpenCV on the CPU. Renting a faster card does not speed
up a Gaussian blur.

**A pod's CPU is not this one, and is not constant across pods.** CLAUDE.md
records the whole compositor-plus-paste-plus-encode bucket at ~20ms on an L4
instance and ~10.3ms on a 4090 instance, and corrects an earlier claim that the
non-GPU portion was a fixed floor. It scaled with the instance.

So read the ratios here, not the magnitudes — and then read the pod's own
report, which now itemises every stage in this document by name.

---

## 8. The GPU question, answered with this data

**At `optimal`: no.** The compositor is 7.4ms of a 50ms budget with both new
realism layers on. Moving it would save single-digit milliseconds against a
felt latency dominated by a ~350ms round trip to Romania.

**At `production`: the compositor cannot fit on the CPU**, and no amount of the
kind of work in §3 changes that. 39ms of elementwise passes against a 33ms
deadline is a factor, not a margin. The options are honest ones:

1. **Treat `production` as a RENDER preset, not a LIVE one.** Offline video does
   not have a frame deadline, and this is the preset whose whole purpose is to
   spend compute on quality.
2. **Port the compositor to torch.** CLAUDE.md's route —
   `affine_grid`/`grid_sample` for the warps, elementwise ops for the rest — and
   it is now a *code* change rather than a dependency one: torch is on the pod,
   `pipeline/core.py` imports it unconditionally, and the instance image supplies
   it. Transfers are trivial (a 640x360 frame is 691KB, ~0.03ms over PCIe 4.0).
   This is where the 28-allocations-per-frame problem in §5 actually goes away.
3. **Do neither until the pod's latency report says compute is the largest
   term.** It currently does not, at `optimal`, which is the default.

If a GPU port ever happens, one item deserves to go with it: **JPEG encode**.
It is CPU, it is on the critical path in both directions, NVJPEG exists, and it
has never been measured on its own because it sat in one bucket with the
compositor. It can be now — every compositor stage is itemised, so encode is the
remainder.


---

## 9. Why the wire format is JPEG, and what should replace it

The obvious question about a 2ms encode and a 1.3ms decode is why they happen at
all: WebSocket carries binary, so why not send the frame raw and skip both?

Because the codec is not what costs. Measured on the same machine as everything
else, on a face-like frame rather than noise (noise is the worst case for any
codec and would flatter raw):

| preset | encode | decode | JPEG/frame | raw/frame | **JPEG** | **raw** |
|---|---|---|---|---|---|---|
| `fast` 480x270 q60 | 0.52ms | 0.57ms | 12.6 KB | 379.7 KB | 1.6 Mbps | **46.7 Mbps** |
| `optimal` 640x360 q70 | 0.76ms | 1.32ms | 29.3 KB | 675.0 KB | 4.8 Mbps | **110.6 Mbps** |
| `production` 960x540 q85 | 2.20ms | 3.61ms | 126.7 KB | 1518.8 KB | 31.1 Mbps | **373.2 Mbps** |

**Raw trades 2.1ms of CPU for 23x the bytes**, on the one leg already suspected
of being the bottleneck. The decisive number is not bandwidth but *serialisation
time* — how long the bytes take to leave the machine, which is latency whatever
the round trip does afterwards. One `optimal` frame, raw:

    20 Mbps uplink (a good home connection)   276 ms   per frame
    100 Mbps                                   55 ms
    1 Gbps                                      5.5 ms

against a 50ms frame budget. Raw needs a sustained **>110 Mbps upstream** merely
to keep pace with the frame rate, before any margin for the round trip. The same
frame as JPEG is 12ms on that 20 Mbps link. So raw is not a smaller overhead, it
is the same overhead moved from a place that costs 2ms to a place that costs
hundreds — and CLAUDE.md already records the uplink as the untested hypothesis
for the latency that did not go away when compute halved.

Bitmap, PNG and lossless WebP all land the same way or worse: PNG is slower to
encode than JPEG *and* several times larger than JPEG on photographic content,
and lossless anything is within a small factor of raw.

### What is actually worth doing

**1. Skip the codec entirely when there is no network.** When the pipeline runs
on the operator's own machine there is no wire, and a frame still goes
JPEG -> socket -> JPEG for a trip between two processes. The desktop already has
`update_from_numpy` for the filtered path, so the receiving half exists. This is
the one case where the question's instinct is exactly right, and it matters
because [LOCAL_GPU_SETUP.md](LOCAL_GPU_SETUP.md) records that a local GPU is not
faster than a rented one, it is *closer* — removing the ~350ms round trip is the
largest single latency improvement available.

**2. Replace JPEG with a video codec, not with raw.** JPEG compresses each frame
alone. A video call is the best case there is for inter-frame prediction: the
background does not move, and most of the frame is identical to the last one.
H.264 at the same perceptual quality runs 5-10x smaller than per-frame JPEG —
4.8 Mbps becomes roughly 0.5-1 Mbps. That is the item that would actually test
the uplink hypothesis, and on the return leg the pod has NVENC sitting idle on a
card it is already renting.

Three things to get right if it is built: **zero-latency tuning** (B-frames
reorder output and would add a frame or more of delay, which is the opposite of
the goal), **keyframe policy** for reconnects and for a client joining mid-stream,
and a **dependency** — PyAV or ffmpeg on the desktop, where the pipeline's
requirements do not currently reach.

**3. Move the codec to the GPU (NVJPEG/NVENC) on the pod.** Keeps the wire
format, removes 2.2ms of encode and 3.6ms of decode from the pod's CPU at
`production`. Modest at `optimal`, and it only helps the pod's half.

### One realism note before anything changes

**The JPEG round trip is partly aligned with the design target, not opposed to
it.** CLAUDE.md names compression artefacts as part of what a real video call
looks like, and the swap currently runs on a JPEG-decoded frame — so the
compositor matches colour, detail and grain against a frame that already carries
those artefacts, and the output inherits them. Switching to H.264 changes the
artefact *character*: blocking and mosquito noise become different blocking and
different temporal smearing. That is not obviously worse and may be better, since
it is what every other participant's video actually looks like. But it changes
what `_match_detail` and `_add_grain` are matching to, so it is a realism change
wearing a bandwidth change's clothes and belongs on footage before it is
believed.

### Not a finding: the return leg is already careful

The desktop decodes each received frame **once** and shares it between the
display and the virtual camera, and when no filter is enabled it does not decode
at all — Qt loads the JPEG directly. There is no redundant codec work to remove
there.

---

## 10. How to reproduce

The per-stage numbers come from `FaceCompositor.last_stage_ms`, which the
pipeline already records for every frame and reports when a stream stops. To take
them on the pod, run a stream and read the `PERF` block; to take them locally,
drive `composite()` in a loop and sum the same dict.

The primitive-level numbers — which Gaussian, which conversion — come from timing
the individual `cv2` calls at the sizes the profile says they run at. That order
matters: profile first to find the stage, then probe the stage to find the
operation. Guessing which `cv2` call is expensive is how the noise generator
survived for as long as it did.


## 11. The measurement history, moved out of CLAUDE.md

These sections lived in CLAUDE.md until it hit its size limit. They are the
working-out behind the headline numbers that file still carries: what each
speed lever was actually worth, whether restoration earns its cost on a
webcam-sized face, the transport diagnosis, and why restoration cost 110ms.

### What the levers are actually worth (same session, same clip)

Read the **frames processed per run**, not the p95 column. `LatencyBudget` was
never reset between streams, so each report covered every frame since the
process started (1018, 4399, 7740, 8781, 9746, 12775) and every p95 was
diluted by its predecessors — a run with restoration *off* still reported a
`restore` percentile, because those were the baseline's frames. Fixed now
(`LatencyBudget.reset()`, called from `_run_stream_impl`), but every sweep
taken before that fix has to be read this way.

| config | frames in its 60s | ms/frame |
|---|---|---|
| baseline | 1018 | 58.9 |
| **no restoration** | 3381 | **17.7** |
| **`restore_min_face=200`** | 3341 | **18.0** |
| `aligned_size=128` | 1041 | 57.6 |
| hyperswap_1a_256 | 965 | 62.2 |
| hyperswap + no restoration | 3029 | 19.8 |

**Restoration off is 3.3x, and it HOLDS the deadline** — the first `[HOLDS]`
verdict this project has produced. `restore_min_face` lands on the same number,
which is the cross-check it was built for: the shippable, config-level lever
reaches the same floor as switching the stage off wholesale.

Two questions closed, both negative. **`aligned_size` is not the cost** — 128
against 256 changes nothing. **hyperswap is slightly worse, not better**: 62.2
against 58.9, and 19.8 against 17.7 with restoration off. The 256px swap costs
2-3ms and buys back *nothing* in restoration time, so on speed grounds it is
just a bigger swap. Its appearance remains unjudged.

### Does restoration actually help a 101px face?

Measured on the recorded webcam clip, 24 frame pairs at identical indices,
restoration off against CodeFormer at 512:

| `tools/compare_frames.py` (ideal = 1.00) | off | CodeFormer 512 | change |
|---|---|---|---|
| high-frequency detail, face / frame | 0.584 | 0.614 | **+0.030** |
| sensor noise, face / frame | 1.500 | 1.500 | **+0.000** |
| gradient at mask edge | 1.028 | 1.038 | +0.010 |

**29.4ms buys +0.03 on the one metric it moves at all.** The face sits 0.42
short of matching the frame's detail; restoration closes 7% of that gap and
leaves noise and seam unchanged to three decimals. Both configurations produce
the same verdict lines — "softer than the frame", "no seam detected", "motion
blur consistent".

That is what the geometry predicts. 86% of what CodeFormer produces is
discarded at `compositor.py:515`, one warp after it is created, so the stage
cannot move the metric much and does not.

**What this does not cover, and should not be read as covering.** These are
per-frame image statistics over 24 frames. They say nothing about **temporal**
behaviour, and shimmer between frames is a large part of what reads as AI. They
are also not a person looking at a face. Strong evidence, not proof.

The noise row reads 1.50x in **both** configurations, so it is not caused by
restoration. It is also not necessarily a defect: this clip was recorded in
poor light, and the source face is fair-complexioned and well lit against a
dark-complexioned, under-lit, visibly noisy target. That is the hardest case
for colour matching and a plausible cause on its own. Re-measure on
better-matched footage before treating grain matching as overshooting.

### Restoration models, benchmarked in isolation (RTX 4090)

Raw inference only — 100 runs, random input, no compositing:

| model | crop | inference | file |
|---|---|---|---|
| codeformer | 512 | 29.4ms | 377 MB |
| gpen_bfr_512 | 512 | 37.5ms | 284 MB |
| **gpen_bfr_256** | **256** | **5.4ms** | **76 MB** |

Note which way `gpen_bfr_512` falls: **slower** than CodeFormer at the same
resolution. The saving is entirely **resolution**, not architecture. GPEN is
not a lighter model; 256 is simply a quarter of the pixels.

**`gpen_bfr_256` has now run in the pipeline** (2026-09-05, RTX 4090, 2267
frames): `restore` p50 **7.7ms**, inside a 25.9ms frame against a 66.7ms
deadline. The 5.4ms isolated figure was honest — compositing adds ~2.3ms of warp
around it — and the ~27ms whole-frame estimate was pessimistic by nearly half.
It still **has not been judged on footage**, which is the part that decides
whether it belongs. Both GPEN files are on the volume at `/workspace/models/`.

Why 256 is the interesting number rather than "off": restoring at 512 and
warping down to a 128-192 aligned space is *supersampling*, and some of that
cost buys antialiasing and stability rather than nothing. At 256 into a 192
aligned space the supersampling margin survives, along with all the
low-frequency work — tone, structure, artifact cleanup — that the downsample
does not destroy. What is given up is the 512-to-256 octave, which is the one
the final resize deletes anyway. It also has **no fidelity weight**: GPEN takes
one input, so `enhancer_weight` would stop meaning anything and only
`enhance_strength` would remain.

### The bottleneck has moved to the transport

With restoration at 256 the frame is ~27ms against a 50ms deadline, and the
reported symptom changed shape with it: **the stutter went away and the lag did
not**. That is the diagnosis. Stutter is throughput — frames arriving faster
than they can be processed. Lag is latency — and halving the compute did not
move it, so compute was not what was holding it.

The chain, end to end, with what each part costs:

    webcam capture
      -> JPEG encode (desktop)
      -> UPLINK          ~30 KB/frame, ~4.8 Mbps at 640x360 q70 @20fps
      -> inbound queue   was 10 deep = 500ms of pure latency
      -> process         ~27ms                    <- no longer the problem
      -> JPEG encode (pod)
      -> DOWNLINK
      -> jitter buffer   started at 400ms, adapted slowly
      -> decode -> display

**None of this was visible.** `RTTTracker` computed true glass-to-glass latency
from a capture timestamp that rides with every frame, and had done all along —
nothing displayed it, logged it to the UI, or reported it. "It feels sluggish"
could not become "RTT is 210ms, the buffer adds 60, the pipeline uses 27".

Fixed, in order of how much they were costing:

- **The readout exists.** `Bridge.latencyText` publishes RTT p50/p95, buffer
  depth and uplink Mbps every two seconds, shown top-right in the viewport
  beside the other badges — never drawn on the frame, for the usual reason.
  Read it against the pipeline's own per-stage report: **the difference between
  the two is network and encode**, and on a remote pod that is most of it.
- **The inbound queue dropped the wrong frame.** On a full queue the handler
  refused the *arriving* frame and kept the backlog, so under pressure the
  pipeline chewed through stale frames while discarding the only current one —
  the face lagged by the whole queue depth and stayed there. It now evicts the
  oldest. Depth went 10 -> 2: anything waiting there is a frame the operator
  has already moved past.
- **The playout buffer started at 400ms** and converged slowly with one
  symmetric alpha, so even a nearby pod felt heavily delayed for the first
  seconds — exactly when an impression forms. Now 120ms initial, a 50ms floor
  (one frame interval at 20fps rather than 80ms), and **asymmetric** smoothing:
  rise fast because a late buffer glitches visibly, fall slow because an early
  one underruns. One alpha has to be slow in a direction; 0.2 was slow in both.

**What is still only a hypothesis: the uplink.** The desktop sends a JPEG per
captured frame, ~4.8 Mbps at the `optimal` preset, and receives about the same
back. Home connections are usually asymmetric with far less upstream. A
saturated uplink queues frames in the OS send buffer, which reads as **latency
while throughput still looks healthy** — the exact reported symptom. The
readout now carries the number; the cheap test is to switch to `fast`
(480x270 q60, ~1.4 Mbps) and see whether latency falls by far more than the
~10ms of compute that saves. If it does, the answer is encoding, not the GPU.

**The term that dominated everything: distance — and it is what the move to
Vast was for.** The pod was `EU-RO-1`, Romania, against an operator in West
Africa: a physical floor of roughly 80-120ms round trip at best and typically
worse.

The obvious fix was a nearer RunPod datacenter, and it does not exist.
Per-datacenter stock, queried directly: **EU-FR-1 carries no eligible GPU at
all**, EU-NL-1 has a single L40S, and RunPod has fifty datacenters and **none
in the UK**. EU-RO-1 was not inertia — it was the only European datacenter
holding 4090s.

Vast has verified 4090s in the UK at $0.31/hr, cheaper than the Romanian card
they replace, on 885 MB/s uplinks. See [docs/VAST_MIGRATION.md](docs/VAST_MIGRATION.md).

**What is still unmeasured is the RTT itself.** Everything above is a proxy for
it. The readout exists; point it at a UK instance and read it before believing
any of this.

**The standing conclusion: stop optimising the pipeline for latency.** There is
~23ms of headroom under the deadline and the felt delay is dominated by terms
the GPU does not touch. Further compute work should be justified by the readout
showing compute as the largest term, which it currently is not.

### Why 110ms: restoration ignores how big the face is

**The dominant cost is spent on interpolated data.** In the measured session the
face was **101x129 px** in a 640x360 frame. The chain it went through:

    face in frame          101 x 129   <- the only real information
    swap native            128 x 128   <- inswapper_128 output, the ceiling
    aligned space          256 x 256   <- follows face size, has a floor
    FFHQ restore crop      512 x 512   <- ALWAYS 512, regardless

`CROP_SIZE = 512` is hard-coded through `_ffhq_geometry` and `_build_ffhq_crop`
(`compositor.py:496`, `:524`), because CodeFormer is trained on FFHQ 512 crops.
So a 101px face is upsampled about 20x in pixel count, the heaviest model in
the pipeline runs on the result, and the output is squeezed back into a 101px
hole. Conv cost scales with pixel count, so this is roughly **4x the compute of
restoring at 256, and 16x of 128** — spent reconstructing detail that was never
in the source.

**The codebase already holds the correct principle and does not apply it here.**
`_aligned_size` (`compositor.py:373`) says the working resolution "follows how
many frame pixels the face actually covers ... and, more importantly, is not
upsampled to a detail level their webcam never captured." That reasoning is
right, and it governs a stage costing a few milliseconds while the 110ms stage
ignores it entirely.

**This is the largest single lever available, and larger than fp16, TensorRT and
a 4090 combined.** Options, cheapest first:

1. **Skip restoration below a face-size threshold.** Config-level, no new
   model. The question it rests on is a footage question, not a latency one:
   does 512-space restoration visibly improve a 101px face whose swap was
   generated at 128? Test with `--debug-frames` before assuming either answer.
2. **Restore every Nth frame**, letting `temporal_alpha`'s aligned-pixel EMA
   carry the gap. Previously declined as too risky to what the operator sees;
   that was decided before knowing restoration is 75% of the frame.
3. **A restoration model that accepts a smaller input**, or a re-export of
   CodeFormer at 256. Changes what the output looks like, so it is an A/B, not
   a swap.

**Option 3 is closed, and option 1 is the live one.** `codeformer.onnx`
declares:

    INPUT  input   [1, 3, 512, 512]   tensor(float)
    INPUT  weight  []                 tensor(double)
    OUTPUT output  [1, 3, 512, 512]

Static and square. So `restore_size` cannot make this model restore at 256 —
`Enhancer.crop_size` warns once and holds at 512, which is the declining path
working as designed rather than a bug. **Restoring smaller needs a re-export,
not a config change.** `restore_min_face` is therefore the only config-level
lever against the 39.5ms, and it needs no new model because skipping is free.

The general lesson is cheaper than the sweep that would have found it: **read a
model's declared input shape before sweeping a shape lever.** Five seconds of
`InferenceSession(...).get_inputs()` replaced a paid measurement run.

`restore_size` and `restore_min_face` are config fields, on `set_realism`, the
CLI, the env and the sweep. Two properties matter:

- **`restore_size` is a request, and the model answers it.**
  `_spatial_size` reads the ONNX input's declared shape at load;
  `Enhancer.crop_size` honours a fixed export over the config and warns **once**
  rather than throwing per frame on the live path. So option 3's real question —
  *is facefusion's `codeformer.onnx` exported with dynamic spatial dims?* — is
  answered by one line of the pod's startup log, and a `restore_256` run whose
  `restore` equals the baseline exactly is what "no, it is fixed" looks like in
  the sweep. It is not a lever that quietly does nothing.
- **The seam is a fraction of the crop, not a pixel count.** `_FFHQ_ERODE` and
  `_FFHQ_FEATHER` reproduce the old 5px erode and 6.0 sigma exactly at 512, so
  nothing changes at the default, and a 256 crop gets the same *seam* rather
  than twice as hard an edge. Otherwise a resolution A/B would also be a
  feathering A/B and neither would be readable.

`_ffhq_geometry` at 256 is exactly half the matrix it is at 512 — the framing is
identical, only the sampling rate changes — so this is a resolution comparison
and nothing else. Both default to current behaviour: 512, never skip.

Note what is *not* the problem, so it does not get optimised by mistake:
transfers are trivial (a 512x512x3 fp32 tensor is 3MB, ~0.1ms over PCIe 4.0,
even six round trips are under 2ms), and the Python layer does not appear in
the measurement at all.


## 12. The L4 session's own plan, superseded

Kept because the *lesson* survived the numbers: the reasoning about
where time goes held, and every prediction of how much a lever would
buy did not. hyperswap has since been run (62.2ms, slightly worse than
inswapper), `no_restore` has been run, and the 4090 figures in §11
replace the estimates below.

### As written after the L4 session

- **The first stream after a pipeline start pays model warm-up**, tens of
  seconds, inside its own window. A 40s capture produced zero frames for this
  reason and looked like a broken config. The sweep hides this with a discarded
  warm-up pass; anything else driving the stream needs its own.
- **Nothing can be copied off the pod.** `orchestrator.py push` is local->pod
  only, port 9000 is the only opening, and the SSH proxy carries no SFTP — so a
  45 KB montage of the comparison frames could not be brought home. An
  `orchestrator.py pull` over the same WebSocket path `push` already uses is
  what makes visual review routine instead of impossible.
- **`orchestrator.py run` used `PATH=... <cmd>`**, which binds only to the
  first word of a line, so the second half of any `&&` chain ran under
  `/usr/bin/python`. Now `export PATH=... && <cmd>`.

**Settled by this:**

- Every model is confirmed on `CUDAExecutionProvider`. No silent CPU fallback.
- **`cuda_graphs` and `cuda_streams` measured flat** (144.4 / 146.1ms — noise).
  A 110ms model is not waiting on kernel launch overhead. Both can be dropped.
- **Numba is closed.** The whole compositor is ~20ms; making it free still
  leaves 126ms. Argued against on reasoning before, now on a number.
- `fp16` and `trt` **never ran** — no converted weights existed, and `trt_gpus`
  correctly declined an engine build on an L4.

**hyperswap has never been run.** Every measurement used the default
`inswapper_128`; the weights are on the pod but unused. It matters for speed,
not just looks: hyperswap is 256px native against inswapper's 128, and its
profile asks for *less* restoration (`enhance_strength` 0.5 vs 0.7) because the
swap needs less. A bigger swap that buys a cheaper restore may be a net win, or
may just be a bigger swap — the sweep now covers both.

**The first run of the next session should be `no_restore`.** It bounds
everything: whatever remains with restoration off is what no amount of work on
restoration can remove. `tools/sweep_levers.py` now leads with it, plus
`aligned_128`, `hyperswap` and `hyperswap+no_restore`.

**Continue from [docs/PENDING_WORK.md](docs/PENDING_WORK.md) §2b.0**, in order:

1. **Convert fp16 on the pod** — the only untested lever aimed at the 110ms:
   `orchestrator.py run "python tools/convert_fp16.py /workspace/models/codeformer.onnx"`
   (needs `pip install onnx onnxconverter-common` there first). Then re-sweep.
2. **Judge it on footage**, not latency alone — restoration is what decides
   whether output reads as a call or as AI.
3. **Measure again on a 4090.** Estimated ~72ms total, so a real 2x but still
   short of 50ms alone; **4090 + fp16** is the combination that plausibly
   holds. Needs `terminate` then `start` — `resume` cannot move a pinned pod.
4. **Reconsider restoration decimation.** Declined earlier as too risky to what
   the operator sees; that was decided before knowing restoration is
   three-quarters of the frame.
5. Only then the XSeg overlap and pipelining — both are bounded by the ~20ms
   that is *not* restoration.

Estimates above are labelled as such. This session's lesson was that the
reasoning about *where* time goes held, and the predictions of *how much* each
lever would buy did not survive contact with a measurement.

**Frame rate, estimated from the measured L4 numbers.** GPU stages scale with
the card; the ~20ms of CPU compositing and encode does not, which is what sets
the floor:

| | restore | detect | CPU | total | fps |
|---|---|---|---|---|---|
| L4 (measured) | 110ms | 16ms | ~20ms | 146ms | **~7** |
| RTX 4090 (est.) | ~44ms | ~6ms | ~20ms | ~70ms | **~14** |
| RTX 4090 + fp16 (est.) | ~24ms | ~6ms | ~20ms | ~50ms | **~20** |

So a 4090 roughly doubles the frame rate and still misses 20fps on its own.
**4090 + fp16 is the first combination that plausibly holds the `optimal`
preset**, and it lands right on the deadline rather than comfortably inside it.
