# Phantom — Realism

## The target

**A normal video call.** Someone's webcam, in their room, with their lighting.
Sensor noise. Compression artefacts. Skin that has pores and blemishes and
uneven tone. A face that is a little soft because the camera is cheap and the
bandwidth is limited.

This is worth stating plainly because the obvious instinct — make the output as
clean and sharp as possible — actively works against it. A face swap fails on a
call in three ways, and only one of them is about the swap being "bad":

1. **It looks too good.** Poreless, evenly lit, perfectly sharp skin on a 720p
   webcam feed. Nothing else in the frame looks like that.
2. **It has a seam.** A visible edge, halo, or colour step where the swapped
   face meets the real head. The eye finds boundaries before it finds anything
   else.
3. **It moves wrong.** Shimmer, jitter, or a face that lags a fraction behind
   the head it is attached to.

Everything in this document exists to fight one of those three. Several stages
deliberately *degrade* the output — adding noise, holding back sharpening,
keeping part of the unrestored face — because matching the frame matters more
than looking good in isolation.

---

## Face restoration

### What it is for

The swapper's output is a 128×128 crop. Upscaled to the size of a face in the
frame it is soft and lacks texture, which reads as a blur patch. Restoration
reconstructs plausible detail — eyes, skin texture, hair edges.

### Why it is dangerous

Restoration models are trained to make faces look *good*, not to make them look
*like the input*. Run at full strength they produce the beautified, poreless,
evenly-lit look that is the single strongest "this is AI" signal on a call.

So restoration is applied and then **partially undone**.

### Two backends

Selected by `enhancer_model`:

| | CodeFormer (default) | GFPGAN |
|---|---|---|
| Runtime | ONNX, on the existing onnxruntime session | torch + `gfpgan` package |
| Extra dependency | None | torch, basicsr, torchvision shim |
| Model | `codeformer.onnx`, downloads on first use | `GFPGANv1.4.pth`, manual |
| Strength control | **Fidelity weight** (`enhancer_weight`) | None |

CodeFormer is the default entirely because of that last row. Its fidelity weight
runs `0.0` (restore hardest, hallucinate most) to `1.0` (stay closest to the
input). GFPGAN v1.4 has one setting — beautify — and no way to ask for less.

GFPGAN is retained so the two can be compared on real footage, not as a
recommendation.

### Two knobs, not one

They do different things and both matter:

- **`enhancer_weight`** (CodeFormer only) — how hard the *model* restores.
  Lower means more invented detail.
- **`enhance_strength`** — how much of the restored result is blended back over
  the unrestored swap. At `0.7`, 30% of the original imperfection survives.

Turning either one down makes the face less impressive and more believable.
Both default to `0.7`, and are **the same in every quality preset** — a preset
buys compute, not a different look. Change them globally in `_LOOK`
(`pipeline/api/schema.py`), or per run with `--enhancer-weight` /
`--enhance-strength` or `set_realism`.

### FFHQ framing

Both models are trained on **FFHQ-aligned 512×512 crops** and have strong priors
about where eyes and mouth sit in the frame. Feeding them the swapper's tighter
arcface crop measurably degrades the result.

So `FaceCompositor` warps into FFHQ space around the restore call. FFHQ framing
is wider than arcface, so warping the swap alone would leave an empty ring — the
crop handed to the restorer is **the real frame in FFHQ framing with the swapped
face composited over it**, with the seam between them feathered so the model does
not try to reconstruct a hard rectangular edge as a feature.

Only what the swap covers survives the mask downstream, so the real face at the
edges never reaches the output.

### Graceful degradation

If the configured backend cannot load, the other is tried. If neither loads,
restoration is disabled and the pipeline keeps running with the raw swap. A
missing model is never fatal.

---

## Killing the seam

The seam is the failure the eye catches fastest, and five separate things work
on it.

It is also the failure that was **reported on real footage** — "very noticeable,
like the face pasted on target" — after everything below was already in place,
which is why items 4 and 5 read the way they do. Two of the three causes were
arithmetic rather than hypothesis, and the third is not fixed yet. The full
diagnosis is in [TEXTURE_PIPELINE.md](TEXTURE_PIPELINE.md) §13.

**1. The mask follows the real face.** `FaceMasker` builds a convex hull from
the 106 landmarks InsightFace already computes, so the boundary tracks the actual
jawline instead of assuming an ellipse. The hull is expanded 10% radially, with
an extra upward push of 16% of hull height to recover forehead.

A **convex** hull, though, and that is a limitation rather than a detail: a
convex hull has no concave points by definition, so it cannot follow an
under-chin or the notch at a temple at any expansion. Those are the two places a
"pasted on" impression usually comes from, and it is the one reported cause of
the seam that is still open.

That forehead push is deliberately moderate. Masking up past the hairline hides
the hairline seam, but produces a much worse tell: swapped skin where hair should
be. A slightly short forehead is the better failure.

**2. Nothing invalid gets composited.** A white frame warped through the same
affine tells the masker exactly which aligned pixels came from inside the frame.
Without this term, a face near the frame edge bleeds a black border into the
composite.

**3. Occlusion is respected.** Optional DFL XSeg segmentation removes hands,
microphones and hair crossing the face from the mask, so they are not painted
over with swapped skin. Degrades to hull-only if the model is unavailable.

**4. The mask is eroded before it is feathered** (`mask_erode`, 3% of the
aligned crop). Without this the hull is expanded 10% and then blurred, which puts
the *midpoint* of the transition outside the landmark silhouette — on neck below
the chin, on hair at the temples. A boundary is most visible exactly where the
material either side of it differs, and that placed it there. Eroding first moves
the soft part onto skin while leaving the outer extent of the transition roughly
where it was, so coverage is preserved and only the softness moves.

Note this is not "extend the mask". Growing the *coverage* is the half that
backfires, for the reason item 1 gives about the forehead: swapped skin where
hair should be is the worse tell.

**5. The edge is feathered twice.** Once in aligned space (5% of crop size) as
anti-aliasing, and again in frame space after warping back — and the second one
is what the eye actually sees, because it is the only one measured against the
face's real size.

That distinction was got wrong once and is worth keeping. Both feathers used to
be fractions of *their own space*, and both spaces are larger than the face: 5%
of a 256 aligned crop lands as 0.91px on a 101px face, and the frame-space blur
was 1% of the region of interest, or 1.1px. Neither constant was wrong alone.
Their product was a **1.4px transition on a 101px face** — 1.4% of its width, a
hard edge, and nothing was looking at the product. `mask_feather` (4%, floored at
2px) now measures against the face's own extent, and the region of interest's
padding grows with it, since a blur wider than its padding reflects off the
border and leaves the mask never reaching zero along it.

---

## Matching the frame

Three stages exist purely to make the composited face belong to the image around
it.

### Colour

Two passes, sampled **inside the mask only**. Sampling a bounding box instead
pulls in hair and background, which is how a bright window behind someone ends up
shifting their skin tone.

**Global** — a mean/std transfer in LAB. Ramps continuously with measured colour
distance rather than triggering on a threshold, so it can never snap on and off
between frames. The ramp starts at 1.5 LAB units and reaches full at 9.5. It
used to start at 4.0, which left a sub-4-unit mean difference corrected by
*nothing* — and a difference that small is invisible across a face and plainly
visible across a boundary, which is where the eye compares hardest. The
anti-snapping property the floor looked like it was protecting is delivered by
the ramp; the floor only has to clear estimator noise. The L-channel standard deviation is damped to 50%: matching L
*mean* fixes brightness and matters; forcing L *std* flattens facial contrast.

**Illumination** — a global shift is only correct when the light is flat, and a
video call almost never is; there is a window or a lamp on one side. Under
directional light the real face carries a brightness gradient the swap does not,
and no single shift can match both ends of it, so it lands correct on average and
visibly wrong at one edge — which is a seam. This second pass corrects the
low-frequency difference that survives the global match.

It is computed on a residual reduced to 1/8 resolution, which is what stops it
copying the target's face onto the swap: facial features cannot survive an 8×
reduction, only illumination can. The blur is normalized by the mask, so
out-of-mask pixels contribute nothing, and the correction is clamped to ±12 LAB
units and faded out with the mask so it cannot introduce an edge of its own.

Note the two are gated differently on purpose. The global pass engages only once
the colours actually differ, since correcting a match is pure risk. The
illumination pass always runs, because the case it exists for is one where the
global means already agree and only their distribution across the face differs —
gating it on the same distance would switch it off exactly when it is needed.

Both controlled by `color_correction` (on/off) and `color_strength` (scale).

### Detail

The swap is *softer* than the frame before restoration and *sharper* after it, so
the correction has to work in both directions. Solving for a blur radius only
corrects one way and is unstable; instead the high-frequency band is separated by
a Gaussian and scaled to match the target's energy, clamped to 0.6–1.6× so a flat
region cannot turn blotchy.

The band split scales with `aligned_size`, so "texture" means the same physical
detail at every preset.

### Grain

The generated face is noise-free. Everything around it carries sensor noise and
JPEG artefacts. That mismatch is read instantly even when a viewer cannot name
what is wrong.

Noise sigma is estimated from the surrounding region via the median absolute
deviation of its Laplacian — median-based so that genuine facial detail and edges
do not inflate it — then added back over the composited area.

Two details matter: grain is added in **frame space**, not aligned space, because
warping a crop down to face size would filter the noise into blobs. And it is
**monochrome** — one luma field added to all three channels — because independent
per-channel noise looks like coloured confetti, nothing like a camera.

Controlled by `grain`. It is on in every preset; it is cheap and it is the single
largest believability win per millisecond spent.

---

## Stability over time

Two EMAs, at different points, doing different jobs.

**Landmark EMA** (`alpha`, `LandmarkStabilizer`) — smooths `kps`, which drives
the swap warp, and `landmark_2d_106`, which drives the mask. Both are smoothed
with the same factor so the warp and the mask can never disagree with each other.

**Pixel EMA** (`temporal_alpha`, `FaceCompositor`) — smooths the aligned crop
itself. Alignment has already removed translation, scale and rotation, so what
remains between frames is expression change plus the generator's own instability.
Smoothing kills the latter, which is the shimmer that reads as fake.

Both release under motion. The pixel EMA's gate is measured on the **real** crops
rather than the fakes — the question is whether the subject actually moved, and
the fakes carry generator noise that would confuse that signal.

The gate is **per region, not per frame**, and this matters more than it sounds.
Someone talking with a still head changes only their mouth, maybe 15% of the
crop. Averaged over the whole crop that lands below any sensible motion floor,
so a frame-wide gate reads "still", leaves smoothing fully on, and blends lips
across frames — smearing the exact thing a viewer on a call is watching. Two
measures are taken and whichever releases more wins:

- **Whole-crop** — did the subject turn or move? Absolute, floor 1.0 to ceiling 6.0.
- **Per-region** — did *this* part move more than the rest? Measured as excess
  over the crop's own median, so it self-calibrates to however noisy the camera
  is instead of assuming a noise level.

Taking the maximum means the gate can only ever smooth *less* than a frame-wide
one, never more. That is the safe direction: under-smoothing costs a little
shimmer, over-smoothing ghosts the mouth.

The change map is built at quarter resolution first. That is not only for speed —
area-averaging suppresses sensor noise, which is spatially incoherent, while
leaving real motion, which is not. The measure therefore reports motion rather
than grain, and a genuinely still subject gets smoothed properly instead of being
held part-way open by whatever noise the camera has.

Both reset on face loss and on source change. Both are bypassed when
`many_faces` is set, since per-frame detection order is not stable and smoothing
would blend between different people.

### Why detection runs every frame

The previous architecture ran a CSRT/KCF/MOSSE correlation tracker between
periodic detections. That warped the swap with a *cached* face — stale landmarks,
which is exactly what makes a swapped face drift relative to the head it is on.

Detection on every frame plus EMA gives temporal continuity without ever using
stale geometry. The tracker was pure cost.

### Why not `cv2.estimateAffinePartial2D`

Its RANSAC and LMEDS methods are randomized and carry no guarantee of returning
the same matrix for the same input. Anything that varies frame to frame feeds
straight back into shimmer. Geometry uses a closed-form Umeyama similarity fit
instead — exact, reproducible, and the same estimator InsightFace uses for its
own alignment, so the two agree by construction.

---

## Matching the call, not only the face

A believable face is necessary and not sufficient. What reaches the other end of
the call is a whole video stream, and a stream that behaves unlike every other
video call is a tell regardless of how good the face is.

Most of that comes free, because the pipeline composites onto a **real webcam
frame** delivered over a **real network**. The room's lighting, the camera's
white balance and exposure hunting, its sensor noise, the codec's blocking, the
irregular arrival of frames — all of it is already in the signal. The job is not
to invent those characteristics but to make sure the swapped face **follows**
them rather than sitting statically on top.

### What the frame already carries

| Characteristic | Where it comes from | What the face must do |
|---|---|---|
| Room lighting, colour temperature | The real frame | Follow it — LAB match, recomputed every frame |
| Directional light, one-sided illumination | The real frame | Follow the gradient — illumination match |
| Auto-exposure and white-balance hunting | The camera, continuously | Follow it, because the match is per-frame and never cached |
| Sensor noise, ISO grain | The camera | Be matched — estimated per frame from the region around the face |
| Codec blocking, compression softness | JPEG in, JPEG out | Share it — the whole frame is re-encoded together after compositing |
| Irregular frame arrival, jitter | The actual network | Inherited; the desktop's jitter buffer already smooths it |

None of these need faking. They need not being *undone*, which is why the colour
and detail matches are recomputed per frame rather than solved once.

### What is not matched yet

Two physical cues survive, and both appear during movement — which is when a
viewer is most likely to be looking.

**Motion blur.** When someone turns their head, the real frame smears; the
generated face does not. The swap is reconstructed from an aligned crop and
comes back sharp, so during fast movement it is *sharper than the face it is
replacing*. Detail matching operates on a static high-frequency band and does
not model direction, so it corrects the average and misses the smear.

The tractable version: estimate inter-frame displacement from the stabilised
landmarks — which the pipeline already computes — and apply a matching
directional blur to the aligned crop before compositing. Magnitude and angle
both fall out of the landmark delta.

**Rolling shutter skew.** A fast pan skews a CMOS sensor's image; a warped crop
has no skew. Much subtler than motion blur and much harder to model. Worth
noting and not worth chasing until motion blur is done and measured.

### Degrade like a network, not like an AI

The unifying rule, and the one that already decides several behaviours
elsewhere:

> When the pipeline cannot do something well, it should fail the way a video
> call fails — not the way a generative model fails.

A frozen picture, a stutter, a moment of softness: these are things every
participant on every call has seen a thousand times and reads past without
thought. A face that flickers between two identities, a seam that appears under
motion, a mouth that smears — these have no innocent explanation.

Applied so far:

| Situation | Behaviour | Reads as |
|---|---|---|
| Multiple faces in frame | Hold the last good frame | Network hiccup |
| Face lost, detection fails | Hold the last good frame | Network hiccup |
| Paid hour expires | Hold the last good frame | Network hiccup |
| Worker dies, pod reclaimed | Hold the last good frame | Network hiccup |

Still to apply:

- **Falling behind.** If the pipeline cannot keep the frame rate, it should drop
  frames evenly rather than accumulate latency. Even dropping reads as
  bandwidth; growing lag reads as a machine struggling, and desynchronises from
  the audio.
- **Quality under load.** Reducing capture resolution or frame rate under
  pressure is exactly what a call client does. Falling back a preset is more
  honest than missing deadlines at the current one.

### The honest caveat

Everything above is reasoning about mechanisms, not observation of output. The
mechanisms are sound and the measurements behind them are real, but **no part of
this document has been checked against a recorded call.** Motion blur may turn
out to be invisible at these resolutions; the illumination match may turn out to
be the thing that matters most; something not listed here may dominate both.

One clip of real footage would settle more than any further analysis. It remains
the single highest-value thing outstanding on the whole project.

---

## Resolution

`aligned_size` sets a **ceiling** on the working resolution (default 256, clamped
128–512). It is not an output resolution — the composited face is warped back to
whatever size the face occupies in the frame.

Within that ceiling the size is chosen per face from how many frame pixels it
actually covers, snapped to steps of 128/192/256/320/384/448/512. Someone sitting
back from the camera is composited at 128 rather than 320: the swapper only
produces 128px, so compositing a small face at 320 upsamples it more than twice
over and then runs every downstream stage on six times the pixels, manufacturing
detail their webcam never captured. Sitting near the face's own resolution is
both cheaper and more honest.

Step changes carry 18% hysteresis, because each change discards the temporal
state — the smoothed buffer is the wrong shape — so a face hovering on a boundary
must not flip back and forth.

Higher ceilings are not automatically better. They cost latency and, past the
face's own resolution, buy nothing. The presets use 192 / 256 / 320.

Capture resolution is separately modest by design (480×270 / 640×360 / 960×540).
A 1080p-sharp face on a video call is itself a tell.

Upscaling the *output* frame is not implemented and is not planned for the live
path — at ~150–300ms/frame it does not fit a 33ms budget, and it works against
the target anyway.

---

## Model files

```
pipeline/models/            (or /workspace/models/ on RunPod)
  inswapper_128.onnx        face swap            — required
  codeformer.onnx           restoration          — auto-downloads
  dfl_xseg.onnx             occlusion masking    — auto-downloads
  GFPGANv1.4.pth            alternate restorer   — manual, optional
```

The two auto-downloading models come from
[facefusion-assets](https://github.com/facefusion/facefusion-assets). RunPod's
network volume is checked first so weights survive pod restarts.

---

## VRAM

| Configuration | Approx VRAM |
|---|---|
| Swap only | ~2–3 GB |
| Swap + CodeFormer | ~4–5 GB |
| Swap + CodeFormer + XSeg | ~5–6 GB |
| Swap + GFPGAN + XSeg | ~6–7 GB |

The default RunPod filter is ≥16 GB, which is comfortable for all of these.

---

## Tuning

There is no UI for these yet. Live A/B tuning goes through the API:

```python
controller.set_realism(enhancer_weight=0.5, enhance_strength=0.6)
controller.set_realism(grain=False)          # to hear how much it was doing
controller.set_realism(enhancer_model='gfpgan')
```

Or at startup, via flag or `.env`:

```bash
python pipeline.py --stream --enhancer-weight 0.5 --no-grain
```

Changes take effect on the next frame. `set_realism` validates and clamps, and
reports unknown fields back rather than ignoring them.

Suggested order when a swap looks wrong:

| Symptom | Try |
|---|---|
| Plastic, too clean | Lower `enhance_strength`, raise `enhancer_weight` |
| Blurry, mushy | Raise `enhance_strength`, lower `enhancer_weight` |
| Visible edge / halo | Check `occluder` loaded; the mask is the first suspect |
| Face shimmers | Lower `temporal_alpha` and `alpha` |
| Face ghosts when moving | Raise `temporal_alpha` |
| Skin tone doesn't match | Raise `color_strength`; confirm `color_correction` on |
| Face too sharp vs. frame | Confirm `grain` on; lower `enhance_strength` |
