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
Defaults are `0.7` / `0.7`; `production` uses `0.6` / `0.8`.

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

The seam is the failure the eye catches fastest, and four separate things work
on it.

**1. The mask follows the real face.** `FaceMasker` builds a convex hull from
the 106 landmarks InsightFace already computes, so the boundary tracks the actual
jawline instead of assuming an ellipse. The hull is expanded 10% radially, with
an extra upward push of 16% of hull height to recover forehead.

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

**4. The edge is feathered twice.** Once in aligned space (5% of crop size), and
again in frame space (1% of ROI width) after warping back. Blurring only at the
aligned resolution leaves a stair-stepped edge as soon as the face in frame is
larger than the aligned crop.

---

## Matching the frame

Three stages exist purely to make the composited face belong to the image around
it.

### Colour

LAB transfer, sampled **inside the mask only**. Sampling a bounding box instead
pulls in hair and background, which is how a bright window behind someone ends up
shifting their skin tone.

Correction ramps continuously with measured colour distance rather than
triggering on a threshold, so it can never snap on and off between frames. The
L-channel standard deviation is damped to 50%: matching L *mean* fixes brightness
and matters; forcing L *std* flattens facial contrast.

Controlled by `color_correction` (on/off) and `color_strength` (scale).

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
the fakes carry generator noise that would confuse that signal. Below a mean
absolute difference of 2.0 the subject is effectively still and smoothing is
full; above 8.0 they are talking or turning and smoothing is fully released, so
it cannot ghost.

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

## Resolution

`aligned_size` sets the working resolution for everything above (default 256,
clamped 128–512). It is not an output resolution — the composited face is warped
back to whatever size the face occupies in the frame.

Higher is not automatically better. It costs latency, and beyond the point where
the face in frame is smaller than the aligned crop it buys nothing. The presets
use 192 / 256 / 320.

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
