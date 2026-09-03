# Phantom — Realism texture pipeline

Assessment of the proposed texture pipeline: reintroducing real, identity-specific
skin detail into the swapped face rather than accepting what restoration leaves
behind.

**Status: phases A0, A (bar A3) and B of §11 are built and unjudged, and C2's
instrumentation is in place waiting for a run.** Selection, extraction and the
frame-space blend ship behind `texture_strength`, which defaults to **0**.
Nothing here has been looked at on footage.

A visible seam was reported on real footage — "like the face pasted on target"
— and it took priority: a seam is failure mode 2, which the eye finds before it
finds anything else, and judging a subtle texture change through one is not
judging it. Two of its three causes are arithmetic and are fixed (§13); the
third, A3, is held back on purpose so it is not confounded with them.

**Everything here still needs footage.** Nothing in phases A or B has been
looked at on a call. The numbers below are measurements of mechanism — a
transition is 10px wide instead of 5, a headroom is 2.4 units — not evidence
that anything reads better.

The proposal document is summarised where it is agreed and quoted where it is
corrected. Read [ENHANCEMENT.md](ENHANCEMENT.md)
first — it describes the stages this would sit between, several of which already
do part of the job under different names.

---

## Verdict

**Build it, and build about half of it, because the other half already exists.**

Three things make this worth doing where most realism proposals in this repo have
not been:

1. **It is aimed at the largest measured realism deficit.** The face/frame
   high-frequency ratio is **0.584**, against an ideal of 1.00. Nothing else
   measured is off by that much.
2. **The per-frame cost is genuinely small here** — 1-3ms of OpenCV against ~23ms
   of headroom — because the expensive part runs once per identity.
3. **It attacks a gap restoration has already been proven not to close.**
   CodeFormer at 512 moves that 0.584 to 0.614. Twenty-nine milliseconds for
   +0.03 of a 0.42 shortfall.

Three things change its shape, and one is structural enough that building to the
document as written would produce a layer that mostly does not survive to the
output. They are §3.1, §3.2 and §3.3 below.

One error should be fixed in the source document before anyone builds from it:
the CodeFormer fidelity weight is specified backwards (§4.1).

---

## 1. What it is aimed at, and why that is the right number

The proposal's premise is that the swap "reads as smooth/plastic rather than
photographic". The measurement agrees, and locates the deficit more precisely than
the document does.

| `tools/compare_frames.py` (ideal = 1.00) | restoration off | CodeFormer 512 |
|---|---|---|
| high-frequency detail, face / frame | 0.584 | 0.614 |
| sensor noise, face / frame | 1.500 | 1.500 |
| gradient at mask edge | 1.028 | 1.038 |

The face carries **58% of the frame's high-frequency energy**. That is the plastic
look, quantified, and it is the one number in this project far enough from ideal
to be worth a new subsystem.

**But restoration is not what is causing it.** Turning restoration off entirely
moves the number by 0.03 — 7% of the gap. So the document's framing, that
restoration "actively removes this detail as a side effect of cleaning up swap
artifacts" and that lowering its strength is the mitigation, is not supported. The
detail was never there. It is missing because the swapper generates at 128
(inswapper) or 256 (hyperswap) native and everything downstream resamples that,
not because a restorer smoothed it away.

This is good news for the proposal and bad news for one of its parameters:

- **Good:** the texture layer is the only thing on the table that can add
  information rather than redistribute it. Scaling a band that contains no pore
  data cannot invent pores.
- **Bad:** §4.2 — the coupling between restoration strength and texture strength
  that the document treats as a design constraint is a ~7% effect, and tuning them
  together on that assumption will mislead.

One caveat in the other direction. Those numbers are **CodeFormer at 512**. The
current default is `gpen_bfr_256`, which was adopted on speed evidence and **has
never been composited or looked at**. Its effect on the detail ratio is unmeasured.
It is plausible that a 256 crop upsampled into a 192 aligned space softens more
than CodeFormer did, in which case the document's premise is more right than the
table above suggests. That is one `compare_frames.py` run, and it should happen
before tuning anything (§10).

---

## 2. What already exists

Roughly half the proposed module list is built, under other names, inside
`FaceCompositor` and `guards`.

| Proposed module | Already in Phantom | Gap |
|---|---|---|
| `selection/best_source_selector.py` | `guards.sharpness(frame, bbox)` (Laplacian variance, face-region only), `guards.estimate_yaw`, `check_source` (size, blur, pose, occlusion) run over **every** uploaded source by `FaceDatabase.review_sources` | The scores exist and are already computed per image. Selection is `max()` over them. **~30 lines, not a module** |
| `reprojection/landmark_smoothing.py` | `LandmarkStabilizer` — EMA on kps/106, releases under motion, resets on identity change with a 3-of-6 confirmation window | **Exists, and is better than proposed.** Do not build a second one |
| Grain, matched to plate noise statistics | `_add_grain` + `_estimate_noise` — MAD of the Laplacian, monochrome, masked, **in frame space** | Exists. And already carries the argument the proposal is missing — see §3.1 |
| Low-frequency tonal variation, LAB a/b | `_match_color` (LAB mean/std transfer inside the mask, ramped by colour distance) and `_match_illumination` (low-frequency LAB gradient at 1/8 scale, ±12 units, 0.7 scale) | Exists, and **works in the opposite direction** — see §5 |
| `confidence/confidence_mask.py` | `FaceMasker.last_coverage` from the XSeg pass, `guards.coverage_ok`, per-frame yaw from `measure_yaw` | Inputs exist; the per-pixel confidence field does not. Real work, but small |
| Frequency separation | `_match_detail` — Gaussian split at `_DETAIL_SIGMA`, high band scaled to the target's energy, clamped `[0.6, 1.6]` | Exists as a *multiplicative* version of the proposed *additive* layer. They must agree on the band — see §6 |
| `combine/blend.py` | `_paste` alpha composite, ROI-confined | Exists |
| `extraction/canonicalization.py` | `estimate_similarity` (closed-form Umeyama), `_ffhq_geometry`, `_build_ffhq_crop` | **Canonical space is already built** — see §3.3 |
| BiSeNet skin parsing, "already available" | **Not available.** Landmark convex hull + DFL XSeg occluder | Real gap — see §8 |

What is genuinely new: **detail extraction, storage, per-frame reprojection,
confidence scoring, procedural fallback, and the diffuse pass.** That is a
worthwhile subsystem. It is not ten modules.

One dependency checked and clear: `handle_upload_source` writes the actual source
image files to disk on the pipeline machine, so **source pixels are available on a
remote pod**, not just the embedding. The proposal's decision 1 holds without a new
transfer path.

---

## 3. Three corrections that change the shape

### 3.1 High-frequency detail must be applied in **frame space**, not aligned space

This is the structural one. Built as written, most of the texture layer would be
destroyed one operation after it was created.

Phantom composites in **aligned space** at 128-320px and then warps the finished
crop down to the face's real size in the frame:

    face in frame          ~101 x 129   <- the only real information
    swap native             128 or 256
    aligned space           128 - 320   <- where the compositor works
    _paste -> frame        ~101 x 129   <- warped back down

A high-frequency field added at 256 and then resampled to 101px is decimated.
Pores at the aligned crop's Nyquist limit alias into blobs or vanish entirely.
This is the same mechanism that makes CodeFormer's 512 crop worth +0.03: **86% of
it is discarded at `compositor.py:515`, one warp after it is created.**

The codebase already knows this, and says so about a field with the same spectral
character as the proposed detail map — `_add_grain`:

> Grain is added here rather than in aligned space because warping a crop down to
> the face's size in frame would filter the noise into blobs.

That comment is the entire argument, already written. **Pores are grain with
structure.** They belong in the same place.

So: `reprojection/warp.py` composes with the **inverse** affine `_paste` already
computes, not with the aligned matrix, and the high-frequency half of the Realism
Layer lives **inside `_paste`, between the alpha composite and `_add_grain`.**

Two consequences worth stating:

- **Under hyperswap the discard is worse, not better.** `aligned_min` is 256 for
  hyperswap against 128 for inswapper, so a 101px face composites at 256 and is
  downsampled ~2.5x at paste. The proposal specifies hyperswap; the correction
  matters more there.
- The proposal's stage order — texture, then grain last — is **right**, and
  survives the move. Both end up in `_paste`, in that order.

### 3.2 The Realism Layer is two layers, at two different insertion points

The document treats the Realism Layer as one block appended after seam blending.
It cannot be, because its sub-stages live in different frequency bands and the
existing pipeline treats those bands at different places.

    ALIGNED SPACE (128-320), before _match_color
      diffuse / light-scatter pass        (low frequency, L channel)
      low-frequency tonal variation       (mid-low, LAB a/b)   <- see §5

    FRAME SPACE (~101px), inside _paste, after the alpha composite
      reprojected high-frequency detail   (pores, micro-texture)
      procedural fallback in low-confidence regions
      grain                               (already built, stays last)

The low-frequency work goes *before* colour matching so that `_match_color` can
reconcile it against the target rather than having it applied on top of a finished
match — otherwise anything injected in the a/b channels is precisely what the
colour stage exists to make agree with the frame, and disagreeing with it is
failure mode 2.

The high-frequency work goes after everything, in frame space, for §3.1.

This split is the single biggest change to the proposed structure. `pipeline.py`
in the proposed module tree does not orchestrate a stage order of its own — it
provides two entry points that `FaceCompositor._composite_impl` and
`FaceCompositor._paste` call.

### 3.3 Canonicalization is already built, and it buys **caching**, not pose robustness

Decision 3 argues for canonical UV face space on the grounds that it "reduces
compounding warp error, especially when source and target poses differ
significantly."

Half right. The machinery exists: `estimate_similarity` fits a closed-form Umeyama
similarity into a fixed template, and `_ffhq_geometry` / `_build_ffhq_crop` already
produce a canonically-framed 256 or 512 crop. **A detail map extracted in FFHQ
space is the canonical map.** No UV unwrap, no 3DMM, no new dependency.

But the stated benefit is wrong. These are **similarity transforms** — 4 degrees of
freedom, and composing two of them yields one of them exactly. So
source→canonical→target has *identical* numerical error to source→target in one
step. Canonicalization buys nothing on that axis.

What it does buy is decision 4, which is the real win: **extraction runs once**, the
map is cached per identity, and the per-frame cost collapses to one warp. That is
worth having on its own.

The thing it explicitly does **not** buy is pose robustness. A similarity transform
does not correct yaw. A source at 20° gives a detail map whose pore geometry is
foreshortened on one cheek, and no affine removes that. Which means:

- **Frontality scoring in source selection (decision 2) is load-bearing**, not a
  nice-to-have. It is the only thing buying pose robustness in this design.
- The confidence mask should fall off with **yaw distance from the source pose**,
  not just with absolute yaw. Two faces at 25° in opposite directions are not
  equally well served by the same map.

---

## 4. Errors to fix in the source document

### 4.1 The CodeFormer fidelity weight is specified backwards

> Light structural cleanup (CodeFormer, low fidelity weight w ≈ 0.1–0.3 …)

CodeFormer's fidelity weight runs **0.0 = restore hardest, hallucinate most** to
**1.0 = stay closest to the input**. So `w ≈ 0.1–0.3` is the *most aggressive, most
hallucinatory* setting available — the maximum-beautification end. It is the
opposite of "light structural cleanup", and it is the exact setting that produces
the poreless output this entire document exists to prevent.

Light cleanup is **`w ≈ 0.8–0.9`**. Phantom's default is 0.7.

This is worth catching now because the error is invisible downstream: the pipeline
would run, the output would look plastic, and the natural diagnosis would be that
the texture layer is too weak.

Note also that this stage puts **two restoration models in series** — GPEN, then
CodeFormer — which is a second full inference on the live path. Given that
restoration was measured at +0.03, a conditional second restorer needs a clear
argument for what it is fixing that the first did not. If the answer is "artifacts,
not smoothing", the gate has to be an artifact detector, and that detector does not
exist yet. Recommend dropping this stage from v1 and revisiting it if artifacts are
actually observed on footage.

### 4.2 The strength-coupling assumption is unsupported

> lowering GFPGAN strength means less damage to undo, so texture/diffuse strength
> should be tuned down correspondingly

Restoration accounts for 0.03 of the 0.42 detail gap. The coupling is real in
principle and negligible in magnitude. Tuning texture strength down because
restoration strength went down would leave the layer under-driven for a reason the
measurement does not support.

The *useful* half of decision 9 stands and should be kept: **independent toggles
per layer, tuned against real footage, so a given artifact can be attributed.**
Keep that. Drop the assumed direction of coupling until a measurement shows one.

### 4.3 "Removes an entire class of temporal-instability risk" is too strong

Decision 4 claims that because the detail map is fixed, "detail content itself
never changes frame to frame" and temporal instability is removed.

Fixed content warped by a per-frame affine produces **texture swimming** — the
pores are locked to the aligned template, so as the head yaws the template slides
across the real face and the pores appear to crawl over the skin. This is caused by
*correct* landmark motion, not by landmark noise, so decision 8's smoothing does
not address it. It is a well-known failure in the same family as the shimmer the
compositor's temporal EMA already fights.

Mitigations, both already available in the design:

- Keep blend strength low enough that swimming sits below threshold. This is the
  main one, and it is why the layer should be conservative by default.
- Let the confidence mask fall off with yaw distance from the source pose (§3.3),
  which attenuates the map exactly where swimming is worst.

Restate the claim as: extraction-once removes *content* flicker, and leaves
*geometric* swimming as the residual temporal risk.

---

## 5. Tonal variation fights two existing stages

The Realism Layer lists "low-frequency tonal variation (skin blotchiness/
unevenness, LAB a/b channel)". Two built stages operate in that band and channel,
and both **remove** variation:

- `_match_color` transfers the target's LAB **mean and standard deviation** into the
  fake inside the mask. Injected a/b variance is variance; the std transfer
  normalises part of it away.
- `_match_illumination` matches the low-frequency LAB residual at **1/8 scale**,
  limited to ±12 LAB units at 0.7 strength. Anything coarser than ~32px in a 256
  crop is visible to it and gets partially corrected out.

So blotchiness injected before colour matching is attenuated twice, and injected
after is unreconciled with the target — a colour patch that does not agree with the
frame, which is failure mode 2.

The band that survives is **between** `_match_illumination`'s 8x downscale and
`_match_detail`'s sigma. That is a real window but it is narrower than the document
implies, and it argues for treating tonal variation as the **lowest priority** item
in the Realism Layer rather than a peer of the pore map.

There is also an internal tension worth resolving in the document: decision 6
excludes mid-frequency structure as expression-dependent, but blotchiness is
mid-frequency. The distinction that actually holds is **surface-locked vs.
expression-driven** — a mole or a patch of redness is surface-locked and
extractable; a nasolabial fold is expression-driven and is not. Restate decision 6
on that axis and blotchiness stops contradicting it.

---

## 6. The additive layer and `_match_detail` must agree on the band

`_match_detail` splits at `_DETAIL_SIGMA = 1.5` scaled by `fake.shape[0] / 256.0`,
and scales the fake's high band toward the target's, clamped to `[0.6, 1.6]`.

Two things follow.

**Choose the extraction sigma to match that scaling.** If the extractor's high-pass
and `_match_detail`'s high-pass describe different bands, the two stages will fight
— one amplifying what the other is normalising. Reuse `_DETAIL_SIGMA` and its `/256`
reference so "texture" means the same physical detail everywhere, exactly as the
existing comment intends.

**The 1.6 clamp is also a free experiment** — see §10. `_match_detail` is already
trying to raise the face's high-frequency energy to match the frame's. If the
measured ratio is 0.584 *after* that stage ran, then either the clamp is binding or
the two measurements describe different bands. Knowing which decides how much of
this document needs building.

---

## 7. The diffuse / light-scatter pass

Decision 7 is agreed: subsurface scattering is a genuinely separate axis from
detail, it is a shading-domain problem, and scoping it to luminance is the right
call. Three additions.

**It goes in aligned space, before `_match_color`** (§3.2).

**Scale the blur radius to face size in frame.** The same `/256` reference
`_DETAIL_SIGMA` uses. Otherwise "soft" means a different physical distance at a 128
aligned crop than at a 320 one, and the look changes when the operator leans back.
Easy to write, hard to see.

**Exclude features from the scatter mask, or it will do exactly what the risk
section fears.** The version that softens identity is

    L + strength * (blur(L) - L)

applied across the whole face. The version that does not restricts that to skin
with eye and mouth polygons removed — and the landmark hull already gives you those
positions for free, no parsing net required. This is the difference between a
working pass and the failure mode decision 7 correctly worries about.

`diffuse_strength` on `FaceSwapConfig`, reachable from `set_realism`, CLI and env,
defaulting **off** until judged on footage. Same treatment every other realism knob
gets.

---

## 8. BiSeNet

Listed as "already available". It is not. Phantom masks with a landmark convex hull
plus the DFL XSeg occluder. Adding BiSeNet to the live path is a new per-frame ONNX
inference, which is the one thing the proposal promises not to add.

The split that avoids it:

- **Extraction (once, on the source image):** use a parsing net if you want one. It
  costs nothing per frame and this is where mask quality actually matters, because
  an error here is baked into the cached map forever.
- **Per frame:** the hull minus eye/mouth polygons, intersected with the XSeg
  occluder that already runs. Not as good as parsing, but free, and the mask only
  has to be right at pore scale over skin the swap already covers.

Do not put BiSeNet on the live path.

---

## 9. Performance, rewritten for this codebase

The proposal's optimisation section is generic and, for Phantom, points at the
wrong machine. Its own decision 1 — "pick one home, matching whatever the models
already output" — has a specific answer here: **the models are ONNX on the GPU, the
compositor is NumPy/OpenCV on the CPU, and pixels come home between models
regardless.** So the texture pipeline's home is CPU/OpenCV, written the way
`_match_detail` and `_add_grain` are already written.

| Proposed | Verdict here |
|---|---|
| #1 device placement | Answered: CPU/OpenCV. Moving this to torch adds two device transfers around ~2ms of work |
| #2 vectorization | Agreed, and satisfied by matching existing style — whole-array `cv2` calls, `addWeighted` fusing multiply-and-add |
| #3 batching across frames | **Not applicable to LIVE.** Batching trades latency for throughput; the live call is a latency problem. Keep for RENDER |
| #4 precomputation | Agreed. Note `_add_grain` currently calls `np.random.normal` per frame at ROI size — a precomputed noise tile with a random offset is cheaper and looks identical |
| #5 conditional inference | Already built as `restore_min_face` / `_restore_worthwhile`, gated on face size. See §4.1 on whether the second restorer should exist at all |
| #6 mixed precision | Not applicable — no torch in this path, and none should be added |
| #7 `torch.compile` / op fusion | Not applicable, same reason |
| #8 resolution-aware | Already done — `_region_of_interest` confines all frame-space work to the face ROI |

**Estimated per-frame cost in this codebase:** one `warpAffine` of a 3-channel map
into the ROI (~101px — trivial), one masked multiply-add, one add; plus, in aligned
space, one Gaussian on the L channel at 128-320 and a blend. **1-3ms** against ~23ms
of headroom under the `optimal` preset's 50ms deadline.

The stated willingness to trade latency for realism is not really tested by this
design. What it costs is complexity and tuning surface, not milliseconds.

---

## 10. The experiment that decides how much of this to build

Before writing any of it, two measurements. Both are cheap, and between them they
bound the whole subsystem.

**1. Is `_DETAIL_RATIO`'s upper clamp of 1.6 binding on real footage?**
`_match_detail` is already scaling the swap's high band toward the frame's. Log the
pre-clamp ratio over a real clip.

- If it is **binding** (pre-clamp ratio > 1.6 routinely), then part of the 0.42 gap
  is simply "the existing stage is not allowed to correct far enough", and raising
  the clamp is a one-line experiment that costs nothing. It will amplify noise along
  with texture — which is exactly why the clamp exists, and exactly the argument for
  a real detail map instead. But measure first.
- If it is **not binding**, the high band genuinely contains nothing to amplify, and
  the case for extracting real detail is made outright.

**2. Re-run `compare_frames.py` against `gpen_bfr_256`.** Every realism number in
this repo was measured with CodeFormer at 512. The current default has never been
composited or looked at. Its detail ratio is the actual baseline this work would be
improving on, and nobody has it.

Both are one instrumented stream each, and they are the difference between building
on a measurement and building on the document's premise.

---

## 11. Build order

Reordered after footage. **An observed defect outranks an unobserved
improvement**, so the seam comes first — it is the failure the eye finds
fastest, and it sits between the operator and any judgement about texture.

### Phase A0 — shipped, unjudged

1. ~~**Best-source selection**~~ — `FaceDatabase.select_texture_source`, a
   `max()` over scores `_review_image` records while it already holds the frame
   and the detection. Sharpness 0.40, size 0.30, frontality 0.20, clipping 0.10;
   weights unmeasured and stated as such in the code.
2. ~~**Extraction + cache**~~ — `pipeline/processing/texture.py`. The crop is
   cached at 512 and the *map* derived per working size, because the band has to
   be chosen at the resolution it will be displayed at. Sigma comes from
   `geometry.DETAIL_SIGMA`, the constant `_match_detail` also reads.
3. ~~**Frame-space reprojection and blend**~~ — `FaceCompositor._add_texture`,
   inside `_paste` between the alpha composite and `_add_grain`.
   Measured with the B1 headroom bound in place: **0.90ms** per frame on a 101px
   face, **7.49ms** on a 460px one. Extraction is 28.2ms, once per source.

### Phase A — the seam (observed)

**A1. Find out what is actually stepping.** *Tooling half done; the capture has
not been taken.* One `--debug-frames` capture, then
sample alpha, LAB mean and high-band energy along the mask normal. This
separates a **geometry** seam (the boundary is in the wrong place) from a
**radiometric** one (the material either side genuinely differs), and the two
have different fixes. Do not guess between A2, A3 and A4 — measure once.

~~Fix the metric in the same pass.~~ **Done.** `compare_frames.py` reported
gradient 1.028 and "no seam detected" on footage where a person sees a seam. **A
metric that disagrees with the eye about the one failure the eye finds fastest is
worse than no metric, because it gets believed.** It measured gradient
*magnitude* — a texture statistic that a 3-unit step across two pixels barely
moves — and divided by the ring *outside* the mask, which contains hair and is
often the larger of the two whatever the composite does. Both effects push it
toward 1.0.

It now reports **`seam_excess`**: the median LAB step across the boundary in the
output, less the step the untouched *input* already carried at the same rings.
That difference is the part the composite is answerable for; a face has real
steps of its own at a jaw shadow or a hairline, and an absolute measure charges
the swap for them. Sampled on a blurred image, because a seam is a
low-frequency event — without that the measure responds to grain instead, since
LAB's transfer is non-linear and a noisier region reports a shifted median even
at identical tone. `seam_ratio` is retained and demoted; its verdict line still
prints, below the new one.

Its threshold is **uncalibrated** and says so in the code. The honest
calibration is a clip somebody has judged by eye, with the number moved to
wherever their verdict flips.

**A2. Widen the transition, and move it inside the face.** **Done**, as
`mask_feather` (0.04) and `mask_erode` (0.03) — knobs rather than constants,
because this is exactly the question only an A/B on a real call settles, and both
reach `set_realism`, the CLI, the env and the pod. Measured on a 100px face, the
transition goes **5px → 10px** at the new default, and 26px at 10%.

The ROI padding now grows with the feather. It was a constant 4px, and a blur
wider than its padding reflects off the region border and leaves the mask never
reaching zero along it — a seam produced by the fix for one.

The arithmetic that motivated it, for a 101px face:

    aligned feather    int(256 x 0.05) | 1     ->  sigma ~2.3px at 256
    scaled into frame  x (101 / 256)           ->  0.91px
    frame feather      max(1.0, roi_w x 0.01)  ->  1.1px
    combined                                       ~1.4px = 1.4% of face width

That is a hard edge. The frame-space feather should be a fraction of **face
extent** with a floor of ~2px, in the region of 4-6% rather than 1%.

**Erode before feathering.** The hull is expanded 10% radially and then blurred,
so the 50%-alpha line sits *outside* the landmark silhouette — on neck at the
chin and on hair at the temples, which is exactly where the material either side
differs most. Eroding first puts the soft region on skin.

This is deliberately **not** "extend the mask". Growing the covered region is the
half that backfires: [ENHANCEMENT.md](ENHANCEMENT.md) already records that
masking past the hairline produces swapped skin where hair should be, a worse
tell than a short forehead. Extend the *transition*, not the *coverage*.

**A3. A boundary that can be concave.** *Held back deliberately.* It changes the
mask's **shape**, where A2 changes only where the softness sits, so shipping them
together would leave no way to tell which one moved the result — and a segmenter
boundary that moves frame to frame trades failure mode 2 for failure mode 3.
Judge A2 first.
 `cv2.convexHull` of the 106 landmarks
cannot follow a jawline, an under-chin or a temple — a convex hull has no
concave points by definition, so at those places the boundary leaves the face
whatever the expansion is set to. Two candidates:

- the ordered 106-point face contour as a polygon, rather than its hull;
- promote the XSeg segmentation from an occlusion *multiplier* to the boundary
  itself where it is confident. It already runs every frame, and its boundary is
  anatomical where the hull's is not.

**A4. Shrink the colour deadband.** **Done** — `_COLOR_FLOOR` 4.0 → 1.5,
`_COLOR_RANGE` 12.0 → 8.0.
 `_COLOR_FLOOR = 4.0` means a LAB mean
difference under 4 units gets **zero** global correction, and the ramp only
reaches full at 16. `_match_illumination` recovers ~70% of what the deadband
leaves (`_ILLUM_SCALE`), so a 3.9-unit difference still lands ~1.2 units
uncorrected — across a 1.4px transition, at a boundary, which is where the eye
compares hardest. The floor's stated purpose is that "correcting a match is pure
risk"; the anti-snapping property it protects is delivered by the *ramp*, not by
the deadband, so the floor can go to ~1.5 with a shorter range.

**Not on this list: gradient-domain blending.** Poisson is the textbook answer
and this codebase already rejected it — it pulses frame to frame, trading
failure mode 2 for failure mode 3.

### Phase B — the texture risks

**B1. Bound the overshoot.** **Done** — `FaceCompositor._texture_headroom`, and
`texture_strength` now means *the fraction of the measured gap to close* rather
than a gain. `_match_detail` runs in aligned space *before*
`_paste`, so it never saw the added texture and nothing pulled the face back
below the frame's high-band energy. Past ratio 1.0 the face is noisier than the
camera that supposedly shot it — failure mode 1 from the other side.

The fix makes the knob better rather than merely safer: measure the target ROI's
high-band deviation (`_estimate_noise` already samples that region for grain) and
cap the added deviation so the composited face lands at or below the frame.
`texture_strength` then means "this fraction of the measured gap" rather than an
open-ended gain — which also removes the unknown `F` from the §12 arithmetic and
makes one setting portable across clips.

As built: independent zero-mean fields add in quadrature, so a face carrying `f`
and about to receive grain `g` can accept `t` before matching a real face
carrying `r`, with `f^2 + g^2 + t^2 = r^2`. The reference `r` is the operator's
own face **in the same pixels** rather than the frame at large — a real face, at
the right size, through the right lens, under the right light, which is a better
statement of "what skin looks like here" than the background could give. Grain
is counted in because it has not been added yet and is about to be; leaving it
out would let the two layers each reach the target and overshoot together.
`TEXTURE_MAX` survives only as a backstop against an estimate gone wrong.

The statistics are taken over a **bounded 160px window**, not the whole region.
A deviation converges on a few thousand pixels, and measuring every pixel of a
500px face cost **16.1ms** — more than the rest of the frame, and paid even on
the path that decides to add nothing. §9 #8 said to do this and the first cut
did not. Cropping rather than downscaling, because a downscale is itself a
low-pass and would measure a different band than the one being added.

**B2. Fade texture with the seam, not just the mask.** **Done, and it came
free** — `_add_texture` already multiplies by the same `warped_mask` the
composite uses, so widening that in A2 widened the texture's own fade with it.
Worth recording as verified rather than assumed: a detail field that stopped
over 1.4px would be a seam made of pores, landing exactly on the boundary A2
just fixed.

**B3. Confidence mask.** **Done**, as `FaceCompositor._pose_confidence` — with
one half deliberately left out.

Yaw distance from the source pose scales the layer: full strength while the
frame is within `_POSE_FULL` (12°) of the photograph the map came from, falling
to nothing by `_POSE_LIMIT` (45°). Ramped rather than switched, because a head
does not sit still and a texture layer appearing and vanishing is more visible
than one that is slightly wrong. This is also the only mitigation for swimming
that is not "turn the strength down": swimming is worst exactly where the pose
has moved furthest from the source, so a term falling off with pose distance
removes the detail at the moment it would start to crawl.

**XSeg coverage needed no new code.** The occlusion mask is already multiplied
into the compositing alpha by `FaceMasker`, and `_add_texture` already multiplies
by that alpha, so texture never lands on a hand or a microphone. Recorded as
verified rather than assumed.

**The directional half is not built, on purpose.** At yaw it is the cheek
turning *away* whose pores are stretched, so the correct term falls off across
the face rather than uniformly. Building it needs the sign convention of
`face.pose` pinned against real footage, and a directional term applied with the
sign backwards would attenuate the half of the face that is still good — worse
than not having it. The uniform version is the conservative form of the same
idea, and the sign is one frame of a debug capture away.

A capability note that is easy to get wrong the other way: a model pack without
`pose` returns **full** confidence, not none. Silently attenuating to zero there
would turn "we cannot measure the angle" into "the layer does nothing", which is
a behaviour change hiding inside a capability gap.

`texture_confidence` is published to the realism readings, because a layer doing
nothing because every frame is off-pose looks identical to one whose strength is
set too low.

### Phase C — judge it

**C1. Footage, with the toggle**, at `texture_strength` 0.3-0.5. Both
`compare_frames.py` (once A1 has made its seam metric trustworthy) and a person
looking at a face — the metric will not see swimming.

**C2. The `_DETAIL_RATIO` clamp reading**, which was §10 item 1. **The
instrumentation is built; the run has not happened.**

`pipeline/services/readings.py` accumulates per-frame scalars and reports
distributions when a stream stops, beside the latency budget under a `REALISM`
scope. `FaceCompositor.last_detail_ratio` publishes the correction
`_match_detail` *wanted*, before its clamp — which is the only way to see this,
since percentiles of the clamped value cannot exceed the clamp. The report says
what share of frames sat at the limit and names the next action rather than
printing a number nobody can interpret:

- clamped on **>50%** of frames: part of the 0.42 detail gap is the clamp rather
  than the swap. Raise `_DETAIL_RATIO` and re-measure **before building anything
  else** — it is a one-constant change and a cheaper lever than this whole
  document.
- clamped on **<5%**: the high band holds nothing more to amplify, and the case
  for extracting real detail is made outright.

`texture_headroom` and `texture_confidence` ride the same mechanism, so one
stream answers whether the texture layer had anything to spend and whether pose
let it spend it.

### Phase D — pose

**D1. Pose-aware canonicalization over `landmark_3d_68`.** This is the UV
question, and the earlier deferral rested on a mistake. It was declined here on
the grounds that per-frame reprojection needs a dense 3D fit, i.e. a new model on
the live path. **`buffalo_l` already computes `landmark_3d_68` every frame** — it
is what sets `face.pose` — and `guards._CAPABILITIES` records it as affecting
"nothing directly". The correspondence is free and unused.

What is actually needed is a fixed reference mesh over those 68 points, a
Delaunay triangulation computed once, and a piecewise-affine warp per frame.
Classical, CPU, no new model.

What it honestly buys is **partial** pose robustness. 68 points is a coarse mesh,
`landmark_3d_68`'s depth is a weak estimate rather than a fitted 3DMM, and it
degrades toward profile. It is not a UV unwrap. But it is the only item on this
list that addresses a source and a target at different angles, which a similarity
transform cannot do at any strength (§3.3).

After B3, because the confidence mask is the cheap version of the same idea:
attenuate where the pose disagrees, rather than correct for it.

### Phase E — unchanged deferrals

Procedural fallback (proposal decision 5), the diffuse/light-scatter pass
(decision 7, §7), and low-frequency tonal variation (§5 — narrowest band, most
existing machinery to fight).

Two items from the source document remain **undefined rather than deferred**, and
need a sentence each before they can be assessed: **"identity / feature
augmentation"**, which appears in the pipeline diagram with no module and no
description; and **"final color grading"**, which has no pipeline-side equivalent
here — the desktop's filters are a separate client-side decorative layer that
[CLAUDE.md](../CLAUDE.md) says to keep off while judging.

## 12. Open risks

Carried from the proposal, with two added.

- Texture quality is bounded by the best source image. Cannot extract what is not
  there.
- Coverage varies per frame with angle and occlusion — expect quality that varies
  shot to shot, not a uniform improvement.
- The diffuse pass can soften identity if it is not strictly scoped to a shading
  channel with features excluded (§7).
- Three-plus interacting strength parameters. Independent toggles are required
  tooling, not optional.
- **Added: texture swimming** (§4.3). Geometric, not content, and not fixed by
  landmark smoothing.
- **Added: the whole design is calibrated against a restoration model nobody has
  looked at.** `gpen_bfr_256` was adopted on speed evidence alone. Partly closed:
  a live run reported the output "sort of generated", which is a footage verdict
  and outranks the metric — but the detail ratio under GPEN is still unmeasured.
- **Added: the layer's cost scales with face size**, which the 1-3ms estimate in
  §9 did not say. Measured **0.90ms** on a 101px face and **7.49ms** on a 460px
  one. At the `production` preset with an operator close to the camera that is
  ~7ms on top of a ~27ms frame against a 33ms deadline. Not free there, and the
  remaining cost is the warp and the multiply-add over the face's area, which is
  inherent rather than reducible by sampling.
- ~~**Added: overshoot is unbounded**~~ — closed by B1.
- **Added: the pose confidence has no direction**, only magnitude, so a frame
  turned away from the source is attenuated uniformly rather than across the
  cheek that is actually stretched. Conservative, and it costs some good detail
  on the near side of an off-pose frame.
- **Added: the seam knobs are two more interacting parameters**, on top of the
  three §12 already warned about. `mask_feather` and `mask_erode` pull against
  each other by construction — erode moves the transition in, feather spreads it
  back out — so sweeping one without pinning the other will produce a confusing
  result. Sweep erode at a fixed feather first.

---

## 13. The seam, as reported

A live run reported the seam as "very noticeable, like the face pasted on
target". That is failure mode 2, it was **seen rather than measured**, and it
reorders §11 — see phase A.

Three things are worth recording before A1's diagnosis runs, because they are
arithmetic rather than hypothesis:

**The transition is ~1.4% of the face's width.** Both feathers are fractions of
their own space, and both spaces are larger than the face: the aligned feather is
5% of 256, but the face is 101px, so it lands as 0.91px; the frame feather is 1%
of the ROI, so 1.1px. Neither constant is wrong on its own; the product is too
small, and nothing was looking at the product.

**The 50%-alpha line sits outside the face.** `_HULL_EXPAND` grows the hull 10%
radially *before* the blur, so the midpoint of the transition lies beyond the
landmark silhouette. At the chin that is neck; at the temples, hair. A boundary
is most visible exactly where the material either side differs, and this places
it there.

**A convex hull has no concave points.** Whatever the expansion, the boundary
cannot follow a jawline or an under-chin — and those are the two places a
"pasted on" impression usually comes from.

One further note, about the tooling rather than the pipeline: **the seam metric
said there was no seam.** `compare_frames.py` reported gradient-at-mask-edge
1.028 and 1.038 in the two configurations measured, both reading as "no seam
detected", and a person then saw one. Until A1 resolves that, treat the seam row
of §1's table as carrying no information — not as evidence the seam is mild.
