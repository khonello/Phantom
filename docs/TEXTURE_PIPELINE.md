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

### 6.1 Agreeing on the band was necessary and not sufficient — they also compete for it

*Added after the layer was first run on real footage and reported as present but
too soft.*

The two stages agreed on the band, as §6 required, and that turned out to be the
setup for a different failure. **They also both aim at the same target** — the
real face's high-frequency energy — and `_match_detail` runs first.

It does not merely aim at it. The pod run measured its clamp binding on **0% of
2267 frames**, which is the stronger statement: it *reaches* parity, on every
frame. So by the time `_texture_headroom` runs in frame space, `fake ≈ real` by
construction, and what it measures as "available" is only what the warp down
happened to lose. Measured p50 **0.78** of an 8-bit unit, against a real face
carrying several. At the 0.3–0.5 working range the layer was contributing well
under 1% of the face's high-frequency energy.

That also explains the observation §10 could not: **texture at 0.3 and 0.5
produced identical readings on every metric.** They were both invisible.

And the composition is worse than the amount. `_match_detail` can only amplify
the band the swap already carries, which is upsampled 128-native output with no
structure in it. The face lands at the correct *energy* with the wrong
*content* — textured by measurement, smooth to the eye. Restoration was already
shown not to close this gap (+0.03 of 0.42); this is the same deficit reappearing
one stage later, disguised as a satisfied statistic.

**The fix is to reserve, not to raise a ceiling.** `_texture_reserve` returns a
share `r`, `_match_detail` aims at `sqrt(1 - r²)` of the target's energy, and the
texture layer fills the rest with detail a camera actually recorded. Quadrature,
so it composes with the `f² + g² + t² = r²` arithmetic already in
`_texture_headroom` rather than fighting it. Total energy still lands at parity;
overshoot is still impossible.

Three things had to come with it, and two were found by measuring rather than by
reasoning:

1. **Reserve only when the layer will run.** A reservation is a deliberate
   undershoot. If texture then declines — no source photograph, pose beyond
   `_POSE_LIMIT`, no map at this working size — nothing fills the gap and the
   face is *softer* than with no texture layer at all. Every gate `_add_texture`
   applies is applied in `_texture_reserve` first, against the same clamped
   values. The one gate that cannot be checked in advance is the headroom
   itself, and that direction is safe: reserving makes headroom more likely.
2. **The band has to travel with the reserve.** Reserving one octave of a field
   measured across three barely moves anything. Measured on a synthetic face in
   exactly the starved regime: reserving 0.4 over the narrow band took the
   headroom from 0.30 to **0.31**. So when a reserve is requested,
   `_match_detail` scales the same span the map will fill; with no reserve it
   splits where it always did, bit-identically.
3. **`_DETAIL_RATIO` has to move with the target it clamps.** The clamp bounds
   how far the stage may deviate from what it is aiming at, and reserving lowers
   what it is aiming at. Left fixed, the 0.6 floor refused the attenuation the
   reservation had just asked for — at `reserve` 0.8 the wanted ratio was 0.54
   against a floor of 0.60, so a quarter of the promised room was quietly kept.

### 6.2 One octave was never enough, and that is the larger term

The same footage report — marks visible in the source, barely visible in the
output — has a second cause that is independent of the budget and, measured,
**larger than it**.

`DETAIL_SIGMA` is 1.5 at a 256px reference. At a 128px working size that is a
high-pass at sigma 0.75, which keeps roughly the finest two pixels. Pore noise
lives there. A freckle, a spot, a mole, a scar, a pockmark or a fine crease is
**2–8px at working resolution**, so the bulk of each one sits *below* the cut and
is subtracted as "shape" at extraction. What survives is its rim. That is the
mechanism behind a face that measures as textured and reads as smooth: the marks
a viewer would actually name were removed before anything else ran.

So the band spans octaves. `texture_band` is the span as a multiple of
`DETAIL_SIGMA`, and 2.0 puts the coarse cut on `_SCATTER_SIGMA` — **this layer
owns everything finer than the distance light diffuses under skin, and §7's
scatter pass owns everything coarser.** Surface against shading. That is a
physical line rather than a chosen constant, and it means the two layers still
cannot reach into each other's band.

**Octaves are cut, normalised and weighted separately.** That makes this a pass
per kind of skin feature, keyed on the property that actually distinguishes them
— scale — rather than on a classifier. A detector for "spot vs mole vs crease"
on one uploaded photograph under unknown lighting is the unreliable part, and
applying the wrong gain to real detail because the classifier was wrong is a
worse failure than a coarse split. `texture_relief` is the mark octave's share
against the pore octave, in quadrature; `0.0` reproduces the shipped pores-only
map exactly, which is what keeps the old behaviour available for A/B.

An octave whose measured deviation falls below `_OCTAVE_FLOOR` is **dropped, not
normalised up**. Per-octave normalisation is what makes the knob portable across
photographs, but applied to a band holding nothing real it would amplify sensor
noise and JPEG ringing — and an octave of amplified ringing, reprojected onto a
moving face, is precisely the crawl §12 warns about.

The octaves are summed into one map at extraction and memoised, so however many
there are, **the live path still does one `warpAffine` of one single-channel
image per frame.**

### 6.3 RMS is the wrong statistic for a mark

The third cause, and the subtlest.

The map is normalised to unit deviation and spent against a budget denominated in
deviation. A *sparse* field spends that budget badly. A dozen spots on an
otherwise flat cheek contribute little to a second moment, so matching second
moments scales them down until they sit at the level of the dense pore noise
around them. Dense-and-uniform spends an RMS budget efficiently; sparse-and-
strong does not — and sparse-and-strong is exactly what a blemish is.

`texture_contrast` expands the amplitude distribution of the mark octave and
renormalises: the same total energy, redistributed toward the structure. On a
Gaussian field a γ of 1.6 raises the p99.5/RMS ratio by about a fifth; on a real
skin band, which is already heavy-tailed where marks exist, more.

**Applied to the mark octave only.** The pore octave is dense filler by nature,
and expanding its amplitude distribution is how a pore field becomes speckle —
the failure the monochrome rule exists to prevent, arrived at from the other
direction.

### 6.4 What the three are worth, measured

On a synthetic face constructed to sit in the starved regime (`_match_detail`
unclamped and reaching parity, which is the real condition), source carrying both
pore noise and 3–8px marks:

| configuration | headroom | reaching the picture |
|---|---|---|
| as shipped (band 1.0, no reserve, no shaping) | 0.30 | 0.364 |
| + wider band alone | 0.78 | 0.378 |
| + reserve alone, narrow band | 0.31 | 0.375 |
| both, strength 0.4 | 0.80 | 0.455 |
| both, strength 1.0 | 0.86 | 0.647 |
| band 3.0, strength 1.0 | 1.17 | 0.770 |

Read the attribution honestly. **The band widening is the dominant term; the
reservation is secondary and does almost nothing without it.** The intuition that
led here — that the budget was being taken by the previous stage — was correct
about the mechanism and wrong about which term mattered most, and only the
factorial showed that.

It is also a synthetic fixture. It establishes the mechanism and the direction.
It does not establish the magnitude on a real face, and nothing here substitutes
for §10.

### 6.5 The reservation was made and never filled

*Added 2026-09-07, after §10 was finally run on a still: freckles obvious in the
source, absent from the output. The measurement is on that source —
`source/Two`, IMG_3745, a 275px face with clear freckles — through the shipped
code rather than a reimplementation of it.*

Everything in §6.1–6.4 was in place, and the layer was still adding **exactly
nothing** at the strengths this document recommends:

| | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|
| headroom | 0.00 | 0.00 | 0.00 | 1.17 | 2.81 | 3.48 |
| added | 0.00 | 0.00 | 0.00 | 0.59 | 1.97 | 3.48 |

Two defects, and the first is the one that made the range dead.

**The grain budget was spent twice.** `_texture_headroom` computes
`real² − fake² − grain²`. The target's band is skin *and* sensor noise, so
`real` already contains the noise — and `_match_detail`, aiming at `real`,
amplified the swap's own structureless band until it covered the noise too.
Grain then added the noise a second time, and the subtraction was left with
nothing:

    real  5.50    grain 2.32    (42% of the amplitude, 18% of the energy)
    reserve 0.3 -> real² - fake² = 3.59  <  grain² = 5.39   ->  0
    reserve 0.4 ->                  4.52  <          5.39   ->  0
    reserve 0.5 ->                  6.76  >          5.39   ->  1.17

Note what this does to §6.1's safety property. That section says a reservation
must only be made when the layer will run, because an unfilled reservation
leaves the face *softer* than with no layer at all — and it lists the gates
checked in advance, then explicitly excuses the headroom: "that direction is
safe: reserving makes headroom more likely to exist, not less." **That is true
of `real² − fake²` and false of the whole expression**, because `grain²` is
subtracted unconditionally and, below reserve 0.5, exceeded what reserving had
freed. The one failure the section was written to prevent was reached through
the one term it declined to check.

`_match_detail` now discounts the noise once, when reserving. Measured on the
**grayscale** band, not the pooled per-channel one it uses for the ratio:
`_estimate_noise` returns a luma sigma and `_add_grain` adds a luma field, so
only a grayscale denominator makes the conventions cancel. Since it is gated on
reserving *and* grain, a run without either is bit-identical — which also means
the pre-existing over-parity (swap matched to skin+noise, then given grain on
top, ~8% over in amplitude) is untouched and remains open.

**`texture_strength` was applied twice.** The reserve is `strength`, and the
spend was `strength × headroom`. Since headroom is what the reserve freed, the
delivered amplitude went as the **square**: 0.4 asked detail matching to stand
back by 40%, then filled 40% of what that freed, delivering 16%. The knob is
documented in §B1 as "the fraction of the measured gap to close", so closing the
gap it opened is what it has to do. Below parity the layer now spends the whole
measured headroom; `f² + g² + t² = r²` still lands the total exactly at parity,
so overshoot remains impossible. Above 1.0 the diagnostic overshoot keeps
multiplying, since past parity there is no reservation left to act through.

Before and after, in one rig, on that source — added deviation in 8-bit units,
against a source freckle carrying 12.8:

| strength | before | after |
|---|---|---|
| 0.3 | **0.00** | 0.56 |
| 0.4 | 0.28 | 1.00 |
| 0.5 | 0.58 | 1.25 |
| 1.0 | 2.30 | 2.34 |

The correction does its work where the knob is meant to be used and converges
where the old behaviour already worked, which is the shape a fix to a budget
should have.

**A reservation that finds no room now says so**, once, naming both numbers.
That state was completely silent before: the readings carry the zero, but until
this was found only a *stream* reported readings at all, so a still render — the
one job shape where a face can be studied closely — printed no `REALISM` block.
Both are fixed together, because neither is much use without the other.

**A third correction, and it is what removed the dead zone entirely.** With both
fixes above, a reserve of 0.2–0.3 *still* found no room. The cause was that the
two stages were reading the same face through three different instruments:

| | `_match_detail` | `_texture_headroom` |
|---|---|---|
| statistic | pooled per-channel deviation | grayscale deviation |
| region | the whole crop | a centred 160px window |
| mask | the full compositing mask | skin only, features cut out |

Each difference is small. Together they left detail matching overshooting its
own aim by ~2.5% — nothing against a large reservation, and the whole of a small
one. **The region was the dominant term**; aligning the mask and the colour
convention alone moved the numbers by under 0.2%. When reserving, `_match_detail`
now measures exactly what `_texture_headroom` will:

| reserve | aim wanted | fake before | fake after | headroom before | after |
|---|---|---|---|---|---|
| 0.2 | 5.21 | 5.35 | 5.09 | **0.00** | 1.56 |
| 0.3 | 5.07 | 5.25 | 4.99 | 0.83 | 1.83 |
| 0.4 | 4.87 | 5.06 | 4.88 | 1.66 | 2.11 |
| 0.6 | 4.26 | 4.21 | 4.17 | 3.25 | 3.30 |
| 0.8 | 3.19 | 3.55 | 3.41 | 3.96 | 4.08 |

Headroom is now `reserve × skin_only` to within a few percent, which is positive
for every reserve above zero — so the knob is linear with no dead zone whose edge
nobody could predict. Cost, measured: `_scatter_weight` 0.2ms, `_estimate_noise`
0.5ms, two band deviations 0.3–0.8ms — about **1.0ms at aligned 128 and 1.2ms at
256**, paid only while the layer is on. The unreserved path is untouched.

**And the reason none of this was noticed for so long.** With texture off, the
composite lands at **1.002** of the target's band: `_match_detail` under-reaches
by 5.6% (the clamp and the warp) and grain adds ~6% back. The double count and
the under-reach cancel almost exactly. That is why the tuning settled where it
did, and why the moment a reservation is made — which lowers `fake` deliberately
while grain stays fixed — the cancellation breaks and the entire reserve
disappears. It is also why the noise discount is gated on `reserve > 0`: applied
unconditionally it would take a composite sitting at 1.002 of parity down to
~0.92, which is strictly worse.

**Two hypotheses tested and dropped, one of them a retraction.**

- **Not chroma.** The extractor is monochrome (`texture.py`, `_build_map`), and
  a freckle is a pigment feature, so a colour-carrying channel looked like the
  answer. At the freckles themselves it is worth **9%** of their ΔE — dL* −3.17
  against |chroma| 0.30 on IMG_3745, and −3.84 against 0.39 on a second source.
  The whole-skin ratio looks far better (chroma 0.58 of luminance) but that is
  measuring uncorrelated chroma *noise*, not freckle signal. Not worth building.
- **The shaping knobs were briefly written off here, wrongly.** p99 of the
  delivered map is ~3.6× its own deviation at every setting of `texture_contrast`
  and `texture_relief`, which reads as inert. It is not: p99 is an order
  statistic over a field the pore octave dominates by count, and the map is
  normalised to unit deviation, so its shape barely moves by construction.
  Scored where it matters — mean |map| at freckle pixels against mean |map| on
  plain skin, the same pixels every run — the knobs work as §6.3 claims:

  | band | relief | contrast | freckle : plain skin |
  |---|---|---|---|
  | 1.0 | any | any | 5.2 |
  | 2.0 | 0.65 | 1.0 | 10.0 |
  | 2.0 | 0.65 | 1.6 | 13.5 (default) |
  | 2.0 | 0.90 | 1.6 | 23.7 |
  | 2.0 | 0.90 | 2.4 | 28.6 |

  Two things follow. **At `texture_band` 1.0 both knobs are exactly inert** —
  there is no mark octave for them to act on — so a BAND slider at the bottom
  silently disables two of the four texture controls. And the defaults are
  conservative: 0.9 / 2.4 puts roughly twice the contrast into the marks, which
  is now a measured range to judge on footage rather than a guess.

### 6.6 What `texture_band` is actually trading

The panel's stated reason for carrying a BAND slider is that its right value
follows how large the operator's face and their skin's marks are, and so differs
per person. **Swept on two people, that is not true.** IMG_3745 (275px face,
strong freckles) and face-19 (350px face, subtle marks), scored at each person's
own marks against their own plain skin:

| band | IMG_3745 | face-19 |
|---|---|---|
| 1.0 | 3.5 | 3.3 |
| 1.5 | 8.4 | 8.0 |
| 2.0 | 9.2 | 8.7 |
| 3.0 | 9.8 | 9.1 |
| 4.0 | 9.7 | 8.9 |

Same shape, same optimum. The per-person argument is retired.

What it does trade only shows up on the composited output, and only against a
swap that has *no* marks of its own — which is the real condition, and which a
stand-in built by attenuating the target's own band quietly fails to reproduce.
Against a markless swap, on IMG_3745 at `texture_strength` 0.5, target freckle
contrast 21.6 and plain skin 3.4:

| band | freckles | plain | map energy above `_SCATTER_SIGMA` |
|---|---|---|---|
| swap alone | 9.2 | 2.6 | — |
| 1.0 | 10.1 | 2.5 | 10.6% |
| 2.0 | 12.3 | 3.3 | 27.1% |
| 3.0 | 13.4 | 3.5 | 33.6% |
| 4.0 | 14.0 | 3.5 | 37.9% |

Mark recovery is monotonic in the band, and so is the share of the map sitting
above the subsurface diffusion length — which is *shading*, and shading in this
map belongs to the source photograph's lighting rather than to the target's
frame. From 2.0 to 4.0: **+14% mark contrast, +11 points of imported shading.**

Note also that 2.0 is not the clean boundary §6.2 implies. A Gaussian band split
has soft edges, so 27% of the map is already above the scatter line at the
default. The line is a gradient, not a wall, and the default sits on it by
construction rather than by measurement.

That trade is a footage question and nothing else, which is the honest
justification for the slider. If footage settles it, `texture_band` becomes a
constant and the panel gets its space back.

**`1.0` is not the low end of that range.** It is the shipped-and-wrong value
§6.2 was written about, and it silently makes `texture_relief` and
`texture_contrast` inert as well — three of the four texture controls disabled
by one slider at its minimum.

### 6.7 What the layer carries, and what it cannot

Asked whether the work done for freckles is *about* freckles, or whether other
skin features come along with it. It is not about freckles at all — nothing in
the layer knows what a freckle is. Measured two ways.

**There is no feature selectivity.** The delivered map correlates **+0.97 to
+0.99** with the source photograph's own band inside the skin mask, at every band
setting. It is a faithful high-pass of that photograph and carries whatever is in
it. §6.3 says as much about the design intent — "keyed on the thing that actually
separates them, which is scale. Not on a classifier" — and this is that intent
holding in the output.

**So the question is scale, not kind.** What share of the source's energy at each
feature size survives into the map, at band 2.0 and the default shaping (1.00
would be the same share the source has):

| feature | px at 256 | share kept |
|---|---|---|
| pore, fine grain | 1–2 | 2.48 |
| freckle, small spot | 2–4 | 2.58 |
| mole, fine crease | 4–8 | 2.27 |
| wrinkle, scar, fold | 8–16 | 1.17 |
| shading, contour | 16–40 | 0.36 |

Everything up to ~8px is carried at roughly two and a half times its source
share — the per-octave normalisation doing what it was built for. Wrinkles,
scars and folds come through at about their source proportion, and are what
`texture_band` above 2.0 buys more of. Shading is suppressed, which is the whole
point of the band.

**Three real exclusions, and they are exclusions of kind rather than of scale:**

- **Colour.** The map is built from `cv2.cvtColor(crop, COLOR_BGR2GRAY)`, so a
  feature that is mostly chroma does not survive. Measured on two sources, a
  freckle is ~91% luminance and comes through; **rash and redness are almost
  entirely chroma and do not**; a pimple contributes only its dark rim, losing
  the redness and the specular highlight that make it read as one. This is the
  monochrome rule, and it is load-bearing — independent per-channel high
  frequency reads as coloured speckle.
- **Anything expression-dependent.** This is the important one and it is
  structural, not a tuning limit. The map is fixed at extraction and warped per
  frame by `canonical_from_frame`, a similarity transform from five keypoints
  with no expression term. A freckle is static on the skin and reprojects
  correctly. **A wrinkle is a deformation of the skin, not a mark on it** — a
  crow's foot appears when the operator smiles and flattens when they stop, and
  a fixed map paints the source photograph's version of it on regardless. Raising
  `texture_band` to reach deeper folds makes that worse, not better, because the
  deeper the fold the more it moves. No band setting and no per-condition
  algorithm fixes this; it would need a map that deforms with expression, which
  is a different subsystem.
- **Three-dimensional relief.** Anything whose appearance depends on the light
  direction — a raised mole, a scar ridge, a pimple — arrives as the shading it
  had in the source photograph, under that photograph's light, added to a face
  lit differently. Small features get away with it; large ones do not, which is
  the same reason the band stops where it does.

**On building a pass per skin condition.** The natural next thought is one
algorithm per condition — freckles, spots, rash, wrinkles. Two of those are
already served by the single scale-keyed pass and would gain nothing from a
detector; one (rash, redness) needs a *chroma* layer rather than a classifier,
because what it lacks is a colour channel and not a category; and one (wrinkles)
is unserved for a reason a detector cannot address. The gap here is
**dimensions — colour and time — not taxonomy.**

---

## 7. The diffuse / light-scatter pass

**Built** — `FaceCompositor._scatter`, `diffuse_strength`, default 0. Decision 7
is agreed: subsurface scattering is a genuinely separate axis from
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

`diffuse_strength` on `FaceSwapConfig`, reachable from `set_realism`, CLI, env
and the pod, defaulting **off** until judged on footage. Same treatment every
other realism knob gets.

**As built.** Aligned space, immediately before `_match_color`, so the colour
stages can still reconcile it. LAB's L channel only. The blur sigma is 3.0 at a
256px reference — roughly 2mm on a real face, which is the order of skin's actual
diffusion length — and scales with the working resolution like every other
spatial constant, so "soft" is the same physical distance whether the operator
leans in or sits back. Deliberately above `_DETAIL_SIGMA`: this is a shading
effect and that is a texture one.

Feature exclusions come from `face.kps` rather than the 106 landmarks, so they do
not depend on a layout that varies between model packs, with radii scaled by the
inter-ocular distance. They are feathered, because an unfeathered exclusion is a
visible disc of "sharp" sitting in softened skin — a seam of its own, at the
eyes.

One ordering property worth stating because it looks like a bug and is not: a
blur at `_SCATTER_SIGMA` does attenuate part of the texture band on its way past.
`_match_detail` runs **after** it and scales that band back up against the real
crop, so what scatter takes out of the texture band is restored and what it takes
out of the shading band stays out — which is exactly the split that was wanted.

**Cost, measured on a laptop CPU** — see §9 on why that is indicative rather
than predictive, and why the pod will print its own figure. **0.07ms at aligned
256** as a marginal cost, because the LAB conversion it needs is shared with
`_match_color` rather than paid twice; 0.65ms at 192 and 0.09ms at 320. In
isolation, with a conversion of its own, it was 3.18ms at 256. About 0.8ms of that is rebuilding the feature-exclusion
weight each frame, which the keypoints moving makes unavoidable without caching
work nobody has justified yet.

**The cheaper variant was measured and rejected.** Adding a monochrome luma delta
to all three BGR channels — the trick `_add_grain` uses — avoids the LAB round
trip and looked like the obvious saving. At 256 it runs 2.29ms against LAB's
2.41ms, a 5% gain, and drifts chroma **three times** as far (max 3 LAB units
against 1). The design's stated preference for an L-channel operation turns out
to be the right one on both counts, which is not what a guess would have
predicted.

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

**Every millisecond in this document was measured on a development laptop's CPU,
and the thing that will run this is a rented GPU instance.** Read them as shape,
not as magnitude.

The reason they transfer at all is that **a GPU does not touch these stages**.
The models are ONNX on the GPU; the compositor is NumPy and OpenCV on the CPU,
and `_scatter`, `_add_texture`, `_texture_headroom` and the seam feather are all
compositor work. Renting a faster card does not speed them up.

The reason they do not transfer exactly is that a pod's CPU is not this one, and
is not even constant across pods: [CLAUDE.md](../CLAUDE.md) records the whole
compositor-plus-paste-plus-encode bucket at ~20ms on an L4 instance and ~10.3ms
on a 4090 instance, and explicitly corrects an earlier claim that the non-GPU
portion was a fixed floor. It scaled with the instance, because the instance's
CPU scaled with it.

So the numbers here bound the *ratios* — texture grows with the face's area in
frame, scatter grows with the aligned size, the headroom sample is flat — and
say nothing reliable about the absolutes.

**What re-examining the CPU work found.** Asked whether anything should move
to the GPU, the answer turned out to be that ~15ms a frame was being *wasted* on
the CPU, and removing waste beats moving work. Four changes, all measured:

| at a large face | before | after |
|---|---|---|
| scatter, marginal cost with colour matching on | 3.18ms | **0.07ms** |
| grain at a 500px region | ~15ms | **2.83ms** |
| texture at a 460px face | 7.49ms | **7.16ms**, and now measured on skin |

- **One LAB conversion for both shading stages.** `_scatter` and `_match_color`
  are both LAB-domain, and each was converting for itself. A round trip is 1.9ms
  at 256 — more than everything scatter does with it. The conversion moved up to
  the caller, and scatter's marginal cost went to nearly nothing.
- **The scatter feature weight is built at a quarter resolution.** It is a smooth
  mask with five soft holes; the blur was running at sixteen times the pixels it
  needed. 0.43ms to 0.08ms at 256, agreeing to within 0.17 on a [0, 1] weight.
- **Grain reuses a cached noise tile** at a random per-frame offset instead of
  calling `np.random.normal` at region size every frame — 1.5ms at 256, scaling
  with area. The offset is what keeps it from reading as fixed-pattern noise,
  which is a worse artefact than none.
- **The noise estimate is bounded before its transform, not after.**
  `_estimate_noise` already knew a sigma converges on far fewer pixels than a
  face region carries — it strided the *result*, while the colour conversion and
  the Laplacian still ran over every pixel. 4.6ms at a 500px region, now flat.
  `_add_grain` also switched to `cv2` ops from numpy broadcasting, which `_paste`
  had already spelled out for exactly this reason: 4.2ms to 2.4ms.

**And the actual answer on the GPU.** It remains no, for now, and the reasons are
unchanged by any of the above: [CLAUDE.md](../CLAUDE.md) records that the felt
delay is dominated by a Romania round trip of ~350ms while compute has ~23ms of
headroom, and that the non-GPU portion already scaled with the card (20ms on an
L4, 10.3ms on a 4090) rather than being a fixed floor.

What *is* now true and was not before: torch is on the pod — `pipeline/core.py`
imports it unconditionally and the instance image supplies it — so the
`affine_grid`/`grid_sample` route is a code change rather than a dependency one.
The trigger to take it is the latency report showing compute as the largest term,
which it does not.

The one candidate worth naming if it ever is: **JPEG encode and decode**. It is
CPU, it is on the critical path in both directions, and NVJPEG exists. It has
never been measured on its own because it sat inside one bucket with the
compositor — and now it can be, since every compositor stage is itemised and
encode is the remainder.

**None of which needs resolving by argument.** `scatter` is its own bucket in
`LatencyBudget` and `texture` is recorded as a subset of `paste`, so the pod's
own latency report prints both, per-stage, against the preset's deadline, on the
hardware that matters. One stream answers it.

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
0. ~~**Diffuse / light-scatter pass**~~ — `FaceCompositor._scatter`,
   `diffuse_strength`, default 0. Built ahead of its trigger on the practical
   argument rather than the tidy one: it is default-off and independently
   toggleable, so it is *available* rather than *stacked*, and a pod session
   costs money — one session can now judge texture and scatter instead of two.
   See §7 for what it does and what it cost.
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

**C1. Footage, with the toggle**, starting at `texture_strength` 0.5 (see §6.5 — 0.2-0.4 was the range in which the layer added nothing at all, and the response is linear from 0.2 now). Both
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

### Phase E — open, but unjustified until footage says otherwise

Two items from the source document are neither built nor rejected. What they
share is that **nothing has yet shown they are needed**, and each would add a
layer to a stack that is already three deep and entirely unjudged. Each is
listed with the observation that would justify it, so the decision has a trigger
rather than a mood.

**Procedural micro-texture fallback** (source decision 5). Justified if footage
shows low-confidence regions reading *worse* than the surrounding face. Note
what B3 already does there: pose confidence reduces the *amount* of real texture
as the angle diverges, so a low-confidence region now gets less detail rather
than wrong detail. That may simply be the right answer, and synthetic fill is
only better if "less texture here" reads worse than "invented texture here" —
which is exactly the generic look this design exists to avoid.

**Low-frequency tonal variation** (§5). Last, always. It has the narrowest band
to live in — between `_match_illumination`'s 8x downscale and `_match_detail`'s
sigma — and two existing stages actively normalise what it would add. Justified
only if the face still reads as *evenly toned* after everything above.

### Phase F — the operator surface, after C1

**Not before C1**, and the reason is the same one that governs the restoration
dropdown: named steps are a promise about what the steps mean. `subtle` and
`balanced` for a layer nobody has looked at would be guesses wearing the clothes
of a calibration, and the operator would trust them.

When it does land, three decisions are already made:

**Fold texture into the existing restoration control, do not add a second one.**
Restoration removes skin texture and this layer puts it back: to the operator
they are one axis — "how synthetic does this look" — and two dropdowns that
interact is a worse control than one that does not. That is a *UX* coupling and
must not leak into the API: §4.2 measured the mechanical coupling at ~7%, so
`enhance_strength` and `texture_strength` stay independent under `set_realism`,
which is where the A/B happens. Presets pick a look; `set_realism` does the
science. This is the same split `PRESETS` and the model profiles already use.

**Do not expose the seam knobs.** `mask_feather` and `mask_erode` are
correctness, not preference — the same category as `color_correction`, which
CLAUDE.md records as removed from the header because "turning it off never makes
output better". There is one right transition width for a given face size and it
is computed from it. If a default is wrong, the fix is the default. They stay on
`set_realism` and the CLI for calibration, which is what that surface is for.

**Do not add a second source picker.** The tempting version — let the operator
choose which photo supplies texture — solves a problem nobody has had: the
scorer has never been watched picking badly, and building an override for a
failure that has not happened is how a UI acquires controls nobody understands.
The pipeline already emits `Texture source: <name> (<N>px face)` under a
`TEXTURE` scope, so the choice is visible before it is overridable, which is the
right order.

If footage does show it picking badly, the affordance is a **selection within
`review.accepted`**, never an independent upload. That constraint is not a
detail: the accepted set has been through the identity-outlier guard, and a
freely chosen texture donor could carry a different person's pores into the face
— a subtle identity leak underneath the guards that exist to prevent exactly
that. The cheaper fix, if the scorer is wrong, is its weights.

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
- **Added: texture and scatter together may miss the `production` deadline.**
  ~7ms combined with a large face after the §9 cuts, down from ~12ms. `optimal`'s 50ms against a
  ~27ms frame absorbs that comfortably; `production`'s 33ms would not. But the
  arithmetic is being done in the wrong units — the pod's CPU is faster than the
  one those were taken on (§9), so this is a flag to *read the latency report*
  at `production`, not a prediction that it misses. Judge realism at `optimal`
  regardless, where there is headroom to spare.
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

---

## 14. What will not be built

Kept as a list rather than deleted, because the reasons are the useful part: a
rejected idea that comes back without its reason is a rediscovery, not a
decision. Everything here is closed unless the evidence behind it changes.

**From the source document**

- **"Identity / feature augmentation."** It appears in the pipeline diagram
  between structural cleanup and colour transformation, with no module, no
  description and no stated effect. Not rejected on its merits — it has none
  stated. One sentence about what it augments would reopen it.
- **"Final colour grading."** There is no pipeline-side equivalent and there
  should not be. The desktop's filters already occupy that slot deliberately
  *client-side*, so a look can be auditioned without reaching a call, and
  [CLAUDE.md](../CLAUDE.md) says to keep them off while judging a swap. A
  pipeline-side grade would compete for the latency budget and be judged on a
  rented GPU, which is the wrong place for a preference.
- **The conditional second restorer** (CodeFormer after GPEN, §4.1). Two
  restoration models in series on the live path, for a stage measured at +0.03
  on the only metric it moves, gated by an artifact detector that does not
  exist. The fidelity weight was also specified backwards, at the
  maximum-hallucination end.
- **The strength-coupling assumption** (§4.2). Measured at ~7%. Independent
  toggles stay; the assumed direction goes.
- **A true UV unwrap.** Superseded rather than refused — the 68-point mesh in
  phase D uses a correspondence `buffalo_l` already computes every frame, where
  a 3DMM fit would be a new model on the live path for a coarser gain.

**From the source document's performance section** (§9)

- **BiSeNet on the live path.** A new per-frame ONNX inference, which is the one
  thing the design promised not to add. The landmark hull minus template feature
  exclusions, intersected with the XSeg pass that already runs, covers it.
- **Batching across frames** (#3). Trades latency for throughput; the live call
  is a latency problem, and the source document says so itself. Still open for
  RENDER.
- **Mixed precision and `torch.compile`** (#6, #7). There is no torch in this
  path and none should be added — the compositor is OpenCV on the CPU, and
  moving it would cost two device transfers around ~2ms of work.

**From this document**

- **Poisson / gradient-domain seam blending.** The textbook answer, already
  rejected in this codebase: it pulses frame to frame, trading failure mode 2
  for failure mode 3.
