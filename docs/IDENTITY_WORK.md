# Making the output look like the source

The complaint this answers: *"the resemblance between the source and the output
is good, but not very good."*

It was raised about the **swap model**, and the first thing worth saying is that
nothing in this repository could have told anyone whether the swap model was the
cause. `compare_frames.py` measures detail, noise and seam. `Readings` records
detail and texture headroom. **Identity was not measured anywhere**, so every
stage between the swapper and the screen was above suspicion by default — and
four of them attack likeness:

| Stage | What it does to identity |
|---|---|
| `gpen_bfr_256` at `enhance_strength` 0.7 | A blind restorer with an FFHQ prior, identity-agnostic by construction, blended at 70% |
| `_match_color` | Transfers the **target's** complexion onto the face. Skin tone is an identity cue |
| `temporal_alpha` 0.6 | EMA across frames; under motion it averages the face with where it used to be |
| The mask | A convex hull of the **target's** landmarks, so the output silhouette is always the target's |
| `inswapper_128` | 128px native, upsampled to 128-320 |

So the work is ordered instrument-first. Anything else is guessing with a
384 MB download attached.

## Status

| # | Item | State |
|---|---|---|
| 1 | Identity readings — ArcFace cosine per stage | **built** |
| 2 | `tools/identity_probe.py` — local A/B harness | **built** |
| 3 | HiFiFace registration | **built** |
| 4 | Shape-following mask | **built** |
| 5 | Identity push (embedding extrapolation) | **built** |
| 6 | Pose-weighted source averaging | **built** |
| 7 | Complexion keeping — chroma held back in the colour match | **built** |
| 8 | `source_blend` — the identity-averaging strategy as a swept field | **built** |
| 9 | Tests, lint, docs | **done** — `tests/test_identity.py`, 61 checks |

**Every one of them is unjudged on footage.** That is the whole of what is left,
and it is deliberately what is left: none of this decides anything until someone
runs it and looks. The numbers below are mechanism, not results.

---

## 1. Identity readings

`pipeline/services/identity.py` measures cosine similarity between the source's
averaged ArcFace embedding and the face that actually comes out, at five points:

    id_swap      the swapper's raw crop, before restoration
    id_restore   after restoration
    id_final     after scatter, colour, detail and texture, before the paste
    id_out       re-detected from the finished frame  <- the one that counts
    id_target    the finished frame against the TARGET's identity

`id_out` is the only one measured the way a viewer sees it: through the mask, at
the face's real size in frame. The three before it exist to attribute a loss to
a stage. `id_target` is the control — a swap that is not taking shows up as
`id_target` rising rather than `id_swap` falling, and those want opposite
remedies.

Numbers to hold them against, for ArcFace cosine on `buffalo_l`:

| | |
|---|---|
| > 0.6 | the same person, comfortably |
| 0.4 - 0.6 | recognisably related; the region a good swap lands in |
| 0.28 | InsightFace's own verification threshold |
| < 0.2 | not this person |

**Absolute values are not the point.** A swap is not a photograph of the source
and will not reach 0.9. What the readings are for is *differences*: between
stages within one run, and between two configurations on the same clip.

The report says so itself rather than printing five cosines and leaving the
reader to subtract them:

```
  -> identity: output 0.461 against the source (p50). Read the drops below.
     the swapper produced 0.611; what happened after:
       restoration            -0.070   (enhance_strength, or restoration_preset)
       colour/detail/texture  -0.010   (color_strength — skin tone is an identity cue)
       mask and paste         -0.070   (mask_erode, mask_feather, mask_shape_growth)
     -> restoration is the largest single loss. Sweep enhance_strength before
        changing the swap model.
```

Off by default. `identity_probe = N` measures every Nth frame; it costs one
ArcFace inference per measured stage, on a model the detector already has
resident.

**The fiddly part, and why it is a service.** The recognition model wants an
`arcface_112` crop — a specific framing, not merely a 112px face — and the
compositor's crops are in the *swapper's* alignment, which is `arcface_128` for
some models and `mtcnn_512` for others. Both are similarity transforms of the
same five points, so the map between them is closed-form: fit the crop's own
template to the 112 one and warp. A probe that ignored the template it was given
would still return a plausible cosine, measured on a face 6% out of frame, which
is worse than no reading. `tests/test_identity.py` pins it in both directions —
each template lands on target, and reading a crop with the *wrong* template does
not.

## 2. `tools/identity_probe.py`

One source, one target image, through the real compositing path, sweeping
whatever you name:

```
python tools/identity_probe.py -s me.jpg -t scene.jpg \
    --sweep enhance_strength=0,0.35,0.7 --sweep identity_push=0,0.2,0.4 \
    --save-frames out/
```

A still rather than a clip because a still needs no stream, no warm-up and no
pod session, and identity is a per-frame property. Temporal behaviour is a
separate question this deliberately does not touch.

It goes through `review_sources` and `get_source_face` rather than a bare
detection, so the guards, the outlier check and the pose weighting are all in
the loop — a sweep that skipped them would be measuring a different identity
from the one a real job uses.

`--save-frames` exists because the cosine measures one axis of three. A higher
number that reads as plastic or seamed is not a better swap, and the tool says
so at the end of every run.

## 3. HiFiFace

`hififace_unofficial_256`, registered in `swapper_models.py`.

| | |
|---|---|
| Weights | 204 MB, `models-3.1.0` — **not** `models-3.3.0`, which does not carry it |
| Converter | `arcface_converter_hififace.onnx`, 21 MB, same tag |
| Inputs | `source` = 512-d embedding, `target` = 256² crop, mean/std 0.5 |
| Template | **`mtcnn_512`** — the first model here that is not arcface |
| Vendor | GuijiAI via facefusion, tagged `license: Unknown` |

**What the 3D actually buys, stated precisely.** HiFiFace is trained with a 3DMM
in the loop, recombining the source's identity coefficients with the target's
expression and pose, so the generator learns to move the face **contour** toward
the source rather than only repainting the interior. Face outline is one of the
strongest identity cues a viewer has, and it is the one thing every other model
registered here leaves at the target's.

Two things not to over-read:

- **The 3D is training-time.** This export takes an embedding and a crop and
  nothing else. There is no 3DMM fit at inference and no per-frame
  reconstruction cost. What ships is a generator that *learned* shape awareness.
- **It keeps the embedding contract**, which is why it could be registered at
  all. `uniface_256` and `blendswap_256` take a source *image* and are
  deliberately absent, because that would break multi-photo averaging, `.npy`
  embeddings and the identity-outlier guard. HiFiFace takes a vector; the
  converter that maps ArcFace's space into its own runs **once per source**, not
  per frame.

The converter is fed the **raw** ArcFace vector, not the normalised one, because
it is a non-linear map fitted on ArcFace's own output scale. That is why
`_average_faces` now carries the raw embedding through — see §6.

## 4. The shape-following mask

**This is the fix that lets §3 matter.** The mask is a convex hull of the
*target's* 106 landmarks (`masking.py:_hull_mask`), intersected with XSeg run on
the *target's* crop. So the output silhouette is the target's face,
unconditionally — and a contour HiFiFace widens is clipped straight back off.
Roughly half of what that model produces would never have reached the screen.

`mask_shape_growth` admits the difference, and every bound on it lives in
`FaceMasker._shape`:

- **Bounded in extent** — at most that fraction of the crop beyond the target's
  hull, dilated with an *elliptical* kernel so the bound is a real radial
  distance. A square kernel reaches `limit * sqrt(2)` diagonally, which is
  exactly at the jaw corners where a wider face differs most; an 8% knob would
  have admitted 11%. The tests caught this.
- **Lower face only** — the ramp starts at the eye line, read from the template
  rather than assumed, because arcface puts it at 0.404 of the crop and mtcnn at
  0.467. Growth at the jaw covers what was neck or background, which is what a
  wider face genuinely looks like; growth at the temples covers **hair**, which
  this codebase already records as the worse tell.
- **Only where the generated face is** — intersected with the hull of landmarks
  detected on the swapped crop, so it can never expand in a direction the model
  did not put a face.
- **Self-neutralising** — a model that does not move the contour produces a hull
  that already matches, so this costs one landmark inference and changes
  nothing. That is what makes it safe to leave on across a model switch, and it
  is also the control: *if `inswapper` output changes when this is turned on,
  the landmark probe is wrong, not the idea.*

**The XSeg exemption is the subtle part.** XSeg segments the face in the crop it
was given, and that crop is the target's. Asked about the band the generated jaw
now occupies, it answers "not face" — correctly, because in the target's frame
it genuinely is not; it is the neck or the background behind a narrower jaw.
That is the right answer to the wrong question, and multiplying by it would
cancel the shape term *exactly*. So the growth band is exempt. It is safe to
exempt because of what bounds it: a few percent of the face's extent, below the
eye line, only where the generated face's own landmarks say there is face. An
occluder large enough to matter crosses the main hull too, where it is still
measured and still guarded — and the occlusion guard's denominator is
deliberately still the *target's* hull, so a shape-aware model does not read as
more occluded than a plain one on identical footage.

**What this does not do.** It cannot make the face *narrower* than the target's
in a way that shows: shrinking the mask reveals the target's real jaw underneath
rather than a background. The growth is one-directional by construction, and a
source with a notably narrower face than the target remains the case this does
not help.

## 5. Identity push

Every swap model lands somewhere *between* the source and the target — that
compromise is what "good, but not very good" is, stated in the model's own
terms. The generator's compromise is not reachable from outside, but the point
it is asked to render is:

    pushed = normalise( source * (1 + k) - target * k )

the source, moved along the line joining them, away from the target. At `k = 0`
it is the source exactly and the code path is bit-identical to not having the
feature. It is the negative half of the range facefusion exposes as
`face_swapper_weight`, which nothing here has ever used.

Two properties worth keeping:

- **Applied in ArcFace space, before any model-specific conversion**, so the two
  vectors being combined always come from the same space. Pushing a *converted*
  source against an unconverted target would be arithmetic on incommensurable
  things.
- **A stand-in, never a mutation.** The source face is the cached identity for
  the whole session and the push depends on *this* target, so writing to it
  would make each frame's push compound on the last one's.

Bounded at `PUSH_MAX = 0.6`: past roughly there the vector leaves the region of
the embedding space the generators were trained on, and the output degrades
toward a face that is nobody rather than one that is more the source. 0.2-0.3 is
where to start.

## 6. Pose-weighted source averaging

The flat mean this replaces treats a sharp frontal portrait and a blurred
three-quarter shot as equal evidence, and they are not: ArcFace embeddings
degrade with pose. The outlier guard already refuses the *wrong person*; nothing
refused a right-person photograph carrying a weaker reading of them, and
averaging pulls the result toward whatever those weaker readings have in common,
which is a blander face.

Weight is detection confidence times `cos²(yaw)` — squared so the penalty stays
gentle through the range people actually photograph themselves in (20° costs
12%) and bites toward profile. Absent either term the weight is 1.0 and this
degrades to exactly the mean it replaces; a capability gap must never become a
silent behaviour change.

A floor at 25% of the best photograph's weight, because the point of accepting
several is that identity is a distributed representation. Without a floor one
sharp frontal shot reduces the others to rounding error, which is multi-photo
averaging switched off by accident.

It also now carries the **raw** embedding through, which HiFiFace's converter
needs. Dropping it was invisible for as long as every registered model took the
normalised vector, and would have surfaced as hififace producing a weaker
identity from four photographs than from one.

## 7. Keeping some of the source's complexion

Skin tone is an identity cue, and `_match_color` spends all of it: the global
mean transfer moves the swapped face onto the **target's** complexion. That is
not a bug — a face whose tone does not match the neck it sits on is failure
mode 2 — but the correction is applied to luminance and chroma alike, and those
are not equally costly to relax.

`complexion_keep` holds back some of the **chroma** correction only. Four things
make it safe:

- **Luminance is still corrected in full.** A brightness step at the jaw is the
  most visible seam there is. Only a/b are touched.
- **Chroma is the cheap channel here for a pipeline-specific reason.** Every
  frame is JPEG-encoded at OpenCV's default 4:2:0, so chroma is already
  subsampled 2x in both axes before it reaches the call. A chroma step at the
  seam is blurred by the transport in a way a luminance step is not.
- **Bounded by the difference, not by the knob.** What is withheld is capped at
  `_COMPLEXION_RESIDUAL` (3.0 LAB units of a/b distance), so close tones keep
  the whole fraction and far-apart tones give way entirely. CLAUDE.md records
  the hard case — a fair, well-lit source against a dark, under-lit target — and
  keeping the source's complexion *there* is a light face on a dark neck, which
  is a broken composite rather than a better likeness. This withdraws exactly as
  the gap grows.
- **`_match_illumination` gets the same factor, and this is the one that would
  have gone wrong silently.** A held-back chroma mean is a low-frequency
  difference, which is precisely what that stage exists to find and correct — so
  without threading the factor through, ~70% of whatever the global pass
  withheld would have been put straight back, and the knob would have looked
  like it barely worked.

It withholds correction rather than importing the source photograph's colour.
The swap already carries some of the source's complexion; steering toward the
donor image's *measured* tone would carry that photograph's white balance with
it, which is a different and worse thing to be right about. That version stays
available — `SourceTexture` already caches an FFHQ crop of the donor — if this
one earns it.

**This is the one item here the cosine cannot judge.** ArcFace is largely a
shape-and-texture model and is fairly insensitive to skin tone, so `id_out` will
barely move while a person may see the difference immediately. `complexion_kept`
in the readings says whether the layer had anything to spend;
`compare_frames.py`'s `seam_excess` says what it cost. The benefit is footage
only. Start at 0.4.

## 8. `source_blend`

The averaging strategy as a swept field rather than a decision, because the
question underneath it is genuinely open. **Averaging is a low-pass filter on
identity**: a feature present in two photographs of five is down-weighted by
five halves, so the average is the most *typical* version of the person — the
same failure mode as blind restoration, arriving one stage earlier.

That matters because averaging is validated for **recognition**, where a robust
estimate of the class centre is exactly what you want. This pipeline wants the
vector that makes a generator produce the most *recognisable* face, and those
two optima are not obviously the same point.

| | |
|---|---|
| `mean` | flat average — what shipped before any of this |
| `weighted` | `det_score × cos²(yaw)` — **default** |
| `norm` | `‖embedding‖ × cos²(yaw)`, the network's own quality signal |
| `best` | the single highest-scoring photograph, no averaging |
| `median` | geometric median (Weiszfeld), resistant to an outlier |

`norm` is the one I would expect to win over `weighted`: the un-normalised
embedding's magnitude is the network's own confidence rather than two hand-picked
correlates of it, which is the relationship MagFace is built on. Verify it
correlates on `w600k_r50` before trusting it.

`median` is included so that "a robust centroid is not worth it at this N" is a
*measurable* claim rather than an opinion. Its whole value is a 50% breakdown
point; with three or four photographs, two would have to be bad before it
diverges from the weighted mean — and at that point `_review_identity`'s
leave-one-out check has already refused them. Expect it to change nothing here,
and to matter if a source ever becomes a video.

**The methodological trap, which matters more than the strategies.** You cannot
score these against the identity they build — the strategy under test would
define its own yardstick and every one of them would win, because every one is
closest to itself. `tools/identity_probe.py --holdout` builds from all but the
last photograph and scores against the one left out, which is the same
leave-one-out structure the outlier guard already uses. **A sweep of
`source_blend` without `--holdout` is meaningless**, and it is easy to run by
accident because it produces confident-looking numbers.

---

## How to spend the next session

In this order, because each step decides whether the next is worth taking.

**1. Measure the current default before changing anything.** `identity_probe=5`
on a stream, or `tools/identity_probe.py` on a still. Read `id_swap` against
`id_out`. That single gap is what the compositor charges for the seam, and until
it is known, "the swap model is the problem" is a hypothesis.

**2. Sweep what is free.** `enhance_strength` at 0.7 / 0.35 / 0, then
`identity_push` at 0 / 0.2 / 0.3, then `complexion_keep` at 0 / 0.4 / 0.8. All
instant, all against the model already loaded, and if restoration is the largest
loss the fix arrives the same afternoon.

Judge `complexion_keep` by eye and by `seam_excess`, **not** by `id_out` — see
§7 for why the cosine is blind to this one. And sweep `source_blend` only with
`--holdout`; see §8 for why it is meaningless without.

**3. Then the models.** `hyperswap_1a_256` has been registered and never judged
on appearance. `hififace_unofficial_256` needs `mask_shape_growth` set to
0.06-0.08 to show what it is for — measure it with the growth at 0 *and* at
0.08, because the difference between those two is the entire argument for
§3 and §4.

**4. Look at the frames.** Every number here measures one axis of three. The
other two are whether it reads as plastic and whether there is a seam, and those
have their own tools and their own history of disagreeing with a good-looking
statistic.
