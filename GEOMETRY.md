# Geometry — whose head shape does the output have?

The complaint this exists to answer came off footage, not off a number: **swaps
read better when the source and target head shapes are similar, and degrade when
they differ.** Nothing in the pipeline measured that, and the instrument that
existed — ArcFace cosine — is trained to be invariant to most of the geometry
involved, so a swap can move the jawline visibly and shift `id_out` by almost
nothing.

Status: **instrument built and corrected twice; one lever measured; the outline
lever untested.**

---

## 1. The central finding: there are two channels, not one

A viewer reads "head shape" from two separate things, and this pipeline treats
them completely differently.

| | What it is | Does the swap move it? |
|---|---|---|
| **Interior contours** | Where the cheekbone sits, how the jaw shadow falls, the width between the cheek lines, nose and mouth structure | **Yes, substantially** |
| **Outline** | The literal silhouette — skin against background, jaw edge, hairline | **No. Not by one pixel.** |

Both were verified independently of the shape metric, on
`mask/swapper_model-alphaface_256_mask_erode-0.0_mask_feather-0.02.png`:

- A Canny edge overlay (target in red, output in green) shows the **entire outer
  boundary in yellow** — face against the wall, jaw, neck, shoulder, hairline all
  coincide exactly. Red/green separation appears only in the interior: nose,
  nasolabial folds, mouth, and a **new cheek contour line** in the output that
  has no counterpart in the target.
- Measuring the skin-to-background boundary per scanline gives a shift of
  **0.00px, max 0.0px** on both sides.

The outline cannot move, and the reason is structural rather than a tuning
failure:

    the swap model generates into a crop framed by the TARGET's five keypoints
    the mask is a convex hull of the TARGET's 106 landmarks
    the mask stops short of the silhouette, so those pixels are never touched

**So the head does read as the source's — via the interior — while the
silhouette stays the target's.** Both statements are true at once, and confusing
them cost this project several rounds of work.

---

## 2. The instrument

`pipeline/services/shape.py`. Landmarks from the pack's own 106-point model on
three faces — source photograph, target, finished output — reduced to pure shape
by fitting away the similarity transform between them.

| Reading | Meaning |
|---|---|
| `shape_mismatch` | How far apart the source's and target's head shapes are. **A property of the pairing** — no setting moves it. This is the quantity behind the original observation |
| `shape_shift` | Whole-face movement off the target's shape toward the source's. 0 kept the target's, 1 took the source's |
| `interior_shift` | The same, interior features only — **the channel that actually moves** |
| `outline_shift` | The same, silhouette only — the channel the mask clips |
| `outline_swap` / `outline_final` | The silhouette measured on the generated crop and after restoration, for attribution |

Gated on `identity_probe=N`, reported in the `REALISM` block and in
`tools/identity_probe.py`.

### Properties that make it trustworthy

- **Similarity-invariant by construction.** The three landmark sets come off
  three different images at three different scales, so each is normalised to
  unit radius *before* the fit rather than the residual being divided after.
  Pinned: scaling the output 3.7x and rotating it 23° must not move the reading.
- **The outline subset is the convex hull, not an index range.** The 106-point
  layout is a property of the model pack, so "0-32 is the jaw" would be right for
  `buffalo_l` and silently wrong for the next one. Points are ranked by distance
  to the hull — which is literally what `FaceMasker` fills — and the nearest
  third taken.
- **Each subset is fitted on itself.** Separation between a contour-only change
  and an interior-only one: **0.983** fitting the outline on the outline, against
  0.821 on all points and 0.724 on the interior.

---

## 3. Known limitations of the instrument

Recorded because two of them cost real time.

- **It measures landmark positions, not appearance.** A swap changes the
  *appearance* of features at roughly fixed landmark positions, and much of what
  a viewer reads as head shape lives there. `outline_shift` reading ~0 is a true
  statement about landmarks and was mistaken for "the head shape did not change".
  **Read `interior_shift` and the frames together; neither alone answers the
  question.**
- **`interior_shift` is the weaker half, knowingly.** A similarity fitted on the
  interior alone can absorb a uniform scaling of the features, so a scale-like
  change is partly fitted away. It points the right way; it is not a calibrated
  fraction.
- **`shape_shift` came back consistently negative** (−0.03 to −0.08) across every
  run. If the true answer were "no change" it should sit at zero ± noise. A
  consistent negative bias suggests something systematic — possibly the landmark
  model placing points differently on a swapped face than a real one. **Unresolved.
  Do not lean on the sign.**
- **Pose inflates `mismatch`.** Out-of-plane rotation changes a face's apparent
  2D shape. It inflates `gap` by nearly the same factor and `shift` is their
  ratio, so the ratio survives pose far better than either absolute.
- **The shape reference is one photograph**, chosen by `select_shape_source`
  weighting frontality 0.65 / size 0.25 / sharpness 0.10, scoring yaw **and**
  pitch together (`off_axis`), roll excluded. Separate from the texture donor,
  which is picked for sharpness — the two want different photographs, and using
  one pick for both was measured wrong (it returned a 28°-off-axis donor).

---

## 4. What has been measured

RTX 5880 Ada, `alphaface_256`, source `source/one` (21 of 30 accepted),
targets `source/two/IMG_3623.jpg` and `IMG_3745.png`.

### The pairing is genuinely mismatched

`shape_mismatch` **0.142** of face size on IMG_3623 (0.192 at the outline) and
**0.157** on IMG_3745. That is large, and it confirms the original observation
had a real quantity behind it.

Note the frontal shape reference *raised* this from 0.099 — the earlier
28°-off-axis reference was **understating** the difference, not inflating it.

### No LIVE model moves the contour

| model | `outline_swap` (what the generator produced) |
|---|---|
| `alphaface_256` | **+0.001** |
| `hififace_unofficial_256` | **+0.001** |

`hififace` is the only registered model advertised as 3D-shape-supervised and it
moved the contour no more than alphaface — while scoring far worse on identity
(`id_swap` 0.565 against alphaface's 0.907, and higher target leakage at 0.374
against 0.289).

**Therefore `mask_shape_growth` is inert.** It exists to stop the target's hull
clipping a widened contour; there is no widened contour to clip. Measured at 0
and 0.08 on both models: `outline_shift` moved from +0.007 to +0.008. That
closes an experiment that had been queued unrun for a long time.

### The mask is the dominant identity cost, and it is a geometry lever

For the baseline: restoration **+0.001**, colour/detail/texture **−0.006**,
**mask and paste −0.101 to −0.161**. Restoration is free; the mask is everything.

`mask_erode` measured 3×3 against `mask_feather`:

| erode | feather | `id_out` | `id_target` |
|---|---|---|---|
| **0.00** | **0.02** | **0.800** | **0.192** |
| 0.00 | 0.04 | 0.790 | 0.211 |
| 0.03 | 0.04 (old default) | 0.713 | 0.293 |
| 0.06 | 0.08 | 0.604 | 0.429 |

**It costs on both axes at once** — the old default gave up 0.087 of source
similarity *and* handed 0.101 back to the target.

**What the knob really controls is how much of `_HULL_EXPAND` survives.** The
hull is grown 10% radially and the erode takes a constant number of pixels back:

| aligned size | 128 | 192 | 256 | 320 |
|---|---|---|---|---|
| expansion (px) | 4.2 | 6.3 | 8.4 | 10.5 |
| erode 0.030 | 4 | 6 | 8 | 10 |
| erode 0.015 | 2 | 3 | 4 | 5 |
| erode 0.0075 | 1 | 1 | 2 | 2 |

So 0.03 cancelled the expansion outright, leaving the mask at the bare landmark
hull — and that expansion exists precisely because "the 106 points stop at the
eyebrows and hug the jaw, so a bare hull clips the swap". **The default was
undoing its own correction.** Now **0.015**, which is a floor rather than a
midpoint: below it the constant-pixel erode is dominated by rounding and the
mask would behave differently by preset and by how close the operator sits.

### XSeg costs identity and compounds with the erode

| erode | `occluder` | `id_out` | `id_target` |
|---|---|---|---|
| 0.0 | on | 0.791 | 0.215 |
| 0.0 | off | 0.803 | 0.184 |
| 0.015 | on | 0.764 | 0.249 |
| 0.03 | on | 0.719 | 0.295 |

0.012 / 0.028 / 0.046 as the erode grows. **This prices XSeg's cost, not its
benefit** — the subject's hair is tied back, so there is nothing over the face
for it to exclude. Never read it as an argument for turning the occluder off.

---

## 5. What is NOT settled

- **Whether `mask_erode` should go to 0.0.** The cosine wants it (0.800 against
  0.764) and by eye the only visible cost is at the hairline, where a smaller
  erode lets smoothed swap skin reach into fine hair at the temples. Not taken
  because three cases are untested: **XSeg is off on the `fast` preset**, which
  is the gear an operator drops to when the link fails, so there the erode is the
  only protection; a **turned head** pushes a radial expansion past the visible
  silhouette into background; and zero leaves no margin for landmark error.
  Also, `id_out` embeds a crop framed on the face, so **more coverage mechanically
  raises the cosine whether or not the extra coverage landed on skin** — the
  metric rewards reaching onto hair.
- **The `shape_shift` negative bias.**
- **Everything at live resolution.** All of the above is a 257×285px face in a
  still. A live call at 640×360 has a ~100px face, `mask_feather` measures
  against face extent, and the verdicts may not carry. Judge final values at
  `optimal`.
- **A loose-hair or turned-head target.** The single case that would settle both
  the erode ceiling and XSeg's benefit, and it has never been run.

---

## 6. Moving the outline — the options

Ranked for a live video-call product.

**1. Let the mask reach further.** The outline is unchanged *because the mask
stops short of it*. This is the cheapest lever, it has measured evidence behind
it already, and it acts on exactly the boundary in question. The open question is
how far is safe — see §5. A jaw-specific growth (where there is no hair) rather
than a uniform one is the obvious refinement and has not been built.

**2. TRAINED tier (`.dfm`, DeepFaceLab).** The only class that moves the
silhouette **at frame rate**. Its `dfl_whole_face` crop takes in jaw and
forehead rather than stopping at the arcface box, and it has no generic-face
manifold to average toward. Economics fit a fixed set of operators: train once,
reuse across every call.
**Cost:** 3,000–10,000 aligned faces, realistically extracted from 5–15 minutes
of varied video rather than assembled from stills — full yaw and pitch range,
expressions, eyes and mouth open and closed, several lightings. Coverage matters
more than raw count. Training ~12–48h on a 4090 from a pretrained model, days
from scratch. The current onboarding flow (30 photos) is two orders of magnitude
short and photos alone cannot cover the pose space.

**3. STUDIO head swap (`reface`, `ghost_2`).** Replaces the whole head including
the silhouette — the only thing that lifts the mask ceiling outright. `reface`
carries the strongest identity figure in the open literature (98.8% top-1 on
CelebA). **~0.6s per image, so RENDER, photo and templates only.** Registered,
never run against real weights. This is the fastest route to head shape on the
non-live paths.

**4. Frame-space warp.** Deform the picture toward the source's proportions,
dragging surrounding pixels so there is no hole. Live-capable but **cosmetic** —
face-slimming, not identity transfer.
Design notes from the session that scoped it: post-warp rather than pre-warp
(geometrically equivalent, and pre-warp additionally takes the generator
off-distribution); a hard magnitude bound as a fraction of face extent; validity
checked by **landmark round-trip** — did the landmarks land where the warp put
them — rather than by "is a face still detected", since detectors run at
`det_thresh=0.35` and tolerate far more distortion than a viewer will; computed
**once per pairing** rather than per frame, because the shape difference between
two people is fixed while only pose changes; and falling back to strength zero
on failure rather than refusing the frame. Budget for jitter from day one — a
landmark-driven field shimmers on the longest boundary in the picture, and
`LandmarkStabilizer` will not catch it because the motion is *correct* landmark
motion.

**5. A different LIVE ONNX model — will not work.** Measured, twice, on two
models. Structural, not tunable.

---

## 7. A note on the alignment itself

`canonical_from_frame` and every other fit here is `estimate_similarity` — **4
degrees of freedom**: one uniform scale, rotation, translation. There is no
independent vertical and horizontal fit anywhere in the chain.

This has two separate consequences that are easy to conflate:

- **Placement.** A donor proportioned differently from the target lands
  misaligned. Relevant to the texture layer, where the map's baked-in skin mask
  travels off its features with it.
- **Shape transfer.** A similarity cannot make one face's proportions into
  another's, by construction. That is why option 4 above requires a genuine
  deformation and not a better fit.

An affine (6 DOF) would add shear and anisotropic scale, covering face
width-to-height ratio — a large share of head-shape difference. The cost is that
`compositor.py` computes working scale as `sqrt(|det|)` in two places, which
assumes uniform scaling and feeds the texture band, the seam feather and the
working resolution. Not a one-line change.
