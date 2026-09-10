# Texture — why the swapped face reads as too clean

**Parked.** Read this before restarting it; two things were fixed, one remains,
and the layer stays off.

The swapped face carries roughly 58% of the frame's high-frequency energy
against an ideal of 1.00. That deficit is what reads as plastic, and
**restoration is not what causes it** — turning restoration off entirely moves
the number by 0.03. The detail was never there: the swapper generates at 128 or
256 native and everything downstream resamples that.

So `pipeline/processing/texture.py` takes it from where it does exist — the
operator's own source photograph — high-passes one image into canonical FFHQ
framing, and `FaceCompositor._add_texture` warps it onto the face every frame.

Status: **`texture_strength` = 0.0 and should stay there.** Now on measurement
rather than caution.

---

## 1. The verdict

Swept 0.0 / 0.5 / 1.0 / 2.0 on two stills, `alphaface_256`, RTX 5880 Ada.
High-pass deviation on a flat cheek patch of `IMG_3623`:

| `texture_strength` | 0.0 | 0.5 | 1.0 | 2.0 | target |
|---|---|---|---|---|---|
| **before** the warp fix | 5.947 | — | 4.114 | — | 7.741 |
| **after** the warp fix | 5.947 | 5.235 | 4.456 | — | 7.741 |

**The layer makes the face smoother, not more detailed, at every strength.**
Both octaves fall together and proportionally, on both targets, so it is not
trading pores for marks — the whole band loses energy.

By eye it is worse than the numbers suggest: at 0.5 a dark line appears under
the eye and across the cheek; at 1.0 the face reads as **aged**, with folds that
are in neither the target nor any photograph of this person at this expression.

---

## 2. What was fixed, and why it was not enough

### Bug 1 — the layer was inert in the measurement tool

`_add_texture` needs a `source_texture`; the pipeline sets one in
`SwappingProcessor._load_texture`, and `tools/identity_probe.py` never went
through it. A sweep would have returned identical rows and read as "the layer
does nothing" — a confident wrong answer about a layer that was never switched
on. Fixed: the rig extracts from the texture pick and reports which photograph,
with its pore and mark deviations.

### Bug 2 — the map was normalised before the warp and spent after it

`detail_for` returns unit deviation in **canonical** space. `warpAffine` then
resamples it **bilinearly**, and bilinear interpolation is a low-pass filter
whose response falls to zero at Nyquist — while this map's finest octave sits at
the resolution limit by construction, `DETAIL_SIGMA` being 1.5. Any rotation,
non-unit scale or sub-pixel offset attenuates exactly the content the layer
exists to add, and `amount` was spent as though it had not.

Measured: the map retained **0.431** of its deviation, so the layer delivered
**41.4%** of a budget `_match_detail` had already stood down to make room for —
identically at strength 0.5 and 1.0, which is what ruled out an amplitude-scaling
explanation and pointed at a constant geometric factor.

`texture_coverage` was added to separate the two possible causes, since a field
that is unit-deviation over a fraction `c` of the region it is measured across
reads `sqrt(c)`. **Coverage came back 0.920**, so area explained nothing —
`sqrt(0.920)` predicts a 96% spend against the 41% measured.

Fixed: `_add_texture` measures what survived the warp inside its own support and
scales by it, capped at `_WARP_GAIN_MAX` (4.0). Past that cap the map has been
destroyed rather than attenuated and multiplying up what is left would amplify
interpolation artefacts, so the shortfall stays visible in `texture_delivered`
rather than being silently corrected.

**Verified: delivery went 41.4% → 91.8%**, the remaining 8% being the feathered
alpha.

### And the face still got worse

Because the problem was never *how much* the layer added. It was *what* it added.
Delivering more of a bad map is not an improvement.

---

## 3. The remaining defect: the map carries creases, not skin

Energy added **coarser than sigma 3.0** — which a high-pass capped at sigma 3.0
should not produce at all:

| | band 1.0 | band 2.0, relief 0.0 | band 2.0, relief 0.65 |
|---|---|---|---|
| coarse energy added | 1.253 | 1.907 | **3.160** ← **shipped default** |

**The shipped default is the worst of the settings tested.** The mark octave is
not carrying marks — it is carrying the donor's expression lines and shading.

This is the exclusion the design already names and cannot enforce: *"anything
expression-dependent — a smiling source's crow's foot is painted on regardless.
No band setting fixes this."* It is now measured rather than predicted.

---

## 4. The leading suspect: the donor

`select_texture_source` chose **face-19.jpeg — a 369px face, upsampled,
28 degrees off axis**, weighting sharpness 0.40 against frontality 0.20.

Three things wrong with that for this purpose:

- **369px against `_SIZE_FULL` 400px**, and *upsampled* into the 512 canonical
  crop — the tool says so itself: "the band is thinner than it looks". Its fine
  octave is interpolated rather than photographed.
- **28° off axis.** Its creases and shading are foreshortened and land wrong on
  a differently-posed face.
- It is the identical mis-weighting already corrected for the shape reference in
  `select_shape_source`. Texture legitimately wants sharpness; it does not follow
  that it wants pose ignored.

**`_pose_confidence` does not catch this.** It measures source-to-**target** pose
*agreement*, so two faces angled the same way score 1.000 while the map is still
foreshortened. Agreement is not frontality. It read 1.000 throughout.

---

## 5. What the readings mean

Gated on `identity_probe=N`, in the `REALISM` block and
`tools/identity_probe.py`.

| Reading | Meaning |
|---|---|
| `texture_headroom` | The budget: the real face's high-frequency energy, less what the swap and grain already carry |
| `texture_delivered` | The spend: what the layer actually put on the face, same units. **Should now land near the budget** |
| `texture_coverage` | The share of the compositing alpha the map's support covers. Separates "attenuated" from "arriving over less of the face than the reserve assumed" |
| `detail_reserve` | How much `_match_detail` stood down to make room |
| `detail_ratio` | The correction `_match_detail` wanted before its clamp |
| `texture_confidence` | Pose agreement — **not** frontality, see §4 |

The report says outright when the spend is under 60% of the budget, including
that raising `texture_strength` will not help because it scales both sides.

---

## 6. Known limitations and traps

- **A reservation that is not filled leaves the face softer than with the layer
  off.** `_match_detail` stands down by `detail_reserve` before the fill is
  attempted, so a layer that then declines or under-delivers is worse than one
  that never ran. This has now recurred twice by different routes.
- **The reserve is uniform, the fill is skin-only.** `_match_detail` stands down
  across the whole face while the map excludes eyes, nostrils and mouth by
  construction, so those regions lose detail with nothing replacing it. **The
  warp gain does not touch this**; it is a separate defect and `texture_coverage`
  is retained to keep it visible.
- **The warp is a similarity — 4 DOF.** A donor proportioned differently from
  the target lands misaligned, and the baked-in skin mask travels off its
  features with it. See GEOMETRY.md §7.
- **`texture_band` 1.0 makes `texture_relief` and `texture_contrast` inert** —
  there is no mark octave for them to act on. Reaching for band 1.0 to escape the
  creases is a retreat, not a fix.
- **`texture_strength` reaches 2.0** over the API as a deliberate overshoot past
  measured parity, to separate "the map is weak" from "the budget is small". It
  is a diagnostic, not a candidate value.
- **The headroom measured 5.3–6.5 here** against the **0.78** recorded on the
  earlier live clip. Whatever that clip was doing, the budget is not the
  constraint on these stills.

---

## 7. Where to restart

In this order, because the first is free and the second decides whether the rest
is worth doing.

1. **Try a better donor.** Large, frontal, sharp, neutral expression, **not
   upsampled** — so a face comfortably over 400px in the original photograph.
   Read the `texture source:` line to confirm which was picked and that it does
   not say "upsampled". If the creases vanish, the picker is the fix.
2. **If they do not vanish, the extraction band is the fix**, not the donor —
   the mark octave is picking up structure that is not skin, and no weighting of
   a bad map helps.
3. **Only then tune.** `texture_relief` and `texture_contrast` are separately
   switchable precisely so one cannot be blamed for another's artefact.

Judge on frames, not on `id_out`: ArcFace is nearly blind to skin texture, the
same blindness that applies to `complexion_keep`.

**Do not turn the layer on to "see how it looks" without reading
`texture_delivered` against `texture_headroom` first.** A short spend means the
face is softer than with the layer off, and that state is invisible in the
picture until you already believe the layer is working.
