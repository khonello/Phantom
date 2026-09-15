# Resemblance — making the output unmistakably the source

The goal this document exists for: **texture and complexion transferred from
the source, target feature leakage driven down, until the output is a striking,
unmistakable resemblance to the source.** Complexion means *all visible skin* —
neck, ears, chest, hands — not the face alone.

Status: **the core (§3) is built. Route A is FINAL, on by default — third footage run 2026-09-15: "done and fully acceptable"** on the hardest pairing, one limitation accepted and stated under Route A. **Live work first, by decision (2026-09-15, §8): Route B and E wait until the live list is clear.** Route B's measurement is scripted and ready; C–E planned.
This is the implementation approach agreed 2026-09-13, written before the
first line of code so the routes are judged against what they were meant to
deliver rather than what they happened to do.
The research behind it is summarised in §1; GEOMETRY.md and TEXTURE.md carry
what was already measured on the two axes this work extends.

---

## 1. The framing, and why it inverts the literature

**This goal is the inverse of what the field optimises.** Face-swap papers
define success as *inject the source identity, strictly preserve the target's
attributes*, and the "leakage" they fight is the **source's** skin tone, hair
and head shape bleeding into the target. The 2026 CASIA benchmark's
cross-ethnicity protocol scores a method *down* for carrying the source's
complexion; CrossSwap criticises E4S because its re-colouring "fuses source
face skin tone into target face attributes" and inflates its identity score.

For this project that is not a flaw, it is the objective. So: **the methods the
field calls leaky are the candidates, and the methods it calls clean are the
wrong tools.** Three families are on our side by construction — identity-first
swappers, head swaps, and reenactment.

### Why the LIVE tier cannot deliver this on its own

Every model in `swapper_models.py` is conditioned on an **ArcFace vector**, and
ArcFace is a recognition network trained to be invariant to exactly what we
want carried: lighting, skin tone, skin texture, most head geometry. The only
appearance information the generator has is **the target crop**. Pores,
complexion and jawline come from the target not because the model is weak but
because nothing else was given to it. The repo's own measurements agree:

| Leak channel | Measured cost | Where |
|---|---|---|
| **Mask** — a hull of the *target's* landmarks | `id_out` −0.101 to −0.161, the largest | GEOMETRY.md §4 |
| **Colour match** — moves the face onto the *target's* complexion | by design; `complexion_keep` buys a bounded part back | CLAUDE.md |
| **Generator** — no LIVE model moves the contour | `outline_swap` +0.001, alphaface *and* hififace | GEOMETRY.md |
| **Texture** — the swap carries 58% of the frame's high-frequency energy | restoration off moves it 0.03 | TEXTURE.md |
| Restoration | +0.001 — free | GEOMETRY.md |

The field confirms the diagnosis from the other side: the strongest 2026
result (CVPR 2026, ID-constrained conditioning, 97.9% ID retrieval on FFHQ)
gets there by adding **DINOv2 image features** beside ArcFace "for spatial
details like skin texture". An image-feature channel is what carries texture.
Switching between ArcFace-conditioned ONNX models is therefore **not a route**:
measured twice here, and the literature's ladder among them is a few points
wide while the mask alone costs 0.10.

---

## 2. The shape of the work: core on main, routes behind flags

**The routes are not peers, and a branch per route would fight the way this
repository tests things.** Its A/B method is one tree, one pod, one clip, flags
flipped live through `set_realism` with no restart — and the routes have to be
judged **in combination**: texture measured on a face that is still the
target's colour does not read as the source, so B measured without A measures
the wrong thing. Long-lived branches also drift from the measurement tooling,
and a sweep has already been lost here to a pod that predated the fields it was
asked to set.

So:

1. **Main carries the core** (§3) — the instrument and the shared services —
   *before* any route starts.
2. **Routes A, B and E are stages**, built on short-lived feature branches and
   merged to main **behind their own flag** as soon as they are testable. Every
   merge leaves main's defaults unchanged.
3. **Routes C and D get long-lived branches**, because they are not stages: C
   changes what the live pipeline *is*, D is training code whose output main
   already knows how to consume.
4. **Testing is a flag matrix on one tree.** A alone, B alone, A+B, A+texture,
   … one session, one clip, read on the readings from §3.

Every route is **independently switchable**. The switch is an environment
key, blank for off, following the one rule for every environment file: added
to `.env.example` and every local `.env`, same section, same order.

---

## 3. The core — on main, first

Everything below is shared by two or more routes and is the same regardless of
which route wins. Nothing here changes how the output looks.

### 3.1 A complexion reading

ArcFace is nearly blind to skin tone, so today nothing can say whether a
complexion transferred. Add to `Readings`, the REALISM block and
`tools/identity_probe.py`:

| Reading | Meaning |
|---|---|
| `complexion_gap` | **Built.** Source skin against target skin — a property of the *pairing*, no setting moves it, and the denominator for the rest. Under ~3 units the two already agree |
| `complexion_face` | **Built.** Chroma distance between the output's face skin and the **source's** — the number to drive toward zero |
| `complexion_target` | **Built.** The same against the **target's** original face skin — the leakage direction; rising is the swap taking |
| `complexion_lum` | **Built.** Output lightness less the source's, signed. Apart from the chroma distances because it is mostly lighting, and lighting is the target's to keep |
| `complexion_neck` | **Built.** The target's neck / visible body skin against the source — whether the face and the rest of the person agree. Absent when no body skin is visible |
| `complexion_seam` | **Built.** The output's face skin against its own neck skin. **This is the seam a face-only transfer creates**, and the number Route A exists to hold at zero. The REALISM verdict calls it out past 4 units |

`pipeline/services/complexion.py`. The three distances are **chroma only** —
the a/b plane of OpenCV's 8-bit LAB, the units `_COMPLEXION_RESIDUAL` and
`complexion_kept` already use — because pigment lives in chroma and luminance
is shading, which a correct swap takes from the target. Skin is the texture
layer's own `skin_mask` (landmark hull minus eyes, nostrils, mouth), so the two
agree on what skin is. Medians, not means: a freckle field and a specular
highlight are not complexion. The source reference is the **median across
every accepted photograph**, with `spread` reporting how much their white
balance disagreed — the honest error bar on `complexion_face`. Measured on the
`identity_probe=N` interval, after the paste, in `FaceCompositor._measure_complexion`;
reported in the REALISM block with a verdict, and per row in
`tools/identity_probe.py` with a `closed` column. `tests/test_complexion.py`.

**Read `id_out` with a new caveat.** ArcFace does carry *some* skin tone —
that is CrossSwap's whole complaint about E4S — so `id_out` will rise when
complexion transfers, for the right reason here, but it can no longer separate
complexion from likeness. Read it beside `complexion_*`, never instead.

### 3.2 A skin segmentation service

`pipeline/services/skin.py`. Face skin *and* body skin, per frame, real-time.
Used by Route A for the neck and hands, by Route C for the head-onto-body
composite, and available to the texture and reshape layers for exclusions.

**Built 2026-09-13: the registry and the `seeded` backend.** `SkinSegmenter`
returns `face` (the texture layer's `skin_mask` warped into frame space — one
definition of skin for every stage) and `body` (classified skin outside the
grown face hull). Selected by `skin_model` / `SKIN_MODEL=`, blank is
`seeded`; forwarded, `set_realism`-reachable, reported by `tools/stats.py`.
Attached to the compositor by the pipeline, reset with its temporal state,
and run only on the probe interval until a stage asks for it every frame.
**~10ms at 640×360 on a laptop CPU; the pod prints its own.**

The seeded model fits per frame from the face's own skin — median and MAD in
the a/b plane, a wide lightness gate — and smooths its **parameters**, not its
matte, so it cannot crawl. Measured on a synthetic scene: neck in shadow 0.77,
hand 0.98, blue wall 0.00, dark shirt 0.00, face-in-body 0.00, a dark
complexion the same. **And a skin-coloured wall patch 1.00** — the stated
blind spot, pinned by `tests/test_skin.py`, and it is worse than a false
positive: with such a wall in shot the body median *is* the wall, so
`complexion_neck` reads the wall's colour (the test pins that too). That is
the argument for the parsing backend, for the reading as well as for Route A.

`mediapipe_multiclass` is registered as a spec and **not built** — its
pre/post-processing is unverified, and `resolve` lands an unbuilt name on the
default rather than trusting a matte from it.

Two candidates, to be measured against each other:

- **MediaPipe multiclass selfie segmentation** — 256 or 512 input, classes
  include face-skin and body-skin, ONNX available, real-time on device. A
  registry entry in the `backgrounds.py` shape: the model owns its input edge,
  normalisation and channel order.
- **A seeded colour model** — CIELab or YCbCr thresholds fitted per frame from
  the pixels inside the face hull, which are known to be this person's skin
  under this light. Measured at 90 fps on a CPU in the literature. No model
  file, and it adapts to the lighting for free; it also fires on skin-coloured
  backgrounds, which the first does not.

Either way: feathered, EMA-smoothed across frames (a matte that crawls on the
neck is failure mode 3 on the longest boundary in the picture), lips, eyes and
hair excluded.

### 3.3 A cross-tone fixture

One source/target pair chosen for **maximum** complexion and head-shape
difference, tracked with the other fixtures, so every route is judged on the
hard case first. The CASIA benchmark's cross-ethnicity protocol is the right
*shape* of test; this is our single-pair version of it.

**Pinned 2026-09-13, in both directions** — docs/REALISM_TESTING.md Pass C.
The tracked fixtures hold three people and every earlier measurement paired
the two closest complexions. **C1 fair → dark:** `source/two/*` onto
`target/face-3.jpeg` (still) and `target/face-6.mp4` (stream). **C2 dark →
fair:** `target/face-1..5.jpeg` onto `source/two/IMG_3623.jpg`, the still
every earlier `id_*` and `shape_*` number was taken on. Chosen by eye —
no detector runs on the development machine — so `complexion_gap` and
`shape_mismatch` for both are **unmeasured**; the first probe run prints
them, and they belong here when they exist. `tests/test_wiring.py` pins that
the files are tracked.

### 3.4 Keys, forwarding and tests

Every flag in §4 declared in `FaceSwapConfig`, read by `core.py`, forwarded by
the orchestrator, documented in `.env.example`, settable through `set_realism`
and reported by `tools/stats.py`. `tests/test_wiring.py` extended so a key that
exists in the code and not in the env file fails, which is the gap
`DIFFUSE_STRENGTH` fell through.

**Built 2026-09-13.** The check found **six more** of the same kind the day it
was written — `TEXTURE_STRENGTH/BAND/RELIEF/CONTRAST`, `MASK_FEATHER`,
`MASK_ERODE` — read by `core.py`, forwarded, and a key in neither env file.
All six now exist blank in both. A second check catches the reverse: a name in
`_FORWARDED_ENV` that nothing in `pipeline/` reads. And the two env files were
brought back into the same key order, which they had drifted out of by one
block. Route keys are added as each route lands.

### 3.5 Core deliverables — the checklist for main

Done means merged to main, tested, defaults unchanged, and the pod prints it.

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 1 | `complexion_gap/face/target/lum/neck/seam` | `pipeline/services/complexion.py`, `FaceCompositor._measure_complexion` | **Done 2026-09-13.** In the REALISM block on stream stop **and** batch finish, on the `identity_probe=N` interval; `tools/identity_probe.py` prints them per configuration with `closed`, `neck`, `seam` columns |
| 2 | `SkinSegmenter` service with a model registry | `pipeline/services/skin.py` | **Seeded backend done 2026-09-13**; `mediapipe_multiclass` registered, not built. Selected by `SKIN_MODEL=`; feathered, parameter-smoothed; eyes, nostrils, mouth excluded. Still to do: the parsing backend, and `skin` as a line in the latency budget once a stage runs it every frame |
| 3 | Cross-tone fixture pair | `source/two/*` ↔ `target/*`, both directions | **Done 2026-09-13**, provisional: chosen by eye, `complexion_gap` and `shape_mismatch` unmeasured until the first probe run. docs/REALISM_TESTING.md Pass C |
| 4 | Keys and wiring | `pipeline/config.py`, `pipeline/core.py`, `vast/orchestrator.py::_FORWARDED_ENV`, `.env.example`, every local `.env`, `pipeline/api/handlers.py` (`set_realism` clamps), `tools/stats.py` | **Done 2026-09-13.** `tests/test_wiring.py` fails on any key `core.py` reads that `.env.example` does not name, and on any forwarded key nothing reads; six missing keys found and added. Route keys land with their routes |
| 5 | The skin mask reaches the layers that want it | `FaceCompositor._add_texture`, `reshape.py`, `_match_color` | each takes the mask when present and behaves bit-identically without it — a capability gap must not become a behaviour change |
| 6 | Documentation | this file, CLAUDE.md pointer, `docs/PENDING_WORK.md` | the runbook's next pod session lists the flag matrix from §2 with the commands |

Nothing in this table changes how a frame looks. That is the test of whether
something belongs in the core: if switching it on alters the output, it is a
route, and it goes behind a flag.

---

## 4. The routes

Each route states what it delivers, what it costs, what it depends on, and its
switch. **Dependencies are on the core and on each other; they are not
optional.** A route measured without the routes it depends on gives a number
that means something else.

### Route A — Complexion across all visible skin, upstream of the swap

**Delivers:** the source's complexion on the face *and* on every other visible
skin pixel — neck, ears, chest, hands — with no colour step at the jaw.

**Under test on footage since 2026-09-14.** Switched on live against the UK
A100 pod (`skin_complexion=1.0` through `tools/realism.py`, mirrored as
`SKIN_COMPLEXION=1.0` in the operator's `.env`), together with the four levers
turned on the day before — `identity_push` 0.25, `complexion_keep` 0.4,
`diffuse_strength` 0.3, `shape_warp` 0.3 — and `identity_probe=5`. The
baseline was being driven from the new dropdown during the session (`mst03`,
then `mst02`). **All five are on together**, so the first readings are one
combined result; attribution needs them flipped one at a time mid-stream,
and Route A is the one to isolate first because it is the only one that
touches skin outside the face. The config default stays `0.0` until the
footage verdict is in; "on" here means the operator's `.env` and the live
pipeline, which is the same mechanism the four levers use. What to read
first: the neck and wrists by eye, then `complexion_seam` (hold under 4),
`complexion_shift` (28 means the cap is binding), and `skin_grade` in the
per-stage latency report for the pod's own cost.

**First footage verdict, 2026-09-14 — positive on the face, two defects found
and fixed the same hour.** UK A100 pod, alphaface_256, 92px face at 640×360,
poor light (a phone torch on the face). The operator's words: *"on the face,
quite good when on auto"*; on any tone step *"the colour change becomes
inconsistent, sort of pulsing"*; and *"below the face, neck going towards
shoulder downwards, not very good."* The REALISM block from the stream stop
put numbers on all three:

| reading | p50 | p95 | what it said |
|---|---|---|---|
| `complexion_gap` | 9.5 | **26.4** | the *target's* measured tone jumped frame to frame under the torch and exposure hunting |
| `complexion_shift` | 6.2 | **19.1** | the grade chased it — that swing IS the pulsing |
| `complexion_gain` | 1.37 | **2.00** | the lightness gain sat at its cap on the far frames |
| `complexion_lum` | −40 | | the face is 40 L units darker than the photographs — lighting, not complexion |
| `complexion_face` / `_target` | 5.7 / 5.0 | | 40% of the way to the source |
| `complexion_seam` | **6.3** | 11.7 | the face and neck disagree — the neck was only partly graded |
| `id_swap` / `id_out` / `id_target` | 0.845 / 0.703 / **0.052** | | leakage very low; the mask still the largest identity loss (−0.121) |
| latency `total` | 96ms | 127ms | **MISSES** a 67ms deadline; `mask` 17ms and `reshape` 14ms pre-existing, and `skin_grade` was **invisible** — `_composite_impl` cleared it |

Two mechanisms, both confirmed by those numbers and both fixed:

1. **A swatch's L is not a scene quantity.** Under a tone step the reference L
   was the Monk swatch (235 for `mst02`) and the stage tried to grade a dim
   webcam face up to it — gain at the 2.0 cap, every exposure wobble doubled.
   Under `auto` the reference L was a real photograph's, so the gain was 1.3
   and the same wobble was a third the size. Now a baseline supplies **chroma
   only**; lightness comes from the photographs as under `auto`. And the gain
   is bounded to **[0.80, 1.25]**: a real complexion difference under the same
   light is a modest ratio, and anything past it is the room, which is the
   target's to keep. The parameter EMA went 0.35 → **0.08** (about a second at
   15fps) so exposure hunting averages out instead of being followed, and the
   stage now reads the target's tone from the segmenter's own smoothed sample
   rather than measuring it a second time.
2. **The neck was under the lightness gate.** Body skin counted only between
   0.45× and 1.45× of the face's median L; a torch-lit face with the neck in
   room light put the neck below that. Gate widened to **[0.30, 1.60]**,
   chroma tolerance 3.0 → 3.5 MADs. And a new reading, **`complexion_coverage`**
   (body skin found, as a share of the face's area), separates "the neck was
   never found" from "the grade did not land there" — the question the
   operator asked and nothing could answer.

`skin_grade` now survives into the latency report. Its cost is real and was a
quarter of the frame; **not optimised yet, on purpose** — the operator's
instruction is to get it working before making it cheap.

**Second footage verdict, 2026-09-14, after those fixes.** Same pod, same
poor light and camera — the operator asked that no abrupt decision be made on
it. **Fair source onto a dark-complexioned target: the C1 direction, the
hardest pairing in §3.3.** In the operator's words: *"face is phenomenal"*,
*"face skin is match closer to source than the rest of the body"*, *"better
than what we had before today's work"*, and the remaining defect is the
*"shoulder area"* — the neck toward the shoulders reads less like the source
than the face does, most visibly when the torch moves from the face to below
it. **Go for the next route** was the call, with this open.

The likely cause of the shoulder step is the lightness bound set that
morning, not the segmenter. `[0.80, 1.25]` was right for the run it came from
(the face 40 L under the *photographs* because of the torch), but on a
fair-on-dark pairing a large L difference IS the complexion, and the cap holds
the neck at most 25% lighter while the swapped face — generated fair, then
colour-matched to graded skin — lands closer to the source. The next REALISM
block separates the two without guessing: `complexion_gain` p50 at 1.25 with
`complexion_lum` still large-negative is the cap binding; low
`complexion_coverage` is the neck not being found. **Not retuned blind.** The
candidate fix, if it is the cap: the baseline dropdown as *permission* — a
source tone step several classes from the target's measured tone licenses a
larger gain than `auto` should ever take, which is what the operator proposed
the dropdown for in the first place.

**CLOSED, end of 2026-09-14 — on by default (`skin_complexion` 1.0), one
limitation accepted.** The operator asked for the close and the record
supports it: every defect footage produced has a measured cause and a
shipped fix, and the face verdict on the hardest pairing in §3.3 is
*"phenomenal"*, *"closer to the source than the rest of the body"*, *"better
than what we had before today's work"*.

**The accepted limitation — the neck and shoulders on a fair-on-dark
pairing.** MY TONE was tried at the darkest step against the source's fair
one; in the operator's words, the result *"still looks quite different from
face, complexion is not very close … neck and shoulder did not meet face."*
Two things about that, both stated rather than argued away:

- It could not be confirmed that the pod carried `997ee11` (the declared-tone
  gain) during the try — it was pushed after the second footage run, reaches
  the pipeline only on a `resume`, and without it `set_realism` rejects the
  field while the dropdown appears to work. The pod was terminated that night,
  so this is unverifiable now. A fresh pod carries everything.
- **But the limit is real either way.** Carrying dark skin 30+ L units up to
  meet a face the swapper generated fair is at the edge of what a lightness
  gain can do believably: it multiplies the camera's noise on that skin with
  it, and lifting L does not supply the translucency fair skin has. The
  operator's own read is the right one — *"if the person testing is fair,
  things will be so much better"* — a pairing without that gap has nothing
  here to fail. Recorded in docs/ACCEPTED_RISKS.md terms: known, bounded, and
  the remedy for the hardest pairing is a route that generates the neck rather
  than grades it (C, head composite; or the STUDIO head swaps in E).

**Reopened the same night at the operator's request, to close it well.** Two
additions, both aimed at the neck and both pinned on a torch-lit synthetic
scene (face under one light, body at a fifth of its lightness in a warmer
hue — the footage case):

- **A second seed** (`skin.py`). The corridor directly under the face hull is
  sampled, admitted by a loose test against the face model, and fitted as its
  own colour model; a pixel is skin if *either* model says so. Without it the
  body in that scene is invisible (0.00 coverage); with it, neck 0.77 and
  chest 1.00. `last_neck_seeded` says whether it engaged.
- **A harmoniser** (`skin_harmonise`, default **0.7**, `SKIN_HARMONISE=`).
  After the grade, the body is corrected toward the *graded face* — chroma
  fully, lightness toward a plausible neck-to-face floor of 0.82, gain capped
  at 2×. This makes `complexion_seam` go to zero *by construction*, whatever
  left the two apart. On the torch scene: face–chest 3.2 → **1.0** units at
  0.7, 0.0 at 1.0; lightness ratio 0.34 → 0.62. A body already in the face's
  light is barely touched. `complexion_harmonise` reports what it corrected.

**Judged 2026-09-15, fresh pod, third footage run — Route A is FINAL.** Same
fair-source / dark-target pairing, same poor light and camera, second seed
and harmoniser on at their defaults. The operator's words: *"perfect, not
100% perfect but as I said the odds are against us, lighting and camera
quality, so I say Route A is done and fully acceptable — and besides, in
actual usage, source and target skin tone will be quite close."* That last
clause is the product truth this whole route was tested against its opposite
of: the test pairing was chosen as the hardest corner of §3.3, and real
pairings sit near the easy end, where the grade is mostly chroma, the lift
is small, and the two renderings of skin are already close. No further work
on Route A's code unless footage from an ordinary pairing asks for it.

**And the question the operator asked next — strong, even light.** Under
blasting light face and body stay consistent *by construction*: one chroma
offset and one gain, applied identically to every skin pixel, cannot change
the relationship between two regions, and the harmoniser does nothing when
the body is already in the face's light (pinned: neck/face L 0.85 → 0.85,
correction ≈ 0). What could go wrong was **clipping**: a declared class ratio
of 2.5 on a face the camera already renders at L 200 flattens the whole
person to white — consistent, and ruined. Fixed with a **headroom guard**
(`_L_HEADROOM` 232): neither gain may carry the face's or body's median past
it, so on a bright frame a large licensed ratio quietly becomes what the
frame can hold (pinned: gain 2.5 → 1.16, face median 231, <5% of pixels at
white). The one residual that stays is the colour space's: near white the
sRGB gamut cannot hold skin chroma, and a brighter face loses ~2 units the
neck keeps. No correction can put back a colour that does not exist at that
brightness; that is the ceiling under blasting light, and it is small.

**What closes with it:** `skin_complexion` default 0 → **1.0**;
`complexion_base` and `complexion_target_base` stay `auto`. `SKIN_COMPLEXION=`
blank now means on; `0` turns it off. The texture donor picker remains
unjudged and moves to the texture un-park (§8 step 4), which the operator's
slider observation has just made the next thing worth footage.

**And a texture verdict that reverses the last one.** The layer's only footage
verdict had been 2026-09-10's *net-negative* — smoother, creases painted on,
parked at 0. On this run, with Route A underneath it, the operator raised the
TUNING slider and reported: *"when I increase the texture slider, I notice
that the texture becomes well good, see some displacement and things that
make the whole thing realistic."* The REALISM blocks agree the layer was
running (`detail_reserve` 0.28–0.35, delivered 62–77% of budget). This is the
sequencing argument of §2 showing up in footage — texture over correct
complexion reads as skin, texture over the target's colour did not. **Caveat
attached:** whether the pod carried the new donor picker (`a6d2c82`) or the
old one during that run is unknown from here; the `Texture source:` line in
the pod log names the photograph and settles it. Either way, `texture_strength`
is now a candidate to un-park rather than a parked layer, and Route B is the
principled version of what it does by hand.

**The block from that run settled it, and the fix landed the same day.**
`complexion_gain` p50 = p95 = max = **1.250** — pinned on every frame;
`complexion_lum` −36; **`complexion_coverage` 4.63**, so the neck and
shoulders *were* found (4.6× the face's area of body skin); `complexion_face`
4.4, 54% closed; `skin_grade` 9.8ms, visible for the first time. The cap, not
the segmenter.

The fix is the symmetric half of the operator's dropdown: **declare the
target's tone too.** `complexion_target_base` / `COMPLEXION_TARGET_BASE=` /
a MY TONE dropdown under COMPLEXION. With both tones declared, the ratio of
the two Monk swatches' L — two paint chips under one canonical light — is
complexion with the room cancelled out, and `lightness_ratio` licenses a gain
as far as the classes are apart, bounded at `CLASS_GAIN_MAX` = 2.5. With
either side `auto`, the narrow [0.80, 1.25] band stays, because
photographs-against-frame is two rooms and cannot be told from lighting. On
the fair-on-dark synthetic scene the neck goes L 63 → 157 declared against 79
under `auto`, shading ratio intact. The REALISM block now says so itself when
the gain is pinned. **Unjudged on footage**, and the thing to watch when it is:
a 2.5× gain on dark skin multiplies the camera's noise there by 2.5 as well —
grain and JPEG blocking on the neck are the expected cost, and whether they
read as skin is the question. Note also that the
texture layer was **on** during this run (`detail_reserve` 0.28 ⇒ the TUNING
slider at ~0.35) and delivered 77% of its budget — the parked layer, running
on footage for the first time.

**Built 2026-09-13** — `pipeline/processing/complexion_stage.py`, called from
`ProcessingPipeline._swap_face` through `FaceCompositor.grade_skin` before
the swap, on both the live and batch paths, and mirrored in
`tools/identity_probe.py` so a sweep measures it. Off by default
(`SKIN_COMPLEXION=`), **never judged on footage.** On the synthetic scene
(`tests/test_complexion_stage.py`, 44 checks): at strength 1 the face and
the neck both land within 0.0 units of the reference chroma with a 0.0 seam,
the neck's shading ratio to the forehead survives (0.747 → 0.750), the wall
and the shirt are byte-identical, half strength closes half the gap, the cap
binds at 28 units and reports it, and the readings measure `complexion_gap`
against the *ungraded* frame (14.1 = the real gap) while `complexion_face`
reads 0.0. ~30ms at 640×360 on a laptop CPU, dominated by the segmenter
(~10ms) and two colour round trips; the pod prints its own, and it is
**not yet in the latency budget's own verdict** — read `skin_grade` in
`last_stage_ms`.

What it deliberately does when the pairing is far apart: bounds the chroma
shift at `_MAX_SHIFT` (28) and the gain at [0.5, 2.0] and says so, rather
than producing an implausible colour. A cross-ethnicity pairing will hit the
cap; that is the reading to look at first on the C1/C2 fixtures.

**Why upstream, and why all the skin.** A face-only complexion transfer is
self-defeating: the jaw becomes a colour step, which is the exact seam
`_match_color` exists to prevent and the reason it matches the face *to* the
target. The way out is to change what it matches to. Recolour every visible
skin pixel of the **target frame** toward the source's complexion *before* the
swap; the existing colour match then pulls the swapped face toward the
now-source-toned surroundings. The seam logic is untouched, the face and the
neck agree because one stage graded both, and `complexion_keep` becomes
unnecessary.

**The baseline comes first, and it is a dropdown (proposed 2026-09-13).** The
measured reference has one weakness the docs already name: uploaded
photographs carry their own white balance, and the median across them is only
as good as the set. So Route A grades toward an anchor built in two parts:

    dropdown  ->  baseline L, a, b and a bound      coarse, stable, operator-chosen
    photos    ->  undertone offset within the bound  fine, measured, `spread` as its error bar
    disagreement past the bound  ->  tell the operator, trust the dropdown

- **`complexion_base`** on `FaceSwapConfig`: `auto` | `mst01` … `mst10`.
  `COMPLEXION_BASE=` in every env file, blank is `auto`; reachable through
  `set_realism`; a dropdown in the sidebar under QUALITY beside the restoration
  preset, and justified the way that one is — a named step an operator
  understands, and a support question has an answer.
- **`auto` is the default and means "measured from the photographs"** — the
  behaviour that exists today. The same pattern as the restoration preset: a
  named scale with an `auto` that lets the pipeline decide, and no silent
  override of a choice the operator made.
- **A prior, not a destination.** Undertone — warm, cool, olive, neutral —
  lives *inside* a scale step, and snapping to the step would erase the thing
  that makes a complexion someone's. The photographs supply it, bounded.
- **Monk Skin Tone, ten steps, not Fitzpatrick.** MST was designed for camera
  and image work, resolves dark tones where Fitzpatrick collapses above type
  IV, and publishes reference sRGB values that convert to the LAB units this
  pipeline already uses.
- **It supplies luminance as well as chroma.** A scale step is an L anchor with
  a known relationship to a/b; a photo median's L is mostly that photograph's
  lighting. This is what lets the large-gap case below move L honestly.
- **It validates the photographs.** Pick tone 4, measure tone 7, and one of
  them is wrong — the operator is told before the session rather than on a
  call. That guard cannot exist without a second, independent reference.
- **The reading reports both.** `complexion_face` is measured against the
  effective reference (baseline plus undertone) *and* against the raw photo
  median, so a disagreement is a number rather than a surprise.

It does nothing until Route A exists: it is the anchor Route A grades toward,
not a stage of its own.

**Built 2026-09-13**, pipeline side: `complexion_base` / `COMPLEXION_BASE=`,
`complexion.MST_SRGB` (the ten published swatches, converted once into the
8-bit LAB used everywhere here), `resolve_reference` (undertone from the
photographs inside `UNDERTONE_BOUND` = 6 a/b units, the baseline's L),
and a warning, once per source, when the photographs and the baseline
disagree past the bound. Note the published scale is not strictly monotone
in L — step 2 → 3 is a hue step — so it is a *tone* scale, not a brightness
ladder. **The desktop dropdown is built** — COMPLEXION, under RESTORATION in
the sidebar, each step drawn as its published swatch; read back on connect
like the restoration preset, never asserted. Also reachable through
`set_realism` and `tools/realism.py`.

**Design constraints, from the literature and from this codebase's own rules:**

- **Transfer chroma, preserve luminance.** Shading and light direction live
  in L; complexion lives in a/b. E4S's re-colouring network reaches the same
  conclusion from the other side, and CLAUDE.md's rule that luminance is
  corrected in full because a brightness step at the jaw is the most visible
  seam is the same physics.
- **Large gaps need luminance too**, or a cross-ethnicity pairing lands on a
  face that is the wrong brightness in the right hue. Move L
  **multiplicatively in log space**, so shading *ratios* survive; never as an
  offset.
- **Bounded and ramped**, like `_COLOR_FLOOR`: a ΔE cap per frame, a ramp so
  it cannot snap, EMA across frames on the transfer parameters.
- **Exclusions:** lips, eyes, brows, hair, beard. Hair and beard colour are a
  separate identity cue and a separate decision.
- **Hands are the risk.** A cross-tone hand recolour is where the eye catches
  it. Hands are in scope by default so the reading in §3.1 can price them;
  `SKIN_COMPLEXION_HANDS=` turns them off without turning the stage off.
- **Pipeline-side, never desktop-side.** It has to precede `_match_color`,
  and `--debug-frames` has to see it.

**Depends on:** core §3.1 (or it cannot be judged), §3.2 (or it has no neck).

**Switch:** `COMPLEXION_BASE=` (the baseline; blank is `auto`) and
`SKIN_COMPLEXION=` (0–1, fraction of the measured complexion gap
to close; blank is off). `SKIN_COMPLEXION_HANDS=` as above.

**Cost:** one segmentation pass and one LAB transfer per frame. Expected
single-digit milliseconds at 640×360; measured before it ships.

**Sequencing note:** this comes **before un-parking the texture layer**.
Pores on a face that is the target's colour do not buy resemblance.

### Route B — Reference-guided restoration

**Delivers:** a restoration stage that pulls the face toward *this person*
rather than toward a generic face manifold — carrying the source's own skin
texture in the same stage that today is identity-neutral (+0.001).

**What it is.** The 2025 reference-restoration literature — FaceMe (5–20
references), RefSTAR, "Copy or Not?" (WACV 2025), InstantRestore (single
step), IConFace, TimeWeaver — restores a face conditioned on the person's own
photographs. The repo already holds those photographs: `review.accepted`. It
is a drop-in replacement for `Enhancer` conceptually — same FFHQ crop in, same
crop out, one more conditioning input — and attacks texture and identity at
one stage.

**Why it is the right answer to the texture layer's failure.** TEXTURE.md's
remaining defect is that the map "carries creases, not skin" — the unreliable
copy that "Copy or Not?" is built to reject: copy reference detail only where
it is reliable, per region. The stronger version of "try a better donor".

**How it lands.** One more entry in `enhancer_models.py`, which is the registry
that exists so a second model is not a fork. Mostly diffusion, so **RENDER and
photo first**; InstantRestore is the one to time on the 4090 for anything near
the live deadline. The registry's `live_capable` fact decides where it may run.

**Depends on:** core §3.3 (the hard pair) and, to be judged correctly, **Route
A underneath it** — texture on a target-toned face is not the measurement.

**Where it got to, 2026-09-14.** Candidate chosen: **RefSTAR** (AAAI 2026) —
complete CLI, weights public, InsightFace-based like this pipeline, and built
for exactly this shape (select a reference, transfer its detail, reconstruct).
FaceMe (SDXL, 7 GB base) and InstantRestore (README gives only a script path)
were the alternatives. The measurement is scripted end to end:

- `tools/routeb_refstar_setup.sh` — its own venv beside the pipeline's,
  torch pinned to the pipeline's 2.2.0 so the wheel cache serves it, and
  every pin that bit on the way recorded in place: `numpy<2` (torch 2.2's
  ABI), `opencv-python==4.10.0.84` (OpenCV 5 wants numpy 2),
  `huggingface_hub<0.26` (diffusers 0.23 imports `cached_download`),
  `peft==0.10.0` (newer imports `EncoderDecoderCache`), the basicsr
  `functional_tensor` import patched by `sed` after locating the file with
  `find_spec` rather than an import, and the repo's `LandmarksType._2D`
  renamed to `TWO_D`. Runs detached with `setsid` — plain `nohup` died with
  the orchestrator's SSH session, silently.
- `tools/routeb_refstar_run.sh` — one swapped still (restoration **off**,
  since RefSTAR is a candidate to *replace* the restorer) plus one source
  photograph, full-frame mode, timed.
- `tools/restore_probe.py` — scores restored stills against the source:
  ArcFace cosine to the source identity, cosine to the reference photograph
  (a restorer that copies its reference scores perfectly and is a different
  picture), cheek high-frequency deviation, face size.
- Inputs already on the pod: `/workspace/routeb/in/enhance-False.png` (the
  unrestored swap) and `enhance-True.png` (GPEN), from `identity_probe.py`
  on the **C1 pair**. Which also gave §3.3 its first numbers:
  **`shape_mismatch` 0.309**, **`complexion_gap` 6.0 chroma / 31 L** — on
  this pairing the complexion difference is almost entirely lightness.

**Blocked on:** `net_g_latest.pth`, **7 GB**, hosted only on Google Drive.
The first download arrived CRC-corrupt (`zipfile.testzip` fails on its first
entry, while RefSel passes); the re-download hit Drive's per-file quota —
*"may exceed the maximum download quota"* — which resets in about 24 hours.
**The pod was terminated that night** (the operator wanted a clean start), so
the venv, weights and stills are gone with its disk. Nothing else is: the
setup script is idempotent and carries every fix. **The retry on a fresh pod
is the setup script first** (~15–20 min, mostly downloads; `setsid`, not
plain `nohup`), then the stills, then the three commands below. Mind the disk
— the main checkpoint is 7 GB and the setup is ~11 GB in total, and pip's
cache filled a 25 GB overlay once already (`rm -rf /root/.cache/pip
/tmp/pip-*` frees it). Drive's quota is per file, so a new pod may not dodge
it; the fallback is a browser download pushed with `orchestrator.py push`.

```bash
python vast/orchestrator.py push tools/routeb_refstar_setup.sh /workspace/routeb_refstar_setup.sh
python vast/orchestrator.py run "setsid nohup bash /workspace/routeb_refstar_setup.sh > /workspace/refstar-setup.log 2>&1 < /dev/null & disown"
# ... wait for '== done' in /workspace/refstar-setup.log, then the stills:
python vast/orchestrator.py run "mkdir -p /workspace/routeb/in && cd /workspace/Phantom && /workspace/venv/bin/python tools/identity_probe.py -s source/two/IMG_3091.jpg source/two/IMG_3623.jpg source/two/IMG_3674.jpg source/two/IMG_3701.jpg source/two/IMG_3745.png source/two/IMG_3751.png -t target/face-3.jpeg --sweep enhance=false,true --save-frames /workspace/routeb/in --execution-provider cuda"
python vast/orchestrator.py push tools/routeb_refstar_run.sh /workspace/routeb_refstar_run.sh
```

**Then:**

```bash
python vast/orchestrator.py run "cd /workspace/RefSTAR/test/pretrained_models && /workspace/venv-refstar/bin/python -m gdown -O net_g_latest.pth 'https://drive.google.com/uc?id=1kKMO9fSUHf5RbpKFPMaPp3D7gC0n99-m' && /workspace/venv-refstar/bin/python -c \"import zipfile; print(zipfile.ZipFile('net_g_latest.pth').testzip())\""
python vast/orchestrator.py run "bash /workspace/routeb_refstar_run.sh /workspace/routeb/in/enhance-False.png /workspace/Phantom/source/two/IMG_3623.jpg /workspace/routeb/out"
python vast/orchestrator.py run "cd /workspace/Phantom && /workspace/venv/bin/python tools/restore_probe.py -s source/two/IMG_3091.jpg source/two/IMG_3623.jpg source/two/IMG_3674.jpg source/two/IMG_3701.jpg source/two/IMG_3745.png source/two/IMG_3751.png --reference source/two/IMG_3623.jpg --frames /workspace/routeb/in/enhance-True.png /workspace/routeb/out/*from_512.png --execution-provider cuda"
```

A `None` from `testzip` is the go signal. Note the pod disk is the only copy
of all of it: a `terminate` starts this over, a `stop` does not.

**Switch:** `ENHANCER_MODEL=<reference model name>` — the existing key. Its
reference photographs come from the source set with no new configuration.

### Route C — Reenactment instead of swapping

**Delivers:** zero target leakage **by construction**. The output *is* the
source photograph animated by the operator's motion — texture, complexion,
head shape, hairline, all the source's, with no target pixel in the face to
leak. The only known route to "unmistakable" on a live budget: LivePortrait is
open and real-time on a 4090; VOODOO XP is one-shot and built for live
telepresence.

**What it costs, and why it is a different product shape.** The operator's
torso, hands and background are the target's, so the reenacted head has to be
**composited onto the target body** — which makes the neck Route A's problem
and the head boundary the mask problem in a new place. The source's lighting is
baked into the source photograph. Hands over the face are hard, since there is
no target face under them to fall back to. And the whole live path changes:
detection and the swap model are replaced, not augmented.

**How it lands.** A long-lived branch, possibly its own service, gated first on
a **one-day latency prototype**: LivePortrait on the 4090, glass-to-glass,
against the 50ms deadline. If it does not hold the deadline the branch waits
for hardware, not for tuning.

**Depends on:** core §3.2 for the composite, **Route A for the neck** — the
reenacted head carries the source's complexion and the target's neck does not.

**Switch:** `REENACT=` (backend name; blank is off). When set it **replaces**
the swap model on the live path the way `IDENTITY_MODEL` does, and is refused
by `_clear_for_live` until its backend declares `live_capable`.

### Route D — The person's texture inside the generator

**Delivers:** identity-specific skin texture produced by the swap model
itself, rather than transferred onto it afterwards.

**Two forms:**

- **TRAINED tier** — already registered. DeepFaceLab learns the person's skin
  and its `dfl_whole_face` crop moves the silhouette at frame rate. The cost
  is onboarding: 5–15 minutes of varied video, hours of training, per person.
- **ReSwapper fine-tune** — an open reimplementation of inswapper *with
  training code* and 256px weights. The cheap experiment: fine-tune on the
  operator's accepted photographs and measure whether identity-specific
  texture appears in the output, on `detail_ratio` and by eye.

**How it lands.** A long-lived branch or `tools/train/` — training code and
weights, not pipeline code. The *output* is consumed by the TRAINED tier's
scanner, which main already has; nothing on the live path changes.

**Depends on:** core §3.3. Independent of A and B; **combines** with A (a
trained model carries the source's face complexion, and the neck still needs
grading).

**Switch:** `IDENTITY_MODEL=<stem>` — the existing key.

### Route E — Identity-first swappers for stills

**Delivers:** the ceiling. What resemblance looks like when the deadline is
removed, so the live routes have a target to be measured against.

- **E4S** (CVPR 2023, code available) — regional texture codes per facial
  component plus a re-colouring network that keeps source skin under target
  lighting; 256 core with super-resolution to 1024. GAN inversion, not live.
- **REFace** — already registered, 89.1% ID retrieval on FFHQ, swaps the
  **head**, so it is the only thing here that raises the mask ceiling.
- **CrossSwap** (2026) — ArcFace plus 3DMM coefficients for *source shape*,
  which is what hififace claimed and did not do. No code, no speed numbers;
  noted, not planned.

**How it lands.** One more entry each in `studio_swappers.py`, subprocess
isolated by design, RENDER and photo only. Never run against real weights so
far — that run *is* the deliverable.

**Depends on:** core §3.1 and §3.3. Independent of every other route, and
their reference point.

**Switch:** `STUDIO_SWAPPER=<name>` — the existing key.

---

## 5. Dependency map

    core §3.1  complexion reading  ─┬─> A ─┬─> B (judged correctly only over A)
    core §3.2  skin segmentation   ─┤      ├─> C (neck)
    core §3.3  cross-tone fixture  ─┤      └─> texture layer un-parked (TEXTURE.md)
    core §3.4  keys + wiring       ─┘
                                    ├─> D (independent; combines with A)
                                    └─> E (independent; the reference point)

Read it as: **nothing before the core, A before anything that touches skin,
D and E whenever there is a pod hour.**

---

## 6. Switches, in one place

| Route | Key | Off | Lives on |
|---|---|---|---|
| A | `COMPLEXION_BASE=` (baseline, blank is `auto`), `SKIN_COMPLEXION=` (+ `SKIN_COMPLEXION_HANDS=`, `COMPLEXION_TARGET_BASE=`) | **`SKIN_COMPLEXION=0`** (default is on) | **main, CLOSED, on by default since 2026-09-14** |
| B | `ENHANCER_MODEL=` | default model | main, registry entry |
| C | `REENACT=` | blank | long-lived branch |
| D | `IDENTITY_MODEL=` | blank | long-lived branch; output consumed on main |
| E | `STUDIO_SWAPPER=` | blank | main, registry entry |

Every key: blank for off, present in every env file, forwarded to the pod,
reachable through `set_realism` for a running pipeline, reported by
`tools/stats.py`.

---

## 7. How each is judged

- **Complexion:** `complexion_face` toward zero, `complexion_seam` held at
  zero, `complexion_target` rising. §3.1.
- **Texture:** `detail_ratio` and `texture_delivered` as now, on the cheek
  patch method TEXTURE.md uses.
- **Leakage:** `id_target` falling. `id_out` rising — with the §3.1 caveat.
- **Shape:** `shape_mismatch` for the pairing, frames for the result; the
  landmark metric cannot grade appearance (GEOMETRY.md §3).
- **The person looking.** Every paper cited reports human preference beside its
  metrics, and every measurement in this repo so far has been overruled by the
  frames at least once. Two to three minutes of live footage per configuration,
  with `--debug-frames`, is the verdict; the numbers say where to look.

---

## 8. Order of work

**Decision, 2026-09-15: live work first.** The live call is the product, and
the operator asked that everything non-live wait until the live list is
clear — the same priority CLAUDE.md has held from the start (*"development is
focused on the live call path; batch follows"*). Route B and Route E are
RENDER/photo by nature, seconds per image, so they wait. Nothing about them is
lost by waiting: B is scripted end to end and E is registry entries.

**The live list, in order:**

1. ~~**Core**~~ — done.
2. ~~**Route A**~~ — **final**, 2026-09-15: *"done and fully acceptable"* on
   the hardest pairing. Nothing owed.
3. ~~**Un-park texture**~~ — **done, 2026-09-15**: on by default at 0.4,
   judged on two live sessions over Route A (*"works quite very well"*).
   TEXTURE.md's header records the reversal and why.
4. **Live cost.** Deferred by instruction until things worked; they do. The
   frame missed its 67ms deadline by 30–47ms and that is felt as lag on a
   call: `mask` 17–26ms, `skin_grade` 10–16ms, `reshape` 14ms.

   **`mask` — found and fixed, 2026-09-15, 18ms → 3.7ms.** Profiled rather
   than guessed (`tools/mask_profile.py`): the CPU work around XSeg was
   0.5ms; the inference was 18ms on an RTX 5880 Ada that should do it in 3.
   ORT's verbose placement log named the cause — **six of the graph's 362
   nodes were on the CPU behind a session that reported CUDA**: every
   `ConvTranspose`, the decoder's upsampling, because the TensorFlow export
   pads them `[0, 0, 1, 1]` and cuDNN's transposed convolution wants
   symmetric pads. Six GPU↔CPU round trips per frame, invisible to
   `execution.verify`, which checks providers and not nodes.
   `pipeline/services/graph_fixes.py` rewrites each one losslessly — zero
   pads plus a `Slice` that drops exactly the rows the pads would have — once,
   beside the original as `dfl_xseg-cuda.onnx`, and the masker prefers it
   when a GPU provider is asked for. Verified on the pod: **bit-identical
   output** (`max |diff| = 0.000e+00`), **zero nodes on the CPU**, `raw
   session.run` 18.3 → 2.96ms, `FaceMasker.build` 18.7 → 3.69ms.
   `tests/test_graph_fixes.py` pins the rewrite numerically where real
   onnx/onnxruntime exist. Worth carrying as a lesson: **a session's provider
   list says nothing about its nodes; the placement log does.**

   Remaining: `skin_grade` (Route A's segmenter, every frame; the every-2nd-
   frame reuse and half-resolution grading are the levers) and `reshape`
   (`shape_warp` remapping the whole frame). Both next.
5. **Route D** — the ReSwapper fine-tune; live by nature, a branch.
6. **Route C** — the reenactment latency prototype; live by nature, a branch.

**Then the non-live list:**

7. **Route B** — the retry is the setup script plus three commands, recorded
   under Route B. Adopt for RENDER and photo if `restore_probe` says it
   beats GPEN on identity *and* cheek detail — or close it as unnecessary if
   the un-parked texture layer already does the job.
8. **Route E** — E4S and REFace on stills, whenever there is a pod hour.

---

## 9. Sources

High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning
(CVPR 2026, arXiv 2503.22179) · Towards High Fidelity Face Swapping: A
Comprehensive Survey and New Benchmark (2026, arXiv 2605.00883) · BlendFace
(ICCV 2023, arXiv 2307.10854) · E4S: Fine-grained Face Swapping via Regional
GAN Inversion (CVPR 2023, arXiv 2310.15081) · CrossSwap (PLOS One 2026) ·
RobustSwap (arXiv 2303.15768) · APPLE (arXiv 2601.15288) · Preserving Source
Video Realism (arXiv 2512.07951) · HS-Diffusion (arXiv 2212.06458) · Zero-Shot
Head Swapping (arXiv 2503.00861) · FaceMe (arXiv 2501.05177) · RefSTAR (arXiv
2507.10470) · Copy or Not? (WACV 2025) · InstantRestore · Reference-Guided
Identity Preserving Face Restoration (arXiv 2505.21905) · IConFace (arXiv
2605.02814) · Arc2Face (arXiv 2403.11641) · LivePortrait (KwaiVGI) · VOODOO XP
(arXiv 2405.16204) · ReSwapper (somanchiu) · MediaPipe Multiclass Segmentation
model card · Lightweight real-time hand segmentation leveraging MediaPipe
landmark detection (2023).
