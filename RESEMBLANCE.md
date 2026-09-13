# Resemblance — making the output unmistakably the source

The goal this document exists for: **texture and complexion transferred from
the source, target feature leakage driven down, until the output is a striking,
unmistakable resemblance to the source.** Complexion means *all visible skin* —
neck, ears, chest, hands — not the face alone.

Status: **core §3.1 face half built (2026-09-13); everything else planned.**
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
| `complexion_neck` | *Waits on §3.2.* The target's neck / visible body skin against the source — whether the face and the rest of the person agree |
| `complexion_seam` | *Waits on §3.2.* The output's face skin against its own neck skin. **This is the seam a face-only transfer creates**, and the number Route A exists to hold at zero |

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

### 3.4 Keys, forwarding and tests

Every flag in §4 declared in `FaceSwapConfig`, read by `core.py`, forwarded by
the orchestrator, documented in `.env.example`, settable through `set_realism`
and reported by `tools/stats.py`. `tests/test_wiring.py` extended so a key that
exists in the code and not in the env file fails, which is the gap
`DIFFUSE_STRENGTH` fell through.

### 3.5 Core deliverables — the checklist for main

Done means merged to main, tested, defaults unchanged, and the pod prints it.

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 1 | `complexion_gap/face/target/lum` **done**; `complexion_neck`, `complexion_seam` | `pipeline/services/complexion.py`, `FaceCompositor._measure_complexion` (face), `ProcessingPipeline` (neck, since it is outside the crop) | in the REALISM block on stream stop **and** batch finish, on the same `identity_probe=N` interval; `tools/identity_probe.py` prints them per configuration. **Face half landed 2026-09-13**; neck and seam wait on #2 |
| 2 | `SkinSegmenter` service with a model registry | `pipeline/services/skin.py` | face-skin and body-skin masks per frame; both candidates (§3.2) registered, one selected by `SKIN_MODEL=`; feathered, EMA-smoothed; lips, eyes, hair excluded; cost printed in the latency budget as `skin` |
| 3 | Cross-tone fixture pair | `tests/fixtures/` beside the existing source and target fixtures | one source set and one target chosen for maximum complexion **and** `shape_mismatch`; named in `docs/REALISM_TESTING.md` as the first pair every route runs on |
| 4 | Keys and wiring | `pipeline/config.py`, `pipeline/core.py`, `vast/orchestrator.py::_FORWARDED_ENV`, `.env.example`, every local `.env`, `pipeline/api/handlers.py` (`set_realism` clamps), `tools/stats.py` | every key in §6 exists blank in both env files; `tests/test_wiring.py` fails on any key `core.py` reads that `.env.example` does not name |
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
| A | `COMPLEXION_BASE=` (baseline, blank is `auto`), `SKIN_COMPLEXION=` (+ `SKIN_COMPLEXION_HANDS=`) | blank | main, behind the flag |
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

1. **Core** — §3.1 first, because it changes what every later result means;
   then §3.2, §3.3, §3.4.
2. **Route A**, and the flag matrix A on/off on the cross-tone pair.
3. **Route B** — time InstantRestore on the 4090; adopt for RENDER and photo
   regardless of the live verdict.
4. **Un-park texture** (TEXTURE.md §7) over A.
5. **Route D** — the ReSwapper fine-tune, cheapest test of texture in the
   generator.
6. **Route C** — the latency prototype. The strategic answer if it holds.
7. **Route E** — E4S and REFace on stills, whenever there is a pod hour.

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
