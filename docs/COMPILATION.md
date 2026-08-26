# Compilation strategy — Nuitka, Numba, and where neither belongs

A proposal to compile the non-model layers of Phantom: Nuitka AOT for the
Python application, Numba JIT for numerical CPU work, leaving the neural
network to ONNX Runtime and CUDA.

The taxonomy is right. Applied to *this* codebase, the two halves are worth
very different amounts, and they are not independent of each other. This
document records what was accepted, what was rejected, why, and what evidence
would overturn either.

**Status: the inference levers are built; Nuitka and Numba are not.**

`pipeline/services/onnx_session.py` now owns session construction, and fp16,
CUDA graphs and TensorRT are opt-in flags on top of it. All three default off,
so the out-of-the-box path is bit-identical to what it was before they existed.
`LatencyBudget` records a per-stage breakdown, which is what makes any of it
falsifiable.

**None of it has run on a GPU yet.** This machine has no onnxruntime; the gates
here are mypy, flake8 and a stubbed test suite, which check the decision-making
— which weights, which providers, in what order, keyed how — and cannot check
that inference got faster. That number comes from a pod session, and until one
has been run every figure below is a prediction.

---

## The shape as proposed

    ML application
    ├── Python application / orchestration -> Nuitka
    ├── Numerical CPU-heavy functions      -> Numba JIT
    └── Neural-network model               -> ONNX Runtime / CUDA

## The shape as it applies here

    desktop/     ships to customers      -> Nuitka standalone        ACCEPTED
    pipeline/    runs in Docker on a pod -> plain CPython, untouched  REJECTED
    numeric      compositor, filters     -> cv2 first, Numba rarely   NARROW
    models       swapper, CodeFormer,
                 XSeg, buffalo_l         -> ONNX Runtime / CUDA       UNCHANGED

The middle two rows are where the proposal changes shape. The reasoning for
each is below.

---

## If the goal is speed, neither of these is the lever

This needs stating before anything else in this document, because the
compilation proposal is easy to mistake for a performance plan and it is not
one. Nuitka is a **distribution** decision. Numba is a micro-optimisation of a
layer that has already been optimised by hand. Neither moves the number that
decides whether a preset holds its deadline.

The per-frame live path runs **four to six separate ONNX inferences**:
`buffalo_l` detection (detector, the `1k3d68` landmark model, recognition), the
swapper, CodeFormer restoration, and XSeg occlusion masking. Every one of them
is invoked the same way:

```python
session.run(None, {...})     # face_swapping.py:287
self._session.run(None, inputs)[0][0]   # enhancement.py:161
session.run(None, {self._input_name: blob})[0]   # masking.py:256
```

Plain numpy in, plain numpy out, fp32, no binding. That means a host→device
copy in and a device→host copy out for each — several full PCIe round trips per
frame — with a CPU-side compositor between them that forces a synchronisation
each time. **This is where the frame time is, and none of it is Python.**

The levers, ranked by expected return against effort and risk:

| # | Lever | Expected | Risk |
|---|---|---|---|
| 1 | **fp16 for CodeFormer and the swapper** | Often ~1.5–2x on these convnets | **Quality.** This is the design target itself — must be A/B'd on real footage, not stills |
| 2 | **Decimate restoration** — restore every Nth frame, let the existing aligned-pixel EMA carry the gap | Up to half the largest single cost | Restoration flicker; but `temporal_alpha` is already the mechanism that would hide it |
| 3 | **IOBinding + pre-allocated output buffers** | Modest, but free | None. No numerical change at all |
| 4 | **TensorRT execution provider** | 2–3x potential | Engine build on first run is minutes, and cold start is already a tracked cost. Needs an engine cache on the network volume |
| 5 | **GPU-side compositing** (`cv2.cuda`) | Removes the sync points between inferences | Large rewrite of a file that is currently correct and readable |

**Item 2 is declined**, not deferred. It is the cheapest large win on the list
and it is left on the table deliberately: it is the only lever that changes what
the operator sees between one frame and the next, and a restoration cadence that
flickers under motion would damage the exact thing the product is judged on.
Recorded here so it is not rediscovered as an oversight.

Item 3 is the one to do first regardless of anything else: it is the only entry
with no quality risk and no architectural cost — and it turns out to matter less
for its own sake than as the precondition for CUDA graphs, which need stable
device buffers to capture against.

Item 1 is the highest-value and the most dangerous, and it deserves the same
treatment restoration strength got — `enhancer_weight` and `enhance_strength`
were tuned to `0.7` because full restoration reads as AI. An fp16 model that is
2x faster and looks marginally different is not obviously a win in a product
whose stated failure mode is "too clean". Measure it on footage.

Note also what is *already* correct here and should not be undone:
`_aligned_size` scales compositing resolution with the face's actual size in
frame, with hysteresis. That is the right instinct — spend compute in
proportion to what is visible — and it is worth extending to the restoration
cadence rather than replacing.

### "Would a better GPU make this unnecessary?"

For the live path, no — and `runpod/orchestrator.py` already says so. `_GPU_PERF`
ranks **RTX 4090 at 100, above H100/H200 at 95 and A100 at 70**, because (per
commit 45ba27c) the workload is "one stream of small ONNX models, bound by
latency rather than throughput: it rewards clocks and cache, and is indifferent
to whether a card carries 24GB or 192GB, because nothing fills 24."

Auto-discovery already picks the top of that ranking, so there is no upgrade to
buy. "More capable" means VRAM and large-batch throughput; this pipeline runs
batch size 1, serialized, and uses neither.

What no GPU changes at any price:

- The whole compositor — `warpAffine`, `cvtColor`, `GaussianBlur`,
  `meanStdDev`, `Laplacian`, `addWeighted`, `resize`. CPU.
- JPEG encode. CPU.
- The host↔device copies per `session.run`. PCIe — and as compute shrinks they
  become a *larger* fraction, so a faster GPU makes the missing IOBinding cost
  proportionally more, not less.
- The WebSocket round trip. The desktop pushes webcam frames to the pod
  (`bridge.py:2660`) and reads them back through RunPod's proxy. Tens of ms,
  GPU-independent, and on a remote pod plausibly the largest single term.

Amdahl sets the ceiling: at 60% inference, an infinitely fast GPU buys 2.5x.

Two hardware moves that *are* real:

1. **Verify what auto-discovery actually selects.** `RUNPOD_MAX_PRICE` defaults
   to $1.00/hr. Landing on an L4 (34), A4000 (32) or V100 (25) instead of a
   4090 (100) is a ~3x gap closable by one `.env` line — cheaper than any
   optimisation in this document. Run `orchestrator.py gpus` before writing code.
2. **`_MAX_SUPPORTED_COMPUTE_CAP = (9, 0)` excludes Blackwell.** 5090s are
   blocked by the base image's PyTorch/ONNX support, not by price, and are often
   cheaper than an H100 while faster at batch 1. Upgrading the image is the only
   upward hardware move left.

**Batch RENDER is the exception.** No per-frame network round trip, no realtime
deadline, and frames could genuinely be batched. There throughput binds and a
bigger card earns its rate.

**On cost efficiency rather than latency:** if the goal is the RunPod bill
rather than the frame deadline, the levers are the same ones. A faster frame
means either a cheaper GPU holds the preset, or a batch render finishes sooner.
The existing GPU auto-discovery already picks on price against eligibility;
making inference cheaper is what widens that field.

---

## Nuitka — accepted for `desktop/`, rejected for `pipeline/`

### The reason is distribution, not speed

The speed argument for Nuitka on the desktop is weak and should not be the
justification. Qt and QML rendering, JPEG decode, the WebSocket transport and
the filter LUTs are all C already. What is left for the interpreter is the
display timer's per-frame glue and the bridge's event handling — real, but
small against a 33ms tick, and already measured to be zero-cost when no
decoration is enabled (see the filters and effects note in `CLAUDE.md`).

The reason to compile the desktop is that **it ships to customers and the
licensing gate lives inside it**. Session hours, the access-code gate and the
Firestore session plane are all enforced desktop-side. Distributed as `.py`
files, that enforcement is removable with a text editor. Nuitka's standalone
build is not a security boundary — nothing that runs on someone else's machine
is — but it moves tampering from "open the file" to "reverse a binary", which
is the difference that matters commercially.

Secondary, and genuinely useful: the customer stops needing a Python install, a
venv, or a matching PySide6 wheel.

### Why `pipeline/` is explicitly excluded

Every argument for compiling the desktop inverts here:

| | `desktop/` | `pipeline/` |
|---|---|---|
| Who sees the source | The customer | Nobody — it is inside a Docker image on a rented pod |
| What gates value | The session clock, in this code | Nothing; the pod is already access-controlled |
| Startup cost | Qt init, ~1s | Model load, tens of seconds — dominated, unmovable |
| Build cost of compiling | Once per release | Every image build, +20 minutes |
| Rate of change | Low | High — this is the layer under active development |
| Traceback quality | Matters little | Matters a lot |

Compiling the pipeline buys nothing measurable and taxes the build and the
debugging loop of the code that changes most. If it is ever revisited, the
trigger would be shipping a *local* pipeline to customer machines, which is a
different product decision than the one this codebase is currently built for.

### Known build hazards

These are the things that will actually cost time, recorded so they are not
rediscovered:

- **PySide6 / QML.** Nuitka's `--enable-plugin=pyside6` handles the common
  case, but QML files, `qmldir` entries and any `qrc` resources must be
  explicitly included as data. A missing QML file fails at runtime, not at
  build.
- **`__file__`-relative paths.** Anything resolving assets relative to its own
  module path behaves differently in a standalone build. `Bridge._cache_dir()`
  and the `prefs.json` path are the first places to check.
- **Binary dependencies.** `opencv-python` and `numpy` ship compiled extensions
  that Nuitka includes as-is rather than compiling — fine, but they dominate
  bundle size.
- **Build time.** Expect 10–30 minutes for a standalone desktop build. This is
  a release step, not a development one; `python desktop.py` stays the loop.
- **Antivirus.** Single-file Nuitka builds of unsigned consumer binaries are
  routinely flagged on Windows. Prefer `--standalone` over `--onefile`, and
  budget for code signing before any customer sees it.

---

## Numba — rejected for the compositor, narrow elsewhere

### The numerical layer is already at its ceiling

The proposal assumes there is interpreted numerical work to accelerate. In the
per-frame path, there is very little. `FaceCompositor` is a sequence of cv2
primitives — `warpAffine`, `cvtColor`, `meanStdDev`, `GaussianBlur`,
`Laplacian`, `addWeighted`, `resize` — with numpy only as the glue between
them. cv2 is SIMD-vectorised, multi-threaded C++. Numba does not beat it at
these operations and should not be expected to.

What Numba *can* beat is unfused numpy glue: chains that allocate a temporary
per operation where one pass would do. That optimisation has already been
performed here, by hand, and the code says so:

- `_match_color` (`pipeline/processing/compositor.py:678`) — the per-channel
  affine transfer was "nine full-resolution array operations per frame", now
  solved as three scalars and applied as one `*=` and one `+=`.
- `_match_detail` (`pipeline/processing/compositor.py:830`) — the high-band
  scale is "rearranged so it is one fused pass rather than a multiply and an
  add over separate temporaries", via `cv2.addWeighted`.
- `_estimate_noise` (`pipeline/processing/compositor.py:962`) — subsampled by
  stride, which "removes two full-size sorts per frame".

A JIT whose main value is fusing what has already been fused is not worth its
cost.

### Scale, which is the decisive argument

Compositing happens in **aligned face space, 256×256**, not on whole frames —
and the size actually used follows the face's size in frame, so it is often
smaller. That is roughly 200k elements per operation. Against a 33ms deadline
at 30fps, in which ONNX inference for the swap and CodeFormer restoration are
the dominant terms, the entire numpy glue budget is small enough that halving
it is not perceptible.

**Optimising the wrong layer is the failure mode this section exists to
prevent.** If a preset misses its deadline, the answer is in the inference
stages or the aligned-size ceiling, not in the glue.

### The one candidate worth measuring

`_add_grain` (`pipeline/processing/compositor.py:924`) is the exception:

```python
noise = np.random.normal(0.0, sigma, mask.shape).astype(np.float32)
return blended + noise[:, :, None] * mask[:, :, None]
```

It allocates a full-size float64 field, casts it to float32, broadcasts across
three channels, and allocates the result — four full-size allocations, with no
cv2 primitive that does the job. A single `@njit(parallel=True, cache=True)`
kernel generating and applying the noise in one pass over the output buffer is
a genuine fusion Numba is good at, and one numpy cannot express.

It is also on the frame-space path rather than the aligned one, so it runs over
the face ROI in the full frame — the largest array in the chain, and the place
where a fused kernel has the most to save.

**This is a candidate, not a decision.** It ships only if the finer-grained
profile below shows grain is a measurable share of the frame.

### What is not a candidate

- **`desktop/effects.py`** — the per-particle loops (60–150 iterations) are
  Python, but every iteration body is a `cv2.rectangle` / `cv2.circle` /
  `cv2.drawContours` call. Numba cannot compile a cv2 call, and the loops are
  already measured at 2.8ms worst case (Bubbles, 960×540) against a 33ms tick.
  Rewriting them as njit rasterisers means reimplementing anti-aliased drawing,
  which is a large amount of new surface for single-digit milliseconds on a
  decorative layer that is off by default.
- **`desktop/filters.py`** — lookup tables and one cached multiply. There is
  nothing to compile.
- **`LandmarkStabilizer`** — EMA over 106 landmark points. Sub-microsecond.
- **Anything in `guards.py`** — scalar predicates over detection metadata.

---

## The conflict: Nuitka and Numba are not independent

The proposed diagram shows these as parallel branches. They are not.

`@njit` compiles a function at first call by reading **its Python bytecode**.
Nuitka compiles Python functions into C — there is no bytecode left to read. A
Numba-decorated function inside a Nuitka-compiled module fails, and it fails at
runtime, on first call, in the customer's hands.

This is not a reason to abandon either. It is a boundary that must be
architectural rather than discovered:

- Numba kernels live in **one module**, `pipeline/kernels.py` (or
  `desktop/kernels.py` if a desktop-side kernel ever justifies itself).
- That module is excluded from the Nuitka build with
  `--nofollow-import-to=pipeline.kernels` and shipped as plain `.py` alongside
  the binary.
- The module contains **kernels only** — no imports from the rest of Phantom,
  no config access, no logging. Arrays and scalars in, arrays out. This keeps
  the excluded surface as small as possible and makes the boundary
  self-evident to a reader.
- Every kernel has a plain-numpy reference implementation and a test asserting
  the two agree within tolerance, so the excluded module can be verified
  independently of whether Numba is installed.

As currently scoped this conflict is theoretical — the only Numba candidate is
in `pipeline/`, which is not being Nuitka-compiled. The rule is written down
because the moment those overlap, the failure is silent until runtime.

### Numba's other runtime cost

First call to an `@njit` function pauses to compile — hundreds of milliseconds
to seconds. On a live stream that is a dropped frame, and it lands on the
*first* frame, which is exactly when the operator is judging whether the thing
works. Mitigations, both required if any kernel ships:

- `cache=True`, so compilation is paid once per machine rather than once per
  process.
- An explicit warm-up call during `_warm_up_models`, alongside the ONNX session
  warm-up that already happens there, so the cost lands in startup where
  seconds are already being spent.

---

## The gate: measure before building either

Phantom already owns the instrument that decides this. `LatencyBudget`
(`pipeline/services/latency.py`) records per-frame stage timings and reports
p50/p95/p99 against the preset's deadline with a HOLDS/MISSES verdict — built
precisely because "does this preset hold" is a different question from "how
long did frame 240 take".

It is currently too coarse to justify any of this work. `record()` takes three
buckets:

```python
def record(self, detect_ms: float, swap_ms: float, total_ms: float) -> None:
```

`swap+composite` is one number covering inference, restoration, smoothing,
colour matching, detail matching, masking, grain and the paste. No decision
about Numba can be made from it.

**Step zero is splitting that bucket**, which is a small change to `latency.py`
and to `_composite_impl`'s call sites. The result is a per-stage distribution
that answers the question directly. The bar to clear:

| Finding | Action |
|---|---|
| Inference dominates the swap bucket | Neither Nuitka nor Numba helps. Look at aligned-size, det_size, provider |
| Grain is >5% of frame time at p95 | Build the `_add_grain` kernel |
| Grain is <5% | Do not build it. Record the number here and close the question |
| The desktop display tick is near budget | Profile the bridge before compiling; Nuitka is not a fix for an O(n) mistake |

`--guard-observe` and `tools/compare_frames.py` are the existing precedent for
this: measure the thing, record the number, then decide. Nine guard thresholds
were chosen without data and that is written down as a known weakness — this
document should not add a tenth.

---

## Implementation order

Steps 1–6 are **done**; the code is in and the suite is green. Step 7 is the
one that matters now, because it is the only one that produces evidence.

1. ~~**Split the latency buckets.**~~ `LatencyBudget.record` takes an `extra`
   breakdown; `FaceCompositor.last_stage_ms` carries per-stage milliseconds for
   mask, restore, smooth, colour, detail and paste, read by the pipeline the
   same way `masker.last_coverage` already was.
2. ~~**Fix the discarded `SessionOptions`.**~~ It was built, configured and
   never passed. Session construction now lives in one place, which is what
   made it visible.
3. ~~**IOBinding and pre-allocated outputs.**~~ `BoundRunner` binds once and
   reuses the buffers, falling back silently to a plain `run` on a symbolic
   output shape or a build without the binding API.
4. ~~**fp16.**~~ `tools/convert_fp16.py` writes a `-fp16.onnx` copy beside the
   original with `keep_io_types=True`, so no calling code changes. Reverting is
   a config flag, not a re-download.
5. ~~**CUDA graphs.**~~ Enabled per model, only where shapes are static — a
   captured graph records fixed buffer addresses, so the detector is excluded.
6. ~~**TensorRT.**~~ Per-architecture engine cache on the network volume, keyed
   by GPU, TensorRT and ORT versions, model fingerprint and precision.
   `TRT_GPUS` bounds which cards are worth a multi-minute build.
7. **Run a session per preset and record the numbers here.** Nothing above has
   been measured. Order: baseline, then `CUDA_GRAPHS=true` (no numerics change,
   so a pure latency read), then `FP16=true` with a `--debug-frames` A/B, then
   `TRT=true` if the first two leave the deadline missed.
8. **Nuitka standalone build of `desktop/`.** Independent of all of it — a
   distribution change, not a performance one.
9. **Code signing.** Follows 8 immediately.
10. **The grain kernel, only if step 7 justifies it.** Realistically last,
    because it is smallest.

### What step 7 has to answer

The per-stage breakdown exists so the question is not "did it feel faster". Read
`restore` against `swap+composite` first: if restoration is not the dominant
term, the whole premise of this document is wrong and the reasoning above
should be rewritten rather than defended.

| Reading | What it means |
|---|---|
| `restore` dominates | As predicted. fp16 and TensorRT are aimed correctly |
| `detect` dominates | Look at `det_size` and the preset, not at any of this |
| `total` >> the sum of stages | The cost is outside the compositor — capture, JPEG encode, or the proxy hop |
| CUDA graphs move nothing | Launch overhead was not the bottleneck; drop the flag rather than keep a lever nobody reads |

## What would overturn this

- **Shipping the pipeline to customer machines.** The Nuitka exclusion for
  `pipeline/` rests entirely on it running inside a Docker image nobody
  inspects. A local-pipeline product reverses it.
- **A CPU-only deployment target.** The whole "the numerical layer is small
  relative to inference" argument assumes CUDA. On CPU, inference balloons and
  the ratio changes — though so does the conclusion, since the answer there is
  a smaller model, not a faster glue layer.
- **A new stage that is genuinely interpreted.** If something lands in the
  per-frame path that is a real Python loop over pixels rather than a cv2 call,
  Numba becomes the right tool immediately. The rule is the loop, not the
  language.
- **Numbers from step 2 that contradict the reasoning above.** This document
  argues from reading the code and from the scale of the arrays. That is an
  informed prediction, not a measurement, and it is written to be checked.
