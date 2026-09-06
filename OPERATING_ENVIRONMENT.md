# Operating environment

Constraints that live on **our side of the wire** — the operator's connection,
their machine, the room they sit in. None of it is a defect in the pipeline, the
GPU or the datacenter, and all of it has at some point been mistaken for one.

This file exists because that mistake is expensive. A session was spent
migrating cloud providers to buy latency that measurement later showed was never
the binding term, and an evening was spent judging presets on a mobile hotspot.
Read this before concluding that something is broken.

Every number here was measured, with the date. Replace them when you measure
again; do not delete the old ones.

---

## 1. What has never been the constraint

Two things get blamed and should not be, both measured 2026-09-05 against a
verified RTX 4090 in Denmark.

**The GPU.** The pipeline held its frame deadline with room to spare:

    detect            8.2ms
    swap+composite   30.3ms      (mask 11.8, restore 7.9, paste 2.2)
    total            38.8ms      p95 42.9ms  ->  7.1ms headroom  [HOLDS]

1105 frames, zero guarded. Restoration is `gpen_bfr_256` at 7.9ms, so the
39.5ms CodeFormer figure that drives much of docs/PERFORMANCE_AUDIT.md no longer
describes the default configuration.

**The datacenter.** Round trip from the operator's machine to every European
region, sampled back to back:

    London     171.6ms      Ireland    175.8ms      Frankfurt  176.9ms
    Zurich     184.1ms      Spain      188.0ms      Paris      191.5ms
    Stockholm  196.8ms      Milan      206.1ms

**The entire European spread is ~35ms.** Moving from Denmark to a UK datacenter
buys roughly 20-40ms. That is real, and it is small next to everything below.

---

## 2. The operator's uplink

This is the binding constraint, and it is not ours to fix.

### What each preset costs

Measured on a real photograph at each preset's own capture settings:

| preset | capture | fps | JPEG q | **uplink needed** |
|---|---|---|---|---|
| `fast` | 480x270 | 15 | 60 | **1.58 Mbps** |
| `optimal` | 640x360 | 15 | 60 | **2.45 Mbps** |
| `production` | 640x360 | 20 | 70 | **3.96 Mbps** |

One JPEG per frame, so bitrate is linear in frame rate and roughly quadratic in
linear resolution. There is no inter-frame compression: a still face costs the
same as a moving one.

**Re-measured 2026-09-05 on three inputs, and the table above is the optimistic
one.** It was taken on `source.jpg`, which is a detailed portrait; the real
camera encodes smaller:

| gear | `source.jpg` | `target.mp4` | **this webcam** |
|---|---|---|---|
| `fast` | 1.45 Mbps | 1.27 | **1.19** |
| `optimal` | 2.34 Mbps | 1.96 | **2.08** |
| `production` | 3.76 Mbps | 3.06 | **3.50** |

Content moves it by ~20%, so treat any single figure as ±20% and the *ratios*
between gears — which are stable to a couple of percent — as the reliable part.
The webcam column assumes `production` reaches 20fps, and on this machine it
does not (below), so its true cost here is nearer **2.6 Mbps**.

### What the link actually delivered

| when | uplink | RTT p50 | RTT p95 | delivered |
|---|---|---|---|---|
| 2026-09-05, fixed line | ~4 Mbps offered | 210ms | 320ms | `fast` 91%, `production` 84% |
| 2026-09-05, **LTE hotspot** | **0.5 Mbps** | **1875ms** | **3386ms** | 801 held frames |

**The same evening, hours apart, RTT moved 9x and uplink fell to a third of what
the cheapest preset requires.** Nothing in the repository changed between those
two readings.

### A third reading, same day, and every gear held

`tools/measure_link.py --presets production,optimal,fast`, run against the same
Denmark 4090. Order reversed deliberately, models warm:

| preset | uplink | delivered | p50 | p95 |
|---|---|---|---|---|
| `production` | 3.96 Mbps | 97% | 286ms | 339ms |
| `optimal` | 2.45 Mbps | 99% | 276ms | 320ms |
| `fast` | 1.58 Mbps | **100%** | 255ms | 291ms |

Network alone that evening was **p50 201ms, p95 242ms**, no frame in it.

Three things follow, and they matter more than the individual numbers.

- **The whole ladder is worth 31ms of p50 against a 201ms network floor.** When
  the link has headroom, preset choice is 11% of the round trip and distance is
  the rest. Bitrate only dominates when the link is *short* of what a preset
  asks — which is what the morning reading was.
- **`optimal` delivered 61% in the morning and 99% in the evening.** Same pod,
  same preset, same machine, nothing changed in the repository. A single fixed
  preset cannot be right on both, which is the case for
  `desktop/uplink.py::UplinkGovernor` in one line.
- **Run-to-run variance swamps small differences.** `production` measured p95
  1293ms in one pass and 339ms ten minutes later. Trust the ordering-controlled
  comparison inside a run; do not compare a row against a row from another day.

**Order matters, and it cost a run here.** `fast` returned **56%** when measured
first after a pipeline restart and **100%** when measured last on warm models —
the first preset was being charged for model load, which is tens of seconds
against a 12s per-preset warm-up. `measure_link.py` now runs a discarded pass
before the first measured one. Any tool driving a fresh pipeline needs the same.

### Rules that follow

- **A mobile hotspot cannot run this product.** 0.5 Mbps is under a third of
  `fast`. No preset, provider, datacenter or codec closes a 3x shortfall.
- **Never judge a preset, a provider or a change on a hotspot session.** The
  variance swamps the effect being measured.
- **Uplink is the asymmetric leg.** Home and mobile connections give far less
  upstream than downstream, and the pipeline sends a JPEG per frame upstream.
  Downstream has never been observed to bind.

---

## 3. The operator's machine

- **No discrete GPU.** Intel UHD Graphics 620, integrated. So running the
  pipeline locally — which would remove the ~200ms round trip entirely, the
  single largest improvement available — needs hardware that does not exist
  here. docs/LOCAL_GPU_SETUP.md covers what it would take.
- The desktop process also carries capture, preview, filters, effects, the
  virtual camera and audio. It is the busiest machine in the chain, while the
  rented 4090 sits at 38.8ms of a 66.7ms budget.
- Only `desktop/.qtcreator/Python_3_12_0venv` has the full desktop dependency
  set. Plain `python desktop.py` fails on imports.

### The webcam, measured 2026-09-05 — three inherited claims are wrong here

Five-second captures at 640x360, each backend, with and without a rate request:

| backend | asked | reports | **delivers** | open + first frame |
|---|---|---|---|---|
| default (resolves to DSHOW) | — | 0 | **15.1 fps** | 0.9s |
| default | 20 | 20 | **15.1 fps** | 1.5s |
| DSHOW explicit | — | 0 | **15.1 fps** | 0.9s |
| DSHOW explicit | 20 | 20 | **15.1 fps** | 1.5s |
| MSMF explicit | either | — | **cannot open the camera at all** | — |

Three things the codebase asserts and this contradicts:

- **"MSMF ignores `CAP_PROP_FPS` — asked for 20 it delivers 30."** Not here. The
  camera delivers 15.1 fps however it is asked, and never 30. So the claim that
  the uplink "was carrying half again as many JPEGs as the preset assumes" does
  not hold on this machine, and **`FramePacer` is inert here** — 15.1 against a
  15 target is inside its 1.1 margin, so it drops nothing. It is still right to
  keep: it costs one comparison per frame and the next camera may differ.
- **The backend is DSHOW, not MSMF.** So `_configure_capture`'s `backend !=
  'MSMF'` branch *does* set the frame rate here — paying ~0.6s for a setting the
  camera then ignores.
- **`production` is unreachable.** It asks 20fps of a camera that gives 15, so
  its real uplink cost here is ~2.6 Mbps rather than 3.5, and the `20 fps` in
  the preset table is aspirational on this hardware.

The 3.9s device-configuration cost recorded in `_configure_capture` is also not
what this camera does: 0.9s to first frame, 1.5s with a set applied. Both are
still worth avoiding per gear change, which is why capture is decoupled from the
preset — but the figure is machine-specific, not a constant.

None of this changes the ladder's ordering, which is the part the governor uses.

---

## 4. Telling environment from product

The badge in the top right of the viewport separates them in one line:

    550ms delay · rtt 1875/3386 · up 0.5Mbps · 801 held · skew ...

| reading | means |
|---|---|
| **up** below the preset's requirement | the link cannot carry this preset. Drop a gear, and do not read anything else as a fault |
| **rtt** far above ~210ms | congestion or a degraded path, not distance |
| **held** climbing | frames arriving after their slot, so the last swapped frame is repeated. Correct behaviour, not a bug |
| **delay** much larger than RTT | the playout buffer, computed as `p95 + spread + 80ms` from **jitter** rather than distance. Pin it with `PHANTOM_PLAYOUT_DELAY_MS` |

Then compare against the pipeline's own report, which lives on the instance:

    python vast/orchestrator.py run "grep -a -A14 'Latency budget' /workspace/phantom-pipeline.log | tail -16"

**The difference between the badge and that report is network and encode.** If
the report says HOLDS and the badge says seconds, nothing about the GPU or the
datacenter is going to help.

`python tools/measure_link.py --out link-history.json` measures the whole thing
deliberately: a network round trip with no frame in it, then each preset's round
trip and delivery rate, appended with a timestamp so days can be compared. It
takes `--host`/`--port`, so it can be aimed at any provider for a like-for-like
comparison on the same afternoon.

---

## 5. Misattribution log

Kept so the same reasoning is not repeated. Each of these looked like a fault in
the product or the provider, and was not.

- **"Vast is unreliable."** Two rented hosts accepted no connections on the
  ports they published. Both were *unverified* machines — one reported
  `192.168.178.106` among its addresses, a home router range, behind a NAT
  nobody had port forwarded. Closed by `VAST_VERIFIED_ONLY=true`, now the
  default. Verified hosts have behaved correctly since.
- **"The move to Vast was for latency."** The distance premise was never
  measured before the move. When it finally was, all of Europe sat within 35ms
  and the felt delay was dominated by uplink and buffering.
- **"`optimal` is too slow, the GPU cannot keep up."** The GPU had 7ms of
  headroom and guarded nothing. `optimal` was losing 16% of frames on the
  uplink.
- **"Bandwidth is not the constraint."** Concluded from a controlled run whose
  p50s differed by 21ms — while delivery differed by 7 points. The operator
  switched presets by hand and reported the lower bitrate as "much smoother"
  immediately. Smoothness is frames arriving consistently; it is not a median.
  `tools/measure_link.py` now weighs delivery beside latency.
- **"Something we changed broke the stream."** Two changes were in flight when
  the link fell to 0.5 Mbps on a hotspot. Neither could affect uplink or RTT —
  one is a display side buffer, the other pipeline side compute — but with both
  outstanding, nothing was attributable. **Change one thing at a time when the
  environment is not stable.**

---

## 6. If the connection improves

The order to re-measure in, because each answer decides whether the next
question matters at all:

1. `tools/measure_link.py` on a normal connection. This is the baseline that
   does not exist yet — everything above came from one bad day and one hotspot.
2. Does `optimal` at 2.45 Mbps hold? It is an estimate, not a measurement. If it
   does, `fast` stops being needed and the resolution question is closed.
3. Only then, is distance worth paying for? It is worth 20-40ms. Judge that
   against whatever the total has become, not against the 350ms figure that
   motivated the original migration.

**The video codec is the lever that actually attacks this.** One JPEG per frame
is why the bitrate is what it is; H.264 at the same visual quality is roughly a
quarter to an eighth of the bytes, which would put `production` inside the
budget `fast` needs today. It is not free — see docs/PENDING_WORK.md, and the
conflict with the inbound queue's drop-oldest policy, which a stateful codec
cannot tolerate — but it is worth hundreds of milliseconds where a datacenter
move is worth twenty.
