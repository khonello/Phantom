# Input Guards

Refusing inputs that would produce a wrong swap, instead of swapping them badly.

**Status: built.** Implemented in `pipeline/services/guards.py`, with source
review in `pipeline/services/database.py` and the runtime path in
`pipeline/processing/pipeline.py`.

**Thresholds are not calibrated, but the measurement for it is built.** This
is a recorded, accepted risk rather than an oversight — see
[ACCEPTED_RISKS.md](ACCEPTED_RISKS.md). Run:

```bash
python pipeline.py --stream --guard-observe --guard-report calibration.json
```

Observe mode evaluates and records every guard while none of them act. That is
not a convenience — a session that *enforces* cannot measure itself, because a
guarded frame emits the held frame and stops being a sample of what the camera
was doing. The report gives a distribution per metric with the percentage that
would fail and the **margin** to the configured threshold; a negative margin
means the threshold sits inside normal operating range.

Nine thresholds, not the two this document originally anticipated. Three can make
things actively worse if mis-set, and none of the three is visible from watching
the output:

| Threshold | Risk if wrong |
|---|---|
| `guard_min_coverage` | Compared against XSeg coverage of the *expanded* hull. What that reads on a completely clear face has never been measured |
| `guard_identity_sim` | Above the same-person similarity under motion blur, the stabilizer resets every frame — the shimmer comes back, caused by a guard |
| `guard_min_confidence` | 0.5 against a detector threshold of 0.35, so everything scoring in between is guarded |

`guard_min_sharpness` and `guard_outlier_sim` are the two that need *upload* data
rather than footage. All start permissive on purpose — see *How aggressive should
source rejection be?* below.

Everything described here is present with one deliberate deviation: **zero faces
is not a guarded frame.** It is the ordinary case of someone stepping out of
shot, already handled by marking the stabilizer missing, and holding the last
good frame there would keep a stale face over an empty chair.

Two things have changed since, both about *multiple faces* specifically — see
[Naming a face](#naming-a-face) and [What a render does](#what-a-render-does)
below.

---

## Why

The pipeline has no notion of an input it should refuse. It finds a face or it
does not; everything in between — the wrong person, an unusable photo, two people
in shot — is processed as though it were fine.

That produces output that is confidently wrong, which is worse than no output. A
frame with no face is obviously broken and the operator fixes it. A frame with a
*stranger's* face swapped in looks like it worked.

---

## The problem in the current code

**When there is more than one face, the pipeline picks the leftmost one.**

```python
# pipeline/services/face_detection.py — detect_one()
return min(detections, key=lambda d: d.bbox.x)
```

Not the largest, not the nearest, not the centre — the smallest x coordinate.
Nothing about how images are composed makes that the subject, so it picks wrong
whenever there is a second candidate: someone beside you, someone walking
behind, a face on a television, a photo on the wall. It arrived with the
original scaffolding and was never revisited.

It is used in two places and fails differently in each:

- **Source images** — `FaceDatabase._extract_from_image()` builds the embedding
  from whatever it returns. One wrong pick at upload time means every subsequent
  frame swaps the wrong identity. Decided once, silently, never re-examined.
- **Runtime** — `DetectionProcessor` calls it on *every frame*. As people move,
  the leftmost face can change between frames, so the swap jumps between
  subjects mid-call and the `LandmarkStabilizer` is fed alternating identities.

**Separately, a batch of source images is averaged with no consistency check.**
`_average_faces()` takes the mean of everything it is given. One photo of a
different person pulls the identity toward a blend of two people — a face that
resembles nobody. Nothing reports it.

Largest-face is a better rule than leftmost and should replace it, but it still
fails silently. The difference between the two call sites is whether a human is
available: at upload the operator is right there, so refuse and say why; at
runtime nobody can be asked, so guard the frame.

---

## Source guards

Applied when images are uploaded, before any embedding is built.

| Guard | Rule |
|---|---|
| **Multiple faces** | Reject — we cannot know which person was meant, and no
picker can fix it: see [Naming a face](#naming-a-face) |
| **No face** | Reject |
| **Face too small** | Reject under ~110 px on the shorter side; the embedding would come from an upscaled blur |
| **Blurred** | Reject below a Laplacian-variance floor; a soft source gives a soft swap on every frame |
| **Extreme pose** | Reject beyond roughly ±35° yaw; ArcFace degrades sharply toward profile |
| **Identity outlier** | With three or more images, reject any that disagrees with the rest |

A rejected image reports **which** image and **why**. This is an upload flow with
a person present, so a bare failure is not good enough — they need to know which
photo to replace.

### The outlier check

Leave-one-out cosine similarity: compare each embedding against the mean of the
others, and reject any that sits too far away. This catches the wrong person
hidden in a batch, which is the failure `_average_faces` cannot currently see.

It needs **three images** to mean anything. With two that disagree there is no
majority and no way to tell which is the intruder — report the disagreement
rather than guessing.

---

## Runtime guards

Applied per frame, before swapping.

| Guard | Rule |
|---|---|
| **Multiple faces** | More than one detection while `many_faces` is off |
| **Low confidence** | Detection score below threshold |
| **Face too small** | Under ~80 px; the swap would be mush |
| **Extreme pose** | Yaw beyond limit |
| **Heavy occlusion** | XSeg coverage below ~40% of the hull |

```
frame ─▶ detect ─┬─ 0 faces ──────────────▶ no swap, mark missing
                 │
                 ├─ >1 face ───────────────▶ GUARDED
                 │
                 └─ 1 face ─┬─ too small ──▶ GUARDED
                            ├─ low score ──▶ GUARDED
                            ├─ bad pose ───▶ GUARDED
                            ├─ occluded ───▶ GUARDED
                            └─ ok ─────────▶ swap · composite · emit
```

Detection already runs every frame, so all of these read data the pipeline
already has. The cost is negligible.

---

## Naming a face

The multi-face guard fires because *"which face did you mean?"* has no safe
default. It follows that anyone who **answers** the question dismisses the
guard, and there are now two ways to answer:

| Who answers | How | Where |
|---|---|---|
| A template's author | `face_point` in the manifest, offline | `config.target_face_point` |
| The operator | Clicks a face over their own photo | `config.target_face_points[i]` |

Both are **normalised points, not indices.** Detection order is not a stable
contract — it can shift with a model pack — and an index that quietly comes to
mean a different person is exactly the silent wrong-person swap this document
exists to prevent. `templates.select_by_point` resolves a point by containment,
then by nearest centre.

The operator's version is a list because photo mode carries up to four targets
and each asks the question separately; a single point would name a face in
photos nobody looked at. It is aligned with `target_paths` by index, `None` for
a photo that was never ambiguous, and cleared by every new upload.

Detection for the picker runs at **upload**, in `handle_upload_target`, not at
swap time. That is where the person is: a photo refused mid-job tells them only
that they already picked the wrong one.

**Sources are deliberately excluded.** `check_source` has no such escape and
should not get one. A source builds the identity that every subsequent frame is
swapped *to*, and averaging that out of a crowd is wrong in a way nothing
downstream recovers from.

---

## What a render does

Batch splits by what an unswapped output would mean, and the split is now three
ways rather than two:

| | Guarded for pose, confidence, occlusion | Guarded for **multiple faces** |
|---|---|---|
| **Video** | Frame passed through, job continues | **Job stops**, no output written |
| **Still** | Nothing written, reason recorded | Nothing written, reason recorded |

Pass-through was the original rule for all of them, and it was wrong for one
case. The other guards describe **one frame** — a turn of the head, a hand, a
blurred moment — and one unswapped frame mid-clip is a smaller defect than a
hole. A second face describes the **target**: it will almost certainly persist,
so every frame it appears in is written unswapped, and the render silently stops
being a swap partway through while reporting success.

The abort names the frame, the timecode and the count, and goes out through
`emit_error` — the desktop reads a batch's success from whether an error
arrived, so a warning would let it render as "processing complete". No partial
file is possible: the abort precedes `create_video`, and the extracted frames
are cleaned in the existing `finally`.

---

## What a guarded frame emits

**The last good swapped frame, unchanged.**

Nothing is drawn onto it — no banner, no border, no text, no tint. The frame goes
straight to the virtual camera and therefore to everyone on the call, so anything
added would be visible to every participant. A held frame reads as a network
hiccup, which is the most innocuous way this can fail in front of other people.

Passing the *raw* frame through is never an option: the operator is on the call
precisely because they do not want their own face transmitted, and showing it
turns the guard into the exposure it exists to prevent.

Two rules go with it:

- **Fail closed.** If a guard cannot be evaluated — occluder model missing,
  detection error, threshold unset — the frame is guarded.
- **Never update temporal state.** A guarded frame must not enter the pixel EMA
  or the landmark EMA, or a stranger gets blended into the smoothed history and
  leaks back out over the following frames. Guards call `FaceCompositor.reset()`
  and `LandmarkStabilizer.reset()`.

---

## The virtual camera invariant

A guarded frame is one case of a rule that has to hold everywhere:

> **The virtual camera shows the last augmented frame, or an augmented frame.
> Never the raw camera, and never nothing.**

It applies to every way processing can stop:

| Event | Virtual camera shows |
|---|---|
| Frame guarded | Last augmented frame |
| **Paid hour expires, pipeline disconnects** | **Last augmented frame** |
| Session ends or is stopped | Last augmented frame |
| Worker dies, pod reclaimed, network drops | Last augmented frame |
| Pipeline crashes | Last augmented frame |

The hour-expiry case is the one most likely to be got wrong, because it is the
only one that is *expected*. It must behave exactly like the failures: the
operator's real face must not appear on the call the moment their time runs out.

**"Never nothing" matters as much as "never raw."** Today `_run_vcam` only calls
`cam.send()` when a frame arrives:

```python
# desktop/bridge.py — _run_vcam()
frame = self._vcam_queue.get(timeout=0.1)
cam.send(frame)
```

When frames stop, it stops sending. The device stalls rather than freezing, and a
call application may report that as a disconnected camera — a louder and stranger
signal than a frozen picture. The loop must instead **hold the last frame and keep
re-sending it** at the normal rate, so the stream stays alive and simply stops
moving.

The raw camera already never reaches the virtual camera: `_push_to_vcam` is only
called with processed frames. That property is easy to break by accident and
worth a test.

---

## Configuration

Thresholds live on `FaceSwapConfig` alongside the realism knobs, settable through
the `set_realism` validation pattern.

```
guard_multi_face        True      reject multi-face sources, guard multi-face frames
guard_min_source_px     110       minimum source face size
guard_min_frame_px      80        minimum runtime face size
guard_min_sharpness     …         Laplacian variance floor — needs calibration
guard_max_yaw           35        degrees
guard_min_confidence    0.5       detection score floor (detect threshold is 0.35)
guard_min_coverage      0.4       fraction of hull unoccluded
guard_outlier_sim       …         cosine floor for source consistency — needs calibration
```

`guard_multi_face` must be switchable off, since the same pipeline serves
`many_faces` deliberately.

---

## Where this lands

| Piece | File |
|---|---|
| Largest-face rule, face count | `pipeline/services/face_detection.py` |
| Source guards, per-image reasons | `pipeline/services/database.py` |
| Outlier rejection | `pipeline/services/database.py::_average_faces` |
| Runtime guards, hold-last-frame | `pipeline/processing/pipeline.py::_process_and_emit` |
| Virtual camera holds last frame | `desktop/bridge.py::_run_vcam` — re-send on empty queue |
| Thresholds | `pipeline/config.py`, `pipeline/api/handlers.py` |
| Rejection reasons in the UI | `desktop/bridge.py`, `desktop/main.qml` |
| Render abort on multi-face | `pipeline/processing/pipeline.py::_process_frame_files` |
| Operator naming a face | `pipeline/api/handlers.py::handle_set_target_faces`, `desktop/bridge.py::chooseFace` |
| Guard reason reaching the operator | `desktop/bridge.py::guardReason` — a badge beside the viewport, never on the frame |

Two fixes belong in the same pass, independent of the guards: replace leftmost
with largest, and reset the stabilizer when the selected face changes identity —
the existing centroid-jump reset does not catch a switch between two people
standing still.

---

## Open questions

**Is largest-face enough, or does selection need continuity?** *Settled: enough.*
Guarding made it moot as expected — two comparable faces in shot guards the frame
regardless of which one selection picks. The remaining case, a switch between two
people of *similar size*, is caught by the stabilizer's identity check rather
than by selection, which is the cheaper place for it.

**Is pose available cheaply?** *Settled: yes, and it was already being computed.*
`buffalo_l` bundles `1k3d68.onnx`, and InsightFace's 3D-landmark model sets
`face.pose` as a side effect of every detection — the pipeline was throwing it
away and approximating it. `guards.measure_yaw` now prefers it.

The keypoint approximation survives as the fallback for packs trimmed with
`allowed_modules`: the nose's offset along the inter-eye axis normalised by eye
span, which is scale and roll invariant, coarse, and increasingly pessimistic
past the threshold — the safe direction. The two are **not on the same scale**,
so a threshold calibrated against one should not be assumed correct for the
other; which source produced each reading is recorded in the telemetry.

The startup capability probe reports whether `pose` is actually present on the
running pack, along with `normed_embedding`, `landmark_2d_106` and `det_score`.
Each of those silently disables a guard when missing, and a disabled guard is
indistinguishable from one that never had cause to fire.

**How aggressive should source rejection be?** *Still open, and only real upload
data settles it.* Every guard that rejects a usable photo is friction at the
moment a new customer is deciding whether this works. Multi-face is unambiguous;
sharpness and pose are judgement calls, so both start permissive.

One caveat on the sharpness floor: Laplacian variance is scale-dependent, so the
same face photographed larger scores higher. That is tolerable for source uploads
— they are all face photographs, and the floor is low — but it means the number
cannot be reused as-is for anything differently framed.
