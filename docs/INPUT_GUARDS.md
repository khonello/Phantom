# Input Guards

Refusing inputs that would produce a wrong swap, instead of swapping them badly.

**Status: design settled, not built.** Thresholds still need calibration against
real uploads and real footage.

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
| **Multiple faces** | Reject — we cannot know which person was meant |
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

Two fixes belong in the same pass, independent of the guards: replace leftmost
with largest, and reset the stabilizer when the selected face changes identity —
the existing centroid-jump reset does not catch a switch between two people
standing still.

---

## Open questions

**Is largest-face enough, or does selection need continuity?** Largest can still
flip between two people of similar apparent size. Preferring the face nearest the
previous frame's position would be steadier, but that is a tracker by another
name. Guarding may make it moot: two comparable faces in shot guards the frame
anyway.

**Is pose available cheaply?** InsightFace exposes it on some configurations. If
`buffalo_l` does not, yaw can be approximated from the five keypoints.

**How aggressive should source rejection be?** Every guard that rejects a usable
photo is friction at the moment a new customer is deciding whether this works.
Multi-face is unambiguous; sharpness and pose are judgement calls, and starting
permissive may convert better than starting strict.
