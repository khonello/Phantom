# Input Guards

Rejecting inputs that would produce a wrong or unusable swap, rather than
swapping them badly.

**Status: proposed. None of this is built.**

---

## The failure this prevents

The pipeline currently has no notion of an input it should refuse. It detects a
face, or it does not. Everything in between — the wrong person, an unusable
photo, two people in shot — is processed as though it were fine, and the result
is a swap that is confidently wrong.

That is worse than no swap at all. A frame with no face is obviously a failure
and the operator knows to fix it. A frame with someone *else's* face swapped
onto it looks like it worked.

---

## What happens today

Three behaviours, all verified against the code, all silent.

**A source photo with several people contributes an arbitrary one.**
`FaceDatabase._extract_from_image()` calls `detector.detect_one()`, which
returns the **leftmost** face in the image:

```python
# pipeline/services/face_detection.py
return min(detections, key=lambda d: d.bbox.x)
```

Upload a photo of yourself with a friend on your left, and you have just built
an embedding of your friend. Nothing reports this.

**A batch of source images is averaged without any consistency check.**
`_average_faces()` takes the mean of every embedding it was given and normalises
it. One photo of a different person pulls the identity toward a blend of two
people, which is an identity that does not exist and will not resemble anyone.
Nothing reports this either.

**At runtime every detected face is swapped, regardless of `many_faces`.**

```python
# pipeline/processing/pipeline.py
for detection in detections:
    face = detection.face
    if not self.config.many_faces:
        face = self._stabilizer.stabilize(face)
    frame = self._swap_face(frame, face)
```

`many_faces` only decides whether landmarks are smoothed — it does not restrict
*which* faces are swapped. So with `many_faces = False` and two people in shot,
both get swapped, and both are pushed through the **same** `LandmarkStabilizer`,
whose EMA state then thrashes between two different people every frame. That is
a bug independent of this feature.

---

## Source guards

Applied when source images are uploaded, before any embedding is built. A
rejected image is **reported to the operator with the reason**, never dropped
silently.

| Guard | Rule | Why it matters |
|---|---|---|
| **Multiple faces** | Reject if more than one face is detected | We cannot know which person was intended. This is the requested behaviour and the most important guard here |
| **No face** | Reject | Already handled, but the reason must reach the UI |
| **Face too small** | Reject if the face box is under ~110 px on its shorter side | Below this the embedding is being computed from an upscaled blur |
| **Blurred** | Reject if the variance of the Laplacian over the face crop is below threshold | A soft source produces a soft swap on every frame forever |
| **Extreme pose** | Reject beyond roughly ±35° yaw | ArcFace embeddings degrade sharply toward profile; a profile source yields a generic face |
| **Identity outlier** | With three or more images, reject any whose embedding is far from the group consensus | Catches the wrong person in a batch — the failure that `_average_faces` cannot currently see |

### The outlier check

Cosine similarity against the mean of the others, leave-one-out:

```
  img1  img2  img3  img4  img5
   │     │     │     │     │
   └─────┴──┬──┴─────┴─────┘
            ▼
     mean embedding (excluding the one under test)
            │
            ▼
   cos(img_i, mean_without_i)  <  threshold   →  reject img_i
```

Two properties worth stating:

- **It needs three images to be meaningful.** With two dissimilar images there is
  no majority and no way to tell which is the intruder — report the
  disagreement and let the operator choose, rather than guessing.
- **The threshold is identity-scale, not pixel-scale.** Same person across
  lighting, age and expression typically sits well above the separation between
  different people, which is what makes this work at all. The exact number has
  to be calibrated against real uploads.

---

## Runtime guards

Applied per frame, before swapping.

| Guard | Rule | Action |
|---|---|---|
| **Multiple faces** | More than one detection and `many_faces` is off | Do not swap this frame — the requested behaviour |
| **Low confidence** | Detection score below threshold | Do not swap |
| **Face too small** | Face box under ~80 px | Do not swap; the result would be mush |
| **Extreme pose** | Yaw beyond limit | Do not swap, or reduce strength |
| **Heavy occlusion** | XSeg coverage below ~40% of the hull | Do not swap; mostly hands or a microphone |

```
frame ─▶ detect ─┬─ 0 faces ────────────▶ no swap, mark missing
                 │
                 ├─ >1 face  ───────────▶ GUARDED
                 │   (many_faces off)
                 │
                 └─ 1 face ─┬─ too small ─────▶ GUARDED
                            ├─ low score ─────▶ GUARDED
                            ├─ extreme pose ──▶ GUARDED
                            ├─ occluded ──────▶ GUARDED
                            └─ ok ────────────▶ swap · composite · emit
```

Detection already runs on every frame, so every one of these is a comparison
against data the pipeline has in hand. The cost is negligible.

---

## What to emit when a frame is guarded

This is the decision that matters most, and it is a product judgement rather
than a technical one.

The naive answer — pass the frame through unswapped — is the wrong one. The
operator is on a video call precisely because they do not want their own face
transmitted. Guarding a frame and then showing their real face turns a safety
feature into the exposure it was meant to prevent.

| Option | Behaviour | Problem |
|---|---|---|
| Pass through unswapped | Real face appears | Defeats the purpose entirely |
| Hold last good swapped frame | Video appears to freeze | Obvious if it lasts |
| Emit nothing | Client freezes on its last frame | Same as holding, decided client-side |

**Recommended:** hold the last good swapped frame for a short window — roughly a
second — which covers the common transient case of someone crossing behind the
operator. Past that window the condition is not transient and the operator needs
to know, loudly and immediately, so they can move, close the door, or end the
call. A frozen picture with a clear on-screen reason is recoverable; a swapped
stranger is not.

Never fail open. If a guard cannot be evaluated, treat the frame as guarded.

---

## Telling the operator

Guards are worthless if they are silent — that is the current failure mode being
fixed, and it would be easy to reproduce.

- **Source rejection** returns per-image reasons, so the UI can say *"3 of 5
  images accepted — `group.jpg` has 2 faces, `blurry.png` is too soft"* rather
  than a bare failure.
- **Runtime guarding** emits an event carrying the reason, so the desktop can
  show a persistent, unmissable state while it lasts. This needs to be more
  prominent than the existing status line: it is the difference between a
  working call and an exposed one.
- Guard activations are worth counting. A source that is guarded 30% of the time
  is a bad source, and the operator should be told to replace it.

---

## Configuration

Every guard needs a threshold and every threshold is a guess until calibrated
against real footage. They belong on `FaceSwapConfig` alongside the realism
knobs, settable through `set_realism`'s validation pattern.

```
guard_multi_face        True      reject multi-face sources, skip multi-face frames
guard_min_source_px     110       minimum source face size
guard_min_frame_px      80        minimum runtime face size
guard_min_sharpness     …         Laplacian variance floor, needs calibration
guard_max_yaw           35        degrees
guard_min_confidence    0.5       detection score floor, above the 0.35 detect threshold
guard_min_coverage      0.4       fraction of hull unoccluded
guard_hold_ms           1000      how long to hold the last good frame
guard_outlier_sim       …         cosine floor for source consistency, needs calibration
```

`guard_multi_face` should be switchable off, since the same pipeline serves
`many_faces` deliberately.

---

## Where this lands

| Guard | File |
|---|---|
| Source guards, per-image reasons | `pipeline/services/database.py` |
| Outlier rejection | `pipeline/services/database.py::_average_faces` |
| Multi-face source detection | `pipeline/services/face_detection.py` — needs a count, not just `detect_one` |
| Runtime guards | `pipeline/processing/pipeline.py::_process_and_emit` |
| Hold-last-frame | `pipeline/processing/pipeline.py` |
| Guard events | `pipeline/events.py`, `pipeline/api/server.py` |
| Operator display | `desktop/bridge.py`, `desktop/main.qml` |
| Thresholds | `pipeline/config.py`, `pipeline/api/handlers.py` |

The `many_faces` bug is worth fixing in the same pass: with it off, the pipeline
should swap only the primary face and never push several people through one
stabilizer.

---

## Open questions

**What counts as "primary" when `many_faces` is off but guarding is disabled?**
Largest face is the usual answer and is more defensible than the current
leftmost. Closest to the previous frame's position would be steadier still, but
that is a tracker, and per-frame detection exists partly to avoid one.

**Should a guarded frame still update the temporal state?**
It should not — feeding a guarded frame into the pixel EMA would blend a
stranger, or a bad detection, into the smoothed history. The compositor's
`reset()` already exists for this.

**Is pose available cheaply?** InsightFace exposes pose on some model
configurations. If `buffalo_l` does not provide it, yaw can be approximated from
the five keypoints, which is cheaper than adding a model but needs its own
calibration.

**How aggressive should source rejection be?** Every guard that rejects a usable
photo is friction at the exact moment a new customer is deciding whether this
product works. Multi-face is unambiguous; sharpness and pose are judgement
calls, and starting permissive with a warning may convert better than starting
strict with a rejection.
