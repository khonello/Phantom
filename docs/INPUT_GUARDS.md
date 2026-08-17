# Input Guards

Rejecting inputs that would produce a wrong or unusable swap, rather than
swapping them badly.

**Status: design settled, not built.** The guard set, the hold-and-alert
policy and the clean-video rule are decided; thresholds still need
calibration against real uploads and real footage.

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

Two behaviours, verified against the code, both silent.

### The single-face rule is "leftmost", and it is not a rule

Everywhere the pipeline needs one face, it takes the one with the smallest x
coordinate:

```python
# pipeline/services/face_detection.py — detect_one()
return min(detections, key=lambda d: d.bbox.x)
```

This is not a heuristic that sometimes fails. It is an arbitrary tie-break
standing in for a decision that was never made. Nothing about how images are
composed makes the leftmost face the subject, and it selects the wrong person
whenever there is more than one candidate: a friend beside you, someone walking
behind you, a face on a poster or a television, a reflection, a photograph on
the wall.

It came in with the original scaffolding (`98892d6`, "Phase 0 & 1") and was
never revisited. Its only virtue is determinism — it is reproducibly wrong
rather than randomly wrong.

It is used in exactly two places, and it fails differently in each.

**In source images, one wrong pick poisons the whole session.**
`FaceDatabase._extract_from_image()` builds the embedding from whatever
`detect_one` returned. Upload a photo with a friend on your left and every
subsequent frame swaps in your friend. The choice is made once, silently, and
never re-examined.

**At runtime, the failure is instability rather than arbitrariness.**
`DetectionProcessor.process()` calls `detect_one` on **every frame** when
`many_faces` is off. As people move, the leftmost face can change from one frame
to the next — so the swap jumps between subjects mid-stream, and the
`LandmarkStabilizer`, which exists on the assumption of one continuous subject,
is fed alternating identities and thrashes. Nothing detects that the target
changed.

> An earlier draft of this document claimed the pipeline swapped *every*
> detected face regardless of `many_faces`. That was wrong: `DetectionProcessor`
> does restrict to one. The stabilizer thrashing is real, but it is caused by
> the selection flipping between frames, not by multiple faces being swapped.

**A batch of source images is averaged without any consistency check.**
`_average_faces()` takes the mean of every embedding it was given and normalises
it. One photo of a different person pulls the identity toward a blend of two
people, which is an identity that does not exist and will not resemble anyone.
Nothing reports this either.

### Why guessing better is not the fix

Largest face is a much better heuristic than leftmost — the subject is usually
nearest the camera — and most-central is defensible too. Both still fail, and
they fail silently in the same way.

The distinction that matters is whether there is a human available to ask:

- **Source images are an upload flow.** The operator is right there, choosing
  files. There is no reason to guess: refuse the ambiguous image and say why.
- **Runtime has nobody to ask.** A rule is unavoidable, so use **largest** as
  the primary-face rule, and guard the frame when the situation is ambiguous
  rather than resolving it by fiat.

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

**Decided: a guarded frame never updates temporal state.** Feeding one into the
pixel EMA would blend a stranger, or a bad detection, into the smoothed history
— and then leak it back out over the following frames once the guard clears.
`FaceCompositor.reset()` and `LandmarkStabilizer.reset()` both already exist for
this; a guard must call them rather than simply skipping the swap.

---

## What a guarded frame emits

**Decided: hold the last good swapped frame. Never pass the raw frame through.**

The operator is on a video call precisely because they do not want their own
face transmitted. Guarding a frame and then showing their real face turns a
safety feature into the exposure it exists to prevent.

| Option | Behaviour | Verdict |
|---|---|---|
| Pass through unswapped | Real face reaches the call | **Never.** Defeats the entire purpose |
| Hold last good swapped frame | Output appears to freeze | **Chosen** |
| Emit nothing | Client freezes on its last frame | Equivalent, but decided in the wrong place |

Holding for roughly a second covers the common transient case — someone crossing
behind the operator — without any visible interruption at all. Past that window
the condition is not transient and the operator has to be told.

There is a second reason to prefer holding: **a frozen picture reads as a
network hiccup**, which is the most innocuous way this can possibly fail in
front of other people. Every alternative is either an exposure or an
announcement.

**Guards fail closed.** If a guard cannot be evaluated — the occluder model
missing, a detection error, a threshold unset — the frame is guarded. An
un-evaluable guard is never treated as a pass.

---

## Making the guard unmissable

A warning the operator does not see is the same as no guard at all, and the
obvious placements both fail:

```
  the operator is looking HERE ─────────┐
                                        ▼
  ┌──────────────────────────┐   ┌──────────────────────┐
  │ Phantom window           │   │ Zoom / Meet / Teams  │
  │  · status line           │   │  · self-view         │
  │  · behind the call, or   │   │  · other participants│
  │    minimised             │   │                      │
  │         ✗ missable       │   │   ✗ burning a banner │
  └──────────────────────────┘   │     in here is seen  │
                                 │     by EVERYONE      │
                                 └──────────────────────┘
```

The virtual camera feed goes straight to the call (`desktop/bridge.py::_run_vcam`
sends whatever frame it is given). So the two intuitive answers are both wrong:
Phantom's own status line sits in a window that is behind the call or minimised,
and anything drawn into the frame is broadcast to every other participant —
which advertises the tool at the worst possible moment.

That gives two hard rules.

**The video output stays clean.** No banner, no border, no tint, no text is ever
composited into a frame that reaches the virtual camera. The guard state must
not be inferable by anyone on the call. A freeze is the entire outward signal,
and it should look like a dropped connection.

**Alerting reaches the operator outside Phantom's window**, through channels
that other participants cannot perceive:

| Channel | Reaches them | Leaks to the call |
|---|---|---|
| Always-on-top overlay, small, local | Yes — over the call window | No |
| Taskbar flash / OS notification | Yes | No |
| Phantom's local preview, clearly marked | Only if visible | No |
| Status line alone | **No** | No |
| Anything drawn into the frame | Yes | **Yes — never** |
| Audible alert | Yes | **Possibly** — an open mic carries it |

An always-on-top overlay is the primary channel, since it is the only one
guaranteed to be in front of the call window. Qt supports it directly
(`Qt.WindowStaysOnTopHint`). Keep it small, put the reason on it, and make it
disappear the instant the guard clears.

An audible alert is worth offering but **off by default**: an open microphone
will carry it into the call, which is precisely the announcement the clean-video
rule exists to avoid.

### Escalation

```
  t=0      guard trips           hold last good frame · no alert yet
  t≈1s     still guarded         always-on-top overlay + taskbar flash
  t≈5s     still guarded         overlay persists, reason shown, optional sound
  clear    guard releases        overlay disappears immediately, swap resumes
```

Below one second nothing is shown, because a transient guard that resolves
itself is not worth interrupting a call over. Above it, the alert stays until
the condition actually clears — it is never auto-dismissed on a timer, and it
is not a toast that disappears while the problem persists.

### The other places a guard must not be missable

- **Source rejection reports per-image reasons.** *"3 of 5 accepted —
  `group.jpg` has 2 faces, `blurry.png` is too soft"*, never a bare failure.
  This is an upload flow with the operator present; there is no excuse for
  ambiguity.
- **A disabled guard is visible.** `guard_multi_face` can be switched off, since
  the pipeline also serves `many_faces` deliberately — but a session running with
  guards off must say so on screen. A guard that is silently disabled is worse
  than one that was never built, because the operator believes they are
  protected.
- **Guard activations are counted.** A source guarded 30% of the time is a bad
  source, and the operator should be told to replace it rather than left to
  wonder why the swap keeps stalling.

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
guard_hold_ms           1000      hold last good frame before alerting
guard_alert_sound       False     off by default — an open mic carries it
guard_overlay           True      always-on-top local alert; the primary channel
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
| Always-on-top alert overlay | `desktop/main.qml` (`Qt.WindowStaysOnTopHint`) |
| Per-image rejection reasons | `desktop/bridge.py`, `desktop/main.qml` |
| Clean-video guarantee | `desktop/bridge.py::_push_to_vcam` — nothing drawn into the frame |
| Thresholds | `pipeline/config.py`, `pipeline/api/handlers.py` |

Two fixes belong in the same pass, independent of the guards themselves:
replacing `detect_one`'s leftmost rule with largest-face, and resetting the
`LandmarkStabilizer` whenever the selected face changes identity — a large
centroid jump between frames already triggers a reset, but a genuine switch
between two people standing still does not.

---

## Open questions

**Is largest-face enough at runtime, or does selection need continuity?**
Largest is a clear improvement on leftmost, but it can still flip if two people
are of similar apparent size. Preferring the face nearest the previous frame's
position would be steadier — though that is a tracker by another name, and
per-frame detection exists partly to avoid one. Guarding may make the question
moot: if two comparable faces are in shot, the frame is guarded anyway.

**Is pose available cheaply?** InsightFace exposes pose on some model
configurations. If `buffalo_l` does not provide it, yaw can be approximated from
the five keypoints, which is cheaper than adding a model but needs its own
calibration.

**How aggressive should source rejection be?** Every guard that rejects a usable
photo is friction at the exact moment a new customer is deciding whether this
product works. Multi-face is unambiguous; sharpness and pose are judgement
calls, and starting permissive with a warning may convert better than starting
strict with a rejection.
