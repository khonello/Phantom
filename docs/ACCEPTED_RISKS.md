# Accepted Risks

Things that are wrong, or could go wrong, that we have decided to live with **for
now** — with the reason, and what would retire each one.

A risk belongs here once it has been *decided* rather than merely noticed. The
value of the list is that it separates two things a reader cannot otherwise
tell apart: a gap nobody has thought about, and a gap someone weighed and
accepted. The second kind must not keep being rediscovered as though it were
the first.

Each entry says what would close it. An entry with no exit condition is not an
accepted risk, it is an unfixed bug.

**Status key**

| | Meaning |
|---|---|
| 🔴 | Blocks a paying customer. Must close before anyone outside the team uses this |
| 🟠 | Fine for an assessment session, not for production |
| 🟡 | Live with it; revisit if it bites |

---

## Security

### 🔴 The WebSocket API has no authentication

`WebSocketAPIServer` binds `0.0.0.0:9000` and RunPod exposes it at
`wss://{pod_id}-9000.proxy.runpod.net/ws`. There is no token, no handshake, no
origin check. Anyone holding that URL can upload a source face, start a stream,
call `cleanup_session`, or `shutdown` the pipeline.

Worse than control: **frames are broadcast to every connected client**, so a
second connection receives the operator's swapped video.

**Why accepted:** the proxy URL is pod-specific and unguessable in practice, and
an assessment session is short and watched. Treat the URL as a credential.

**Note what does *not* mitigate this.** The access-code gate lives in
`desktop/bridge.py` — it is client-side. It gates the UI, not the pod. A
customer with a valid code and a customer with none reach the same
unauthenticated socket.

**Closes when:** the WebSocket sits behind an authenticated reverse proxy, or
the server requires a token in the first frame and drops connections that do
not present one. Required before the first paying user.

### 🟠 `RUNPOD_API_KEY` is forwarded into the pod, unscoped

The pod needs it to stop itself when `RUNPOD_MAX_UPTIME` expires. RunPod API
keys are account-wide, so a pod that can stop itself can stop, start, or
terminate **every other pod on the account**.

**Why accepted:** self-stopping is what prevents an unattended pod billing a
full day, which is a larger and more likely loss than the key being extracted
from a pod only we can reach.

**Closes when:** RunPod offers scoped keys, or the auto-stop moves to something
outside the pod that watches it and stops it from the account side.

### 🟠 An SSH key registered on RunPod grants root on the pod

`RUNPOD_SSH_KEY_PATH` must be an unencrypted key, because the orchestrator
loads it without a passphrase prompt.

**Why accepted:** SSH mode is a development convenience. Docker mode does not
need it.

**Closes when:** deployment is docker-mode only, or the orchestrator learns to
prompt.

### 🟠 The repo token persists on the network volume

A private repo is cloned with the token in the URL
(`RUNPOD_REPO_URL=https://<token>@github.com/khonello/Phantom.git`), so git
writes that URL into `/workspace/Phantom/.git/config` — on the **network
volume**, which survives `stop` and `terminate` and is reused by every future
pod. `startup.sh` needs it there: `git pull --ff-only` runs on every launch and
would otherwise prompt.

**Why accepted:** it sits inside the same trust boundary as the forwarded
`RUNPOD_API_KEY` — anyone who can read it already has root on the pod. Keeping
the repo private is still the right call; this is the cost of that, not an
argument against it.

**Closes when:** the token is a fine-grained PAT scoped to this one repository
with read-only contents access, so extracting it grants nothing else. Worth
doing now — it is a GitHub settings change, not code.

### 🟡 Uploads share one directory on the pod

`_UPLOAD_DIR = '/tmp/phantom_uploads'`. Target photos get a per-job
`mkdtemp` subdirectory — added because two photos with the same camera filename
would otherwise overwrite each other — but **sources do not**, so an uploaded
source overwrites any earlier one of the same name.

**Why accepted:** one pod serves one operator at a time. It becomes real the
moment that is not true.

**Closes when:** the pod serves more than one session, at which point uploads
need per-session isolation and a lifetime.

### 🟡 Source uploads have no size cap

Target photos are capped at `MAX_PHOTO_BYTES` (6 MB) and refused above it.
Sources are read, base64-encoded and sent whole, with no ceiling — a 40 MB
image becomes a ~53 MB WebSocket frame. The server's `max_size` is 64 MB, so
the failure mode is a dropped connection rather than a rejection with a reason.

**Why accepted:** a face photo that large is unusual, and the guards refuse
most unusable sources for better reasons first.

**Closes when:** `upload_source` gets the same cap-and-re-encode ladder
`_encode_photo` already implements for targets.

---

## Correctness and calibration

### 🟠 Guard thresholds were chosen without data

Nine thresholds, none calibrated. Three can make things actively *worse* if
mis-set and none of the three is visible from watching the output:
`guard_min_coverage`, `guard_identity_sim`, `guard_min_confidence`.
`guard_min_sharpness` and `guard_outlier_sim` need upload data rather than
footage.

**Why accepted:** all start permissive, and the measurement for closing this is
built — `--guard-observe` with `--guard-report` records what every guard *would*
have done without any of them acting.

**Closes when:** a pod session produces reports across the three presets and the
thresholds are set from the distributions. See
[PENDING_WORK.md](PENDING_WORK.md) §2.4 and [INPUT_GUARDS.md](INPUT_GUARDS.md).

### 🟠 Batch video has never run with real models

The FFmpeg plumbing, frame ordering, audio sync, cancellation, cleanup and the
multi-face abort are all verified — against a **stubbed** swapper. The ML layer
is stubbed in every test (`tests/conftest.py`), so no swap in the render path
has ever produced a real face.

**Why accepted:** the stub proves the parts that break silently — ordering,
sync, cleanup. The swap itself is shared with the live path, which has run.

**Closes when:** one real render on the pod, checked for audio drift and frame
ordering at the end of the clip rather than the start.

### 🟡 An animated webp is accepted as its first frame, silently

OpenCV decodes frame one and the guards judge that frame, which is a sound
result — the first frame is a real still of the person. Nothing tells the
operator the animation was flattened.

`.gif` and `.heic` are refused instead, and the difference is not arbitrary:
OpenCV cannot decode either at all, so accepting one guarantees a wasted round
trip. `tests/test_wiring.py` round-trips every accepted extension and asserts a
well-formed gif still fails, so if a future OpenCV gains a decoder the check
flips and says gif can be *added* rather than refused.

**Closes when:** the desktop probes frame count at selection and says
"portrait.webp is animated — using the first frame".

### 🟡 A target photo over 6 MB is re-encoded to JPEG under its original name

`_encode_photo` re-encodes only when a photo exceeds the cap, and keeps the
original filename — so an oversized `.webp` is staged on the pod as JPEG bytes
in a file called `.webp`. Nothing decodes by extension (`cv2.imread` sniffs
content) and the output is a genuine webp because `imwrite` encodes by the
output path, so it works end to end.

**Why accepted:** it is cosmetic, confined to a temp file, and pre-existing —
a 10 MB PNG behaves the same way.

**Closes when:** the re-encode renames to match what it produced.

### 🟡 Nothing locks the detector against concurrent use

`face_boxes` runs detection on the API thread while the stream thread may be
detecting on frames. Today this cannot happen through the UI — `setMode` stops
the pipeline before switching to the IMAGE tab — but that is the *desktop*
preventing it, not the pipeline.

**Why accepted:** one operator, one client, and the UI closes the path.

**Closes when:** more than one client can drive a pod, at which point
`FaceDetector` needs a lock around inference rather than only around
initialisation.

---

## Product surface

### 🟠 No templates are bundled

The machinery runs against an empty library, so the TEMPLATES tab shows "no
scenes available". `tools/validate_templates.py` runs the real guards over the
library and exits non-zero, so a scene that would be refused cannot ship.

**Why accepted:** the assets are a content decision — including where they come
from and how they are licensed for this use — and that, not the code, is the
cost.

**Closes when:** scenes are sourced and licensed.

### 🟡 Video targets still have no transfer path

`set_target` validates with `os.path.exists` against the *pipeline's*
filesystem, so a desktop-chosen file only resolves when the pipeline runs
locally. Photos sidestep it by uploading inline; a 2 GB video cannot.

**Why accepted:** photo and template jobs are the paths being assessed first.

**Closes when:** chunked, resumable, progress-reporting transfer exists in both
directions. See [TODO.md](TODO.md).

### 🟡 `many_faces` and `keep_frames` are CLI-only, on purpose

Both were declared in `COMMANDS` with no handler, so calling either returned
`Unknown command`. Rather than implement them: `many_faces` bypasses every
runtime guard and both temporal EMAs, and `keep_frames` is a debugging flag
that fills a pod's disk at roughly 4 MB per 1080p frame.

**Why accepted:** neither is a choice a consumer should be making, and the CLI
already offers both.

**Closes when:** never, unless a case appears. `tests/test_wiring.py` pins the
absence with the reason so it does not read as an oversight.

---

## Reading this list

Two entries are 🔴 or would become so under load: **the unauthenticated
WebSocket** is the one that must close before a paying customer, and **shared
upload directories** become real the moment one pod serves more than one
session. Everything else is either waiting on the pod session that will
calibrate it, or genuinely small.

When an entry closes, delete it rather than marking it done — this file is
about what is *currently* accepted, and a list of resolved items buries the
live ones.
