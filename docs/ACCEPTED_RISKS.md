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

**What it does when the token goes stale**, which has now happened once: the
pull stops being able to authenticate and git falls back to *asking*. There is
no terminal to ask on, so it blocks on stdin rather than failing, and the boot
stops dead under `Pulling latest changes...` — the orchestrator prints
`[still running...]` every 30 seconds until the 1800s command timeout. A fresh
GPU changes nothing, because the checkout and its stored URL are on the volume,
not the pod.

`startup.sh` now sets `GIT_TERMINAL_PROMPT=0` and bounds the pull with
`timeout 120`, so a dead token fails in seconds and says so instead of hanging
for half an hour. Note the ordering trap that creates: the fix lives in
`startup.sh`, which reaches the pod *through the pull it is fixing*, so the
first recovery has to be manual.

**Recovery:** delete the checkout and let it re-clone. The clone URL is read
from the orchestrator's own `.env` at deploy time and the clone step only runs
when the directory is absent, so a fresh token in `.env` is picked up with no
surgery on the pod's git config:

```bash
python runpod/orchestrator.py run "rm -rf /workspace/Phantom"
python runpod/orchestrator.py start
```

The venv, models and templates live elsewhere on the volume
(`/workspace/venv`, `/workspace/models`, `/workspace/templates`), so this costs
one clone and nothing else.

**Closes when:** the token is a fine-grained PAT scoped to this one repository
with read-only contents access, so extracting it grants nothing else. Worth
doing now — it is a GitHub settings change, not code. Note that fine-grained
PATs **expire**, so whatever is chosen, the failure above is the one to expect
and the reason it now fails loudly.

### 🟡 Uploads share one directory on the pod

Sources and targets each get a per-upload `mkdtemp` subdirectory now, and names
are made unique within it, so two photos with the same camera filename no longer
overwrite each other. What remains is that **every session shares one root**
(`<temp>/uploads`), and `cleanup_session` empties the whole thing.

**Why accepted:** one pod serves one operator at a time, so there is nothing to
isolate from. It becomes real the moment that is not true — at which point the
shared root also means one operator's cleanup deletes another's uploads.

**Closes when:** the pod serves more than one session, at which point uploads
need per-session isolation and a lifetime of their own.

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

### 🟡 A microphone and a virtual cable fixed at different rates cannot both be used

The audio path takes its sample rate from the **output** device, because that is
the one with no alternative: `find_virtual_output` picks the lowest-latency
instance of the cable, which on Windows is the WASAPI one, and WASAPI in shared
mode will not resample. The microphone is the system's and usually opens at
whatever it is asked for. When it will not — a mic locked to 44.1 kHz against a
cable locked to 48 — the desktop says so at startup, naming both devices, both
rates and the fix, and audio does not start.

It is not repaired, and that is deliberate, because the two ends share one ring
buffer. Filling it at one rate and draining it at another produces two faults
that compound:

- **A pitch shift.** Reading 48000 samples a second out of audio recorded at
  44100 plays everything about 8.8% fast — roughly a tone and a half up. Ugly,
  but at least stable.
- **A drift with no bound.** It consumes ~3,900 more samples per second than
  arrive, so the playout cursor walks steadily away from the fixed 550ms target
  and audio separates from video without limit. The fixed-delay design has no
  mechanism to pull it back; not having one is the point of it being fixed.

**Why accepted:** both devices are settable to a common rate from the Windows
sound control panel in about fifteen seconds, and the message says so. The
alternative is a resampler on the sync-critical audio callback, which has to be
cheap enough for that thread *and* rate-tracking, since two clocks drift against
each other even when they agree nominally. That is real machinery to carry
speculatively, on the one path where a mistake is continuously audible.

**Closes when:** a machine turns up whose two devices genuinely cannot be set to
the same rate. `AudioCapture._drift_samples` already measures the clock
difference this would have to correct; nothing acts on it yet.

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
