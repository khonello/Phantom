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

### 🟠 The pod's git remote is whatever it was cloned with, forever

`git clone` writes the URL it was given into `/workspace/Phantom/.git/config`,
on the **network volume** — which survives `stop` and `terminate` and is reused
by every future pod. Nothing ever revisits it. If that URL is wrong, or stops
resolving, every pod from then on inherits the problem and a fresh GPU changes
nothing.

**The failure mode is a hang, not an error**, which is what makes it expensive.
When GitHub will not serve a repository anonymously — private, renamed, deleted,
or simply mistyped — it answers 401 rather than 404, so git falls back to
*asking* for a username. There is no terminal to ask on, so it blocks on stdin.
The boot stops dead under `Pulling latest changes...` and the orchestrator
prints `[still running...]` every 30 seconds until the 1800s command timeout.

`startup.sh` now sets `GIT_TERMINAL_PROMPT=0` and bounds the pull with
`timeout 120`, so this fails in seconds with
`could not read Username ... terminal prompts disabled` instead of hanging for
half an hour. Note the ordering trap: that fix reaches the pod *through the pull
it fixes*, so the first recovery is always manual.

**Diagnosis** — one command, and it is the only one that matters:

```bash
python runpod/orchestrator.py run "git -C /workspace/Phantom remote get-url origin"
```

**Recovery.** Retry first — it is transient, and `startup.sh` now retries three
times over ~20s on its own. If it persists, authenticate: an authenticated pull
is not subject to the anonymous rate limit, whether or not the repository needs
a token to be *read*.

```bash
python runpod/orchestrator.py run "git -C /workspace/Phantom remote set-url origin https://<token>@github.com/khonello/Phantom.git"
```

Set `RUNPOD_REPO_URL` in `.env` to the same value, or any pod that re-clones
goes straight back to being anonymous. Both `start` and `resume` run the clone
step, so this is not only a first-deploy concern. The token then lives in
`.git/config` on the volume for every future pod to read — the trade recorded
below.

Deleting the checkout and redeploying also works, and picks up whatever
`RUNPOD_REPO_URL` currently says, since the clone step only runs when the
directory is absent:

```bash
python runpod/orchestrator.py run "rm -rf /workspace/Phantom"
python runpod/orchestrator.py resume
```

The venv, models and templates live elsewhere on the volume, so that costs one
clone and nothing else.

**Why accepted:** a clone URL is set once and is not usually interesting. What
was not acceptable was the failure being silent and unbounded, and that part is
fixed rather than accepted.

**Closes when:** the deploy verifies the remote resolves before depending on it,
rather than discovering it inside a pull that cannot report.

### 🟡 A private repo would put a token on the network volume

Not the current configuration — `RUNPOD_REPO_URL` carries no token and the
repository is public — but it is the documented way to use a private one, so the
cost is worth stating before someone reaches for it. `git clone` would write
`https://<token>@github.com/...` into `.git/config` on the volume, which
survives `terminate` and is readable by every future pod.

**Why accepted:** it would sit inside the same trust boundary as the forwarded
`RUNPOD_API_KEY` — anyone who can read it already has root on the pod.

**Closes when:** if the repo is ever made private, the token used is a
fine-grained PAT scoped to that one repository with read-only contents access,
so extracting it grants nothing else. Note those expire, which turns into the
hang described above.

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
  arrive, so the playout cursor walks steadily away from the calibrated target
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
difference this would have to correct; nothing acts on it.

That last clause used to be untrue in a way that cost latency.
`AudioPlayback._output_callback` *did* act on it — `playback_point = now -
target + drift_ns` — which is a half-correction that cannot work: the read
cursor is continuous, so the term only moved the seek point, and seeking is not
resampling. What it did instead was fold the input device's open latency into
the position permanently, since the drift baseline was taken before the stream
delivered its first block. Both are fixed; drift is measured and reported and
nothing reads it back into the timebase.

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
