# Setup checklist

What has to be installed, **on which machine**, for a call to work — and where
the app puts the files it produces.

---

## Two machines, two different jobs

Phantom runs as two processes, and they are usually not on the same computer.

| | **Operator machine** | **Pipeline machine** |
|---|---|---|
| Runs | `desktop.py` | `pipeline.py` |
| Typically | your laptop | a rented GPU pod |
| Has | webcam, microphone, the conferencing app | a GPU and the model weights |
| Sees | the person | frames over a WebSocket |

**OBS and VB-Audio Cable go on the operator machine only.**

That is worth stating plainly because it is the natural thing to get wrong. The
pipeline is headless: it receives JPEG frames, swaps the face, and sends them
back. It has no virtual camera, no audio path, no conferencing app, and no
screen. It never touches a microphone — **audio is never uploaded to the
pipeline at all** — and `pyvirtualcam` and `sounddevice` appear nowhere in its
imports or its requirements files.

Installing either driver on a pod does nothing. Forgetting them on the operator
machine breaks the call.

---

# Part 1 — The operator machine

The machine with the camera and the person in front of it. Everything in this
part is about that machine.

## 1.1 Video — OBS Studio

The desktop writes swapped frames into OBS's virtual camera device, and the
conferencing app selects that device as its webcam.

1. Install [OBS Studio](https://obsproject.com/).
2. Open it once and press **Start Virtual Camera**, then close it. This
   registers the device with the system; OBS does not need to stay running.
3. In the conferencing app, select that device as the camera. In Zoom:
   **Settings → Video → Camera → OBS Virtual Camera**. See
   [the Zoom settings table](#zoom-in-one-place) for both halves together.

The badge in the bottom-left of the viewport shows whether the device is open.

**The preview takes a few seconds to appear**, and that is the camera opening,
not the app hanging. OpenCV's Windows backend renegotiates the device format on
every property it is asked to change, at about 1.7 seconds each — measured from
`VideoCapture(0)` to the first frame at 640x360:

| | to first frame |
|---|---|
| setting resolution and frame rate unconditionally | 7.4s |
| skipping the frame-rate set, which MSMF ignores | 5.6s |
| also skipping anything already correct — what it does now | 3.9s |

All three give the same 640x360 at the same 30fps. It runs on a worker thread,
so the window and everything else are usable while it happens.

**The camera is deliberately always on** — opened when the app opens, released
only when it closes. It is not tied to a session or a mode. An open device
nobody has selected costs nothing, while a device that comes and goes is what
makes a conferencing app go looking for another one, and the next one it finds
is your real webcam. That is the exact failure this product exists to prevent,
so there is no button to turn it off.

## 1.2 Audio — a virtual cable

This is the counterpart of the virtual camera, and the reason it is needed is
not obvious.

Your swapped video arrives from the pipeline **~350–400ms late** — that is the
network round trip, not the processing. If your microphone goes straight to the
call, your voice arrives **ahead of your face**: the other person hears the word
and then sees your lips form it, like a badly dubbed film. So the desktop
captures your microphone, holds it by exactly the amount the video was held,
and releases both together.

That re-timed audio has to reach the call, which means it has to go to a device
the conferencing app can select as a microphone.

### Windows

1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (donationware).
   It creates `CABLE Input` (a playback device) and `CABLE Output` (a recording
   device).
2. Restart the desktop. It finds the cable automatically and logs:

   ```
   [AUDIO] Playing to virtual output: CABLE Input (VB-Audio Virtual Cable)
   ```

3. In the conferencing app, select the *other* end of the cable as the
   microphone. In Zoom: **Settings → Audio → Microphone → CABLE Output
   (VB-Audio Virtual Cable)**.

   Which end matters. The app **plays into `CABLE Input`**; the conferencing app
   **records from `CABLE Output`**. They are the two ends of one pipe, and
   swapping them gives a call with no audio and no error message.

4. Leave the conferencing app's **speaker** on your headphones or speakers.
   Pointing it at `CABLE Input` feeds the call's audio back into the pipe the
   desktop is already writing to, which is a loop.

Windows lists the same cable once per audio API, and the difference is large: on
one machine the same device measured **90ms on MME, 120ms on DirectSound and
2ms on WASAPI**. The app picks the lowest-latency instance automatically, so the
name alone does not identify which one is in use.

### macOS

Install [BlackHole](https://existential.audio/blackhole/) (2ch is enough) and
select it as the microphone in the conferencing app. Discovery matches it by
name the same way.

### If you skip this

The app says so plainly rather than failing quietly:

```
[AUDIO] No virtual audio output found — playing to the system default.
The call will NOT receive time-aligned audio: your microphone still reaches
it undelayed, ahead of the swapped video.
```

In that state the delayed audio goes to your **speakers** — you hear yourself
half a second late, which sounds broken — while the call still hears your real
microphone with no delay at all. **The delay makes the desync worse, not
better.** It is the one configuration that is worse than having no audio
handling at all.

### Both devices must agree on a sample rate

The app takes its rate from the **cable**, not from a constant, because the
cable is the one that cannot bend: the lowest-latency instance of it is the
WASAPI one, and WASAPI will not resample. Your microphone then has to open at
that same rate.

Almost always it does, and there is nothing to configure. When it does not, the
app says so at startup rather than producing silence:

```
[AUDIO] CABLE Input (VB-Audio Virtual Cable) wants 48000 Hz but the microphone
(Microphone Array) will not open at that rate. Audio may not start. Set both
devices to the same rate in the system sound settings.
```

The fix is in Windows, not in the app: **Sound → Recording / Playback → the
device → Properties → Advanced → Default Format.** Set the microphone and
`CABLE Input` to the same rate — 48000 Hz for both is the safe pair — and
restart the app.

It is not repaired automatically on purpose. Capture and playback share one
buffer, so running them at different rates would shift the pitch of your voice
*and* drift it further out of sync with the video every second it ran. Saying so
is better than doing that quietly. See
[ACCEPTED_RISKS.md](ACCEPTED_RISKS.md).

### Monitoring yourself

Sending your voice into a cable means you stop hearing it, which is normal — you
do not usually hear your own microphone. VB-Audio's **VoiceMeeter** can route
one input to two outputs if you want to monitor, but what you would hear is
~400ms delayed, which most people find harder to speak over than silence.

## 1.3 Python

```bash
pip install -r requirements-desktop.txt
```

PySide6, opencv-python, numpy, pyvirtualcam, sounddevice, websockets,
python-dotenv, praat-parselmouth. `sounddevice` is optional in the sense that
the app starts without it — capture and playback disable themselves and say so
— but then none of 1.2 applies.

**No face models here.** The desktop sends JPEG frames and displays what comes
back; it never loads onnxruntime, insightface or torch. `tests/test_wiring.py`
asserts both halves of that: every third-party import is declared, and no ML
runtime is.

## 1.4 Three environments, not one

This is the thing that costs an afternoon on a new machine. **No single
environment runs everything**, and the failure is always the same
`ModuleNotFoundError` from a script you did not expect to be picky.

| Environment | Runs | Needs |
|---|---|---|
| **Test / lint** (system or pyenv) | `pytest tests/`, `flake8`, `mypy` | `requirements-ci.txt` plus pytest, flake8, mypy |
| **`environ-orchestrator/`** | `runpod/orchestrator.py`, `tools/sweep_levers.py`, `tools/stats.py`, `tools/realism.py` | `requirements-orchestrator.txt` |
| **Desktop venv** | `desktop.py`, `tools/build_desktop.py` | `requirements-desktop.txt` |

Two things worth knowing before you copy the layout:

- **The test environment deliberately has no ML libraries.** `tests/conftest.py`
  stubs insightface, onnxruntime, torch and tensorflow, which is what lets the
  suite run in ~40s on any machine instead of needing a GPU box. It is not an
  incomplete install.
- **The desktop venv on this machine is named `Python_3_12_0venv` and is
  actually Python 3.10.0.** Qt Creator named it; nothing reads the name. Do not
  install 3.12 on a new machine because a directory says so.

They can be one environment if you would rather — nothing prevents installing
all three requirement sets together. The split here is history, not design.

## 1.5 What is NOT installable from Python

`ffmpeg` must be on `PATH` for RENDER mode (this machine has 7.1). OBS and the
audio cable are the other two, covered above. Everything else is pip.

---

# Part 2 — The pipeline machine

**Nothing from Part 1 belongs here.** No OBS, no VB-Cable, no webcam, no
microphone.

## 2.1 On a rented pod

`runpod/orchestrator.py start` provisions everything — CUDA image, venv,
dependencies, model weights on the network volume. See
[RUNPOD_DEPLOYMENT.md](../RUNPOD_DEPLOYMENT.md), which is the authority for that
side. You install nothing by hand.

## 2.2 On your own GPU machine

```bash
pip install -r requirements-pipeline-gpu.txt    # or -cpu without CUDA
python pipeline.py --execution-provider cuda
```

Plus **FFmpeg** on `PATH`, required for video encode and decode in RENDER mode.

Model weights download on first use into `pipeline/models/`, or
`/workspace/models/` when that exists. Confirm what actually loaded:

```bash
python tools/stats.py --host <ip> --port <port>
```

That exits non-zero if an accelerator was requested and is not available — the
silent CPU fallback, which bills a GPU hour and produces nothing usable.

---

# Part 3 — Both on one machine

Supported, and the simplest way to develop. Run `pipeline.py` and `desktop.py`
side by side and point the desktop at `localhost`.

You still need OBS and the virtual cable, because you are still the operator —
they are needed by the *desktop*, and the desktop is here. What changes is that
the network round trip disappears, so the playout delay is mostly wasted; set
`DEFAULT_PLAYOUT_DELAY_NS` lower (or 0 for adaptive) if the lag is distracting.

Renders are **not** downloaded in this configuration — the file is already at
the output path, and copying it beside itself would be noise.

---

# Part 4 — Moving to a new machine

### A new operator machine

1. **Python 3.10** — the version everything here is built and tested against
2. **Git**, and clone the repository
3. **FFmpeg** on `PATH` (RENDER mode decodes and encodes with it)
4. **OBS Studio**, virtual camera started once (1.1)
5. **VB-Audio Cable** or **BlackHole** (1.2)
6. `pip install -r requirements-desktop.txt` in the desktop environment
7. `pip install -r requirements-orchestrator.txt` if this machine will also
   manage pods or run the measurement tools
8. Conferencing app: camera → OBS Virtual Camera, microphone → CABLE Output
   (speaker stays on your headphones — pointing it at `CABLE Input` is a loop)
9. Microphone and `CABLE Input` set to the **same sample rate** (1.2). The app
   reports a mismatch at startup and will not start audio
10. **Rehearse it** — Part 6 — before the first real call. A local recording is
    the only check that shows video and audio together, the way a participant
    gets them
9. Copy `.env` across — **it is gitignored**, so it does not arrive with a
   `git clone`, and it holds the API key and `RUNPOD_POD_ID`

For running the test suite as well, add `requirements-ci.txt` plus `pytest`,
`flake8` and `mypy` — see 1.4 for why that is a separate environment.

### A new pipeline machine

1. `requirements-pipeline-gpu.txt` and FFmpeg (2.2)
2. Nothing from Part 1
3. `python tools/stats.py` to confirm the GPU and both models

`.env` deserves its own warning on either machine: it is read by the
orchestrator at **pod creation time only**, so editing it never affects a pod
that already exists. On a running pipeline use `tools/realism.py`. See
RUNPOD_DEPLOYMENT.md, "Changing settings on a pod that already exists".

---

# Part 5 — Where the output files go

| Mode | Written to |
|---|---|
| **LIVE** | Nothing. Frames go to the virtual camera and are gone |
| **VIDEO → RENDER** | Beside the video you selected, `_swapped` suffix |
| **IMAGE → UPLOAD** | Beside the photo you picked, `_swapped` suffix |
| **IMAGE → TEMPLATES** | `Pictures/Phantom/` |

A render happens on the *pipeline's* filesystem, which on a pod is another
machine, so the desktop reads it back in chunks and saves it locally:

```
C:\Videos\interview.mp4          <- the target you picked
C:\Videos\interview_swapped.mp4  <- the result
```

The status line names the full path when it lands (`saved to …`), the panel
shows the filename with the full path on hover, and **OPEN OUTPUT** opens the
containing folder.

Template results go to `Pictures/Phantom/` rather than beside the template,
because a template's target is a shared asset and writing next to it would leave
your face there for the next job.

---

# Part 6 — Checking it all works

Ask the pipeline what it is running:

```bash
python tools/stats.py --host <ip> --port <port>
```

GPU, both models, whether restoration is on, requested vs available execution
providers, and how long before the pod stops itself.

On the desktop, the viewport shows:

- **top-right** — playout delay, RTT, uplink Mbps, frames held
- **bottom-left** — virtual camera state
- **bottom-centre** — detection, and why the swap is paused if it is

And the console carries, every two seconds:

```
[SYNC] playout delay calibrated to 575ms (rtt p50 348 / p95 383 over 60 samples, was 550ms)
[SYNC] delay=575ms rtt=348/383ms buf=1 held=0 up=6.0Mbps/30fps
[SYNC] av video=+591ms audio=+578ms skew=+13ms (aligned)
[SYNC] audio buf=210ms underruns=0 trims=0 resyncs=0 drift=2.1ms out=CABLE Input (VB-Audio Virtual Cable) (+2.0ms)
```

`skew` is the one to read. It is how far apart the two streams' *presented*
material is: negative means the sound is behind the picture. Within one frame
interval (50ms at `optimal`) is discrete frames and not a fault; a steady
several hundred means video and audio are not being held to the same delay.

Two lines to check after setting up a new machine:

- **`out=`** — anything other than a virtual device means the call is not
  getting time-aligned audio.
- **`held=`** — climbing steadily means the playout delay is too tight for the
  link. It raises itself and says so, but a high number is the signal that the
  network, not the app, is the problem.

---

## Zoom in one place

Everything the conferencing app needs, both halves together. The wording is
Zoom's; every app has the same three settings under different names.

| Zoom setting | Set it to | Why |
|---|---|---|
| **Settings → Video → Camera** | `OBS Virtual Camera` | Where the swapped video comes out (1.1) |
| **Settings → Audio → Microphone** | `CABLE Output (VB-Audio Virtual Cable)` | Where your delayed voice comes out (1.2) |
| **Settings → Audio → Speaker** | your headphones — **not** `CABLE Input` | Pointing it at the cable loops the call's audio back into the pipe |

Two things about the microphone row that catch people:

- **`CABLE Output`, not `CABLE Input`.** The desktop plays *into* `CABLE
  Input`; Zoom records *from* `CABLE Output`. Two ends of one pipe. Choosing
  the wrong one gives a call with no audio and nothing reported.
- **Both devices must agree on a sample rate.** The app takes its rate from the
  cable and needs the microphone to open at the same one. It says so at startup
  if they disagree — see 1.2.

Optional, and only if audio sounds processed: Zoom applies its own noise
suppression to whatever it is given, and it is being given an already-clean
recording. **Settings → Audio → Audio Profile → Original sound for musicians**
turns that off. Try it if voices sound gated or thin; leave it alone otherwise.

---

## Rehearsing before a real call

You can prove the whole chain alone, without a second person. Do this once on
every new machine — it takes about two minutes and it is the only way to see
what a participant actually gets.

**1. Point the conferencing app at the two virtual devices** — the three
settings in [the table above](#zoom-in-one-place).

**2. Check video in the self-preview.** With Phantom running and a source set,
the Zoom video preview shows the swapped face. That is the whole video path
confirmed — camera, pod, virtual camera — with nobody else present.

**3. Check audio on the input meter.** Still in **Settings → Audio**, speak and
watch the input level bar for `CABLE Output`.

It should move — **about half a second after you speak**, on a remote pod. That
lag is the expected signature, not a fault: audio is deliberately held to the
calibrated playout delay so it lands with video that took a round trip to the
pod. The exact figure is in the `[SYNC] playout delay calibrated to` line, and
it is smaller on a closer link. A meter that responds *instantly* is the thing to worry about, since
it means Zoom is on your real microphone and the call will hear you ahead of
your own face.

A meter that never moves means the chain is broken upstream — check the
`[AUDIO]` lines in the console first, and the `out=` line described above.

**4. Record yourself and watch it back.** Start a meeting alone (**New
Meeting**), record locally, talk for thirty seconds, stop, and play the file.

This is the only check that shows video and audio *together*, as a participant
receives them, and it is what catches lip-sync being off — which neither the
preview nor the level meter can tell you, because each shows one stream in
isolation.

### You will not hear yourself, and that is correct

Your voice goes into the cable, so it stops coming out of your speakers. That is
normal — you do not usually hear your own microphone.

Do not solve this by monitoring the delayed audio. What you would hear is your
own voice ~550ms late, and speaking over a delayed copy of yourself is markedly
harder than speaking into silence; broadcasters treat it as a fault condition.
If you want to confirm the audio sounds right, use the recording in step 4
rather than live monitoring.
