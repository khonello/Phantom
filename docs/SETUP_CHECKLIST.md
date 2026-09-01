# Setup checklist

What has to be installed on the **operator's machine** for a call to work, and
where the app puts the files it produces.

Everything here is about the local machine. The GPU pod needs none of it —
see [RUNPOD_DEPLOYMENT.md](../RUNPOD_DEPLOYMENT.md) for that side.

---

## The short version

| | Software | Why | Without it |
|---|---|---|---|
| **Video** | **OBS Studio** | Provides the virtual camera the call sees | No swapped video reaches the call |
| **Audio** | **VB-Audio Virtual Cable** (Windows)<br>**BlackHole** (macOS) | Provides the virtual microphone the call hears | Your real voice reaches the call **undelayed**, ahead of your face |

Both are third-party and both are drivers. Nothing in this application can
create either one — a virtual camera and a virtual microphone are kernel-level
devices, so they are installed once, per machine.

---

## 1. Video — OBS Studio

The desktop writes swapped frames into OBS's virtual camera device, and the
conferencing app selects that device as its webcam.

1. Install [OBS Studio](https://obsproject.com/).
2. Open it once and press **Start Virtual Camera**, then close it. This
   registers the device with the system; OBS does not need to stay running
   afterwards.
3. In the conferencing app, set **Camera → OBS Virtual Camera**.

The badge in the bottom-left of the viewport shows whether the device is open.

**The camera is deliberately always on** — opened when the app opens, released
only when it closes. It is not tied to a session or a mode. An open device
nobody has selected costs nothing, while a device that comes and goes is what
makes a conferencing app go looking for another one, and the next one it finds
is your real webcam. That is the exact failure this product exists to prevent,
so there is no button to turn it off.

---

## 2. Audio — a virtual cable

This is the counterpart of the virtual camera, and the reason it is needed is
not obvious.

Your swapped video arrives from the pod **~350–400ms late** — that is the
network round trip, not the processing. If your microphone goes straight to the
call, your voice arrives **half a second ahead of your face**. So the desktop
captures your microphone, delays it by exactly the amount the video was
delayed, and plays it out again.

That re-timed audio has to reach the call, which means it has to go to a device
the conferencing app can select as a microphone.

### Windows

1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (donationware).
   It creates two devices: `CABLE Input` (a playback device) and `CABLE Output`
   (a recording device).
2. Restart the Phantom desktop. It finds `CABLE Input` automatically and logs:

   ```
   [AUDIO] Playing to virtual output: CABLE Input (VB-Audio Virtual Cable)
   ```

3. In the conferencing app, set **Microphone → CABLE Output**.

Windows lists the same cable once per audio API, and the difference matters:
on one machine the same device measured **90ms on MME, 120ms on DirectSound
and 2ms on WASAPI**. The app picks the lowest-latency instance automatically,
so the name in the log is not enough to tell them apart — the latency it
reports is.

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
half a second late, which sounds broken, while the call hears your real
microphone with no delay at all. **The delay makes the desync worse, not
better.** It is the one configuration that is worse than having no audio
handling at all.

### Monitoring yourself

Sending your voice into a cable means you no longer hear it, which is normal —
you do not usually hear your own microphone. If you want to monitor the delayed
audio, VB-Audio's **VoiceMeeter** can route one input to two outputs. Note that
what you would hear is delayed by ~400ms, which most people find harder to
speak over than silence.

---

## 3. Where the output files go

### LIVE

Nothing is written. Frames go to the virtual camera and are gone.

### VIDEO → RENDER

The render happens on the **pipeline's** filesystem, which on a pod is another
machine. The desktop reads the finished file back in chunks and saves it
**beside the video you selected**, with a `_swapped` suffix:

```
C:\Videos\interview.mp4          <- the target you picked
C:\Videos\interview_swapped.mp4  <- the result
```

The status line names the full path when it lands (`saved to …`), the panel
shows the filename with the full path on hover, and **OPEN OUTPUT** opens the
containing folder.

If the pipeline is running locally rather than on a pod, nothing is downloaded
— the file is already at the output path, because copying it beside itself
would be noise.

### IMAGE → UPLOAD and TEMPLATES

Swapped photos are returned inline over the socket and written **beside the
photo you picked**, with the same `_swapped` suffix. Template results go to
`Pictures/Phantom/`, because a template's target is a shared asset and writing
next to it would leave your face there for the next job.

---

## 4. Checking it is all working

With the pipeline running, ask it what it has:

```bash
python tools/stats.py --host <ip> --port <port>
```

That reports the GPU, both models, whether restoration is on, and whether the
models are on CUDA or quietly on CPU.

On the desktop side, the viewport shows:

- **top-right** — round-trip latency, buffer depth, uplink Mbps
- **bottom-left** — virtual camera state
- **bottom-centre** — detection, and why the swap is paused if it is

And the console carries a line every two seconds:

```
[SYNC] delay=396ms rtt=348/383ms buf=1 up=6.0Mbps/30fps
[SYNC] audio buf=210ms underruns=0 trims=0 resyncs=0 out=CABLE Input (VB-Audio)
```

`out=` is the one to check after setting up the cable. If it says anything
other than a virtual device, the call is not getting time-aligned audio.

---

## 5. Python dependencies

```bash
pip install -r requirements-pipeline-cpu.txt   # or -gpu on a CUDA machine
```

The desktop additionally needs `PySide6`, `pyvirtualcam`, `sounddevice` and
`websockets`. `sounddevice` is optional in the sense that the app starts
without it — audio capture and playback disable themselves and say so — but
then none of section 2 applies.
