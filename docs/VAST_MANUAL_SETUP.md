# What you have to do by hand

Everything on Vast.ai that no script in this repo can do for you, in the order
it has to happen. Roughly 20 minutes, once.

[VAST_DEPLOYMENT.md](../VAST_DEPLOYMENT.md) is the full reference for how the
deployment works. This file is only the human part: accounts, keys, money, and
the handful of ongoing duties the orchestrator cannot take over.

**None of it is needed to look first.** `python vast/orchestrator.py offers`
runs against the live API with no account and no key, and prints exactly what
`start` would rent. Run that before spending anything.

---

## 0. The one that will cost you the most if you skip it

**Save a credit card and turn on autobilling.**

Vast is prepaid, and when the balance reaches zero it stops your instances,
keeps billing storage against a negative balance, and then — **if there is no
card on file** — destroys the instances and the data on them.

That matters more here than it would on most projects. Nothing is baked into a
Docker image, so **the instance disk is the only copy** of the venv and the
model weights. Losing it is not an inconvenience; it is a full cold start,
every time, until you rent again and wait for pip and ~2 GB of weights.

With a card on file the top-up happens and nothing is destroyed. So:

- [ ] Add a card at <https://cloud.vast.ai/billing/>
- [ ] Turn on **autobilling**, threshold ≈ your average daily spend
- [ ] Set the low-balance email alert to ~75% of that threshold

A grace period exists before deletion and scales with your spending history, so
a new low-spend account has the *shortest* grace period. That is precisely when
this is easiest to forget.

---

## 1. Account and credit

- [ ] Sign up at <https://cloud.vast.ai/> and verify the email
- [ ] Add credit — **$5 minimum**. Card via Stripe, or crypto via BitPay /
      Crypto.com

Budget check, so the number is not a surprise. At the UK 4090 seen on
2026-09-03:

| | |
|---|---|
| GPU, while running | ~$0.31/hr |
| Bandwidth, `optimal` preset | ~$0.014/hr (about 4% on top) |
| Storage, 25 GB, **whether running or stopped** | ~$5–10/month |

$20 is a comfortable first load: enough for several hours of real sessions plus
a month of keeping the models warm.

---

## 2. API key

- [ ] Create a key at <https://cloud.vast.ai/manage-keys/>
- [ ] Paste it into `.env` as `VAST_API_KEY`

## 3. A scoped key for the instance

The instance needs a key of its own so it can stop itself when
`VAST_MAX_UPTIME` expires. If you leave `VAST_SCOPED_API_KEY` empty, the
orchestrator forwards `VAST_API_KEY` instead — which works, and puts an
account-wide key on a rented machine that could then destroy every other
instance you own.

This is the step that closes a risk the project carried through the whole
RunPod era, so it is worth the two minutes.

```bash
pip install vastai          # the CLI is not in requirements-orchestrator.txt
cat > perms.json <<'EOF'
{"api": {"instance_read": {}, "instance_write": {}}}
EOF
vastai create api-key --name phantom-instance --permission_file perms.json
```

- [ ] Put the resulting key in `.env` as `VAST_SCOPED_API_KEY`
- [ ] Delete `perms.json` — it is not secret, but it is clutter

## 4. SSH key

Every deploy goes over SSH: the instance is a stock CUDA image that
`vast/startup.sh` builds on, so there is no image-only path.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

- [ ] **Must be unencrypted** (`-N ""`). The orchestrator loads it without a
      passphrase prompt and will exit if it cannot
- [ ] Paste the **public** half (`.pub`) at <https://cloud.vast.ai/manage-keys/>
- [ ] Confirm `VAST_SSH_KEY_PATH` in `.env` points at the **private** half

Two things to know:

- **A key added to your account only applies to instances created afterwards.**
  If you add a key while an instance is running, that instance will not accept
  it. Add the key before your first `start`.
- Password authentication is disabled account-wide, so the key is the only way
  in. On Windows, a key generated inside WSL lives in the WSL home, not the
  Windows one:

  ```bash
  wsl -e cp ~/.ssh/id_ed25519 /mnt/c/Users/<you>/.ssh/id_ed25519
  ```

---

## 5. `.env`

- [ ] `cp .env.example .env`
- [ ] Fill in `VAST_API_KEY`, `VAST_SCOPED_API_KEY`, `VAST_SSH_KEY_PATH`
- [ ] Check `VAST_DISK`. **Set it before your first rental** — an instance disk
      is fixed at creation and cannot be grown, unlike a RunPod network volume.
      25 GB fits the venv and weights; raise it to 60+ if you intend to render
      long video, which extracts ~4 MB per 1080p frame

Everything else has a working default. **Do not hand-edit**
`VAST_INSTANCE_ID`, `PHANTOM_API_URL`, `PHANTOM_TLS_FINGERPRINT` or
`PHANTOM_API_TOKEN` — the orchestrator writes those, and a hand-edited
fingerprint will simply be refused.

## 6. Choose a host

```bash
python vast/orchestrator.py offers
```

- [ ] Pick a row and put its host id in `.env` as `VAST_PREFERRED_HOST`

Pinning a host is what gives you a stable IP and a warm disk across
`stop`/`resume`. The filtered search still runs whenever that host has nothing
free, so pinning costs nothing when it is busy.

Weigh three columns, not just the price:

- **`$/hr`** — the GPU
- **`up MB/s`** — the host's uplink. The app needs about 0.6, so anything here
  is plenty; a low number is a signal about the host, not a capacity limit
- **`$/mo st`** — standing storage while stopped. This varies by more than 3x
  between otherwise identical offers and is the number people forget

---

## 7. Operator machine

Separate from all of the above, and easy to skip because the app appears to
work without it. See [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md).

- [ ] **OBS Studio** with the virtual camera
- [ ] **VB-Audio Cable**

Both belong on **your** machine, never the instance. The instance is headless:
it receives JPEG frames, swaps, and sends them back. It has no virtual camera
and no audio path at all, because audio is never uploaded to it.

The audio one is the easiest to skip and the worst to skip. The desktop delays
your microphone to match video that arrives late; with no virtual audio output
that delayed audio goes to your *speakers* while the call still receives your
real microphone undelayed — so the delay makes the desync worse rather than
better. The app says so at startup rather than appearing to work.

---

## 8. First run

```bash
pip install -r requirements-orchestrator.txt
python vast/orchestrator.py start
python desktop.py
```

Then verify, in this order — each answers a different question:

- [ ] `start` finished and wrote four values into `.env`
- [ ] It printed **"Certificate matches the recorded fingerprint"**. If it
      warned instead, do not disable the pin — re-run `start`
- [ ] `python tools/stats.py --host <ip> --port <port>` **exits zero**. This is
      the silent-CPU-fallback check: ONNX Runtime falls back to CPU without
      erroring, and a pod on CPU bills a full GPU hour while producing unusable
      output. Non-zero here means stop and read `logs`
- [ ] The desktop connects and the viewport shows a swapped face
- [ ] **Read the latency badge, top-right.** This is the number the whole
      migration was for. Compare it against the ~350ms you had on the Romanian
      pod. Nothing in this repo has measured it yet

`status` gives you the host and port for the `stats.py` line.

---

## 9. Ongoing, and genuinely manual

Four things no script does for you.

**Destroy the orphan after a fallback resume.** If `resume` finds your host has
no GPU free, it rents a new instance and rewrites `VAST_INSTANCE_ID`. The old
one is left **stopped and still billing for its disk**, and `terminate` can no
longer reach it because `.env` now names the new one. The orchestrator prints
this when it happens.

- [ ] Delete it at <https://cloud.vast.ai/instances/>

**Decide `stop` vs `terminate` for a gap.** `stop` keeps the models warm and
costs $5–10/month. `terminate` costs nothing and makes the next session a full
cold start. Days: stop. Weeks: terminate.

**Watch the balance.** Covered by step 0, and the reason it is worth repeating
is that the consequence here is data loss rather than an outage.

**Re-run `start` after any rebuild.** The certificate is generated once and
reused, so a fingerprint mismatch means either the instance was genuinely
rebuilt or something is sitting in the middle of the connection. `start`
rewrites both values. Never work around the pin.

---

## What is *not* manual

So you do not do these twice:

| | |
|---|---|
| Choosing a GPU | `offers` filters and ranks; `start` takes the top row |
| Waiting for capacity | `start` retries the whole search every 60s for `VAST_GPU_WAIT`. Waiting is free — billing starts when an instance runs |
| Installing anything on the instance | `vast/startup.sh`: ffmpeg, venv, pip, cuDNN, model pre-warm |
| TLS certificates and the API token | Generated on the instance, pinned into `.env` |
| Finding the WebSocket address | Written to `.env`; the port is random and changes per rental |
| Stopping an idle session | The pipeline's own timer stops it at `VAST_MAX_UPTIME` (default 120 min), even with no desktop connected |
| Adding a region | One entry in `VAST_GEOLOCATIONS`. No volumes to create or seed — that was the RunPod procedure |
