# Access codes

One code buys one hour. You sell it however you like, the customer types it
into the desktop once, and the hour runs from the moment the pipeline is
actually up.

---

## The rule that shapes everything

**A code is spent when the pipeline is running, not when it is typed.**

The desktop holds the code from entry until the WebSocket is connected and the
pipeline reports running, and only then calls `/redeem`. A pod that fails to
provision, a GPU that never appears, a connection that never completes — none
of those cost the customer their code. It is still unused and they can try
again.

Everything else follows from that: the two-step `Redemption` in `desktop/auth.py`
exists for this reason, and `/redeem` is idempotent for the same machine inside
its own live hour so a dropped reply cannot spend an hour twice.

## The other rule

**The hour belongs to the machine, not to the app session.**

The desktop calls `/session` on every launch. If the machine has time left, the
auth screen never appears — the customer closes the app, reopens it, and carries
on. The answer lives in Firestore keyed by machine id, not in a local file,
because a local file would mean a reinstall costs them an hour they paid for.

---

## The flow

```
  you                                   customer
  ───                                   ────────
  mint codes ──────────────────────────► gets a code, however you sell it
  (tools/mint_codes.py)                        │
                                               ▼
                                         opens desktop.py
                                               │
                                    ┌──────────┴──────────┐
                                    │  POST /session      │
                                    │  {machine_id}       │
                                    └──────────┬──────────┘
                                               │
                       active: true ◄──────────┴──────────► active: false
                              │                                    │
                              ▼                                    ▼
                     straight in, no prompt              auth screen, types code
                     "34 minutes left"                            │
                                                    checksum checked locally
                                                                  │
                                                       pipeline connects
                                                       and reports running
                                                                  │
                                                       ┌──────────┴──────────┐
                                                       │  POST /redeem       │
                                                       │  {code, machine_id} │
                                                       └──────────┬──────────┘
                                                                  │
                                                       code burned, hour starts
```

---

## Pieces

| File | What it is |
|---|---|
| `firebase/functions/main.py` | The two endpoints. The only thing that touches Firestore |
| `firebase/firestore.rules` | Deny-all. No client reaches the database |
| `desktop/codes.py` | Code format, shared by minting and the desktop |
| `desktop/auth.py` | Machine id, `/session` and `/redeem` calls, the held-code logic |
| `tools/mint_codes.py` | Operator tool. Mints codes into Firestore |

## Data

Two collections, both trivial.

```
codes/{CODE}
  used          false -> true, once
  machine_id    null until burned
  duration_min  60
  started_at    server time at burn
  expires_at    started_at + duration_min

sessions/{machine_id}
  code
  started_at
  expires_at
```

`sessions` is a lookup index — `/session` is a single document read by id, no
query and no composite index. The code documents are the ledger.

Codes are stored under their own text rather than hashed, so the Firestore
console doubles as your inventory: you can see what is unsold and what is
spent. The only readers are you and the functions, which is what makes that
safe here.

## The code format

Ten characters of Crockford base32, shown as `XXXXX-XXXXX`:

```
ZQSHJ-RV6N0      8RKYJ-9WCMA      GXJJG-XWH0V
TW075-7Z18H      260F7-KY6MH      0AN6P-GWE7X
```

Crockford because these get read aloud — its alphabet has no `I`, `L`, `O` or
`U`, so there is no "one or ell" over a phone line, and input is folded the
other way too: a customer who types `O` for `0` is understood, not rejected.
Case and hyphens are ignored.

Eight characters are payload (40 bits, about a trillion codes) and the last two
are a checksum. **The checksum is what makes a typo local** — a mistyped code is
rejected by the desktop with no round trip, so it never counts as a failed
redemption and never reaches the server. Verified: every single-character
substitution and every adjacent transposition is caught, and a random string
passes about 1 in 877 times.

> The first version used a single check character modulo 32, which is
> degenerate against a base-32 payload — `value % 32` *is* the last character,
> so the check digit was a copy of it and caught nothing. The modulus has to
> exceed 32 and be coprime to it; 1021 is prime and fits in two symbols.

---

## Setup

Once, and only the person selling codes does it.

### 1 — Firebase project

[console.firebase.google.com](https://console.firebase.google.com) → **Add
project**. Then **Build → Firestore Database → Create database**, production
mode, any region.

Cloud Functions require the **Blaze** (pay-as-you-go) plan, so a card has to be
on file. The free allowance is 2M invocations a month against your handful, so
the bill is $0 — but the plan upgrade is a real step and it will stop you if you
skip it.

### 2 — Deploy

```bash
npm install -g firebase-tools
firebase login
cd firebase
firebase use --add          # pick the project
firebase deploy --only firestore:rules,functions
```

Deploy prints the function URLs. The base is what goes in `.env`:

```env
PHANTOM_AUTH_URL=https://us-central1-<project-id>.cloudfunctions.net
```

### 3 — Admin credentials, for minting only

**Project settings → Service accounts → Generate new private key.** Save the
JSON somewhere on your machine and point at it:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # bash
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"  # PowerShell
pip install firebase-admin
```

**This file never leaves your machine and never ships with the desktop.** It is
the one credential with direct database access.

### 4 — Mint

```bash
python tools/mint_codes.py 10                # ten one-hour codes
python tools/mint_codes.py 3 --hours 2       # three two-hour codes
python tools/mint_codes.py 5 --dry-run       # print, write nothing
```

`--dry-run` needs no credentials and no Firebase at all — useful for seeing the
format before any of the above.

---

## Three ways to run

`PHANTOM_AUTH_URL` decides all of it, and **the variable must exist in `.env`,
not just in `.env.example`** — `.env` is gitignored, so a key added to the
template never reaches an existing config. A missing variable reads exactly like
a disabled feature, which is the one failure mode of this design that looks like
nothing at all.

| `PHANTOM_AUTH_URL` | Behaviour |
|---|---|
| blank or absent | **No gate.** Local development, and why nothing in `pipeline/` depends on this module |
| `mock` | Gate shown, both endpoints answered in-process. No Firebase, no network |
| a URL | The real thing |

### Mock mode

For seeing and testing the gate before Firebase exists — the same reason
`mint_codes.py` has `--dry-run`.

```env
PHANTOM_AUTH_URL=mock
PHANTOM_AUTH_MOCK_MINUTES=2
```

Any code that passes its checksum is accepted once; a second use is refused, and
so is a second code while time remains. State lives in
`~/.phantom/mock-auth.json` rather than in memory, because the behaviour most
worth testing is what happens across a **restart** — close the desktop, reopen
it, and the hour should still be running. Delete that file to start over.

`PHANTOM_AUTH_MOCK_MINUTES=2` makes the countdown chip and the expiry reachable
without waiting an hour.

Generate codes to type in with no credentials at all:

```bash
python tools/mint_codes.py 5 --dry-run
```

It triggers on the exact literal only (`mock`, `demo`, `local`, `offline`). A
real URL that happens to be unreachable still reports as unreachable — mock is a
development mode, never a fallback.

---

## Deliberate gaps

Named here so they are decisions rather than oversights.

**No rate limit, no lockout.** A code is 40 bits and single-use, so guessing is
not the exposure worth building against yet. If it becomes one, the place to
put it is `redeem` in `functions/main.py`, keyed by machine id. A client-side
lockout — registry, config file, anything on the customer's machine — would be
worth nothing, and a machine id the client asserts is not a stronger version of
the same idea.

**Machine id is spoofable in principle.** It is a hashed OS identifier, and a
determined customer can present a different one. That buys them a fresh *auth
prompt*, not a fresh *hour* — the code is still burned. The identifier answers
"is this the same machine as before", which is all the session lookup needs.

**A machine that never gets a stable OS id gets a generated one** in
`~/.phantom/machine-id`. Deleting it yields a new identity and a new prompt, and
again, no free time.

**The hour is not refundable.** If the pipeline dies twenty minutes in, the
customer has lost forty minutes. That is a product decision to make, not a bug —
and it is the reason the burn waits for the pipeline to be running, which
removes the most likely way for a session to fail before it starts.

**This does not enforce the hour.** `expires_at` is what the desktop reads to
know when to stop; the pod is stopped independently by `VAST_MAX_UPTIME`. Two
clocks that currently agree. Keeping them separate is what lets a two-hour code
work later without touching the shutdown path.
