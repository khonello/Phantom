"""
Access control on the desktop side.

Three things happen here, and the order between them is the whole design:

1. **On launch, ask the server whether this machine already has time left.**
   A customer buys an hour, not a launch of the app. Closing the desktop and
   opening it again inside that hour must not ask for anything, so the answer
   lives in Firestore keyed by machine id — not in a local file, which would
   also mean a reinstall costs them their hour.

2. **A code is checked locally before it is sent.** The checksum in codes.py
   catches a mistyped character with no round trip, so a typo is never a
   failed redemption.

3. **A code is burned only once the pipeline is connected and running.**
   `begin` holds the code; `commit` spends it. If the pod never comes up, the
   code is still unspent and the customer has lost nothing. This is the reason
   redemption is two calls instead of one.

The clock is the server's throughout. `seconds_remaining` is what the function
returned, decremented locally only for display; anything that decides whether
the session is still live re-asks. A desktop with its clock wound back gets the
same answer as one without.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from desktop.codes import is_valid, normalize

_TIMEOUT = 15  # seconds; a slow answer here delays the whole app launch

# Where a generated fallback id is kept when the OS has no stable one to offer.
_FALLBACK_ID_PATH = Path.home() / ".phantom" / "machine-id"


@dataclass
class SessionState:
    """What the server says about this machine right now."""

    active: bool
    seconds_remaining: int
    error: Optional[str] = None

    @property
    def minutes_remaining(self) -> int:
        return self.seconds_remaining // 60

    @property
    def reachable(self) -> bool:
        return self.error is None


@dataclass
class RedeemResult:
    ok: bool
    seconds_remaining: int = 0
    reason: Optional[str] = None

    @property
    def message(self) -> str:
        """What to actually show the customer."""
        if self.ok:
            return "Session started — {} minutes".format(self.seconds_remaining // 60)
        return {
            "unknown_code": "That code is not recognised.",
            "code_used": "That code has already been used.",
            "session_active": "You already have {} minutes left.".format(
                self.seconds_remaining // 60),
            "invalid_format": "That code is not complete — check the characters.",
            "unreachable": "Cannot reach the licence server. Check your connection.",
        }.get(self.reason or "", self.reason or "Could not start the session.")


def _windows_machine_guid() -> Optional[str]:
    try:
        import winreg
    except ImportError:
        return None
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        )
        with key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
        return str(value) or None
    except OSError:
        return None


def _linux_machine_id() -> Optional[str]:
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            continue
    return None


def _macos_platform_uuid() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "IOPlatformUUID" in line and '"' in line:
            return line.rsplit('"', 2)[-2]
    return None


def _generated_id() -> str:
    """
    Last resort: a random id kept in the user's home directory.

    Weaker than the OS-provided ones — deleting the file yields a new machine
    and therefore a fresh auth prompt. Accepted because the alternative is an
    app that will not start at all on a platform we did not anticipate, and
    because the code itself is still single-use: a new identity does not
    conjure an unburned code.
    """
    try:
        existing = _FALLBACK_ID_PATH.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass

    fresh = str(uuid.uuid4())
    try:
        _FALLBACK_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FALLBACK_ID_PATH.write_text(fresh, encoding="utf-8")
    except OSError as exc:
        print("[AUTH] could not persist fallback machine id: {}".format(exc),
              file=sys.stderr)
    return fresh


_cached_machine_id: Optional[str] = None


def machine_id() -> str:
    """
    A stable identifier for this computer.

    Hashed before it leaves the machine. The server only ever needs to know
    that two launches came from the same place, and a raw hardware GUID in a
    database is more than that question requires.
    """
    global _cached_machine_id
    if _cached_machine_id is not None:
        return _cached_machine_id

    system = platform.system()
    raw = None
    if system == "Windows":
        raw = _windows_machine_guid()
    elif system == "Linux":
        raw = _linux_machine_id()
    elif system == "Darwin":
        raw = _macos_platform_uuid()

    if not raw:
        raw = _generated_id()

    _cached_machine_id = hashlib.sha256(
        ("phantom:" + raw).encode("utf-8")
    ).hexdigest()[:32]
    return _cached_machine_id


def _auth_url() -> str:
    """Base URL of the deployed functions, e.g. https://us-central1-<proj>.cloudfunctions.net"""
    return os.getenv("PHANTOM_AUTH_URL", "").rstrip("/")


def is_enabled() -> bool:
    """
    Whether access control is configured at all.

    Unset means an ungated build: local development, and the CI end-to-end run.
    Nothing in the pipeline depends on this module, so the whole feature is one
    environment variable away from not existing.
    """
    return bool(_auth_url())


# ── Mock mode ─────────────────────────────────────────────────────────────
# PHANTOM_AUTH_URL=mock answers both endpoints in-process, so the gate can be
# exercised before Firebase exists — the same reason mint_codes.py has
# --dry-run. It is a development aid, not a fallback: it triggers only on that
# exact literal, so a real URL that happens to be unreachable still reports as
# unreachable rather than quietly letting anyone in.
#
# State is file-backed rather than in-memory because the behaviour most worth
# testing is what happens across a *restart* — close the desktop, open it, and
# the hour should still be running.
#
# PHANTOM_AUTH_MOCK_MINUTES shortens the hour. Set it to 2 to watch the
# countdown and the expiry without waiting.
_MOCK_URLS = {"mock", "demo", "local", "offline"}
_MOCK_STATE_PATH = Path.home() / ".phantom" / "mock-auth.json"


def _is_mock() -> bool:
    return _auth_url().lower() in _MOCK_URLS


def _mock_load() -> Dict[str, Any]:
    try:
        data = json.loads(_MOCK_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("sessions", {})
            data.setdefault("used", [])
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {"sessions": {}, "used": []}


def _mock_save(state: Dict[str, Any]) -> None:
    try:
        _MOCK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MOCK_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        print("[AUTH] mock state not saved: {}".format(exc), file=sys.stderr)


def _mock_post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = _mock_load()
    machine = str(payload.get("machine_id", ""))
    now = time.time()
    expires = float(state["sessions"].get(machine, 0.0))
    remaining = max(0, int(expires - now))

    if endpoint == "session":
        return {"active": remaining > 0, "seconds_remaining": remaining}

    if remaining > 0:
        return {"ok": False, "reason": "session_active", "seconds_remaining": remaining}

    code = str(payload.get("code", ""))
    if code in state["used"]:
        return {"ok": False, "reason": "code_used"}

    # Any well-formed code works here. Real validation is "does this document
    # exist in Firestore", which is precisely what a mock cannot answer.
    if not is_valid(code):
        return {"ok": False, "reason": "unknown_code"}

    minutes = int(os.getenv("PHANTOM_AUTH_MOCK_MINUTES", "60"))
    state["used"].append(code)
    state["sessions"][machine] = now + minutes * 60
    _mock_save(state)
    print("[AUTH] mock: {} redeemed for {} minute(s)".format(code, minutes),
          file=sys.stderr)
    return {"ok": True, "seconds_remaining": minutes * 60}


def _post(endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """POST JSON, return the parsed reply, or None if the server is unreachable."""
    base = _auth_url()
    if not base:
        print("[AUTH] PHANTOM_AUTH_URL is not set", file=sys.stderr)
        return None

    if _is_mock():
        return _mock_post(endpoint, payload)

    request = urllib.request.Request(
        "{}/{}".format(base, endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        print("[AUTH] {} failed: {}".format(endpoint, exc), file=sys.stderr)
        return None


def check_session() -> SessionState:
    """
    Ask whether this machine is inside a paid hour.

    Called on every launch. `reachable` is False when the licence server could
    not be reached at all, which is a different thing from "no time left" and
    should be presented differently — one is the customer's problem, the other
    is ours.
    """
    reply = _post("session", {"machine_id": machine_id()})
    if reply is None:
        return SessionState(active=False, seconds_remaining=0, error="unreachable")
    return SessionState(
        active=bool(reply.get("active")),
        seconds_remaining=int(reply.get("seconds_remaining") or 0),
    )


class Redemption:
    """
    A code held between entry and the pipeline coming up.

    `begin` validates the shape locally and remembers the code. `commit` is
    what actually spends it, and is called only once the pipeline is connected
    and running. A Redemption that is never committed costs the customer
    nothing.
    """

    def __init__(self) -> None:
        self._code: Optional[str] = None
        self._committed = False

    @property
    def pending(self) -> bool:
        return self._code is not None and not self._committed

    @property
    def committed(self) -> bool:
        return self._committed

    def begin(self, raw_code: str) -> bool:
        """Accept a typed code if it is well-formed. No network call."""
        if not is_valid(raw_code):
            return False
        self._code = normalize(raw_code)
        self._committed = False
        return True

    def commit(self) -> RedeemResult:
        """
        Spend the held code. Safe to call more than once — the server treats a
        repeat from the same machine inside its own live hour as success, so a
        dropped reply does not cost an hour.
        """
        if self._code is None:
            return RedeemResult(ok=False, reason="invalid_format")
        if self._committed:
            return RedeemResult(ok=True, seconds_remaining=0)

        reply = _post("redeem", {"code": self._code, "machine_id": machine_id()})
        if reply is None:
            return RedeemResult(ok=False, reason="unreachable")

        if reply.get("ok"):
            self._committed = True
            return RedeemResult(
                ok=True,
                seconds_remaining=int(reply.get("seconds_remaining") or 0),
            )

        return RedeemResult(
            ok=False,
            seconds_remaining=int(reply.get("seconds_remaining") or 0),
            reason=str(reply.get("reason") or "unknown_code"),
        )

    def discard(self) -> None:
        """Drop an uncommitted code — the session never started."""
        if not self._committed:
            self._code = None


def wait_for_expiry(seconds_remaining: int) -> float:
    """Wall-clock deadline for a session with this many seconds left."""
    return time.time() + max(0, seconds_remaining)
