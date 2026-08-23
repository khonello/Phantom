"""
Phantom access codes — the only thing that touches Firestore.

Two endpoints, both POST, both JSON:

    /session   {machine_id}          -> is this machine already in a paid hour?
    /redeem    {code, machine_id}    -> burn a code and start the hour

Why a function rather than letting the desktop talk to Firestore directly:
the desktop is on the customer's machine, so any credential it holds is a
credential they hold. Firestore rules can express "read one document if you
already know its id", but not "burn this code, start a clock, and refuse if
the machine already has time left" — that is three writes and a decision, and
it belongs somewhere the customer cannot reach. `firestore.rules` denies all
client access; this is the only door.

The clock is this function's clock. Expiry is computed here and stored, so a
desktop with its system time wound back reads the same answer as one without.

No API key and no lockout on purpose. A code is 45 bits of entropy and is
burned on first use, so guessing is not the exposure worth building against
yet. If that changes, rate-limit `redeem` by machine_id here.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from firebase_admin import firestore, initialize_app
from firebase_functions import https_fn
from google.cloud.firestore_v1.transaction import Transaction

initialize_app()

# A code with no explicit duration is worth one hour.
DEFAULT_DURATION_MIN = 60


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json(payload: Dict[str, Any], status: int = 200) -> https_fn.Response:
    import json
    return https_fn.Response(
        json.dumps(payload),
        status=status,
        mimetype="application/json",
    )


def _read_body(req: https_fn.Request) -> Dict[str, Any]:
    body = req.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def _remaining(expires_at: Optional[datetime], now: datetime) -> int:
    """Whole seconds left, floored at zero."""
    if expires_at is None:
        return 0
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return max(0, int((expires_at - now).total_seconds()))


def _active_session(db: Any, machine_id: str, now: datetime) -> Tuple[bool, int, Optional[datetime]]:
    """Look up the machine's session. Returns (active, seconds_left, expires_at)."""
    snap = db.collection("sessions").document(machine_id).get()
    if not snap.exists:
        return False, 0, None
    expires_at = (snap.to_dict() or {}).get("expires_at")
    left = _remaining(expires_at, now)
    return left > 0, left, expires_at


@https_fn.on_request()
def session(req: https_fn.Request) -> https_fn.Response:
    """
    Does this machine already have time left?

    Called every time the desktop starts. An active answer means the auth
    screen is never shown — the customer paid for an hour, not for one launch
    of the app.
    """
    if req.method != "POST":
        return _json({"error": "POST only"}, 405)

    machine_id = str(_read_body(req).get("machine_id", "")).strip()
    if not machine_id:
        return _json({"error": "machine_id required"}, 400)

    db = firestore.client()
    now = _now()
    active, left, expires_at = _active_session(db, machine_id, now)

    return _json({
        "active": active,
        "seconds_remaining": left,
        "expires_at": expires_at.isoformat() if expires_at else None,
    })


@https_fn.on_request()
def redeem(req: https_fn.Request) -> https_fn.Response:
    """
    Burn a code and start the machine's hour.

    The desktop calls this only once the pipeline is connected and running, so
    a code is never spent on a session that failed to come up.

    Three outcomes worth naming:

    - The machine already has time left. Refused, with the remaining seconds.
      Entering a second code mid-hour is an accident, and spending it would be
      the wrong way to resolve one.
    - The code was already burned by *this* machine and is still live. Treated
      as success, not as reuse. A dropped response on the caller's side must
      not cost the customer their hour.
    - The code was burned by another machine, or does not exist. Refused.
    """
    if req.method != "POST":
        return _json({"error": "POST only"}, 405)

    body = _read_body(req)
    code = str(body.get("code", "")).strip().upper()
    machine_id = str(body.get("machine_id", "")).strip()

    if not code or not machine_id:
        return _json({"error": "code and machine_id required"}, 400)

    db = firestore.client()
    now = _now()

    active, left, expires_at = _active_session(db, machine_id, now)
    if active:
        return _json({
            "ok": False,
            "reason": "session_active",
            "seconds_remaining": left,
            "expires_at": expires_at.isoformat() if expires_at else None,
        })

    code_ref = db.collection("codes").document(code)
    session_ref = db.collection("sessions").document(machine_id)

    @firestore.transactional
    def burn(tx: Transaction) -> Dict[str, Any]:
        snap = code_ref.get(transaction=tx)
        if not snap.exists:
            return {"ok": False, "reason": "unknown_code"}

        data = snap.to_dict() or {}

        if data.get("used"):
            # Already spent. The one acceptable case is this same machine
            # retrying inside its own live hour.
            owner = data.get("machine_id")
            still_live = _remaining(data.get("expires_at"), now)
            if owner == machine_id and still_live > 0:
                return {
                    "ok": True,
                    "seconds_remaining": still_live,
                    "expires_at": data["expires_at"].isoformat(),
                    "replayed": True,
                }
            return {"ok": False, "reason": "code_used"}

        duration = int(data.get("duration_min") or DEFAULT_DURATION_MIN)
        expires = now + timedelta(minutes=duration)

        tx.update(code_ref, {
            "used": True,
            "machine_id": machine_id,
            "started_at": now,
            "expires_at": expires,
        })
        tx.set(session_ref, {
            "code": code,
            "started_at": now,
            "expires_at": expires,
        })

        return {
            "ok": True,
            "seconds_remaining": duration * 60,
            "expires_at": expires.isoformat(),
            "replayed": False,
        }

    result = burn(db.transaction())
    return _json(result, 200 if result.get("ok") else 200)
