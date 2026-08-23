#!/usr/bin/env python3
"""
Mint access codes into Firestore.

    python tools/mint_codes.py 10                 # ten one-hour codes
    python tools/mint_codes.py 3 --hours 2        # three two-hour codes
    python tools/mint_codes.py 5 --dry-run        # print, write nothing

Operator tool. It is the only thing besides the Cloud Functions that talks to
Firestore, and the only thing that needs admin credentials — point
GOOGLE_APPLICATION_CREDENTIALS at the service-account JSON, which stays on
your machine and never ships with the desktop.

Codes are stored under their own text, so the Firestore console doubles as
your inventory: you can see at a glance what is unsold and what is spent. That
is worth more here than hashing them would be, because the only readers are
you and the functions.
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from desktop.codes import format_code, generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint Phantom access codes")
    parser.add_argument("count", type=int, help="how many codes to mint")
    parser.add_argument("--hours", type=float, default=1.0,
                        help="hours each code is worth (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the codes without writing to Firestore")
    args = parser.parse_args()

    if args.count < 1:
        print("ERROR: count must be at least 1")
        sys.exit(1)

    duration_min = int(round(args.hours * 60))
    if duration_min < 1:
        print("ERROR: --hours is too small to be worth anything")
        sys.exit(1)

    codes = [generate() for _ in range(args.count)]

    if args.dry_run:
        # Plain ASCII in tool output: Windows consoles default to cp1252 and
        # render an em dash as a replacement character.
        print("Dry run - nothing written. {} code(s), {} minutes each:\n".format(
            len(codes), duration_min))
        for code in codes:
            print("  {}".format(format_code(code)))
        return

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS is not set.")
        print("  Point it at your Firebase service-account JSON:")
        print("  Firebase console -> Project settings -> Service accounts -> Generate new private key")
        sys.exit(1)

    try:
        import firebase_admin
        from firebase_admin import firestore
    except ImportError:
        print("ERROR: firebase-admin not installed. Run: pip install firebase-admin")
        sys.exit(1)

    firebase_admin.initialize_app()
    db = firestore.client()

    # One batch, so a network failure part-way through does not leave you
    # unsure which codes exist. Either they are all there or none are.
    batch = db.batch()
    for code in codes:
        batch.set(db.collection("codes").document(code), {
            "used": False,
            "machine_id": None,
            "duration_min": duration_min,
        })
    batch.commit()

    print("Minted {} code(s), {} minutes each:\n".format(len(codes), duration_min))
    for code in codes:
        print("  {}".format(format_code(code)))


if __name__ == "__main__":
    main()
