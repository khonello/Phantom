"""
Access code format — shared by the minting tool and the desktop.

Ten characters of Crockford base32, shown as two groups of five:

    K7QM4-9XRHT

Crockford because these get read aloud. Its alphabet drops I, L, O and U, so
there is no "is that a one or an ell" over a phone line, and its decoder maps
the confusable characters back anyway: a customer who types O for 0 or l for 1
is understood rather than rejected.

The last character is a checksum over the other nine. That is what makes a
The last two characters are a checksum over the first eight. That is what makes
a typo a *local* event: the desktop rejects a mistyped code without a round
trip, so a wrong character never counts as a failed redemption and never
reaches Firestore. Eight payload characters is 40 bits, which is the actual
entropy — about a trillion codes.
"""

import secrets
from typing import Optional

# Crockford base32: no I, L, O, U.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_PAYLOAD_LEN = 8
_CHECK_LEN = 2
CODE_LEN = _PAYLOAD_LEN + _CHECK_LEN

# The checksum modulus. Two things make 1021 the right choice and are easy to
# get wrong:
#
#   It is larger than 32. A modulus of 32 or below is degenerate against a
#   base-32 payload — `value % 32` *is* the last character, so the check digit
#   would be a copy of it and catch nothing. This was the first version.
#
#   It is prime, so coprime to 32. A single wrong character shifts the value by
#   delta * 32**i, and no such shift can land back on a multiple of 1021. Every
#   single-character error is caught, and so is every transposition of two
#   adjacent characters, since that shifts by a multiple of 31 * 32**i.
#
# 1021 < 1024 = 32**2, so the remainder always fits in two symbols.
_CHECK_MOD = 1021

# Forgiving input: the letters Crockford excludes map to the digits they are
# mistaken for. Applied before decoding, so "OL" and "01" are the same code.
_CONFUSABLE = {"O": "0", "I": "1", "L": "1"}


def _decode(payload: str) -> Optional[int]:
    value = 0
    for char in payload:
        index = _ALPHABET.find(char)
        if index < 0:
            return None
        value = value * 32 + index
    return value


def normalize(raw: str) -> str:
    """
    Fold user input into canonical form: upper case, no separators,
    confusable letters mapped to digits.

    Does not validate — a normalized string may still be the wrong length or
    fail its checksum.
    """
    cleaned = "".join(ch for ch in raw.upper() if ch.isalnum())
    return "".join(_CONFUSABLE.get(ch, ch) for ch in cleaned)


def _checksum(payload: str) -> Optional[str]:
    """The two check characters for a payload, or None if it is not decodable."""
    value = _decode(payload)
    if value is None:
        return None
    remainder = value % _CHECK_MOD
    return _ALPHABET[remainder // 32] + _ALPHABET[remainder % 32]


def is_valid(raw: str) -> bool:
    """True if this could be a real code: right length, right alphabet, checksum agrees."""
    code = normalize(raw)
    if len(code) != CODE_LEN:
        return False
    expected = _checksum(code[:_PAYLOAD_LEN])
    return expected is not None and expected == code[_PAYLOAD_LEN:]


def format_code(code: str) -> str:
    """Group as XXXXX-XXXXX for display. Input is assumed normalized."""
    return "{}-{}".format(code[:5], code[5:]) if len(code) == CODE_LEN else code


def generate() -> str:
    """Mint one code. Canonical form, no separator."""
    payload = "".join(secrets.choice(_ALPHABET) for _ in range(_PAYLOAD_LEN))
    check = _checksum(payload)
    assert check is not None  # every character came from _ALPHABET
    return payload + check
