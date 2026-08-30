"""NIP-01 event crypto: key parsing, signing, and signature verification.

Nostr identities are secp256k1/BIP340 x-only keys. Outbound writes go
through the buzz CLI (see outbound.py) and never need this, but the
inbound WebSocket client (inbound.py) talks to the relay directly --
for NIP-42 AUTH and for verifying the signatures on incoming events --
so this module implements those primitives from scratch.
"""

from __future__ import annotations

import hashlib
import json

import bech32
from coincurve import PrivateKey, PublicKeyXOnly


class NostrCryptoError(Exception):
    """Raised when a key can't be parsed or signing/verification fails."""


def parse_private_key(raw: str) -> bytes:
    """Parse a private key given as bech32 nsec1... or 64-char hex; return 32 bytes."""
    raw = raw.strip()
    if raw.startswith("nsec1"):
        hrp, data = bech32.bech32_decode(raw)
        decoded = bech32.convertbits(data, 5, 8, False) if data is not None else None
        if hrp != "nsec" or decoded is None or len(decoded) != 32:
            raise NostrCryptoError(f"invalid nsec private key: {raw!r}")
        return bytes(decoded)
    try:
        key_bytes = bytes.fromhex(raw)
    except ValueError as exc:
        raise NostrCryptoError(
            f"private key is not valid hex or nsec: {raw!r}"
        ) from exc
    if len(key_bytes) != 32:
        raise NostrCryptoError(f"private key must be 32 bytes, got {len(key_bytes)}")
    return key_bytes


def pubkey_hex(private_key: bytes) -> str:
    """Return the 64-char hex x-only pubkey for `private_key`."""
    return PublicKeyXOnly.from_valid_secret(private_key).format().hex()


def _serialize(
    pubkey: str, created_at: int, kind: int, tags: list, content: str
) -> bytes:
    payload = [0, pubkey, created_at, kind, tags, content]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


def event_id(
    pubkey: str, created_at: int, kind: int, tags: list, content: str
) -> bytes:
    """Return the 32-byte NIP-01 event id: sha256 of the canonical serialization."""
    return hashlib.sha256(_serialize(pubkey, created_at, kind, tags, content)).digest()


def sign_event(
    private_key: bytes, created_at: int, kind: int, tags: list, content: str
) -> dict:
    """Build and sign a NIP-01 event with `private_key`; return the full event dict."""
    pubkey = pubkey_hex(private_key)
    event_id_bytes = event_id(pubkey, created_at, kind, tags, content)
    sig = PrivateKey(private_key).sign_schnorr(event_id_bytes)
    return {
        "id": event_id_bytes.hex(),
        "pubkey": pubkey,
        "created_at": created_at,
        "kind": kind,
        "tags": tags,
        "content": content,
        "sig": sig.hex(),
    }


def verify_event(event: dict) -> bool:
    """Verify an event's id matches its contents and its signature is valid."""
    try:
        event_id_bytes = event_id(
            event["pubkey"],
            event["created_at"],
            event["kind"],
            event["tags"],
            event["content"],
        )
        if event_id_bytes.hex() != event["id"]:
            return False
        pubkey = PublicKeyXOnly(bytes.fromhex(event["pubkey"]))
        sig = bytes.fromhex(event["sig"])
        if len(sig) != 64:
            return False
        return pubkey.verify(sig, event_id_bytes)
    except (KeyError, ValueError):
        return False
