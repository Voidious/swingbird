"""Outbound relay posting via the buzz CLI.

The daemon owns its own Nostr identity (BUZZ_PRIVATE_KEY in its own
environment) and posts through the `buzz` CLI subprocess rather than
talking to the relay directly for writes (§4.2, §7, open question #2).
The inbound WebSocket client (a later step) handles reads.
"""

from __future__ import annotations

import json
import subprocess


class RelayError(Exception):
    """Raised when a buzz-cli invocation fails or returns something unusable."""


def _run_buzz(args: list[str], stdin: str | None = None) -> dict:
    try:
        result = subprocess.run(
            ["buzz", *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RelayError("buzz CLI not found on PATH") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RelayError(f"buzz {' '.join(args)} failed: {detail}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RelayError(
            f"buzz {' '.join(args)} returned unparseable output: {result.stdout!r}"
        ) from exc


def open_dm(pubkey: str) -> str:
    """Open (or resurface) a DM conversation with `pubkey`; return its channel id."""
    return _run_buzz(["dms", "open", "--pubkey", pubkey])["dm_id"]


def send_message(channel_id: str, content: str, reply_to: str | None = None) -> str:
    """Post `content` into `channel_id`; return the new event id.

    `content` goes over stdin (`--content -`) rather than argv, so
    arbitrary message text never has to survive shell-style quoting.
    """
    args = ["messages", "send", "--channel", channel_id, "--content", "-"]
    if reply_to is not None:
        args += ["--reply-to", reply_to]
    return _run_buzz(args, stdin=content)["event_id"]


def relay_dispatch(channel_id: str, instruction: str, requested_by: str) -> str:
    """Post `instruction` into `channel_id`, attributed to `requested_by`.

    Per §5: a dispatched instruction must make clear it's relaying the
    user's own directive, not the TPM agent's own initiative, so the
    receiving coding agent treats it as an actual instruction.
    """
    content = f"Relaying instruction from {requested_by}: {instruction}"
    return send_message(channel_id, content)
