"""Outbound relay posting via the buzz CLI.

The daemon owns its own Nostr identity (BUZZ_PRIVATE_KEY in its own
environment) and posts through the `buzz` CLI subprocess rather than
talking to the relay directly for writes (§4.2, §7, open question #2).
The persistent WebSocket client (inbound.py) handles live reads.
`run_buzz_cli` is also reused by history.py for one-shot historical
reads (recap doesn't need a live subscription), so both modules share
one place that knows how to invoke and parse `buzz` CLI output.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


class RelayError(Exception):
    """Raised when a buzz-cli invocation fails or returns something unusable."""


def run_buzz_cli(args: list[str], stdin: str | None = None) -> Any:
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
    return run_buzz_cli(["dms", "open", "--pubkey", pubkey])["dm_id"]


def set_presence(status: str) -> None:
    """Publish the daemon's own presence (kind:20001) so its availability dot
    in Buzz Desktop reflects whether it's actually up, not just deployed."""
    run_buzz_cli(["users", "set-presence", "--status", status])


def send_message(channel_id: str, content: str, reply_to: str | None = None) -> str:
    """Post `content` into `channel_id`; return the new event id.

    `content` goes over stdin (`--content -`) rather than argv, so
    arbitrary message text never has to survive shell-style quoting.
    """
    args = ["messages", "send", "--channel", channel_id, "--content", "-"]
    if reply_to is not None:
        args += ["--reply-to", reply_to]
    return run_buzz_cli(args, stdin=content)["event_id"]


def relay_dispatch(
    channel_id: str,
    instruction: str,
    requested_by: str,
    target_agent: str | None = None,
) -> str:
    """Post `instruction` into `channel_id`, attributed to `requested_by`.

    Per §5: a dispatched instruction must make clear it's relaying the
    user's own directive, not the TPM agent's own initiative, so the
    receiving coding agent treats it as an actual instruction.

    Buzz agents only react to @mentions by default, so when a
    `target_agent` was identified the relayed message leads with an
    `@name` mention -- otherwise it's relayed but nothing in the channel
    is guaranteed to ever look at it.
    """
    prefix = f"@{target_agent} " if target_agent else ""
    content = f"{prefix}Relaying instruction from {requested_by}: {instruction}"
    return send_message(channel_id, content)
