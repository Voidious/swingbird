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
import re
import subprocess
from typing import Any

# Matches an `@` that `buzz messages send` would try to resolve as a member
# mention: at start-of-string or after whitespace, immediately followed by a
# name character. Mirrors buzz-cli's own `extract_at_names` matcher (see
# buzz-sdk/src/mentions.rs) so this catches exactly what would otherwise
# error out.
_STRAY_MENTION_RE = re.compile(r"(?:^|(?<=\s))@(?=[A-Za-z0-9._-])")


class RelayError(Exception):
    """Raised when a buzz-cli invocation fails or returns something unusable."""


def _escape_stray_mentions(content: str) -> str:
    """Defang `@word` tokens in free-form (LLM-generated) content.

    `buzz messages send` treats any `@word` in `content` as an attempted
    member mention and hard-fails (non-retryable) if it doesn't resolve to
    exactly one channel member -- and recap/chit-chat/reply-summary text can
    easily contain an incidental `@word` (quoting another channel's mention,
    or a name that isn't a member of the DM it's being posted into). None of
    that text is ever meant to notify anyone, so a zero-width space is
    inserted right after the `@` to break the match while leaving the text
    visually unchanged.
    """
    zero_width_space = "\u200b"
    return _STRAY_MENTION_RE.sub("@" + zero_width_space, content)


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


def join_channel(channel_id: str) -> None:
    """Join `channel_id`; a no-op if the identity is already a member.

    The relay accepts a join for an open channel unconditionally and
    silently no-ops a join for a channel the identity already belongs to,
    so the only way this raises `RelayError` is a real problem: notably a
    private channel the identity isn't already in ("restricted: channel is
    private"). Callers (see `daemon.py`'s `_join_project_channels`) treat
    that as an expected, reportable condition rather than a crash.
    """
    run_buzz_cli(["channels", "join", "--channel", channel_id])


def set_presence(status: str) -> None:
    """Publish the daemon's own presence (kind:20001) so its availability dot
    in Buzz Desktop reflects whether it's actually up, not just deployed."""
    run_buzz_cli(["users", "set-presence", "--status", status])


def get_own_display_name() -> str | None:
    """Return the current identity's Buzz display name, or None if it has
    never set one (`buzz users get` with no `--pubkey` returns the caller's
    own profile)."""
    profiles = run_buzz_cli(["users", "get"])
    if not profiles:
        return None
    return profiles[0].get("display_name")


def set_display_name(name: str) -> None:
    """Update the current identity's Buzz display name."""
    run_buzz_cli(["users", "set-profile", "--name", name])


def send_message(
    channel_id: str,
    content: str,
    reply_to: str | None = None,
    mentions: list[str] | None = None,
) -> str:
    """Post `content` into `channel_id`; return the new event id.

    `content` goes over stdin (`--content -`) rather than argv, so
    arbitrary message text never has to survive shell-style quoting.

    `mentions` are pubkeys passed as explicit `--mention` flags, which
    notify their owner even if `content`'s `@name` text can't be resolved
    against the channel's membership (e.g. the owner posting into their
    own DM, then having that instruction relayed into a project channel
    they aren't a member of).

    When `mentions` is empty, `content` is free-form (recap/chit-chat/reply
    -summary text with no intended live mention) and any stray `@word` in
    it is defanged via `_escape_stray_mentions` -- see that function for
    why. Callers that build an intentional `@name` mention (`relay_dispatch`)
    always pass `mentions`, so they're unaffected.
    """
    if not mentions:
        content = _escape_stray_mentions(content)
    args = ["messages", "send", "--channel", channel_id, "--content", "-"]
    if reply_to is not None:
        args += ["--reply-to", reply_to]
    for pubkey in mentions or ():
        args += ["--mention", pubkey]
    return run_buzz_cli(args, stdin=content)["event_id"]


def relay_dispatch(
    channel_id: str,
    instruction: str,
    requested_by_name: str,
    requested_by_pubkey: str,
    target_agent: str | None = None,
) -> str:
    """Post `instruction` into `channel_id`, attributed to the requester.

    Per §5: a dispatched instruction must make clear it's relaying the
    user's own directive, not the TPM agent's own initiative, so the
    receiving coding agent treats it as an actual instruction. The
    attribution is a real `@mention` (via `requested_by_pubkey`), not just
    name text, so the requester is notified and easy to follow back to --
    and so a working agent's own reply is more likely to @mention them
    back -- even in a project channel the requester never joined.

    Buzz agents only react to @mentions by default, so when a
    `target_agent` was identified the relayed message leads with an
    `@name` mention -- otherwise it's relayed but nothing in the channel
    is guaranteed to ever look at it.
    """
    prefix = f"@{target_agent} " if target_agent else ""
    content = f"{prefix}Relaying instruction from @{requested_by_name}: {instruction}"
    return send_message(channel_id, content, mentions=[requested_by_pubkey])
