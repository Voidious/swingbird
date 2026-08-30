"""Fetch recent channel activity via buzz-cli, for the recap flow (§4.2).

Recap is a one-shot "what happened recently" read, not a live
subscription, so it reuses outbound.py's buzz-cli plumbing rather than
the persistent WebSocket client in inbound.py.
"""

from __future__ import annotations

from swingbird.outbound import run_buzz_cli


def fetch_recent_messages(channel_id: str, limit: int | None = None) -> list[dict]:
    """Return recent messages in `channel_id`, most-recent-last."""
    args = ["messages", "get", "--channel", channel_id]
    if limit is not None:
        args += ["--limit", str(limit)]
    return run_buzz_cli(args)
