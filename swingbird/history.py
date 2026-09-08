"""Fetch recent channel activity via buzz-cli, for the recap flow (§4.2).

Recap is a one-shot "what happened recently" read, not a live
subscription, so it reuses outbound.py's buzz-cli plumbing rather than
the persistent WebSocket client in inbound.py.
"""

from __future__ import annotations

from swingbird.outbound import run_buzz_cli

# `buzz messages get --limit N` silently clamps to 200 server-side no
# matter how large N is (verified against the relay directly, see
# swingbird-dev thread 2026-09-08) -- this is the real per-call ceiling,
# not a value swingbird can raise by asking for more.
DEFAULT_PAGE_SIZE = 200


def fetch_recent_messages(
    channel_id: str, limit: int | None = None, before: float | None = None
) -> list[dict]:
    """Return recent messages in `channel_id`, most-recent-last.

    `before`, when given, is a unix timestamp: only messages strictly
    before it are returned, for paging further back via
    `fetch_messages_since`.
    """
    args = ["messages", "get", "--channel", channel_id]
    if limit is not None:
        args += ["--limit", str(limit)]
    if before is not None:
        args += ["--before", str(int(before))]
    return run_buzz_cli(args)


def fetch_messages_since(
    channel_id: str,
    since_ts: float,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_messages: int | None = None,
) -> list[dict]:
    """Return every message in `channel_id` at or after `since_ts`.

    A single page tops out at `page_size` (see `DEFAULT_PAGE_SIZE`), so
    this pages backwards with `--before` until a page's oldest message is
    at or before `since_ts`, a page comes back empty (channel exhausted),
    or `max_messages` have been collected -- a safety cap so one very
    chatty channel can't page indefinitely. Result is most-recent-last,
    same ordering as `fetch_recent_messages`, deduped by event id in case
    a boundary timestamp is shared by messages on both sides of a page.
    """
    collected: list[dict] = []
    seen_ids: set[str] = set()
    before: float | None = None
    while True:
        page = fetch_recent_messages(channel_id, limit=page_size, before=before)
        if not page:
            break
        new_events = [event for event in page if event["id"] not in seen_ids]
        seen_ids.update(event["id"] for event in new_events)
        collected = new_events + collected
        oldest = page[0]["created_at"]
        if oldest <= since_ts:
            break
        if before is not None and oldest >= before:
            break  # no progress -- avoid looping forever on a stuck boundary
        if max_messages is not None and len(collected) >= max_messages:
            break
        before = oldest
    return [event for event in collected if event["created_at"] >= since_ts]
