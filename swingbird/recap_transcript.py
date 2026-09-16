from __future__ import annotations

import time

from swingbird.config import ChannelConfig
from swingbird.history import fetch_messages_since


def _build_transcript(
    channels: list[ChannelConfig],
    stale_after_days: int,
    max_messages_per_channel: int,
    explicit: bool,
    closed_ids_by_channel: dict[str, set[str]],
) -> tuple[str, dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Return the transcript text and `{channel_name: {tag: event_id}}` /
    `{channel_name: {tag: content}}` maps.

    Each message is tagged with a short local id ("m1", "m2", ...) rather
    than its real (64-char) event id -- cheap for the LLM to copy back
    verbatim in `source_id` (see `_item_extraction_instructions`) without
    risking a garbled hex string. Both maps only get an entry for messages
    that actually carry an `"id"` -- real `buzz messages get` events always
    do; this just means a message without one can't be cited as a source,
    never a crash. The content map exists alongside the id map so a
    resolved `source_id` can carry the message's own text forward too (see
    `RecapItem.source_content`), not just its event id.

    Tags are numbered once, globally, across every channel's section --
    never reset back to "m1" at the start of the next channel. A global
    recap concatenates many channels' sections into one transcript, and
    `_resolve_tag` looks a cited tag up as `mapping[claimed_channel][tag]`,
    trusting the LLM's own "channel" field completely (see that function's
    own "never guess" docstring, which only ever covered the tag half of
    that lookup). Observed live: on a busy multi-channel detailed recap,
    extraction wrote a real, correctly-formatted item for one channel's
    open work but named a *different* channel in "channel" -- the same
    "attention degrades across a long generation" failure this module
    already documents elsewhere (`_ensure_channel_paragraphs`,
    `_backfill_missing_sources`), just landing on the "channel" field
    instead. With per-channel-reset numbering, that other channel's own
    unrelated "m3" was a real hit, so the item silently linked to a real
    but wrong message in a real but wrong channel -- worse than no link at
    all. Numbering globally means a tag exists in exactly one channel's
    sub-map; a mislabeled "channel" now finds no such tag there at all, so
    `_resolve_tag` returns `None` (triggering `_backfill_missing_sources`
    to try grounding it again, scoped to that same claimed channel) instead
    of resolving to a real message that has nothing to do with the item.

    A message whose own id is in `closed_ids_by_channel` for its channel --
    the same message that grounded a now-closed item -- gets a literal
    "[closed]" suffix (see `_CLOSED_MARKER_GUARD`), a deterministic signal
    alongside `_format_closed_items_guard`'s prose-only guard for the
    common case where the closed item's own source message is still within
    this recap's window.
    """
    cutoff = time.time() - stale_after_days * 86400
    sections = []
    id_map: dict[str, dict[str, str]] = {}
    content_map: dict[str, dict[str, str]] = {}
    tag_counter = 1
    for channel in channels:
        # Same time-windowed fetch either way, paging past the relay's
        # 200-per-call cap as needed (see history.fetch_messages_since) so
        # a chatty channel can't push a still-relevant item out of the
        # window. Only whether an empty result is skipped differs: a
        # channel the user didn't name is silently dropped from an
        # all-channels recap, but one named explicitly still gets a
        # paragraph below ("(no recent activity)") since silence isn't a
        # useful answer to a question about a specific project.
        events = fetch_messages_since(
            channel.id, cutoff, max_messages=max_messages_per_channel
        )
        if not events and not explicit:
            continue
        header = f"## {channel.name}"
        if channel.goal:
            header += f" (goal: {channel.goal})"
        if events:
            closed_ids = closed_ids_by_channel.get(channel.name, set())
            lines = []
            tags = {}
            contents = {}
            for event in events:
                tag = f"m{tag_counter}"
                tag_counter += 1
                closed_suffix = " [closed]" if event.get("id") in closed_ids else ""
                lines.append(
                    f"[{tag}] [{event['created_at']}] {event['content']}{closed_suffix}"
                )
                if "id" in event:
                    tags[tag] = event["id"]
                    contents[tag] = event["content"]
            body = "\n".join(lines)
            if tags:
                id_map[channel.name] = tags
                content_map[channel.name] = contents
        else:
            body = "(no recent activity)"
        sections.append(f"{header}\n{body}")
    transcript = (
        "\n\n".join(sections) if sections else "(no channels with recent activity)"
    )
    return transcript, id_map, content_map
