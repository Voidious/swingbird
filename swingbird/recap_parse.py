from __future__ import annotations

import json
from dataclasses import dataclass

from swingbird.closed_items import ClosedItem

from .recap_counts import RecapItem
from .recap_paragraphs import _normalize_label, _strip_closed_paragraphs
from .recap_tags import _resolve_tag


def _parse_keywords(raw: object) -> tuple[str, ...]:
    """Coerce the LLM's "keywords" field into a tuple of non-empty strings,
    silently dropping anything malformed (a non-list, or non-string/blank
    entries) rather than raising -- keywords are a matching aid, not load-
    bearing content, so a malformed entry should never fail the whole
    recap."""
    if not isinstance(raw, list):
        return ()
    return tuple(k.strip() for k in raw if isinstance(k, str) and k.strip())


def _add_if_unique(seen: set, key) -> bool:
    if key in seen:
        return False
    seen.add(key)
    return True


def _closed_label_keys(closed: tuple[ClosedItem, ...]) -> set[tuple[str, str]]:
    """Return `{(channel, normalized label)}` for every closed item with a
    non-empty label, so `_parse_recap` can drop an extracted item that
    exactly matches one instead of trusting the LLM to honor
    `_format_closed_items_guard`'s prose.

    `_format_closed_items_guard` asks the LLM to never re-list a closed
    item, but that's a prompt request, not an enforced constraint --
    observed live, the LLM re-extracted an item under the *exact* label
    the close flow itself recorded, despite the guard naming that label
    explicitly. Same "instruction gets skipped under attention
    degradation" failure class as `_parse_recap`'s own per-channel
    label-dedup backstop, so it gets the same deterministic treatment: an
    exact (channel, normalized label) match against something already
    closed is dropped before it can consume a slot. Anything short of an
    exact label match (a paraphrase, a reopened item under a new label)
    still relies on the prose guard -- there's no reliable label to key a
    deterministic check off of there.
    """
    return {
        (item.channel, key)
        for item in closed
        if (key := _normalize_label(item.label).strip().casefold())
    }


class RecapError(Exception):
    """Raised when a recap is requested for an unknown channel, or the LLM's
    response can't be trusted as a recap."""


@dataclass(frozen=True)
class Recap:
    text: str
    items: tuple[RecapItem, ...] = ()


def _build_recap(closed_keys, closed_lead_channels, detail, items, response):
    text = response.get("text")
    if not isinstance(text, str):
        # Observed live: a detailed recap with several channels' worth of
        # items came back with "items" fully populated but "text" missing
        # entirely, rather than malformed/truncated JSON (see llm.py's own
        # truncation handling for that separate failure mode) -- the same
        # "attention degrades across a long single generation" pattern
        # _ensure_channel_paragraphs already works around for a single
        # missing channel paragraph, just total instead of partial this
        # time. "items" is already fully grounded independent of "text" (it
        # comes first in both the extraction instructions and the response
        # shape -- see _RESPONSE_SHAPE_INSTRUCTIONS), so when at least one
        # item parsed, an empty "text" is used instead of failing outright:
        # _ensure_channel_paragraphs then reconstructs every primary item's
        # own fallback paragraph from scratch, the same as it already does
        # per-item. Only raise when there's truly nothing to build a recap
        # from -- no grounded items either.
        if not items:
            raise RecapError(
                f"LLM response is missing recap text: {json.dumps(response)}"
            )
        text = ""
    text = _strip_closed_paragraphs(text, closed_keys, closed_lead_channels, detail)
    return Recap(text=text, items=tuple(items))


def _parse_recap(
    response: dict,
    id_map: dict[str, dict[str, str]],
    content_map: dict[str, dict[str, str]],
    max_items_per_channel: int,
    closed: tuple[ClosedItem, ...] = (),
    detail: str = "concise",
) -> Recap:
    closed_keys = _closed_label_keys(closed)
    items = []
    channel_counts: dict[str, int] = {}
    seen_labels: dict[str, set[str]] = {}
    channel_first_seen: set[str] = set()
    closed_lead_channels: set[str] = set()
    for item in response.get("items") or []:
        if not isinstance(item, dict) or not item.get("instruction"):
            continue
        channel = item.get("channel", "")
        label = _normalize_label(item.get("label", ""))
        is_first_for_channel = channel not in channel_first_seen
        channel_first_seen.add(channel)
        # _item_extraction_instructions asks the LLM to fold every
        # restatement of the same underlying work into one "items" entry,
        # but that's a prompt request, not an enforced constraint --
        # observed live, a detailed recap still came back with two entries
        # for the same status update, one labeled "recap-close" and the
        # other "recap close". _normalize_label already treats those as
        # the same label (hyphenated slug vs plain words), so reusing it
        # here to key a per-channel "already added" set is a deterministic
        # backstop for exactly the failure the prompt guard was meant to
        # prevent, without discarding a second, genuinely distinct item
        # that happens to be unlabeled (an empty label never dedupes
        # against another empty one). Checked before this entry can
        # consume a `channel_counts`/`is_primary` slot, so a duplicate
        # never bumps a later, real item out of "text".
        label_key = label.strip().casefold()
        if label_key:
            if (channel, label_key) in closed_keys:
                if is_first_for_channel:
                    # In concise mode "text" only ever narrates the
                    # channel's lead item (see _strip_closed_paragraphs),
                    # so a closed match on that very first entry means the
                    # channel's "**channel**:" paragraph almost certainly
                    # describes the closed work, not whatever surviving
                    # item ends up promoted to primary below.
                    closed_lead_channels.add(channel)
                continue
            channel_seen = seen_labels.setdefault(channel, set())
            if not _add_if_unique(channel_seen, label_key):
                continue
        # The LLM lists a channel's items in priority order (see
        # _item_extraction_instructions) -- the first max_items_per_channel seen for a
        # channel are the ones "text" itself narrates, everything after is
        # one of the "additional" items (see RecapItem.is_primary).
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        is_primary = channel_counts[channel] <= max_items_per_channel
        items.append(
            RecapItem(
                channel=channel,
                label=label,
                summary=item.get("summary", ""),
                instruction=item.get("instruction", ""),
                is_primary=is_primary,
                source_event_id=_resolve_tag(id_map, channel, item.get("source_id")),
                source_content=_resolve_tag(
                    content_map, channel, item.get("source_id")
                ),
                keywords=_parse_keywords(item.get("keywords")),
            )
        )
    return _build_recap(closed_keys, closed_lead_channels, detail, items, response)
