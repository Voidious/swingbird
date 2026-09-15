"""Persistent record of recap items the user has marked closed, so a future
recap stops re-surfacing them as open work (swingbird-dev, 2026-09-14/15).

A recap is fully stateless -- `recap.py`'s `build_recap` re-extracts every
item from the raw transcript window on every call, and `RecapActionStore`
only ever holds the *last* recap's items per thread, replaced wholesale by
the next one (see `recap_actions.py`). So "closed" can't be a row deleted
from some queryable item list -- there isn't one. Instead, closing an item
appends a durable snapshot of it here, and `build_recap` consults this store
on every future call, in every thread, to tell the extraction LLM which
work is already done (see `recap.py`'s closed-items guard).

`ClosedItemStore` mirrors `audit.py`'s append-only-JSON-lines-file pattern,
with one difference: `audit.jsonl` is a write-only debugging trail nothing
ever reads back, while this file *is* read back -- fully into memory at
startup, then consulted (and appended to) on every close. A modest local
file (one line per closed item, ever) is cheap to hold entirely in memory;
there's no need for anything fancier at this scale.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from swingbird.recap_counts import RecapItem


@dataclass(frozen=True)
class ClosedItem:
    """A snapshot of a `RecapItem` at the moment it was closed.

    Stored as a full snapshot (channel, label, summary, instruction,
    keywords), not just an id, since the recap that produced the original
    `RecapItem` is long gone by the time a future recap needs to consult
    this -- there's nothing else to look the content back up from.
    `source_event_id` is carried forward too: it's what lets `recap.py` tag
    a transcript message as `[closed]` when the same message that grounded
    the original item resurfaces in a later recap's window (see
    `RecapItem.source_event_id` for why it can be `None`).
    """

    channel: str
    label: str
    summary: str
    instruction: str
    source_event_id: str | None
    keywords: tuple[str, ...]
    closed_at: float


class ClosedItemStore:
    """Every closed item, ever, loaded fully into memory at startup and
    appended to (in memory and on disk) as items are closed.

    Deliberately holds every closed item forever on disk -- narrowing what's
    *sent to the LLM* is `for_channels`'s job (via its `since` cutoff), not
    something enforced at storage time, so a shrunk `closed_item_window_days`
    can never lose history that a later config change might want back.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[ClosedItem] = list(_load(self._path))

    def close(self, item: RecapItem) -> ClosedItem:
        """Append `item` as newly closed, both in memory and on disk."""
        closed = ClosedItem(
            channel=item.channel,
            label=item.label,
            summary=item.summary,
            instruction=item.instruction,
            source_event_id=item.source_event_id,
            keywords=item.keywords,
            closed_at=time.time(),
        )
        self._items.append(closed)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_serialize(closed)) + "\n")
        return closed

    def for_channels(
        self, channel_names: set[str] | None, since: float
    ) -> tuple[ClosedItem, ...]:
        """Return closed items closed at or after `since`, narrowed to
        `channel_names` when given (`None` means every channel) -- see
        `recap.py`'s `build_recap`, which scopes this to whichever channels
        a given recap call actually covers, so a channel's own closed items
        never leak into another channel's guard."""
        return tuple(
            item
            for item in self._items
            if item.closed_at >= since
            and (channel_names is None or item.channel in channel_names)
        )


def _load(path: Path) -> list[ClosedItem]:
    if not path.exists():
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            items.append(
                ClosedItem(
                    channel=record["channel"],
                    label=record["label"],
                    summary=record.get("summary", ""),
                    instruction=record.get("instruction", ""),
                    source_event_id=record.get("source_event_id"),
                    keywords=tuple(record.get("keywords", [])),
                    closed_at=record["closed_at"],
                )
            )
    return items


def _serialize(item: ClosedItem) -> dict:
    return {
        "channel": item.channel,
        "label": item.label,
        "summary": item.summary,
        "instruction": item.instruction,
        "source_event_id": item.source_event_id,
        "keywords": list(item.keywords),
        "closed_at": item.closed_at,
    }
