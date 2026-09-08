"""Recap follow-up commands: "go ahead with X" / "tell me more about X".

A `recap_action` or `recap_detail` intent from the router (see `router.py`)
doesn't carry its own content -- it just carries the user's raw reference
text ("F4", "all", "the duplicate extractor issue"). The actual content
lives in the most recent recap's structured `RecapItem`s (see `recap.py`),
so resolving a follow-up is a two-step job: look up the items stored for
this thread, then match the reference against them.

`RecapActionStore` holds the last recap's items per thread, mirroring
`PendingActionStore`'s per-thread shape. `resolve_reference` does the
matching deterministically -- no LLM judgment involved -- per the same
"no guessing on ambiguity" invariant `pending_actions.py` already enforces:
it raises rather than guessing when nothing matches, but deliberately
leaves the "how many matches is too many" policy to the caller (`daemon.py`)
rather than baking one in here, since a caller might reasonably want "all"
to succeed with multiple items while an ambiguous partial reference
shouldn't. `daemon.py` currently treats any multi-item result the same way
(ask which one) -- turning "all" into a batch of dispatch proposals would
need `PendingActionStore` extended beyond one proposal per thread, which is
deliberately out of scope here (see the recap follow-up plan).
"""

from __future__ import annotations

from swingbird.recap import RecapItem

_ALL_MARKERS = {"", "all", "everything"}


class RecapActionError(Exception):
    """Raised when a recap-item reference can't be resolved."""


class RecapActionStore:
    """The last recap's items per thread id, replaced wholesale by each new
    recap for that thread."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[RecapItem, ...]] = {}

    def set(self, thread_id: str, items: tuple[RecapItem, ...]) -> None:
        self._items[thread_id] = items

    def get(self, thread_id: str) -> tuple[RecapItem, ...] | None:
        return self._items.get(thread_id)


def resolve_reference(
    items: tuple[RecapItem, ...], reference: str | None
) -> list[RecapItem]:
    """Return the `RecapItem`s `reference` refers to.

    An empty reference or "all"/"everything" (case-insensitive) refers to
    every item. Otherwise, `reference` is matched case-insensitively against
    each item's `label` or `channel`, and every match is returned -- zero
    matches raises rather than guessing, but whether more than one match is
    acceptable is the caller's policy to enforce, not this function's (see
    module docstring).
    """
    normalized = (reference or "").strip().lower()
    if normalized in _ALL_MARKERS:
        if not items:
            raise RecapActionError("no items in the last recap")
        return list(items)
    matches = [
        item
        for item in items
        if normalized in item.label.lower() or normalized in item.channel.lower()
    ]
    if not matches:
        raise RecapActionError(f"no recap item matches {reference!r}")
    return matches
