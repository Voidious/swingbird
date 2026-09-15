"""Pending-close store and deterministic yes/no resolution for "close F4" /
"mark the duplicate extractor fix as done" (swingbird-dev, 2026-09-14/15).

A `recap_close` intent from the router (see `router.py`) resolves to one
`RecapItem` the same way `recap_action`/`recap_relay` do (`daemon.
_resolve_or_store_single_recap_item`), then `daemon._propose_close` expands
that to every item sharing the same `source_event_id` -- "all the work items
grounded on this message" -- and asks the user to confirm closing the whole
batch before persisting anything (see `closed_items.py`).

That confirmation deliberately isn't routed back through the router's own
`confirm`/`cancel` intents or `pending_actions.PendingActionStore` -- closing
an item never relays anything (there's nothing to send to a coding agent),
so overloading the dispatch confirm/cancel machinery would conflate two
unrelated kinds of "yes". Instead this is its own small pending-state store,
resolved deterministically in `daemon._process` before the message ever
reaches the router -- the same idiom `recap_disambiguation.DisambiguationStore`
already uses for its own "which did you mean" follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

from swingbird.recap import RecapItem

_CONFIRM_WORDS = {
    "yes",
    "y",
    "yeah",
    "yep",
    "confirm",
    "confirmed",
    "do it",
    "close it",
    "close them",
    "close all",
    "go ahead",
}
_CANCEL_WORDS = {
    "no",
    "n",
    "nope",
    "cancel",
    "cancelled",
    "canceled",
    "never mind",
    "nevermind",
    "don't",
    "dont",
}


@dataclass(frozen=True)
class PendingClose:
    """An open "close these items?" question for one thread -- `items` is
    the whole batch that will be closed together if confirmed (see module
    docstring)."""

    items: tuple[RecapItem, ...]


class PendingCloseStore:
    """One open close confirmation per thread, replaced or cleared like
    `PendingActionStore`'s one pending dispatch proposal per thread."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingClose] = {}

    def set(self, thread_id: str, pending: PendingClose) -> None:
        self._pending[thread_id] = pending

    def get(self, thread_id: str) -> PendingClose | None:
        return self._pending.get(thread_id)

    def clear(self, thread_id: str) -> None:
        self._pending.pop(thread_id, None)


def resolve_close_reply(text: str) -> bool | None:
    """Match `text` -- a reply to an open close confirmation -- against a
    small fixed vocabulary of confirm/cancel wording, or return `None` if it
    doesn't answer the question at all (the caller then falls through to
    ordinary intent routing, per the module docstring).

    Deliberately a plain word/phrase lookup, not an LLM call -- same
    "cheap and deterministic" reasoning as `recap_disambiguation.
    resolve_choice`'s number-word matching, and there's no ambiguity to
    weigh here that would justify the cost of a model call. An unrecognized
    reply (e.g. a new, unrelated message) is never guessed as an answer.
    """
    normalized = text.strip().lower().rstrip(".!")
    if normalized in _CONFIRM_WORDS:
        return True
    if normalized in _CANCEL_WORDS:
        return False
    return None
