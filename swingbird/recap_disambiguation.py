"""Pending disambiguation store for an ambiguous recap follow-up.

`_resolve_single_recap_item` (`daemon.py`) raises `AmbiguousRecapReference`
when a `recap_action`/`recap_relay` reference matches more than one recap
item -- until now, the "which did you mean" question that produced was a
dead end: nothing remembered the candidates, so the user's answer (a label,
or a bare "1") went straight back through the intent router as a brand-new
message and got misclassified, most often as a fresh dispatch of the
answer's own text (see the swingbird-dev bug report, 2026-09-12).

`DisambiguationStore` closes that loop the same way `PendingActionStore`
closes it for confirm/cancel: `daemon._process` checks it before routing at
all, and only falls through to the router when the message doesn't answer
the open question. Answers are matched against a numbered list
(`format_choices`) rather than free-text label matching -- per Voidious,
this also needs to work over voice eventually, and a spoken "one"/"two" is
far more reliable than a spoken repeat of a label string, so
`resolve_choice` accepts a small vocabulary of number words alongside
digits, plus an exact label/`channel/label` match for a typed reply that
just repeats what was shown.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from swingbird.recap import RecapItem
from swingbird.recap_actions import RecapActionError
from swingbird.router import Intent


class AmbiguousRecapReference(RecapActionError):
    """Raised instead of a plain `RecapActionError` when a reference
    matches more than one recap item, so the caller can remember the
    candidates for a follow-up disambiguation answer (see module
    docstring) alongside formatting the "which did you mean" message."""

    def __init__(self, message: str, candidates: tuple[RecapItem, ...]) -> None:
        super().__init__(message)
        self.candidates = candidates


@dataclass(frozen=True)
class PendingDisambiguation:
    """An open "which did you mean" question for one thread.

    `kind` is always `"recap_action"` or `"recap_relay"` -- the two
    `daemon.py` handlers that can hit `AmbiguousRecapReference` -- and
    tells `daemon._resume_disambiguation` which handler to resume once
    `candidates` narrows to one. `intent` is the original router `Intent`
    that triggered the ambiguity, kept as-is so resuming can reuse its
    `message`/`item_reference`/`channel` exactly like the first attempt did.
    """

    kind: str
    candidates: tuple[RecapItem, ...]
    intent: Intent


class DisambiguationStore:
    """One open disambiguation per thread, replaced or cleared like
    `PendingActionStore`'s one pending dispatch proposal per thread."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingDisambiguation] = {}

    def set(self, thread_id: str, pending: PendingDisambiguation) -> None:
        self._pending[thread_id] = pending

    def get(self, thread_id: str) -> PendingDisambiguation | None:
        return self._pending.get(thread_id)

    def clear(self, thread_id: str) -> None:
        self._pending.pop(thread_id, None)


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def format_choices(candidates: tuple[RecapItem, ...]) -> str:
    """Render `candidates` as a numbered list for a disambiguation
    question, e.g. "1. backend/F4, 2. frontend/F5" -- the same numbering
    `resolve_choice` accepts back."""
    return ", ".join(
        f"{i}. {item.channel}/{item.label}"
        for i, item in enumerate(candidates, start=1)
    )


def resolve_choice(candidates: tuple[RecapItem, ...], text: str) -> RecapItem | None:
    """Match `text` -- the user's reply to an open disambiguation question
    -- against `candidates`, or return `None` if it doesn't answer the
    question at all (the caller then falls through to ordinary intent
    routing, per the module docstring).

    Accepts a 1-based index into `candidates` as a bare digit ("1"), a
    lightly decorated one ("1.", "#1", "option 1"), or a spelled-out number
    word up to nine (voice-friendly, per Voidious) -- checked against the
    reply's last whitespace-separated token, so a short sentence like "the
    first one" or "option 2" still resolves. Separately, an exact
    (case-insensitive) match of `item.label` or `"{channel}/{label}"`
    resolves a reply that just repeats what was shown instead of picking a
    number. Nothing here guesses a partial or fuzzy match -- an
    unrecognized reply is treated as a new message, not a wrong answer.
    """
    normalized = text.strip().lower()
    if not normalized:
        return None
    last_token = normalized.split()[-1].strip(string.punctuation)
    index = None
    if last_token.isdigit():
        index = int(last_token)
    elif last_token in _NUMBER_WORDS:
        index = _NUMBER_WORDS[last_token]
    if index is not None and 1 <= index <= len(candidates):
        return candidates[index - 1]
    for item in candidates:
        if normalized in (item.label.lower(), f"{item.channel}/{item.label}".lower()):
            return item
    return None
