"""Recap follow-up commands: "go ahead with X" / "tell me more about X" /
"for X, ...".

A `recap_action` or `recap_detail` intent from the router (see `router.py`)
doesn't carry its own content -- it just carries the user's raw reference
text ("F4", "all", "the duplicate extractor issue"). A `recap_relay` intent
carries a reference too (`Intent.item_reference`, same handling), alongside
separate content of its own (`Intent.message`) to forward -- see
`recap_relay.py`. Either way, the actual item content lives in the most
recent recap's structured `RecapItem`s (see `recap.py`), so resolving a
follow-up is a two-step job: look up the items stored for this thread, then
match the reference against them.

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

from typing import NamedTuple

from swingbird.recap import RecapItem

_ALL_MARKERS = {"", "all", "everything"}

# Connector words stripped out when falling back to word-level label matching
# (see `resolve_reference`) -- common enough to show up in almost any
# channel-qualified reference ("F4 for dripbird") without carrying any of
# the reference's actual meaning, so treating them as match candidates would
# just invite coincidental false positives.
_STOPWORDS = {"for", "the", "a", "an", "in", "on", "of", "to", "with", "about"}

# Generic nouns stripped out the same way as _STOPWORDS, for the same reason
# -- unlike a connector word, these do carry some meaning, but they're
# common enough as a label's own final word (almost any recap item can be
# called a "fix", a "bug", an "issue", etc.) that matching on one alone
# produces false positives across unrelated items in the same channel
# (observed live: "the swingbird fix" matched both the disambiguation fix
# and the async freeze fix, since both labels end in "...fix"). Still
# usable as part of the full bidirectional substring test above -- only
# excluded from the single-word fallback, where a false positive is
# cheapest to cause.
_GENERIC_WORDS = {
    "fix",
    "bug",
    "issue",
    "item",
    "items",
    "task",
    "feature",
    "problem",
    "update",
    "change",
}

# Words that signal "give me the rest", not "give me one specific item" --
# covers every phrasing the recap's own "N additional/open items" prose uses
# (see recap.py's _CONCISE_SYSTEM_PROMPT/_DETAILED_SYSTEM_PROMPT) plus the
# router's recap_detail enumeration examples (see router.py). Checked as
# whole reference words (post-stopword-strip), not a substring, so it can't
# accidentally fire just because a label happens to contain one of these --
# but a plural-intent word is also excluded from the separate identifying-
# word fallback below (like _GENERIC_WORDS), for the mirror-image reason:
# an item literally *about* the recap system (e.g. a work item labeled
# "recap-additional-items-followup") can contain one of these words in its
# own label, and matching on it there would wrongly resolve a genuine "the
# other items" request to that one coincidentally-worded item instead of
# ever reaching the "give me every non-primary item" fallback (observed
# live: exactly this item swallowed every "tell me about the other/
# additional/open swingbird items" phrasing tried).
_PLURAL_INTENT_WORDS = {
    "additional",
    "other",
    "others",
    "open",
    "remaining",
    "backlogged",
    "follow-up",
    "follow-ups",
    "rest",
}


class RecapActionError(Exception):
    """Raised when a recap-item reference can't be resolved."""


class ResolvedReference(NamedTuple):
    """The result of matching a reference against a recap's items.

    `degraded` is True only when `reference` used plural-intent wording
    (asking for "the other/additional items") but the channel it narrowed
    to has no non-primary items at all, so `items` is just the sole
    existing item handed back again rather than a genuine additional one.
    `daemon._recap_detail` uses this to tell the user there's nothing else
    instead of silently re-elaborating on the one item as if it were new
    (see `resolve_reference`'s docstring for the exact fallback this
    covers). Every other return path -- an explicit label match, "all", a
    real non-primary-items match, or the single-candidate fallback for a
    generic non-plural reference -- is `degraded=False`.
    """

    items: list[RecapItem]
    degraded: bool


class RecapActionStore:
    """The last recap's items per thread id, replaced wholesale by each new
    recap for that thread."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[RecapItem, ...]] = {}

    def set(self, thread_id: str, items: tuple[RecapItem, ...]) -> None:
        self._items[thread_id] = items

    def get(self, thread_id: str) -> tuple[RecapItem, ...] | None:
        return self._items.get(thread_id)


def _matches_text(candidate: str, normalized: str, words: list[str]) -> bool:
    """Bidirectional + word-level match of `normalized` (and its non-stopword
    `words`) against one piece of candidate text -- the same three-way check
    `resolve_reference` applies to `label`, factored out so it can also be
    applied to each of an item's `keywords` without duplicating the logic
    (see `resolve_reference`'s docstring for why each half of the check
    exists). Can't spuriously match an empty `candidate` -- no word is a
    substring of ""."""
    lowered = candidate.lower()
    return (
        normalized in lowered
        or (candidate and lowered in normalized)
        or any(word in lowered for word in words)
    )


def resolve_reference(
    items: tuple[RecapItem, ...],
    reference: str | None,
    channel: str | None = None,
) -> ResolvedReference:
    """Return the `RecapItem`s `reference` refers to, plus whether that was
    a degraded fallback (see `ResolvedReference`).

    An empty reference or "all"/"everything" (case-insensitive) refers to
    every item. Otherwise, `reference` is matched case-insensitively against
    each item's `label`, `keywords`, or `channel`, and every match is
    returned -- zero matches raises rather than guessing, but whether more
    than one match is acceptable is the caller's policy to enforce, not this
    function's (see module docstring).

    The match (`_matches_text`) is bidirectional (does `reference` contain
    the label, or does the label contain `reference`) since a bare label
    like "F4" needs the former and a reference that also names the channel
    ("F4 for dripbird") needs the latter -- neither string is a substring of
    the other in that second case. Candidates are narrowed to a single
    channel's items first whenever exactly one channel is identified, so a
    label that happens to recur across channels (or a channel name that
    happens to look like another item's label) can't cross-match. `channel`,
    when given, is the authoritative source for that narrowing -- it's the
    router's own classification of the message (`Intent.channel`), which can
    identify the channel even when the leftover reference text doesn't name
    it at all (e.g. "tell me more about the open items" once the router has
    already resolved which channel "the open items" belongs to). Only when
    `channel` is `None` does this fall back to searching for a channel name
    as a substring of `reference` itself.

    A channel-qualified reference can still fail the whole-string
    bidirectional test even after narrowing: a bundled multi-option item
    (e.g. one item's label covering a "F4 or F5 or F6" decision) isn't a
    substring of "F4 for dripbird", and "F4 for dripbird" isn't a substring
    of it either -- only the "F4" fragment actually identifies the item. So
    matching also falls back to individual words of `reference`: if any
    single word is itself a substring of the label, that's enough. This
    fallback uses a narrower word list (`match_words`) than the plural-intent
    check below (`words`): both drop stopwords, but `match_words` also drops
    `_GENERIC_WORDS` ("fix", "bug", "issue", ...) and `_PLURAL_INTENT_WORDS`
    ("other", "additional", ...). Generic words are common enough as a
    label's own last word that matching on one alone produces false
    positives across unrelated items (e.g. "the swingbird fix" against two
    items whose labels both end in "...fix"). Plural-intent words are
    excluded for the mirror-image reason: an item that happens to be *about*
    the recap system itself (e.g. labeled "recap-additional-items-followup")
    can contain one in its own label, and matching on it there would resolve
    a genuine "the other items" request to that one item instead of ever
    reaching the "give me every non-primary item" fallback below. This can't
    spuriously match an empty label (no word is a substring of "").

    Each of an item's `keywords` (see `recap.py`'s `_ITEMS_INSTRUCTIONS`) is
    checked with this exact same three-way test, independently of `label` --
    a keyword is just another name for the item, extracted by the LLM at
    recap time from the item's own content rather than the user's later
    phrasing, so it deserves the same substring/word-level leniency `label`
    gets, not a stricter one.

    If no label match was found but the reference itself asks for "the rest"
    (a word in `_PLURAL_INTENT_WORDS`, e.g. "the additional items", "what
    other items") and the channel narrowing above left exactly one channel,
    every non-primary item (`RecapItem.is_primary` is `False`) for that
    channel is returned -- this is what lets "tell me more about the
    additional items" resolve to real items without needing its own thread
    of message ids (see `recap.py`'s `is_primary`). If that channel has no
    non-primary items (e.g. it only ever had one open item), this falls
    through to the next rule instead of returning an empty list -- with
    `degraded=True` on the result, since asking for "the rest" and getting
    the same one item back again isn't a real match.

    If none of that identifies a match, the reference has no plural-intent
    wording, and the channel narrowing above left exactly one candidate
    marked `is_primary` (the common case: `text` always narrates the
    primary item, so a generic non-plural reference like "the open item",
    "the first one", or a description of the primary item's own content
    reasonably defaults to it), that primary candidate is returned --
    `degraded=False`, since this is exactly what the reference meant, not a
    fallback of last resort. This fires even when the channel has other
    (non-primary) candidates too, unlike the single-candidate case below --
    a reference to "the" item for a channel means the one `text` already
    described, not "whichever one happens to be the only candidate." If
    more than one candidate is marked primary (only possible with
    hand-built `RecapItem`s outside `recap.py`'s own invariant of one
    primary per channel), this doesn't guess between them.

    Failing that, if the channel narrowing left exactly one candidate
    total, that candidate is returned anyway -- a generic reference can
    only be pointing at that channel's one item once the channel is
    unambiguous. This mirrors the empty/"all" handling above, just scoped
    to one channel's candidates instead of every item, and it only fires
    when the channel is unambiguous -- a bare, channel-less generic
    reference still raises rather than guessing across channels.
    """
    normalized = (reference or "").strip().lower()
    if normalized in _ALL_MARKERS:
        if not items:
            raise RecapActionError("no items in the last recap")
        return ResolvedReference(list(items), degraded=False)
    if channel is not None:
        channels_named = {
            item.channel for item in items if item.channel.lower() == channel.lower()
        }
    else:
        channels_named = {
            item.channel for item in items if item.channel.lower() in normalized
        }
    candidates = (
        [item for item in items if item.channel in channels_named]
        if len(channels_named) == 1
        else items
    )
    words = [w for w in normalized.split() if len(w) > 1 and w not in _STOPWORDS]
    match_words = [
        w for w in words if w not in _GENERIC_WORDS and w not in _PLURAL_INTENT_WORDS
    ]
    matches = [
        item
        for item in candidates
        if _matches_text(item.label, normalized, match_words)
        or normalized in item.channel.lower()
        or any(
            _matches_text(keyword, normalized, match_words) for keyword in item.keywords
        )
    ]
    plural_intent = any(word in _PLURAL_INTENT_WORDS for word in words)
    if not matches and len(channels_named) == 1:
        if plural_intent:
            non_primary = [item for item in candidates if not item.is_primary]
            if non_primary:
                return ResolvedReference(non_primary, degraded=False)
        else:
            primary = [item for item in candidates if item.is_primary]
            if len(primary) == 1:
                return ResolvedReference(primary, degraded=False)
        if len(candidates) == 1:
            return ResolvedReference(candidates, degraded=plural_intent)
    if not matches:
        raise RecapActionError(f"no recap item matches {reference!r}")
    return ResolvedReference(matches, degraded=False)
