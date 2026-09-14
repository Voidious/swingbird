from __future__ import annotations

import re
from dataclasses import dataclass


def _item_paragraph_prefix(channel: str, label: str, detail: str) -> str:
    """Return the bold Markdown prefix that starts an item's own paragraph
    in recap "text", per `_FORMAT_GUARD` (concise: "**channel**:") or
    `_DETAILED_FORMAT_GUARD` (detailed: "**channel -- label:**", matching
    `recap_detail.py`'s own per-item format).
    """
    if detail == "detailed":
        return f"**{channel} -- {label}:**"
    return f"**{channel}**:"


def _additional_items_paragraph(channel: str, labels: list[str]) -> str:
    """Build the standalone paragraph a detailed recap's fold note renders
    as, e.g. "**backend -- 2 additional open items:** F6, F7".

    A concise recap's fold note stays a short inline "(N additional open
    items.)", since concise is explicitly the terse mode -- but a detailed
    recap already gives each shown item its own paragraph and names it by
    label, so folding several unnamed items into the tail of the last
    paragraph's own sentence was hard to notice at a glance and gave no way
    to identify them. This mirrors that same "**channel -- label:**" prefix
    shape (`_DETAILED_FORMAT_GUARD`) so the note reads as one more entry in
    the same followable list, and names each folded item the same way
    `resolve_reference` (`recap_actions.py`) already matches a later
    "tell me more about F6" against -- an unlabeled placeholder is used only
    for the rare item the LLM left without one, never fabricated.
    """
    noun = "item" if len(labels) == 1 else "items"
    names = ", ".join(label or "(unlabeled)" for label in labels)
    return f"**{channel} -- {len(labels)} additional open {noun}:** {names}"


def _last_primary_by_channel(items: tuple[RecapItem, ...]) -> dict[str, RecapItem]:
    """Return the last (highest-priority-order) primary item per channel --
    the item whose own paragraph a detailed recap's additional-items note
    (`_additional_items_paragraph`) immediately follows, since that's the
    last paragraph shown for that channel before the fold
    (see `_append_item_counts`)."""
    last: dict[str, RecapItem] = {}
    for item in items:
        if item.is_primary:
            last[item.channel] = item
    return last


@dataclass(frozen=True)
class RecapItem:
    """One channel's current actionable item, extracted alongside the recap
    text so a later "go ahead with X" DM can refer back to it (see
    `recap_actions.py`) without re-parsing rendered prose.

    `is_primary` marks the items per channel that "text" itself narrates --
    the first item the LLM listed for that channel, per `_ITEMS_
    INSTRUCTIONS` (see `_parse_recap`), plus up to `max_items_per_channel`
    - 1 more after it. A concise recap always caps that at 1, so it's a
    single lead item there, same as before; a detailed recap allows up to
    `config.recap.max_detailed_items` (see `build_recap`), so more than one
    item per channel can be `is_primary` there. Every other item for that
    channel -- beyond however many got shown -- is one of the "N
    additional/open items" the recap text only counts (`_append_item_
    counts`). `resolve_reference` (`recap_actions.py`) uses this to answer
    "what are the other items" with exactly the ones that weren't shown.

    `source_event_id` is the transcript message the instruction was grounded
    in, when the LLM could point to one specific message -- it's what lets a
    later `recap_action` thread its relayed dispatch as a reply to that
    message instead of posting disconnected from it (see `daemon.py`).
    `source_content` is that same message's own text, carried alongside so a
    later dispatch-phrasing rewrite (see `dispatch_phrasing.py`) can ground
    itself in the coding agent's own wording instead of just the recap's
    condensed summary/instruction. Both are `None` under the same
    conditions -- nothing single message grounds the item; never guessed.

    `keywords` are alternate phrases the LLM thought of at extraction time
    for referring to this same item later (see `_item_extraction_instructions`) --
    unlike every other field here, they're deliberately *not* required to be
    grounded in the transcript's literal wording, since their whole job is
    covering synonyms/categories the transcript never used. `resolve_
    reference` (`recap_actions.py`) matches a follow-up reference against
    these the same way it matches `label`, so a user can say "the lint
    issue" for an item whose label is "F6" without needing to guess the
    LLM's exact label string."""

    channel: str
    label: str
    summary: str
    instruction: str
    is_primary: bool = True
    source_event_id: str | None = None
    source_content: str | None = None
    keywords: tuple[str, ...] = ()


# A trailing "(N more/additional/other/remaining open items.)"-shaped
# parenthetical the LLM sometimes narrates on its own, despite
# _CONCISE_SYSTEM_PROMPT explicitly telling it not to ("a count of how many
# more there are is appended separately, not narrated by you") -- that's a
# prompt request, not an enforced constraint, so it can still leak through
# (observed live: a "(1 more open item.)" from the LLM stacked right next to
# our own correct "(1 additional open item.)", a bare, occasionally wrong,
# "(0 more open items.)" for a channel with nothing else, a countless
# "(Additional open items remain.)" -- no leading number/no/zero at all, and
# "remain" instead of "remaining" -- a "(+1 more open item.)" with a leading
# "+" the plain \d+ alternative didn't match, and a countless "(Additional
# open items omitted.)" -- same shape as "remain(ing)" but naming the fold
# itself as an omission instead of describing what's left over). The leading
# count and trailing verb are both optional here to catch the countless
# shapes too -- either can be absent from what the LLM narrates, but "more/
# additional/other/remaining" plus "item(s)" together are specific enough to
# this one note that a false strip elsewhere isn't a real risk. Stripped
# before _append_item_counts adds the real, grounded count, so the two can
# never stack and a wrong LLM-invented note is never left standing on its
# own.
_LLM_COUNT_NOTE_RE = re.compile(
    r"\s*\(\s*(?:(?:\+?\d+|no|zero)\s+)?(?:more|additional|other|remaining)\s+"
    r"(?:open\s+)?items?(?:\s+(?:remain(?:s|ing)?|omitted|excluded|hidden|"
    r"skipped|not\s+shown))?\.?\s*\)\s*$",
    re.IGNORECASE,
)

# A full sentence the LLM sometimes narrates as its own declarative aside
# instead of the parenthetical `_LLM_COUNT_NOTE_RE` shape -- observed live
# both as a whole standalone paragraph right after a detailed recap's own
# deterministic "**channel -- N additional open items:** ..." paragraph
# ("There are 7 more open items for swingbird." -- with a count that didn't
# even match our real one -- and "There are 8 additional open swingbird
# items beyond the three described above.") and, in concise mode, tacked
# onto the *end* of the primary item's own paragraph, right before our own
# correct parenthetical note ("...push or pull that branch and integrate/
# live-test it. There are 10 additional open items. (9 additional open
# items.)"). Since it's declarative prose, not a trailing parenthetical,
# `_LLM_COUNT_NOTE_RE` (anchored on a `(...)` at the end of a paragraph)
# never matches it. Anchored on a sentence boundary (start of paragraph or
# right after a ".", "!", or "?") rather than the whole paragraph, so
# `.sub()` below can drop it -- and everything after it, via the trailing
# `.*` -- whether it's the paragraph's only content or trails real content
# that came before it. The subject alternation covers both phrasings
# observed live -- "there is/are ..." and "<channel> has/have ..." (e.g.
# "swingbird has 6 additional open items beyond these three.") -- rather
# than only the former.
_LLM_COUNT_SENTENCE_RE = re.compile(
    r"(?:^|(?<=[.!?]\s))(?:there\s+(?:is|are)|\S+\s+(?:has|have))\s+"
    r"(?:(?:\+?\d+|no|zero)\s+)?(?:more|additional|other|remaining)\s+"
    r"(?:open\s+)?(?:\S+\s+)?items?\b.*",
    re.IGNORECASE,
)


def _append_item_counts(text: str, items: tuple[RecapItem, ...], detail: str) -> str:
    """Append a deterministic additional-items note to `text`, computed from
    the real non-primary items in `items` rather than left to the LLM's own
    prose judgment.

    The LLM was previously asked to narrate this itself (see
    _CONCISE_SYSTEM_PROMPT's history), but proved unreliable in practice --
    it would fold another item's content into the leading paragraph instead
    of counting it, or drop the mention entirely, while the grounded
    `items` list was correct the whole time. Called for both "concise" and
    "detailed" recaps (see `build_recap`), but rendered differently: a
    concise recap shows exactly one item per channel, so a short "(N
    additional open items.)" is appended inline to that channel's one
    paragraph (`**channel**:`); a detailed one shows up to `config.recap.
    max_detailed_items` per channel, each in its own paragraph (`**channel
    -- label**:`, see `_DETAILED_FORMAT_GUARD`), so the fold instead gets its
    own paragraph naming each folded item's label
    (`_additional_items_paragraph`), inserted right after the *last* of
    those (`_last_primary_by_channel`) -- the paragraph the fold immediately
    follows. Either way, anything beyond what got shown is a non-primary
    item here (see `RecapItem.is_primary`) and belongs in this same note,
    not narrated by the LLM itself.

    Every paragraph first has any `_LLM_COUNT_NOTE_RE`-shaped trailing note
    stripped, regardless of whether that channel has a real count to append
    afterward -- the LLM can narrate a bogus count even for a channel that
    ends up with zero real additional items, and leaving it in place would
    be worse than the stacked-duplicate case, since nothing would ever
    correct it.

    Observed live: in detailed mode, the LLM sometimes narrates the count as
    an entire standalone paragraph -- a bare "**channel**: (+1 more open
    item.)" using the concise "no open item" shape -- instead of appending
    it inline to the last shown item's own "**channel -- label**:"
    paragraph. Stripping the trailing note from a paragraph like that leaves
    a dangling, content-free "**channel**:" header; since that channel
    already has its own item paragraph(s) elsewhere (it's in `last_primary`)
    this stray paragraph is pure noise once stripped, so it's dropped
    outright rather than kept -- the real note still lands correctly after
    the last primary item's own paragraph via the loop below.

    Relies on the format guard's paragraph-prefix contract to find the
    right paragraph; one that doesn't start that way (the LLM ignoring the
    format guard) is silently left without a note rather than guessing
    which paragraph it meant.

    Observed live: the LLM also sometimes narrates the count as a whole
    extra sentence-paragraph on its own -- "There are 7 more open items for
    swingbird." right after our own correct fold-note paragraph, with a
    count that didn't even agree with ours -- and, separately, in concise
    mode, tacked onto the *end* of the primary item's own paragraph, right
    before our own correct parenthetical note ("...it. There are 10
    additional open items. (9 additional open items.)"). `_LLM_COUNT_NOTE_RE`
    only matches a trailing `(...)` on an existing paragraph, not a
    freestanding sentence, so `_LLM_COUNT_SENTENCE_RE` is applied as a
    `.sub()` instead of a whole-paragraph `fullmatch` -- it strips the
    sentence and everything after it (its trailing `.*`) wherever it starts
    a sentence within the paragraph, leaving any real content that came
    before it intact. A paragraph that turns out to be nothing but this
    narration (no real content before it) becomes empty and is dropped
    outright, same "stray paragraph, no other content worth keeping"
    treatment as the bare "**channel**:" case above.

    Observed live: right after our own correct "**swingbird -- N additional
    open items:** label, label, ..." fold paragraph (the real one this
    function generates below, not an LLM invention), the LLM separately
    narrated its own leftover "has"-phrased version of the sentence case
    above -- "swingbird has 6 additional open items beyond these three."
    `_LLM_COUNT_SENTENCE_RE`'s subject alternation (`there is/are` or
    `<word> has/have`) now catches this phrasing the same way it already
    caught "there are".
    """
    additional: dict[str, list[str]] = {}
    for item in items:
        if not item.is_primary:
            additional.setdefault(item.channel, []).append(item.label)
    last_primary = _last_primary_by_channel(items) if detail == "detailed" else {}
    stray_bare_prefixes = {f"**{channel}**:" for channel in last_primary}
    paragraphs = text.split("\n\n")
    kept_paragraphs = []
    changed = False
    for paragraph in paragraphs:
        cleaned = _LLM_COUNT_NOTE_RE.sub("", paragraph)
        if cleaned != paragraph:
            changed = True
        without_sentence = _LLM_COUNT_SENTENCE_RE.sub("", cleaned).rstrip()
        if without_sentence != cleaned:
            changed = True
        cleaned = without_sentence
        if not cleaned.strip() or cleaned.strip() in stray_bare_prefixes:
            changed = True
            continue
        extra_paragraph = None
        for channel, labels in additional.items():
            if detail == "detailed":
                last_item = last_primary.get(channel)
                prefix = (
                    _item_paragraph_prefix(channel, last_item.label, detail)
                    if last_item is not None
                    else None
                )
                if prefix and cleaned.startswith(prefix):
                    extra_paragraph = _additional_items_paragraph(channel, labels)
                    changed = True
                    break
            else:
                prefix = _item_paragraph_prefix(channel, "", detail)
                if prefix and cleaned.startswith(prefix):
                    noun = "item" if len(labels) == 1 else "items"
                    cleaned = f"{cleaned} ({len(labels)} additional open {noun}.)"
                    changed = True
                    break
        kept_paragraphs.append(cleaned)
        if extra_paragraph is not None:
            kept_paragraphs.append(extra_paragraph)
    if not changed:
        return text
    return "\n\n".join(kept_paragraphs)
