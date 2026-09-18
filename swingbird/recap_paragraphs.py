from __future__ import annotations

import re


def _normalize_label(label: str) -> str:
    """Flatten a hyphenated label like "fix-message-id-reliability" into
    "fix message id reliability" -- but only when the whole label is a
    single hyphen-joined slug (no spaces of its own), never a normal phrase
    that merely contains a hyphenated word.

    `_item_extraction_instructions` asks for "a few words naming the item,"
    but the LLM sometimes echoes a hyphenated, git-branch-shaped slug from
    the transcript instead (project channels are full of literal branch
    names) rather than phrasing it in words -- observed live, inconsistently
    even across two recap calls for the same underlying work: one call's
    labels for it came back hyphenated, another's came back as plain words.
    A label meant to stay exactly as given (e.g. "F4", reused verbatim per
    the same instructions) is never hyphenated in the first place, so a
    blanket replace here can't clash with that case.

    A blanket `.replace("-", " ")` over-corrected this: observed live, a
    label like "Integrate recap follow-up reliability" -- already plain
    words, just with one legitimately hyphenated compound word in it -- came
    back from this function as "Integrate recap follow up reliability",
    while the LLM's own "text" narration kept writing the paragraph header
    with the hyphen intact (`"**swingbird -- Integrate recap follow-up
    reliability:**"`), since nothing renormalizes that copy. The two no
    longer matched byte-for-byte, so `_ensure_channel_paragraphs` treated
    the already-narrated item as missing and appended a duplicate fallback
    paragraph for it -- one that, being appended unconditionally at the end
    of "text", also landed after a later channel's own paragraph, breaking
    the "grouped by channel" contract. Restricting the flatten to labels
    with no spaces at all keeps the slug case (never has spaces) working
    exactly as before while leaving any label that's already phrased in
    words -- hyphenated compound word or not -- untouched, so it stays
    identical to whatever the LLM wrote for it in "text"."""
    if " " in label:
        return label
    return label.replace("-", " ")


_DETAILED_PARAGRAPH_HEADER_RE = re.compile(
    r"^\*\*(?P<channel>[^*\n]+?) -- (?P<label>[^*\n]+?):\*\*"
)

# Concise mode's own paragraph header (`_FORMAT_GUARD`, `_item_paragraph_
# prefix`) has no label to key on -- just "**channel**:" -- so
# `_strip_closed_paragraphs` uses this to find a channel's lead paragraph
# without being able to confirm by label the way the detailed header can.
_CONCISE_PARAGRAPH_HEADER_RE = re.compile(r"^\*\*(?P<channel>[^*\n]+?)\*\*:")


def _strip_closed_paragraphs(
    text: str,
    closed_keys: set[tuple[str, str]],
    closed_lead_channels: set[str],
    detail: str,
) -> str:
    """Drop a "text" paragraph that narrates an item `_parse_recap`'s
    `closed_keys` filter already dropped from "items", so a promoted
    item's own fallback paragraph (`_ensure_channel_paragraphs`) doesn't
    end up sitting next to -- or, in concise mode, invisibly replaced by --
    stale prose about the closed work.

    `_parse_recap`'s closed-label filter (see `_closed_label_keys`) only
    ever touches "items" -- "text" is the LLM's own verbatim prose, not
    reconstructed from "items" (see `Recap`'s own field docs), so dropping
    an item there doesn't touch whatever paragraph the LLM wrote about it.
    Observed live as a real risk, not just theoretical: if the closed item
    was a *primary* one, the surviving next item for that channel is
    promoted to primary by `_parse_recap`'s ordinal counting (dropping a
    slot before `channel_counts` increments), but "text" still narrates
    the closed item's own paragraph under its own label.

    Detailed mode can confirm the match directly -- each paragraph is
    headed `**channel -- label:**` (`_DETAILED_PARAGRAPH_HEADER_RE`), so a
    header whose (channel, normalized label) is in `closed_keys` is
    dropped outright. `_ensure_channel_paragraphs` then appends the
    promoted item's own fallback paragraph as usual; without this step it
    would instead leave the stale paragraph standing untouched
    (`_dedupe_item_paragraphs` only rewrites/dedupes a paragraph matching a
    *known* item, and an orphaned one is explicitly left as written) while
    a second, fallback paragraph landed right next to it for the same
    channel.

    Concise mode has no label in its `**channel**:` header to confirm
    against, since only one item per channel is ever primary there -- so
    `closed_lead_channels` (built by `_parse_recap` alongside `closed_
    keys`, from whichever raw item came first for a channel before any
    filtering) stands in: a channel's lead paragraph is dropped when its
    very first extracted item -- primary or not -- was the one that got
    closed-filtered, since that's the item concise "text" would have been
    narrating. `_ensure_channel_paragraphs`'s own channel-prefix-only
    check (see its docstring) can't tell a stale paragraph from a correct
    one, so without this step it would treat the promoted item as already
    covered and never append its fallback at all -- the recap would keep
    describing the closed work as current.
    """
    if not text.strip():
        return text
    paragraphs = text.split("\n\n")
    kept = []
    for paragraph in paragraphs:
        if detail == "detailed":
            match = _DETAILED_PARAGRAPH_HEADER_RE.match(paragraph)
            if match:
                key = (
                    match.group("channel").strip(),
                    _normalize_label(match.group("label")).strip().casefold(),
                )
                if key in closed_keys:
                    continue
        else:
            match = _CONCISE_PARAGRAPH_HEADER_RE.match(paragraph)
            if match and match.group("channel").strip() in closed_lead_channels:
                continue
        kept.append(paragraph)
    return "\n\n".join(kept)
