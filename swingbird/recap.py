"""Naive recap flow (§4.2, §7, step 7).

The initial implementation skips the incrementally-maintained
project-state cache from §4.2 entirely: each `recap` request
recomputes a short, prioritized summary from recent channel activity
via a single LLM call. Correct and simple; the incremental cache is
deferred to the "refine prompts and tooling" follow-on once real recap
output quality has been seen.

Alongside the rendered text, that same call also extracts a structured
`RecapItem` per channel's current actionable next step. This is what
lets a later DM ("go ahead with F4") refer back to a specific item from
the last recap without re-parsing rendered prose -- see
`recap_actions.py`. One JSON call does both jobs rather than two: the
transcript is the same either way, and a second LLM round trip would
double the cost for no benefit.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace

from swingbird import outbound
from swingbird.config import ChannelConfig, Config
from swingbird.history import fetch_messages_since
from swingbird.llm import LLMClient, LLMError

_LAUNDERING_GUARD = (
    "Never describe work that is drafted, proposed, or awaiting the user's "
    "answer to a direct question as finished or done -- if the most recent "
    "activity for a project includes an unanswered question or something "
    "explicitly not yet committed/confirmed, that belongs under what needs "
    "attention, regardless of how much unrelated work has landed since."
)

_QUESTION_GUARD = (
    "If the most recent or most relevant message from a coding agent poses "
    'a direct question to the user (e.g. "would you like me to implement '
    'A, B, and/or C?") or gives an explicit recommendation (e.g. '
    '"Recommendation: use brute force for now, implement the analyzer '
    'when performance becomes an issue"), include that question or '
    "recommendation in the recap -- rephrasing for clarity is fine, but "
    "preserve the actual options or recommendation rather than just noting "
    "that one was given."
)

_RESOLUTION_GUARD = (
    "A thread's status is set by its most recent resolution, not by the "
    'most detailed or alarming message in it: if the user later says "go '
    'ahead", "that\'s fine", "you can ignore X", or otherwise dismisses or '
    "answers an earlier question or concern, treat that concern as closed "
    "-- don't lead with it or describe it as still needing attention just "
    "because it generated the most discussion. This isn't only about the "
    "user dismissing something -- the same rule applies whenever a later "
    "message reports something as committed, merged, done, or fixed: that "
    "supersedes an earlier message describing the same thing as blocked, "
    "staged, or in-progress, even if the earlier message is longer or more "
    "detailed, and regardless of whether the user or the agent sent the "
    'later message. Never reconstruct a "still blocked, next step X" '
    "status from an earlier message once a later message in the same "
    "thread says that exact thing is already done. If the transcript's "
    "last message is an interruption rather than a substantive reply (e.g. "
    "a session-limit, retry, or error notice), the open item is whatever "
    "the user's last instruction before that was, not any earlier "
    'resolved concern -- phrase status like "told to do X, interrupted '
    'before doing it," not the concern that preceded that instruction.'
)

_FORMAT_GUARD = (
    'Format "text" as one paragraph per channel, each starting with the '
    'channel name in bold Markdown (e.g. "**backend**: ..."), separated '
    "from the next channel's paragraph by a blank line (a literal \\n\\n "
    "between them in the JSON string). Never merge multiple channels into "
    "a single run-on paragraph."
)

# Detailed recap's own version of _FORMAT_GUARD -- a concise recap only ever
# shows one item per channel, so "one paragraph per channel" and "one
# paragraph per item" are the same instruction there. A detailed recap can
# show several items per channel (up to config.recap.max_detailed_items), so
# the two diverge: without this, the LLM folds every shown item for a
# channel into one run-on paragraph instead of a followable list, which is
# indistinguishable from the old free-form prompt this replaced. Mirrors
# recap_detail.py's own _FORMAT_GUARD (same "**channel -- label:**" prefix,
# unconditional even for a single item) so a detailed recap's per-item look
# matches its "tell me more" follow-up's look.
_DETAILED_FORMAT_GUARD = (
    'Format "text" as one paragraph per item: each item\'s paragraph '
    "starts with its channel name and label in bold Markdown (e.g. "
    '"**swingbird -- F4:** ..."), separated from the next paragraph -- '
    "even one for the same channel -- by a blank line (a literal \\n\\n "
    "between them in the JSON string). List each channel's items together "
    "and in the priority order given, but never merge more than one item "
    "into a single paragraph. If a channel has no open item, give it one "
    'paragraph instead, starting "**channel**: ..." the same way a concise '
    "recap does. Never merge multiple channels' content together either."
)

# Two-phase in a single call: extraction first, prose second. An earlier
# version of this prompt asked for "text" and "items" as siblings with
# "text" listed (and, by the JSON shape's own field order, generated) first
# -- observed live to make source_id citation unreliable, especially for a
# channel's second/third item and for a global recap's later channels,
# since by the time the model reached "items" it had already composed the
# full narrative from its own read of the transcript and was now writing
# citations for content it had already settled, rather than the reverse.
# Reordering only the JSON *shape* to put "source_id" before the
# descriptive fields within one item (an earlier fix) helped but didn't
# close the gap, because the deeper problem was the *outer* order: "text"
# fully formed before "items" ever started. This version asks for "items"
# -- fully grounded, source_id first within each -- before "text", and
# tells "text" to narrate what "items" already extracted rather than
# re-deriving it from the transcript independently. That's the same
# "ground the citation before describing it" principle _ITEM_EXTRACTION_
# INSTRUCTIONS already applies per-item, just applied once more to the
# call as a whole, without a second LLM round trip (see this module's own
# docstring for why one call does both jobs).
_ITEM_EXTRACTION_INSTRUCTIONS = """

Before writing "text" (described above), extract every channel's current
open/actionable item(s) into "items". Each transcript message is tagged
with a short id like "m3" (e.g. "[m3] [<timestamp>] some message"). For
every item, work message-first: find the one transcript message that most
directly states or requests that item's next step, note its tag, and only
then write the item's other fields to describe what that specific message
actually says. Never write an item's label/summary/instruction first and
go looking for a citation afterward -- pick the grounding message before
you describe it. Apply this same message-first process to every item you
list for a channel, not just the first (primary) one -- a channel with
several open items needs its second, third, and later items grounded
exactly as carefully as its first, not skimmed through faster. Extract
every item this way, for every channel, before you write any part of
"text" -- "text" only narrates what you've already grounded here, so
nothing about it should be decided first.

Give each item in "items" this shape: {"channel": "<the channel name from
a \\"## <name>\\" transcript heading>", "source_id": "<the tag (e.g.
\\"m3\\") of the single transcript message you identified first, that most
directly states this item's instruction -- omit or use an empty string
only if you genuinely cannot find one specific message that states it,
which should be rare>", "label": "<a short identifier the user could refer
to later -- reuse an id like \\"F4\\" if the transcript already uses one,
otherwise a few words naming the item>", "summary": "<one clause
describing the item>", "instruction": "<the actual next step or
recommendation, preserved as closely to the transcript's own wording as
possible -- extract it, don't paraphrase or invent it>", "keywords":
["<2-4 short alternate phrases someone might later use to refer to this
item -- synonyms, a category, or a plainer description of this same item's
own label/summary/instruction, not new claims about the work; may be
empty>"]}

Include every currently open/actionable item for each channel, not just one
-- list the item "text" will narrate first for that channel, then any
other open items for that channel afterward, in whatever order they matter
most. Omit a channel from "items" entirely if it has no open/actionable
item (e.g. it said "no open item"). Never fabricate an item, a label, an
instruction, or a "source_id" that isn't grounded in the transcript --
every item is grounded independently. "keywords" is the one exception: it
doesn't need to be grounded in the transcript's own wording -- ground it in
the item's own label/summary/instruction instead, listing other natural
ways someone might refer to that same item later.

Once "items" is fully extracted, write "text" as described above, using
the item(s) you already grounded there as its source instead of
re-deriving them from the transcript a second time. Respond with JSON
only, matching this shape -- "items" before "text", since that's the order
you should actually produce them in, not just how the response is shaped:
{"items": [<one object per item, in the shape given above>], "text": "<the
recap text described above>"}"""

_CONCISE_SYSTEM_PROMPT = f"""You are a TPM agent's recap assistant. For \
each project channel, give at most one most-recent, immediately-\
actionable item: current status in one clause, then a proposed next \
step, distilled into a single decision where possible -- 1-2 sentences, \
phrased as status then next step. Describe only that one leading item -- \
if the channel has other open items, don't mention them or fold their \
content into this paragraph; a count of how many more there are is \
appended separately, not narrated by you. If a channel has no open \
item, say so briefly, and if a goal is given for it, add one short \
sentence naming that goal as what's next for the project. \
{_LAUNDERING_GUARD} {_QUESTION_GUARD} {_RESOLUTION_GUARD} {_FORMAT_GUARD} \
Skip routine chatter. Be concise -- 1-2 sentences per channel, not a \
transcript.{_ITEM_EXTRACTION_INSTRUCTIONS}"""


def _detailed_system_prompt(max_items: int) -> str:
    """Build the "detailed" recap prompt for `max_items` per channel
    (`config.recap.max_detailed_items` -- see `build_recap`).

    Deliberately built the same way `_CONCISE_SYSTEM_PROMPT` is -- lead
    item(s) then next step, per channel, sharing every guard but the format
    one -- rather than the old free-form three-bucket ("what needs
    attention / in flight / finished") prompt this replaced. That older
    prompt didn't narrate individually-referenceable items at all, so a
    "detailed recap" couldn't support the same "tell me more about F4"
    follow-ups a concise one could. Uses `_DETAILED_FORMAT_GUARD` instead of
    `_FORMAT_GUARD` since it can narrate several items per channel, each
    needing its own paragraph -- see that constant.
    A function instead of a module-level constant only because `max_items`
    is configurable and has to reach the LLM's own instructions, not just
    `_parse_recap`'s bookkeeping.
    """
    return f"""You are a TPM agent's recap assistant. For each project \
channel, give up to {max_items} of its most-recent, immediately-\
actionable items, in priority order: for each, current status in one \
clause, then a proposed next step -- 2-4 sentences per item, phrased as \
status then next step, with more concrete detail than a one-line summary \
(what was tried, why, what's blocking it, relevant numbers or file/\
function names, when the transcript has them). Describe only those \
leading items (up to {max_items} per channel) -- if the channel has more \
open items beyond that, don't mention them or fold their content into \
another item's paragraph; a count of how many more there are is appended \
separately, not narrated by you. If a channel has no open item, say so \
briefly, and if a goal is given for it, add one short sentence naming \
that goal as what's next for the project. \
{_LAUNDERING_GUARD} {_QUESTION_GUARD} {_RESOLUTION_GUARD} {_DETAILED_FORMAT_GUARD} \
Skip routine chatter. Be thorough but concise -- 2-4 sentences per item, \
not a transcript.{_ITEM_EXTRACTION_INSTRUCTIONS}"""


# Fired only for items the main extraction call already tried and failed to
# ground (RecapItem.source_event_id is None) -- never for one that already
# has a source_id, so a recap where extraction worked the first time (the
# common case, especially for a channel's primary item) never pays for a
# second call at all. This is the module docstring's "one call does both
# jobs" tradeoff applied selectively rather than abandoned: a second full
# extraction+text call for every recap would double cost unconditionally;
# a small, grounding-only call that only fires on the actual failure mode
# (observed live: worse for non-primary items and later channels in a
# global recap, where the main call's attention has already moved on to
# writing "text") costs nothing when extraction already succeeded and asks
# a much narrower question -- "which one message states this already-
# written item" -- than the main call's "read this whole transcript, write
# a recap, and cite everything as you go."
_BACKFILL_SYSTEM_PROMPT = """You are grounding work items that an earlier \
extraction pass described but couldn't confidently cite a source message \
for. You'll be given, per channel, that channel's own tagged messages \
(each shown as "[m3] <message text>") and a numbered list of items from \
that channel, each with the summary/instruction already extracted for it. \
For each item, find the one message that most directly states or \
requests it, and report that message's tag. If genuinely no single \
message in the given transcript states it, report a null source_id for \
that item rather than guessing the closest one.

Respond with JSON only, one entry per item you were given (in any order): \
{"groundings": [{"index": <the item's index number, as given>, \
"source_id": "<tag, or null if none clearly grounds it>"}]}"""


def _needs_backfill(item: RecapItem, content_map: dict[str, dict[str, str]]) -> bool:
    """An item is worth a backfill attempt only if the main call left it
    ungrounded and its channel actually has tagged messages to search --
    a channel with none (e.g. "no recent activity") has nothing a backfill
    call could find either, so it's excluded rather than sent as an empty
    section."""
    return item.source_event_id is None and bool(content_map.get(item.channel))


def _backfill_missing_sources(
    llm: LLMClient,
    items: tuple[RecapItem, ...],
    id_map: dict[str, dict[str, str]],
    content_map: dict[str, dict[str, str]],
) -> tuple[RecapItem, ...]:
    """Make one best-effort follow-up call to (re)ground any item the main
    extraction call left without a `source_event_id`, per channel that has
    at least one such item -- see `_BACKFILL_SYSTEM_PROMPT` for why this is
    cheap and narrow rather than the module docstring's usual "one call
    does both jobs" concern.

    Returns `items` unchanged (same tuple, same objects) when nothing needs
    backfilling, when the call itself fails (`LLMError` -- a recap should
    never fail because a reliability nicety on top of it did), or when the
    response doesn't parse as expected -- exactly `_resolve_tag`'s own
    "never guess" rule (`RecapItem.source_event_id` is never guessed),
    just applied to this call's response instead of the main one's.
    """
    missing = [
        (i, item) for i, item in enumerate(items) if _needs_backfill(item, content_map)
    ]
    if not missing:
        return items
    sections: dict[str, list[str]] = {}
    for index, item in missing:
        sections.setdefault(item.channel, []).append(
            f"Index {index}: summary: {item.summary}; instruction: {item.instruction}"
        )
    parts = []
    for channel, item_lines in sections.items():
        transcript = "\n".join(
            f"[{tag}] {content}" for tag, content in content_map[channel].items()
        )
        parts.append(
            f"## {channel}\n{transcript}\n\nItems needing a source:\n"
            + "\n".join(item_lines)
        )
    messages = [
        {"role": "system", "content": _BACKFILL_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    try:
        response = llm.complete_json(messages)
    except LLMError:
        return items
    groundings = response.get("groundings")
    if not isinstance(groundings, list):
        return items
    by_index = dict(missing)
    result = list(items)
    for entry in groundings:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if not isinstance(index, int) or index not in by_index:
            continue
        item = by_index[index]
        tag = entry.get("source_id")
        source_event_id = _resolve_tag(id_map, item.channel, tag)
        if source_event_id is None:
            continue
        result[index] = replace(
            item,
            source_event_id=source_event_id,
            source_content=_resolve_tag(content_map, item.channel, tag),
        )
    return tuple(result)


class RecapError(Exception):
    """Raised when a recap is requested for an unknown channel, or the LLM's
    response can't be trusted as a recap."""


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
    for referring to this same item later (see `_ITEM_EXTRACTION_INSTRUCTIONS`) --
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


@dataclass(frozen=True)
class Recap:
    text: str
    items: tuple[RecapItem, ...] = ()


def build_recap(
    llm: LLMClient,
    config: Config,
    channel_names: list[str] | None = None,
    detail: str = "concise",
) -> Recap:
    """Return a short, prioritized recap of recent channel activity.

    `channel_names` restricts the recap to those configured channels;
    omit it to recap every channel in the config. Message selection uses
    the same `config.recap.stale_after_days` time window either way -- the
    only difference is that a channel with nothing in that window is
    silently dropped from an all-channels recap, but still gets a "(no
    recent activity)" paragraph when named explicitly, since the user
    asked about it by name and should get an answer, not silence.
    `detail` selects "concise" (default, one actionable item per project)
    or "detailed" (up to `config.recap.max_detailed_items` per project,
    each with more detail -- anything else falls back to concise).
    """
    channels = _select_channels(config, channel_names)
    transcript, id_map, content_map = _build_transcript(
        channels,
        stale_after_days=config.recap.stale_after_days,
        max_messages_per_channel=config.recap.max_messages_per_channel,
        explicit=channel_names is not None,
    )
    max_items_per_channel = (
        config.recap.max_detailed_items if detail == "detailed" else 1
    )
    system_prompt = (
        _detailed_system_prompt(max_items_per_channel)
        if detail == "detailed"
        else _CONCISE_SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    recap = _parse_recap(
        llm.complete_json(messages), id_map, content_map, max_items_per_channel
    )
    recap = Recap(
        text=recap.text,
        items=_backfill_missing_sources(llm, recap.items, id_map, content_map),
    )
    recap = Recap(
        text=_ensure_channel_paragraphs(recap.text, recap.items, detail),
        items=recap.items,
    )
    recap = Recap(
        text=_append_item_counts(recap.text, recap.items, detail), items=recap.items
    )
    recap = Recap(
        text=_append_source_links(recap.text, recap.items, config, detail),
        items=recap.items,
    )
    return recap


def _ensure_channel_paragraphs(
    text: str, items: tuple[RecapItem, ...], detail: str
) -> str:
    """Append a deterministic fallback paragraph for any channel whose
    primary item(s) came back in `items` but never got a paragraph in
    `text` at all.

    `_ITEM_EXTRACTION_INSTRUCTIONS` has the LLM extract `items` before
    writing `text`, so `items` can be correctly grounded for a channel even
    when the LLM's own narration skips that channel entirely -- observed
    live on a global, multi-channel recap: the busiest channel (most items,
    most competing content) was left out of `text` two times out of three,
    while `items` still had its entries. This is the same "attention
    degrades across a long single generation" failure `_backfill_missing_
    sources` already addresses for a single item's citation, just at the
    coarser "did this channel get a paragraph at all" level -- and, like
    that backfill, it's a deterministic patch rather than a second LLM
    call, since the content to render (`item.summary`) is already grounded.

    Only fires for a channel that's missing outright -- a channel `_append_
    item_counts` can already find a paragraph for (even a stray or
    malformed one) is left alone here. Checks for the same `"**{channel}"`
    prefix `_append_item_counts`/`_append_source_links` match against, so a
    fallback paragraph this adds is itself indistinguishable from an
    LLM-written one to those two passes that run after it.
    """
    paragraphs = text.split("\n\n")
    channels_with_items = {item.channel for item in items if item.is_primary}
    channels_present = {
        channel
        for channel in channels_with_items
        if any(p.startswith(f"**{channel}") for p in paragraphs)
    }
    missing_items = [
        item
        for item in items
        if item.is_primary and item.channel not in channels_present
    ]
    if not missing_items:
        return text
    fallback = "\n\n".join(
        f"{_item_paragraph_prefix(item.channel, item.label, detail)} {item.summary}"
        for item in missing_items
    )
    return f"{text}\n\n{fallback}" if text.strip() else fallback


def _parse_recap(
    response: dict,
    id_map: dict[str, dict[str, str]],
    content_map: dict[str, dict[str, str]],
    max_items_per_channel: int,
) -> Recap:
    text = response.get("text")
    if not isinstance(text, str):
        raise RecapError(f"LLM response is missing recap text: {json.dumps(response)}")
    items = []
    channel_counts: dict[str, int] = {}
    for item in response.get("items") or []:
        if not isinstance(item, dict) or not item.get("instruction"):
            continue
        channel = item.get("channel", "")
        # The LLM lists a channel's items in priority order (see
        # _ITEM_EXTRACTION_INSTRUCTIONS) -- the first max_items_per_channel seen for a
        # channel are the ones "text" itself narrates, everything after is
        # one of the "additional" items (see RecapItem.is_primary).
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        is_primary = channel_counts[channel] <= max_items_per_channel
        items.append(
            RecapItem(
                channel=channel,
                label=item.get("label", ""),
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
    return Recap(text=text, items=tuple(items))


def _parse_keywords(raw: object) -> tuple[str, ...]:
    """Coerce the LLM's "keywords" field into a tuple of non-empty strings,
    silently dropping anything malformed (a non-list, or non-string/blank
    entries) rather than raising -- keywords are a matching aid, not load-
    bearing content, so a malformed entry should never fail the whole
    recap."""
    if not isinstance(raw, list):
        return ()
    return tuple(k.strip() for k in raw if isinstance(k, str) and k.strip())


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


def _item_paragraph_prefix(channel: str, label: str, detail: str) -> str:
    """Return the bold Markdown prefix that starts an item's own paragraph
    in recap "text", per `_FORMAT_GUARD` (concise: "**channel**:") or
    `_DETAILED_FORMAT_GUARD` (detailed: "**channel -- label:**", matching
    `recap_detail.py`'s own per-item format).
    """
    if detail == "detailed":
        return f"**{channel} -- {label}:**"
    return f"**{channel}**:"


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
        if cleaned.strip() in stray_bare_prefixes:
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


def _append_source_links(
    text: str, items: tuple[RecapItem, ...], config: Config, detail: str
) -> str:
    """Append a Buzz message link to `text` for every primary item's
    `source_event_id`, when the LLM grounded it in one specific transcript
    message (see `RecapItem.source_event_id`).

    A concise recap has one paragraph per channel (`**channel**:`), so every
    primary item for that channel shares it and a link lands on the same
    paragraph, one per line, in the order `items` lists them. A detailed
    recap gives each primary item its own paragraph (`**channel --
    label**:`, see `_DETAILED_FORMAT_GUARD`), so each item's link lands on
    its own paragraph instead -- mirrors `recap_detail.py`'s own
    `_append_source_links`, which uses this same per-item format
    unconditionally since it only ever elaborates on already-selected items.

    Only ever a primary item -- those are the only ones `text` actually
    narrates (see `_ITEM_EXTRACTION_INSTRUCTIONS`); a non-primary item is just
    counted by `_append_item_counts`, never described, so there's no
    content of its own for a link to attach to.

    `config.channel_by_name(item.channel)` is guaranteed to succeed
    whenever `source_event_id` is set: `_resolve_tag` only ever resolves
    one from `id_map`, whose keys are exactly the names of this same
    `config`'s own channels (see `_build_transcript`) -- so there's no
    `None`-channel case to guard here, unlike a `RecapItem` that might have
    come from anywhere else (see `recap_detail._append_source_links`,
    which does need that guard).
    """
    for item in items:
        if not item.is_primary or item.source_event_id is None:
            continue
        channel = config.channel_by_name(item.channel)
        link = outbound.message_link(channel.id, item.source_event_id)
        prefix = _item_paragraph_prefix(item.channel, item.label, detail)
        text = outbound.append_paragraph_link(text, prefix, link)
    return text


def _resolve_tag(
    mapping: dict[str, dict[str, str]], channel: str, tag: object
) -> str | None:
    """Resolve an LLM-cited `tag` (e.g. "m3") against `mapping` for `channel`.

    Never trusts the LLM's tag blindly -- a missing tag, an empty string, an
    unknown channel, or a tag that doesn't match any message actually shown
    for that channel all resolve to `None` (a plain dict miss, in the latter
    three cases) rather than a guess. Shared by both `RecapItem.
    source_event_id` and `source_content`, resolved from the same tag against
    two parallel maps built in `_build_transcript`."""
    if not isinstance(tag, str):
        return None
    return mapping.get(channel, {}).get(tag)


def _select_channels(config: Config, channel_names: list[str] | None):
    if channel_names is None:
        return config.channels
    selected = []
    for name in channel_names:
        channel = config.channel_by_name(name)
        if channel is None:
            raise RecapError(f"unknown channel: {name!r}")
        selected.append(channel)
    return selected


def _build_transcript(
    channels: list[ChannelConfig],
    stale_after_days: int,
    max_messages_per_channel: int,
    explicit: bool,
) -> tuple[str, dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Return the transcript text and `{channel_name: {tag: event_id}}` /
    `{channel_name: {tag: content}}` maps.

    Each message is tagged with a short per-channel local id ("m1", "m2",
    ...) rather than its real (64-char) event id -- cheap for the LLM to
    copy back verbatim in `source_id` (see `_ITEM_EXTRACTION_INSTRUCTIONS`) without
    risking a garbled hex string. Both maps only get an entry for messages
    that actually carry an `"id"` -- real `buzz messages get` events always
    do; this just means a message without one can't be cited as a source,
    never a crash. The content map exists alongside the id map so a
    resolved `source_id` can carry the message's own text forward too (see
    `RecapItem.source_content`), not just its event id.
    """
    cutoff = time.time() - stale_after_days * 86400
    sections = []
    id_map: dict[str, dict[str, str]] = {}
    content_map: dict[str, dict[str, str]] = {}
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
            lines = []
            tags = {}
            contents = {}
            for i, event in enumerate(events, start=1):
                tag = f"m{i}"
                lines.append(f"[{tag}] [{event['created_at']}] {event['content']}")
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
