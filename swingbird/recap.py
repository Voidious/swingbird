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
from swingbird.closed_items import ClosedItem, ClosedItemStore
from swingbird.config import ChannelConfig, Config
from swingbird.history import fetch_messages_since
from swingbird.llm import LLMClient, LLMError

from .recap_counts import (
    RecapItem,  # fmt: skip
    _append_item_counts,
    _item_paragraph_prefix,
)

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

_CLOSED_MARKER_GUARD = (
    'A transcript message tagged "[closed]" describes work the user has '
    "already marked closed (see the closed-items list below, when there is "
    "one) -- never extract a new item grounded only in that message. Only "
    "extract an item touching that same work if a different, un-tagged "
    "message in the same transcript shows it was reopened or something "
    "new is being asked, genuinely distinct from what was closed."
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


# Two-phase in a single call: extraction first, prose second, and -- as of
# this version -- extraction is fully described and *capless* before any
# per-channel display limit is mentioned anywhere in the prompt. An earlier
# version put "text"'s own "give up to N items" framing in the prompt's
# opening sentence and only appended _item_extraction_instructions (with its
# own "include every item, not just one") afterward. Observed live
# (`max_detailed_items` set to 1, a channel with several genuinely open
# items): the model treated "up to N" as the task's overall scope rather
# than a "text"-only limit, so "items" itself came back capped at N too,
# indistinguishable from there being no more open items at all -- silently
# breaking the "N additional open items" fold (`_append_item_counts`), which
# depends on "items" listing everything, and `recap_actions.py`'s "what are
# the other items" lookup, which depends on the same thing. Reordering only
# the JSON *shape* to put "source_id" before the descriptive fields within
# one item (a still-earlier fix) helped source citation the same way for the
# same reason, but didn't close this gap, because the deeper problem was
# again the *outer* order: whichever instruction the model reads as framing
# the whole task first is the one whose limit it applies everywhere, not
# just where that instruction actually says. This version puts capless
# extraction first, with the per-channel display limit introduced afterward
# and stated explicitly as bounding only "text", never "items" -- the same
# "ground the citation before describing it" principle applied once more to
# the call as a whole, without a second LLM round trip (see this module's
# own docstring for why one call does both jobs).
#
# Capless extraction fixed one over-counting bug but exposed another:
# observed live on this very channel's own recap, a detailed recap named 18
# distinct "items" for a single project, but several of those names --
# e.g. "Message-ID reliability" / "Merge recap follow-up fixes" / "Improve
# source-message citations for recap items" -- were the same underlying
# branch restated across this channel's own status updates as work on it
# progressed (proposed, then implemented, then tested, then ready to
# merge). Each restatement independently satisfied "an open/actionable
# item," so extraction correctly found many messages but incorrectly
# treated each message as its own item, rather than recognizing several of
# them as later updates on the same open thread of work. The dedup
# paragraph below asks the model to fold those together *before* deciding
# how many entries a channel gets, using its current status rather than
# picking one restatement arbitrarily -- the same "don't reconstruct an
# earlier status once a later message updates it" principle _RESOLUTION_
# GUARD already applies to a single item's own status, extended here to
# recognizing *that two messages are about the same item* in the first
# place, which has to happen before _RESOLUTION_GUARD's rule can even
# apply.
def _item_extraction_instructions(detail: str) -> str:
    """Build the "items" extraction instructions shared by both system
    prompts, appending an extra paragraph in detailed mode asking each
    item's own "summary"/"instruction" to carry the same 2-4-sentences'
    worth of concrete detail the detailed "text" narration below asks for
    its own (primary) items -- not just the fixed one-clause/one-line
    version this instruction set otherwise asks for regardless of mode.

    Without this, `RecapItem.summary`/`instruction` came out at the same
    fixed detail level in concise and detailed recaps alike, since this
    instruction text used to be a single module-level constant shared
    verbatim by both prompts. That's fine for the *primary* item(s) a
    recap actually narrates -- "text" is written fresh by the LLM at the
    right level either way -- but `recap_list.render_items` builds "the
    other items" directly from `summary`/`instruction` with no LLM call of
    its own (see that module's docstring for why), so those items stayed
    stuck at the concise level even in a detailed recap. Asking for more
    detail here, still grounded in the transcript the same way, fixes that
    without a second LLM call.
    """
    detailed_note = (
        """

Since this is a detailed recap, write each item's "summary" and \
"instruction" with the same richness the detailed narration below gives \
its own items -- 2-4 sentences' worth of concrete detail (what was tried, \
why, what's blocking it, relevant numbers or file/function names, when \
the transcript has them), not just a one-clause/one-line version. This \
keeps a plain listing of this item (e.g. "what are the other items") at \
the same detail level as the recap that surfaced it, since that listing \
never re-elaborates beyond what's extracted here."""
        if detail == "detailed"
        else ""
    )
    return (
        """

Extract every channel's current open/actionable item(s) into "items" --
every one of them, with no cap on how many -- before writing anything
else. Each transcript message is tagged with a short id like "m3" (e.g.
"[m3] [<timestamp>] some message"). For every item, work message-first:
find the one transcript message that most directly states or requests
that item's next step, note its tag, and only then write the item's other
fields to describe what that specific message actually says. Never write
an item's label/summary/instruction first and go looking for a citation
afterward -- pick the grounding message before you describe it. Apply
this same message-first process to every item you list for a channel, not
just the first (primary) one -- a channel with several open items needs
its second, third, and later items grounded exactly as carefully as its
first, not skimmed through faster.

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

Before deciding how many entries a channel gets, check whether two or
more candidate items are actually the same underlying piece of work --
the same PR, branch, commit, or feature -- rather than genuinely separate
open items, even when different messages phrase it differently, name it
differently, or catch it at a different stage of its own progress (e.g.
proposed, then committed, then merged, then integrated). A channel's own
history often restates the same open thread of work several times as
work on it advances; that is one item, not a new one each time it's
mentioned again. Fold every restatement of the same underlying work into
a single entry, grounded (via the message-first rule above) in whichever
one message states that work's most current status and next step, not
whichever message happens to be easiest to cite or most recent in the
transcript. Only give separate entries to genuinely separate pieces of
open work -- distinct PRs, branches, or features -- never split one
because it was discussed more than once.

Include every currently open/actionable item for each channel, however
many that is -- list the item you'll narrate first for that channel (the
instructions below explain how many of them "text" actually narrates,
which is a separate question from how many belong here), then any other
open items for that channel afterward, in whatever order they matter
most. Nothing below -- including any limit on how many items "text"
describes -- ever shrinks this list: a channel with ten open items still
gets ten entries in "items" even when "text" only narrates its first one
or two. Omit a channel from "items" entirely if it has no open/actionable
item (e.g. it said "no open item"). Never fabricate an item, a label, an
instruction, or a "source_id" that isn't grounded in the transcript --
every item is grounded independently. "keywords" is the one exception: it
doesn't need to be grounded in the transcript's own wording -- ground it in
the item's own label/summary/instruction instead, listing other natural
ways someone might refer to that same item later."""
        + detailed_note
    )


# Placed after both "items" and "text" are fully described, so "described
# above" always means what it says regardless of which recap mode built the
# rest of the prompt (see _item_extraction_instructions's own comment for
# why that ordering matters).
_RESPONSE_SHAPE_INSTRUCTIONS = """

Ground "text" in the "items" you already extracted above, instead of
re-deriving them from the transcript a second time. Respond with JSON
only, matching this shape -- "items" before "text", since that's the order
you should actually produce them in, not just how the response is shaped:
{"items": [<one object per item, in the shape given above>], "text": "<the
recap text described above>"}"""

_CONCISE_SYSTEM_PROMPT = f"""You are a TPM agent's recap assistant.\
{_item_extraction_instructions("concise")}

For each project channel, narrate in "text" only its one most-recent, \
immediately-actionable item: current status in one clause, then a \
proposed next step, distilled into a single decision where possible -- \
1-2 sentences, phrased as status then next step. This limits only what \
"text" describes, never how many items belong in "items" above -- \
describe only that one leading item here -- if the channel has other \
open items (already captured in "items"), don't mention them or fold \
their content into this paragraph; a count of how many more there are is \
appended separately, not narrated by you. If a channel has no open \
item, say so briefly, and if a goal is given for it, add one short \
sentence naming that goal as what's next for the project. \
{_LAUNDERING_GUARD} {_QUESTION_GUARD} {_RESOLUTION_GUARD} {_FORMAT_GUARD} \
{_CLOSED_MARKER_GUARD} \
Skip routine chatter. Be concise -- 1-2 sentences per channel, not a \
transcript.{_RESPONSE_SHAPE_INSTRUCTIONS}"""


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
    `_parse_recap`'s bookkeeping. `max_items` only ever bounds the "text"
    paragraph below -- see `_item_extraction_instructions`'s own docstring
    for why it's never mentioned any earlier in this prompt, where
    extraction is described.
    """
    return f"""You are a TPM agent's recap assistant.\
{_item_extraction_instructions("detailed")}

For each project channel, narrate in "text" up to {max_items} of its \
most-recent, immediately-actionable items, in the priority order you \
already extracted them in above: for each, current status in one clause, \
then a proposed next step -- 2-4 sentences per item, phrased as status \
then next step, with more concrete detail than a one-line summary (what \
was tried, why, what's blocking it, relevant numbers or file/function \
names, when the transcript has them). This {max_items} cap limits only \
how many items "text" describes, never how many belong in "items" above \
-- a channel can have far more than {max_items} entries there; describe \
only the leading {max_items} of them here -- if the channel has more open \
items beyond that, don't mention them in "text" or fold their content \
into another item's paragraph; a count of how many more there are is \
appended separately, not narrated by you. If a channel has no open item, \
say so briefly, and if a goal is given for it, add one short sentence \
naming that goal as what's next for the project. \
{_LAUNDERING_GUARD} {_QUESTION_GUARD} {_RESOLUTION_GUARD} {_DETAILED_FORMAT_GUARD} \
{_CLOSED_MARKER_GUARD} \
Skip routine chatter. Be thorough but concise -- 2-4 sentences per item, \
not a transcript.{_RESPONSE_SHAPE_INSTRUCTIONS}"""


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


def _format_closed_items_guard(closed: tuple[ClosedItem, ...]) -> str:
    """Build the dynamic (per-call) system prompt addition listing every
    closed item in scope for this recap (see `build_recap`'s `closed_items`
    handling), so extraction never re-lists that work as open even when a
    transcript message restates it -- mirrors `_RESOLUTION_GUARD`'s "a
    later resolution supersedes an earlier open question" reasoning, just
    anchored on a durable closed-items record instead of a within-
    transcript resolution. Belt-and-suspenders alongside `_CLOSED_MARKER_
    GUARD`'s `[closed]` transcript tag (`_build_transcript`) -- that tag
    only fires when the exact message that grounded the closed item is
    still in this recap's window, while this guard's prose covers every
    closed item regardless, including one restated on a brand-new message.
    A guard, not transcript surgery: stripping the original message out of
    the transcript would risk losing context other, still-open items in
    the same message need.

    Empty (returns "") when there's nothing closed in scope, so a recap
    with no closed items doesn't grow its prompt for no reason.
    """
    if not closed:
        return ""
    lines = [f"- {item.channel}: {item.label} -- {item.instruction}" for item in closed]
    return (
        "\n\nThe following work has already been marked closed by the "
        "user and must never be listed as an open/actionable item again, "
        "even if a transcript message restates or re-describes it -- only "
        "include it if the transcript shows something genuinely new (the "
        "work was reopened, or there's a new, distinct ask), not just a "
        "restatement of what's below:\n" + "\n".join(lines)
    )


class RecapError(Exception):
    """Raised when a recap is requested for an unknown channel, or the LLM's
    response can't be trusted as a recap."""


@dataclass(frozen=True)
class Recap:
    text: str
    items: tuple[RecapItem, ...] = ()


def build_recap(
    llm: LLMClient,
    config: Config,
    channel_names: list[str] | None = None,
    detail: str = "concise",
    closed_items: ClosedItemStore | None = None,
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

    `closed_items` (see `closed_items.py`) supplies every item the user has
    marked closed within `config.recap.closed_item_window_days`, scoped to
    `channels` -- omitted (`None`) only by tests that don't care about
    closing; `daemon.py` always passes the real store. The window is
    floored at `config.recap.stale_after_days` (never narrower than the
    recap's own message window) so a closed item still young enough for
    its own restatement to appear in `transcript` can never fall outside
    the guard meant to suppress it -- see `RecapConfig.closed_item_window_
    days`'s own docstring.
    """
    channels = _select_channels(config, channel_names)
    closed = (
        ()
        if closed_items is None
        else closed_items.for_channels(
            {channel.name for channel in channels},
            since=time.time()
            - max(config.recap.closed_item_window_days, config.recap.stale_after_days)
            * 86400,
        )
    )
    transcript, id_map, content_map = _build_transcript(
        channels,
        stale_after_days=config.recap.stale_after_days,
        max_messages_per_channel=config.recap.max_messages_per_channel,
        explicit=channel_names is not None,
        closed_ids_by_channel=_closed_ids_by_channel(closed),
    )
    max_items_per_channel = (
        config.recap.max_detailed_items if detail == "detailed" else 1
    )
    system_prompt = (
        _detailed_system_prompt(max_items_per_channel)
        if detail == "detailed"
        else _CONCISE_SYSTEM_PROMPT
    ) + _format_closed_items_guard(closed)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    recap = _parse_recap(
        llm.complete_json(messages), id_map, content_map, max_items_per_channel
    )
    recap = Recap(
        text=_dedupe_item_paragraphs(recap.text, recap.items, detail),
        items=recap.items,
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


_DETAILED_PARAGRAPH_HEADER_RE = re.compile(
    r"^\*\*(?P<channel>[^*\n]+?) -- (?P<label>[^*\n]+?):\*\*"
)


def _add_if_unique(seen: set, key) -> bool:
    if key in seen:
        return False
    seen.add(key)
    return True


def _dedupe_item_paragraphs(
    text: str, items: tuple[RecapItem, ...], detail: str
) -> str:
    """Drop a later detailed-recap paragraph that restates an earlier one's
    same item under a differently-punctuated label.

    `_item_extraction_instructions` asks the LLM to fold every restatement
    of the same underlying work into one "items" entry before writing
    "text" at all -- but that's a prompt request, not an enforced
    constraint, and it governs "items", not the free-form "text" prose
    written from it. Observed live: a detailed recap wrote two separate
    paragraphs for the very same status update, headed "**swingbird --
    recap-close:**" and "**swingbird -- recap close:**", differing only by
    a hyphen vs a space in the label. `_parse_recap` already applies this
    same normalize-and-compare logic to dedupe "items" itself (see its own
    docstring for that half of the fix), but "text" is the LLM's own
    verbatim wording, not reconstructed from "items" -- so a duplicate
    paragraph there isn't guaranteed to disappear just because "items" no
    longer has a matching duplicate entry.

    Rewrites the surviving paragraph's header to its matching item's own
    canonical prefix (`_item_paragraph_prefix`, built from the already-
    normalized `item.label`), not just whichever raw spelling the LLM wrote
    -- otherwise the paragraph that survives dedup could keep a hyphenated
    header like "**swingbird -- recap-close:**" while `item.label` was
    normalized to "recap close", and `_ensure_channel_paragraphs`'s own
    exact-prefix match would then treat the item as still missing and
    append a second, fallback paragraph for it right back. A paragraph
    whose header doesn't match any known item (e.g. a "no open item"
    paragraph, or the LLM naming a channel/label "items" doesn't have) is
    still deduped by its own normalized header but otherwise left as
    written, since there's no canonical form to rewrite it to.

    Never touches a concise recap's "**channel**:" paragraphs (no label to
    compare) or a detailed "no open item" paragraph in that same shape --
    only the per-item header `_DETAILED_FORMAT_GUARD` actually asks for.
    """
    if detail != "detailed":
        return text
    canonical: dict[tuple[str, str], str] = {}
    for item in items:
        key = (item.channel.strip().casefold(), item.label.strip().casefold())
        canonical.setdefault(
            key, _item_paragraph_prefix(item.channel, item.label, detail)
        )
    paragraphs = text.split("\n\n")
    seen: set[tuple[str, str]] = set()
    kept = []
    for paragraph in paragraphs:
        match = _DETAILED_PARAGRAPH_HEADER_RE.match(paragraph)
        if match:
            key = (
                _normalize_label(match.group("channel")).strip().casefold(),
                _normalize_label(match.group("label")).strip().casefold(),
            )
            if not _add_if_unique(seen, key):
                continue
            prefix = canonical.get(key)
            if prefix is not None:
                paragraph = prefix + paragraph[match.end() :]
        kept.append(paragraph)
    return "\n\n".join(kept)


def _ensure_channel_paragraphs(
    text: str, items: tuple[RecapItem, ...], detail: str
) -> str:
    """Append a deterministic fallback paragraph for any primary item that
    never got its own paragraph in `text`.

    `_item_extraction_instructions` has the LLM extract `items` before
    writing `text`, so `items` can be correctly grounded for a channel even
    when the LLM's own narration skips it, or part of it, entirely --
    observed live on a global, multi-channel recap: the busiest channel
    (most items, most competing content) was left out of `text` two times
    out of three, while `items` still had its entries. This is the same
    "attention degrades across a long single generation" failure
    `_backfill_missing_sources` already addresses for a single item's
    citation, just at the "did this item get its own paragraph" level --
    and, like that backfill, it's a deterministic patch rather than a
    second LLM call, since the content to render (`item.summary`) is
    already grounded.

    Checks each primary item's own paragraph prefix (`_item_paragraph_
    prefix`), not just whether the channel has any paragraph at all --
    observed live in detailed mode (`max_detailed_items` > 1): a channel
    with several primary items can get a paragraph for some of them but not
    all, and a channel-level check would wrongly treat the whole channel as
    covered by the ones that did land, silently dropping the rest along
    with the "N additional open items" fold note that's anchored to the
    last primary item's own paragraph (`_append_item_counts`). A concise
    recap only ever has one primary item per channel, so this is equivalent
    to the old per-channel check there. Matches the same per-item approach
    `recap_detail._backfill_missing_paragraphs` already uses for the "tell
    me more" elaboration case.

    Inserts each fallback right after that item's own channel's last
    existing paragraph (or at the very end if the channel has none yet),
    rather than always at the end of the whole `text` -- observed live: a
    fallback for an earlier channel, added after a later channel's
    paragraph was already in `text`, landed after that later channel's
    paragraph too, breaking `_FORMAT_GUARD`/`_DETAILED_FORMAT_GUARD`'s
    "grouped by channel" contract instead of merely restating the missing
    item. Keeping every channel's paragraphs contiguous also keeps
    `_append_item_counts`'s own "last primary item's own paragraph" lookup
    correct for that channel regardless of which other channels come after
    it in `text`.
    """
    paragraphs = text.split("\n\n") if text.strip() else []
    missing_items = [
        item
        for item in items
        if item.is_primary
        and not any(
            p.startswith(_item_paragraph_prefix(item.channel, item.label, detail))
            for p in paragraphs
        )
    ]
    if not missing_items:
        return text
    for item in missing_items:
        fallback = (
            f"{_item_paragraph_prefix(item.channel, item.label, detail)} {item.summary}"
        )
        insert_at = len(paragraphs)
        for i, paragraph in enumerate(paragraphs):
            if paragraph.startswith(f"**{item.channel}"):
                insert_at = i + 1
        paragraphs.insert(insert_at, fallback)
    return "\n\n".join(paragraphs)


def _parse_recap(
    response: dict,
    id_map: dict[str, dict[str, str]],
    content_map: dict[str, dict[str, str]],
    max_items_per_channel: int,
) -> Recap:
    items = []
    channel_counts: dict[str, int] = {}
    seen_labels: dict[str, set[str]] = {}
    for item in response.get("items") or []:
        if not isinstance(item, dict) or not item.get("instruction"):
            continue
        channel = item.get("channel", "")
        label = _normalize_label(item.get("label", ""))
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
    return Recap(text=text, items=tuple(items))


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


def _parse_keywords(raw: object) -> tuple[str, ...]:
    """Coerce the LLM's "keywords" field into a tuple of non-empty strings,
    silently dropping anything malformed (a non-list, or non-string/blank
    entries) rather than raising -- keywords are a matching aid, not load-
    bearing content, so a malformed entry should never fail the whole
    recap."""
    if not isinstance(raw, list):
        return ()
    return tuple(k.strip() for k in raw if isinstance(k, str) and k.strip())


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
    narrates (see `_item_extraction_instructions`); a non-primary item is just
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


def _closed_ids_by_channel(closed: tuple[ClosedItem, ...]) -> dict[str, set[str]]:
    """Return `{channel_name: {source_event_id, ...}}` for every closed item
    that was grounded in one specific message -- an item closed without a
    `source_event_id` (see `RecapItem.source_event_id`) has no transcript
    message to tag, so it's covered only by `_format_closed_items_guard`'s
    prose, not this per-message marker (see `_build_transcript`)."""
    by_channel: dict[str, set[str]] = {}
    for item in closed:
        if item.source_event_id is not None:
            by_channel.setdefault(item.channel, set()).add(item.source_event_id)
    return by_channel


def _build_transcript(
    channels: list[ChannelConfig],
    stale_after_days: int,
    max_messages_per_channel: int,
    explicit: bool,
    closed_ids_by_channel: dict[str, set[str]],
) -> tuple[str, dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Return the transcript text and `{channel_name: {tag: event_id}}` /
    `{channel_name: {tag: content}}` maps.

    Each message is tagged with a short per-channel local id ("m1", "m2",
    ...) rather than its real (64-char) event id -- cheap for the LLM to
    copy back verbatim in `source_id` (see `_item_extraction_instructions`) without
    risking a garbled hex string. Both maps only get an entry for messages
    that actually carry an `"id"` -- real `buzz messages get` events always
    do; this just means a message without one can't be cited as a source,
    never a crash. The content map exists alongside the id map so a
    resolved `source_id` can carry the message's own text forward too (see
    `RecapItem.source_content`), not just its event id.

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
            for i, event in enumerate(events, start=1):
                tag = f"m{i}"
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
