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
import time
from dataclasses import dataclass

from swingbird.config import ChannelConfig, Config
from swingbird.history import fetch_messages_since, fetch_recent_messages
from swingbird.llm import LLMClient

# Only used for an explicitly-named channel, where staleness never applies
# and we just want "whatever's most recent" rather than a time window.
EXPLICIT_CHANNEL_MESSAGE_LIMIT = 50

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

_FORMAT_GUARD = (
    'Format "text" as one paragraph per channel, each starting with the '
    'channel name in bold Markdown (e.g. "**backend**: ..."), separated '
    "from the next channel's paragraph by a blank line (a literal \\n\\n "
    "between them in the JSON string). Never merge multiple channels into "
    "a single run-on paragraph."
)

_ITEMS_INSTRUCTIONS = """

Each transcript message is tagged with a short id like "m3" (e.g. "[m3]
[<timestamp>] some message"). Respond with JSON only, matching this shape:
{"text": "<the recap text described above>", "items": [{"channel":
"<the channel name from a \\"## <name>\\" transcript heading>", "label":
"<a short identifier the user could refer to later -- reuse an id like
\\"F4\\" if the transcript already uses one, otherwise a few words naming
the item>", "summary": "<one clause describing the item>", "instruction":
"<the actual next step or recommendation, preserved as closely to the
transcript's own wording as possible -- extract it, don't paraphrase or
invent it>", "source_id": "<the tag (e.g. \\"m3\\") of the single transcript
message that most directly states this instruction -- omit or use an empty
string if it isn't clearly grounded in one specific message>"}]}

Include every currently open/actionable item for each channel, not just one
-- list the same leading item "text" already narrates for that channel
first, then any other open items for that channel afterward, in whatever
order they matter most. Omit a channel from "items" entirely if it has no
open/actionable item (e.g. it said "no open item"). Never fabricate an item,
a label, an instruction, or a "source_id" that isn't grounded in the
transcript -- every item is grounded independently, exactly like the single
leading item was before."""

_CONCISE_SYSTEM_PROMPT = f"""You are a TPM agent's recap assistant. For \
each project channel, give at most one most-recent, immediately-\
actionable item: current status in one clause, then a proposed next \
step, distilled into a single decision where possible -- 1-2 sentences, \
phrased as status then next step. If there are additional open items \
beyond the one you lead with, note how many there are rather than \
listing them. If a channel has no open item, say so briefly, and if a \
goal is given for it, add one short sentence naming that goal as what's \
next for the project. {_LAUNDERING_GUARD} {_QUESTION_GUARD} {_FORMAT_GUARD} \
Skip routine chatter. Be concise -- 1-2 sentences per channel, not a \
transcript.{_ITEMS_INSTRUCTIONS}"""

_DETAILED_SYSTEM_PROMPT = f"""You are a TPM agent's recap assistant. \
Given recent messages from one or more project channels, write a short, \
prioritized summary: lead with what needs the user's attention \
(blockers, decisions needed, open questions), then what's in flight, \
then what finished recently. {_LAUNDERING_GUARD} {_QUESTION_GUARD} \
{_FORMAT_GUARD} Skip routine chatter. Be concise -- a few sentences per \
channel, not a transcript.{_ITEMS_INSTRUCTIONS}"""

_SYSTEM_PROMPTS = {
    "concise": _CONCISE_SYSTEM_PROMPT,
    "detailed": _DETAILED_SYSTEM_PROMPT,
}


class RecapError(Exception):
    """Raised when a recap is requested for an unknown channel, or the LLM's
    response can't be trusted as a recap."""


@dataclass(frozen=True)
class RecapItem:
    """One channel's current actionable item, extracted alongside the recap
    text so a later "go ahead with X" DM can refer back to it (see
    `recap_actions.py`) without re-parsing rendered prose.

    `is_primary` marks the one item per channel that "text" itself narrates
    (the first item the LLM listed for that channel, per `_ITEMS_
    INSTRUCTIONS` -- see `_parse_recap`); every other item for that channel
    is one of the "N additional/open items" the recap text only counts.
    `resolve_reference` (`recap_actions.py`) uses this to answer "what are
    the other items" with exactly the non-primary ones.

    `source_event_id` is the transcript message the instruction was grounded
    in, when the LLM could point to one specific message -- it's what lets a
    later `recap_action` thread its relayed dispatch as a reply to that
    message instead of posting disconnected from it (see `daemon.py`).
    `source_content` is that same message's own text, carried alongside so a
    later dispatch-phrasing rewrite (see `dispatch_phrasing.py`) can ground
    itself in the coding agent's own wording instead of just the recap's
    condensed summary/instruction. Both are `None` under the same
    conditions -- nothing single message grounds the item; never guessed."""

    channel: str
    label: str
    summary: str
    instruction: str
    is_primary: bool = True
    source_event_id: str | None = None
    source_content: str | None = None


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
    omit it to recap every channel in the config (channels stale per
    `config.recap.stale_after_days` are silently dropped from that
    all-channels case only -- naming a channel explicitly always
    includes it, ignoring staleness). `detail` selects "concise"
    (default, one actionable item per project) or "detailed" (today's
    three-bucket summary).
    """
    channels = _select_channels(config, channel_names)
    transcript, id_map, content_map = _build_transcript(
        channels,
        stale_after_days=config.recap.stale_after_days,
        max_messages_per_channel=config.recap.max_messages_per_channel,
        explicit=channel_names is not None,
    )
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPTS.get(detail, _CONCISE_SYSTEM_PROMPT),
        },
        {"role": "user", "content": transcript},
    ]
    return _parse_recap(llm.complete_json(messages), id_map, content_map)


def _parse_recap(
    response: dict,
    id_map: dict[str, dict[str, str]],
    content_map: dict[str, dict[str, str]],
) -> Recap:
    text = response.get("text")
    if not isinstance(text, str):
        raise RecapError(f"LLM response is missing recap text: {json.dumps(response)}")
    items = []
    seen_channels: set[str] = set()
    for item in response.get("items") or []:
        if not isinstance(item, dict) or not item.get("instruction"):
            continue
        channel = item.get("channel", "")
        # The LLM lists a channel's leading item first (see
        # _ITEMS_INSTRUCTIONS) -- the first item seen for a channel is its
        # primary one, everything after is one of the "additional" items.
        is_primary = channel not in seen_channels
        seen_channels.add(channel)
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
            )
        )
    return Recap(text=text, items=tuple(items))


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
    copy back verbatim in `source_id` (see `_ITEMS_INSTRUCTIONS`) without
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
        if explicit:
            # Staleness never applies to a channel the user named on
            # purpose -- just show whatever's most recent, however old.
            events = fetch_recent_messages(
                channel.id, limit=EXPLICIT_CHANNEL_MESSAGE_LIMIT
            )
        else:
            # Time-windowed, paging past the relay's 200-per-call cap as
            # needed (see history.fetch_messages_since) so a chatty
            # channel can't push a still-relevant item out of the window.
            events = fetch_messages_since(
                channel.id, cutoff, max_messages=max_messages_per_channel
            )
            if not events:
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
