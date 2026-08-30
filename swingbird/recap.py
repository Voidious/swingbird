"""Naive recap flow (§4.2, §7, step 7).

The initial implementation skips the incrementally-maintained
project-state cache from §4.2 entirely: each `recap` request
recomputes a short, prioritized summary from recent channel activity
via a single LLM call. Correct and simple; the incremental cache is
deferred to the "refine prompts and tooling" follow-on once real recap
output quality has been seen.
"""

from __future__ import annotations

from swingbird.config import Config
from swingbird.history import fetch_recent_messages
from swingbird.llm import LLMClient

DEFAULT_MESSAGE_LIMIT = 50

_SYSTEM_PROMPT = """You are a TPM agent's recap assistant. Given recent \
messages from one or more project channels, write a short, prioritized \
summary: lead with what needs the user's attention (blockers, decisions \
needed, open questions), then what's in flight, then what finished \
recently. Skip routine chatter. Be concise -- a few sentences per \
channel, not a transcript."""


class RecapError(Exception):
    """Raised when a recap is requested for an unknown channel."""


def build_recap(
    llm: LLMClient,
    config: Config,
    channel_names: list[str] | None = None,
    limit: int = DEFAULT_MESSAGE_LIMIT,
) -> str:
    """Return a short, prioritized recap of recent channel activity.

    `channel_names` restricts the recap to those configured channels;
    omit it to recap every channel in the config.
    """
    channels = _select_channels(config, channel_names)
    transcript = _build_transcript(channels, limit)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    return llm.complete(messages)


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


def _build_transcript(channels, limit: int) -> str:
    sections = []
    for channel in channels:
        events = fetch_recent_messages(channel.id, limit=limit)
        body = (
            "\n".join(f"[{event['created_at']}] {event['content']}" for event in events)
            if events
            else "(no recent activity)"
        )
        sections.append(f"## {channel.name}\n{body}")
    return "\n\n".join(sections)
