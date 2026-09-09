"""Elaborate on a recap item beyond its own stored summary/instruction
(see `daemon._recap_detail`).

`RecapItem.summary`/`instruction` are already the recap's own condensed
words -- echoing them back to a "tell me more" follow-up just replays the
recap in its own wording, saying nothing new. This single LLM call grounds
a genuinely fuller answer in context the recap itself had no room for: the
full thread `item.source_event_id` belongs to (when there is one -- see
`RecapItem`'s "never guessed" invariant, `daemon._fetch_item_thread` skips
this when there isn't), plus the DM conversation so far (the recap that
surfaced the item, and this follow-up question).
"""

from __future__ import annotations

from swingbird.llm import LLMClient
from swingbird.recap import RecapItem

_SYSTEM_PROMPT = """You are a TPM agent elaborating on one item from a \
recap you already gave the user, who's now asking for more detail on it. \
You'll be given the recap item's own summary/instruction, optionally the \
full thread it was grounded in, and the DM conversation so far (the \
recap, and this follow-up question).

Write a few genuinely additional sentences -- new specifics drawn from \
the thread (what was tried, why, what's blocking it, relevant numbers or \
file/function names) that the recap's own summary/instruction didn't \
already say. Do not just restate or reword the summary or instruction you \
were given -- if the thread doesn't offer anything beyond what's already \
there, say so briefly rather than padding with a reworded repeat. No \
preamble, no quotes -- just the elaboration."""


def elaborate(
    llm: LLMClient,
    item: RecapItem,
    thread_messages: list[dict],
    dm_messages: list[dict],
    reference: str | None,
) -> str:
    """Return an elaborated answer for a "tell me more about X" follow-up.

    `thread_messages` is the full thread `item.source_event_id` belongs to
    (empty if the item wasn't grounded in one specific message);
    `dm_messages` is recent history from the DM the follow-up arrived in.
    Either can be empty without the call failing -- there's just less to
    ground the elaboration in.
    """
    parts = [
        f"Recap item -- {item.channel}: {item.summary}",
        f"Recap instruction: {item.instruction}",
    ]
    if thread_messages:
        transcript = "\n".join(
            f"[{message['created_at']}] {message['content']}"
            for message in thread_messages
        )
        parts.append(f"Full thread this was grounded in:\n{transcript}")
    if dm_messages:
        conversation = "\n".join(
            f"[{message['created_at']}] {message['content']}" for message in dm_messages
        )
        parts.append(f"DM conversation so far:\n{conversation}")
    parts.append(f"User's reference: {reference or 'all'}")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return llm.complete(messages)
