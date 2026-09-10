"""Elaborate on one or more recap items beyond their own stored
summary/instruction (see `daemon._recap_detail`).

`RecapItem.summary`/`instruction` are already the recap's own condensed
words -- echoing them back to a "tell me more" follow-up just replays the
recap in its own wording, saying nothing new. This single LLM call grounds
a genuinely fuller answer in context the recap itself had no room for: each
item's own full thread (when `item.source_event_id` is set -- see
`RecapItem`'s "never guessed" invariant, `daemon._fetch_item_thread` skips
this when there isn't one), plus the DM conversation so far (the recap that
surfaced the item(s), and this follow-up question).

`items` can hold more than one `RecapItem` -- e.g. "tell me more about the
additional items" resolves (via `recap_actions.resolve_reference`) to every
non-primary item for a channel, not just one, so this elaborates on all of
them in a single call rather than needing one call per item.
"""

from __future__ import annotations

from swingbird.llm import LLMClient
from swingbird.recap import RecapItem

_SYSTEM_PROMPT = """You are a TPM agent elaborating on one or more items \
from a recap you already gave the user, who's now asking for more detail. \
You'll be given each item's own summary/instruction, optionally the full \
thread it was grounded in, and the DM conversation so far (the recap, and \
this follow-up question).

Write a few genuinely additional sentences per item -- new specifics drawn \
from its thread (what was tried, why, what's blocking it, relevant numbers \
or file/function names) that the item's own summary/instruction didn't \
already say. Do not just restate or reword the summary or instruction you \
were given -- if a thread doesn't offer anything beyond what's already \
there, say so briefly for that item rather than padding with a reworded \
repeat. When there's more than one item, address each by its channel/label \
so the user can tell them apart. No preamble, no quotes -- just the \
elaboration."""


def elaborate(
    llm: LLMClient,
    items: list[RecapItem],
    threads: list[list[dict]],
    dm_messages: list[dict],
    reference: str | None,
) -> str:
    """Return an elaborated answer for a "tell me more about X" follow-up.

    `threads[i]` is the full thread `items[i].source_event_id` belongs to
    (empty if that item wasn't grounded in one specific message) -- `threads`
    must be the same length as `items`, aligned by index. `dm_messages` is
    recent history from the DM the follow-up arrived in. Any of these can be
    empty without the call failing -- there's just less to ground the
    elaboration in.
    """
    multiple = len(items) > 1
    parts = []
    for i, (item, thread_messages) in enumerate(zip(items, threads), start=1):
        prefix = f"Item {i}/{len(items)} -- " if multiple else ""
        parts.append(f"{prefix}Recap item -- {item.channel}: {item.summary}")
        parts.append(f"{prefix}Recap instruction: {item.instruction}")
        if thread_messages:
            transcript = "\n".join(
                f"[{message['created_at']}] {message['content']}"
                for message in thread_messages
            )
            parts.append(f"{prefix}Full thread this was grounded in:\n{transcript}")
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
