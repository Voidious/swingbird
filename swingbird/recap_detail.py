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

Each item's own paragraph in the LLM's answer also gets a Buzz message
link appended, when `item.source_event_id` is set -- see
`_append_source_links`.
"""

from __future__ import annotations

from swingbird import outbound
from swingbird.config import Config
from swingbird.llm import LLMClient
from swingbird.recap import RecapItem

# Mirrors recap.py's own _FORMAT_GUARD pattern -- the labeled, one-paragraph-
# per-item look used to be an accident of multi-item input (the LLM only
# reached for it when "Item i/N --" prefixes gave it something to visually
# separate); a single-item elaboration fell back to plain prose instead,
# which read as an inconsistent format across channels even though the
# underlying instruction was the same. Making it explicit and unconditional
# is what actually makes every channel's "tell me more" look the same.
_FORMAT_GUARD = (
    "Format your answer as one paragraph per item, each starting with that "
    "item's channel and label in bold Markdown (e.g. \"**crispen -- "
    'selfcheck-20:** ..."), separated from the next by a blank line (a '
    "literal \\n\\n between them). Use this format even when you were only "
    "given one item -- don't switch to plain prose just because there's "
    "nothing else to visually separate it from."
)

_SYSTEM_PROMPT = f"""You are a TPM agent elaborating on one or more items \
from a recap you already gave the user, who's now asking for more detail. \
You'll be given each item's own summary/instruction, optionally the full \
thread it was grounded in, and the DM conversation so far (the recap, and \
this follow-up question). You may also be told the user asked for other or \
additional items but the channel doesn't have any beyond the one below -- \
if so, say that plainly in one sentence first, then still give the \
elaboration for the one item you do have.

Write a few genuinely additional sentences per item -- new specifics drawn \
from its thread (what was tried, why, what's blocking it, relevant numbers \
or file/function names) that the item's own summary/instruction didn't \
already say. Do not just restate or reword the summary or instruction you \
were given -- if a thread doesn't offer anything beyond what's already \
there, say so briefly for that item rather than padding with a reworded \
repeat. {_FORMAT_GUARD} No preamble, no quotes -- just the elaboration."""


def elaborate(
    llm: LLMClient,
    items: list[RecapItem],
    threads: list[list[dict]],
    dm_messages: list[dict],
    reference: str | None,
    config: Config,
    no_other_items: bool = False,
) -> str:
    """Return an elaborated answer for a "tell me more about X" follow-up.

    `threads[i]` is the full thread `items[i].source_event_id` belongs to
    (empty if that item wasn't grounded in one specific message) -- `threads`
    must be the same length as `items`, aligned by index. `dm_messages` is
    recent history from the DM the follow-up arrived in. Any of these can be
    empty without the call failing -- there's just less to ground the
    elaboration in.

    `no_other_items` is `ResolvedReference.degraded` from `recap_actions.
    resolve_reference` -- True when the user asked for "the other/additional
    items" but the channel only ever had the one item already in `items`.
    Without this, the elaboration would silently re-describe that one item
    as if it were something new; with it, the LLM is told to say upfront
    that there's nothing else before elaborating on what it does have.
    """
    multiple = len(items) > 1
    parts = []
    if no_other_items:
        parts.append(
            "The user asked for other/additional items, but this channel "
            "has no items beyond the one below -- say that plainly before "
            "elaborating on it."
        )
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
    return _append_source_links(llm.complete(messages), items, config)


def _append_source_links(text: str, items: list[RecapItem], config: Config) -> str:
    """Append a Buzz message link to each item's own paragraph in `text`,
    pointing at `item.source_event_id` when the recap grounded it in one
    specific transcript message (see `RecapItem.source_event_id`) --
    mirrors `recap.py`'s own `_append_source_links`, keyed on `_FORMAT_
    GUARD`'s "**channel -- label:**" paragraph prefix instead of recap.py's
    plain "**channel**:", since that's the format this module's own prompt
    asks for. A channel name `config` doesn't recognize is silently
    skipped, same rationale as recap.py's version.
    """
    for item in items:
        if item.source_event_id is None:
            continue
        channel = config.channel_by_name(item.channel)
        if channel is None:
            continue
        link = outbound.message_link(channel.id, item.source_event_id)
        prefix = f"**{item.channel} -- {item.label}:**"
        text = outbound.append_paragraph_link(text, prefix, link)
    return text
