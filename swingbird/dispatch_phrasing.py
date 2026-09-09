"""Rephrase a recap item into a dispatch instruction (see `daemon._recap_action`).

`RecapItem.instruction` is written for the *owner* reading a recap, and per
`recap.py`'s extraction prompt it's preserved close to the transcript's own
wording -- when a channel's actionable next step was itself a decision among
several candidates (e.g. "decide which issue to fix first -- F4 (...), F5
(...), F6 (...); recommend starting with F4"), that whole bundle ends up in
one item's `instruction`. Relaying it verbatim once the user picks one
candidate ("go ahead with F4") sends the owner-facing decision summary back
to the agent that raised it, not a directive scoped to what was actually
picked. This single LLM call narrows and rephrases: given the item's
summary/instruction (and, when available, the exact transcript message it
was grounded in -- `RecapItem.source_content`, richer than the recap's own
condensed wording) plus the user's own reference text, it returns one
directive addressed to the target agent, scoped to what the user asked for.
"""

from __future__ import annotations

from swingbird.llm import LLMClient
from swingbird.recap import RecapItem

_SYSTEM_PROMPT = """You are a TPM agent turning the user's follow-up into an \
instruction to relay to a coding agent. You'll be given a recap item -- a \
short summary and an instruction/recommendation extracted from that \
project's recent activity, which may describe more than one candidate next \
step -- possibly alongside the original transcript message it was drawn \
from, and the user's own reference to which part of it they mean (e.g. \
"F4", "the duplicate extractor fix", or "all" if they didn't narrow it \
down).

Write one directive, addressed directly to the agent doing the work (e.g. \
"Go ahead and implement F4: ..."), scoped to only what the user referenced \
-- drop any other candidate the instruction mentioned. Preserve the \
technical substance and, where given, the original message's own wording \
for the part the user picked; don't invent detail that isn't there. If the \
user's reference doesn't narrow anything (e.g. "all", or the instruction \
only ever described one thing), relay the instruction essentially as-is. \
Respond with the instruction text only -- no preamble, no quotes."""


def rephrase_for_dispatch(
    llm: LLMClient, item: RecapItem, reference: str | None
) -> str:
    """Return the instruction to relay for `item`, scoped to `reference`."""
    parts = [
        f"Recap summary: {item.summary}",
        f"Recap instruction: {item.instruction}",
    ]
    if item.source_content:
        parts.append(f"Original message this was drawn from: {item.source_content}")
    parts.append(f"User's reference: {reference or 'all'}")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return llm.complete(messages)
