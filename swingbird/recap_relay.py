"""Forward the user's own message about a recap item to its agent (see
`daemon._recap_relay`).

Unlike `dispatch_phrasing.rephrase_for_dispatch` (which rewrites the
*recap's own* instruction once the user picks a candidate, for
`recap_action`), this handles a different follow-up shape entirely: the
user has a new question, comment, or pushback of their own about something
the recap surfaced (e.g. "for dripbird F4, couldn't we just pre-compile
it?") that should reach the agent close to verbatim, not be rewritten into
a directive. Per Voidious (swingbird-dev, 2026-09-11): since the relayed
message threads to the item's `source_event_id`, the receiving agent
already has the original conversation via the reply chain, so the default
is to change nothing at all -- the one exception is an ambiguous reference
within the user's own wording (e.g. "it", "that") whose target isn't clear
from the message alone, which this resolves using the recap item's own
context rather than leaving the receiving agent to guess.
"""

from __future__ import annotations

from swingbird.llm import LLMClient
from swingbird.recap import RecapItem

_SYSTEM_PROMPT = """You are a TPM agent forwarding the user's own message \
to a coding agent, about a specific item from a recap you gave earlier. \
You'll be given that recap item -- a short summary and \
instruction/recommendation from that project's recent activity -- \
possibly alongside the original transcript message it was drawn from, the \
user's reference to which item they mean, and the user's own message to \
relay.

Relay the user's message close to verbatim -- it's their own question or \
comment, not something to rewrite, soften, or turn into a directive. The \
only thing to change is an ambiguous reference within it (e.g. "it", \
"that", "this") whose target isn't clear from the message's own wording -- \
resolve that using the recap item's summary/instruction/original message, \
so an agent reading only this message doesn't have to guess what "it" \
means. If the message is already unambiguous, change nothing at all -- \
don't add context that isn't needed, don't restate the recap item's \
summary or instruction, and don't answer the user's question yourself. \
Respond with the relayed text only -- no preamble, no quotes."""


def relay_with_context(
    llm: LLMClient, item: RecapItem, reference: str | None, message: str
) -> str:
    """Return `message`, forwarded near-verbatim with any ambiguous
    reference resolved against `item`'s context (see module docstring)."""
    parts = [
        f"Recap summary: {item.summary}",
        f"Recap instruction: {item.instruction}",
    ]
    if item.source_content:
        parts.append(f"Original message this was drawn from: {item.source_content}")
    parts.append(f"User's reference to the item: {reference or item.label}")
    parts.append(f"User's message to relay: {message}")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return llm.complete(messages)
