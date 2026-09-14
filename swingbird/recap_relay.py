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

A second exception surfaced 2026-09-12: `router.py`'s own extraction for
this intent strips the leading verb the user framed their ask with ("ask
if...", "ask whether...") the same way it strips "go ahead with" for
`recap_action`, since that verb is the user instructing the TPM agent, not
part of what should reach the coding agent. But unlike "go ahead with F4"
-> "F4", dropping "ask if"/"ask whether" leaves a dangling subordinate
clause ("if Kimi 2.6 will work as well as Kimi 2.5") that was never a
complete sentence on its own -- relaying that "verbatim" reads as a
fragment, not a question. So a message that arrives as such a fragment
also isn't left alone; it's inverted into the direct question it was
always asking.
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
relay. That message has already had any framing like "ask if..."/"tell \
them..." stripped out before it reached you, since that was the user \
instructing the TPM agent, not part of what the coding agent should read.

Relay the message close to verbatim -- it's the user's own question or \
comment, not something to rewrite, soften, or turn into a directive. Only \
fix what's needed for it to read as a complete, standalone message \
addressed directly to the agent, since they see only this text, never the \
user's original wording or the framing that was stripped from it:
- An ambiguous reference (e.g. "it", "that", "this") whose target isn't \
clear from the message's own wording -- resolve it using the recap item's \
summary/instruction/original message.
- A dangling subordinate clause left over from stripping a leading "ask \
if..."/"ask whether..." (e.g. "if Kimi 2.6 will work as well as Kimi \
2.5") -- invert it into the direct question it was always asking (e.g. \
"Will Kimi 2.6 work as well as Kimi 2.5?"), filling in from the recap item \
whatever context it needs to stand alone (e.g. what this is the default \
for, or what it's being compared against).

If the message is already a complete, standalone question or statement, \
change nothing at all -- don't add context that isn't needed, don't \
restate the recap item's summary or instruction, and don't answer the \
user's question yourself. Respond with the relayed text only -- no \
preamble, no quotes."""


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
