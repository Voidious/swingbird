"""Intent router: classifies an incoming DM per §4.2 of the design doc.

An LLM call (forced JSON output via `LLMClient.complete_json`) sorts an
inbound message into one of `VALID_INTENTS`, with the known channel/agent
list from config injected into the prompt so the model can recognize
valid dispatch targets instead of guessing at names. Per §5 ("no guessing
on ambiguity"), the prompt instructs the model to leave `channel` /
`target_agent` null rather than pick a plausible-looking match -- the
pending-action store (a later step) is responsible for turning an
ambiguous or unknown target into a clarifying question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from swingbird.audit import AuditLog
from swingbird.config import Config
from swingbird.llm import LLMClient

VALID_INTENTS = (
    "recap",
    "dispatch",
    "clarify_response",
    "confirm",
    "cancel",
    "recap_action",
    "recap_detail",
    "chit_chat",
)

_SYSTEM_PROMPT = """You are the intent classifier for a TPM agent on Buzz.
Classify the user's message into exactly one of these intents:

- recap: asking for a status/summary of one or more project channels. Set
  "detail" to "detailed" only when the user explicitly asks for more (e.g.
  "detailed", "full", "give me everything", "in depth"); default to
  "concise" otherwise, including when detail isn't mentioned at all --
  don't infer a fuller level from tone or message length alone.
- dispatch: anything meant to be relayed to a specific project
  channel/agent for it to act on or answer -- an instruction (e.g. "tell
  backend to fix the login bug") *or* a question addressed to a named
  project/channel/agent (e.g. "verify swingbird is written in Python",
  "ask frontend if the tests pass"). A message naming exactly one known
  channel/agent from the list below, and asking or telling it something,
  is dispatch even if it isn't phrased as an imperative command -- don't
  reserve dispatch for commands only and fall back to chit_chat just
  because the message is a question.
- clarify_response: answering a clarifying question the agent asked.
- confirm: approving a previously proposed action (e.g. "yes", "do it") --
  only when the agent has already proposed a specific dispatch to relay
  (it said "Confirm to send, or cancel"). If nothing has been proposed yet
  and the user is instead reacting to a *recap* (e.g. it mentioned an
  agent's recommendation), that is recap_action, not confirm.
- cancel: rejecting/withdrawing a previously proposed action.
- recap_action: telling the agent to proceed with something the *recap*
  itself surfaced -- e.g. "go ahead with F4", "let's do the duplicate
  extractor fix for dripbird", "do all of them". This is not yet a
  dispatch proposal (nothing has been proposed to confirm/cancel) -- it's
  a reference back to an item the recap already described, which the
  agent will resolve itself. Put the user's reference to *which item(s)*
  they mean into "message", preserved as closely to their own wording as
  possible (e.g. "F4", "the duplicate extractor fix for dripbird", "all")
  -- do not try to identify the channel or agent yourself, and do not
  restate the recommendation's content, since only the raw reference is
  needed to look it back up.
- recap_detail: asking for more detail on something the *recap* surfaced,
  without asking to proceed with it -- e.g. "tell me more about F4", "what
  did dripbird say about the duplicate extractor". Same "message" handling
  as recap_action: preserve the user's own reference to which item, don't
  restate its content.
- chit_chat: anything else, out of scope for this agent.

Known project channels and their agents:
{channel_list}

Respond with JSON only, matching this shape:
{{"intent": "<one of the intents above>", "channel": "<channel name or null>",
"target_agent": "<agent name or null>",
"message": "<instruction text to relay, or null>",
"detail": "<\"concise\" or \"detailed\", default \"concise\">"}}

Only set "channel" or "target_agent" to a name from the known list above,
and only when the message clearly identifies it. If the message names a
channel or agent that isn't in the list, or is ambiguous about which one
it means, leave that field null rather than guessing -- do not invent or
assume a target.

"message" must preserve the user's own wording as closely as possible --
extract it, don't paraphrase or rewrite it. In particular, never replace a
project, channel, or agent name the user used with "you" or another
second-person pronoun just because the message is being routed to that
same target. The user may be asking a question *about* something named
that also happens to be a routing target (e.g. "is swingbird written in
Python", asking about the swingbird project) rather than addressing it
directly -- collapsing the name into "you" erases that distinction for
the agent that receives the relayed message."""


class RouterError(Exception):
    """Raised when the LLM's response can't be trusted as a classification."""


@dataclass(frozen=True)
class Intent:
    kind: str
    channel: str | None = None
    target_agent: str | None = None
    message: str | None = None
    detail: str = "concise"


class IntentRouter:
    def __init__(
        self, llm: LLMClient, config: Config, audit: AuditLog | None = None
    ) -> None:
        self._llm = llm
        self._audit = audit
        self._system_prompt = _SYSTEM_PROMPT.format(
            channel_list=_format_channel_list(config)
        )

    def route(self, text: str, thread_id: str | None = None) -> Intent:
        """Classify `text` into an `Intent`, calling the configured LLM.

        Every call is recorded as a `transcript_in` audit entry (if an
        `AuditLog` was configured) before the LLM call, so the audit trail
        covers what came in even if classification itself fails.

        Retries the classification once if the first attempt comes back
        `chit_chat`, since that's the catch-all bucket an under-confident
        or momentarily-flaky classification collapses into (there's no
        equivalent "maybe" bucket for the other intents to fall back to).
        The configured model doesn't support lowering temperature to
        reduce this kind of run-to-run variance (kimi-k2.6 rejects any
        value other than its default), so a same-input retry is the
        cheaper lever available. If the retry also comes back
        `chit_chat`, that result is trusted and returned as-is.
        """
        if self._audit is not None:
            self._audit.log_transcript_in(thread_id, text)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text},
        ]
        intent = _parse_intent(self._llm.complete_json(messages))
        if intent.kind == "chit_chat":
            intent = _parse_intent(self._llm.complete_json(messages))
        return intent


def _format_channel_list(config: Config) -> str:
    lines = [
        f"- {channel.name}: agents = {list(channel.agents)}"
        for channel in config.channels
    ]
    return "\n".join(lines)


def _parse_intent(response: dict) -> Intent:
    kind = response.get("intent")
    if kind not in VALID_INTENTS:
        raise RouterError(f"LLM returned unknown intent: {json.dumps(response)}")
    detail = response.get("detail")
    if detail != "detailed":
        detail = "concise"
    return Intent(
        kind=kind,
        channel=response.get("channel"),
        target_agent=response.get("target_agent"),
        message=response.get("message"),
        detail=detail,
    )
