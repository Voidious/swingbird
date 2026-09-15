"""Intent router: classifies an incoming DM per §4.2 of the design doc.

An LLM call (forced JSON output via `LLMClient.complete_json`) sorts an
inbound message into one of `VALID_INTENTS`, with the known channel/agent
list from config injected into the prompt so the model can recognize
valid dispatch targets instead of guessing at names. Per §5 ("no guessing
on ambiguity"), the prompt instructs the model to leave `channel` /
`target_agent` null rather than pick a plausible-looking match -- the
pending-action store (a later step) is responsible for turning an
ambiguous or unknown target into a clarifying question.

`route()` also takes `has_open_recap`, whether the calling thread currently
has a stored recap it can still reference (see
`recap_actions.RecapActionStore`). The classification call otherwise has
zero visibility into anything outside the single message being classified,
so wording that could equally describe "elaborate on what the recap just
said" or "give me a fresh recap" (e.g. "tell me more about the open items
for dripbird") was observed to flip unpredictably between `recap` and
`recap_detail` on identical input -- the model had no way to know a recap
had just been given. Appending an explicit conversation-state note to the
system prompt per call, rather than trying to infer that state from the
message's wording alone, gives the model the one piece of context it
actually needs to disambiguate deterministically.
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
    "recap_list",
    "recap_relay",
    "recap_close",
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
  because the message is a question. Exception: when there's an open
  recap (see the conversation-state note below) and the message is
  building on a specific item that recap surfaced, classify it as
  recap_relay instead, even though it also names a channel.
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
- recap_detail: asking this agent to genuinely elaborate on something the
  *recap* surfaced -- go deeper than the recap's own wording, without
  asking to proceed with it -- e.g. "tell me more about F4", "what did
  dripbird say about the duplicate extractor", "tell me more about the
  additional items" (explicit "tell me more"/"elaborate"/"go deeper"
  phrasing, even when it's about more than one item). Same "message"
  handling as recap_action: preserve the user's own reference to which
  item(s), don't restate its content.
- recap_list: asking to see/enumerate the recap's additional/open items
  themselves, without asking for deeper detail on any of them -- e.g. "what
  are the other items", "what else is there", "show me the rest", "what
  other items". The distinction from recap_detail is "tell me more"/
  "elaborate" wording: if the message just asks *which* other items exist,
  it's recap_list; if it also asks to go deeper on them, it's recap_detail.
  Same "message" handling as recap_action/recap_detail: preserve the user's
  own reference, don't restate content.
- recap_relay: forwarding the user's own new question, comment, or
  pushback about a specific item the *recap* surfaced, on to the agent
  that owns it -- e.g. "for dripbird F4, couldn't we just pre-compile
  it?", "ask backend if F4 still needs the migration", "tell frontend
  that sounds risky, what about caching instead". Unlike recap_action
  (proceeding with the recap's own recommendation, nothing new from the
  user) and recap_detail/recap_list (asking *this* agent about its own
  recap, nothing gets relayed anywhere), recap_relay is new content of the
  user's own that should be sent on to the coding agent. Put the user's
  reference to *which item* they mean into "item_reference" -- same
  handling as recap_action's reference (preserve wording, e.g. "F4",
  "dripbird F4"), do not identify the channel yourself. Put the actual
  text to relay into "message", preserved as closely to the user's own
  wording as possible, same rule as dispatch -- extract it, don't
  paraphrase, and don't fold the item reference into it (e.g. for "for
  dripbird F4, couldn't we just pre-compile it?", "item_reference" is "F4"
  or "dripbird F4" and "message" is "couldn't we just pre-compile it?").
- recap_close: telling the agent to mark a *recap* item as closed/done, so
  it stops being surfaced as open work in future recaps -- e.g. "close
  F4", "mark the duplicate extractor fix as done", "that's already fixed,
  you can close it". This never relays or dispatches anything -- unlike
  recap_action (which proceeds with the recap's own next-step instruction
  by sending it somewhere), recap_close only records that the item itself
  is finished. Same "message" handling as recap_action: put the user's own
  reference to *which* item(s) into "message", preserved as closely to
  their own wording as possible, and don't restate the item's content.
- chit_chat: anything else, out of scope for this agent.

Known project channels and their agents:
{channel_list}

Respond with JSON only, matching this shape:
{{"intent": "<one of the intents above>", "channel": "<channel name or null>",
"target_agent": "<agent name or null>",
"message": "<instruction text to relay, or null>",
"item_reference": "<recap_relay's reference to which item, or null>",
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


_OPEN_RECAP_NOTE = """

Conversation state: this thread already has an open recap -- a structured
recap with per-channel items was given earlier in this conversation, and the
user can still reference it. Prefer recap_detail, recap_list, recap_action,
recap_close, or recap_relay over a plain recap or a plain dispatch when the
wording could describe either (e.g. "tell me more about the open items for
dripbird", "what's the status on F4", "go ahead with the duplicate extractor
fix", "close F4", "what are the other items", "what else is there", "for
dripbird F4, couldn't we just pre-compile it?") -- treat these as referring
back to what that recap already surfaced, not as a request to regenerate a
fresh one or post a fresh, unrelated dispatch. Only classify as recap when the user is
clearly asking for a new or refreshed summary instead (e.g. "give me an
update", "what's changed since then", naming a channel that wasn't part of
the open recap).

The open recap's actual items are listed below (channel: label). A message
naming one of these channels and asking or saying something about it is
building on that item -- classify it recap_relay, not dispatch, even when
it's phrased as a plain question addressed to the channel (e.g. "ask
frontend if the tests pass" is dispatch on its own, but if "frontend: fix
the login timeout bug" is an open item, "does frontend's login fix need a
migration?" is recap_relay) and even when the wording doesn't quote the
item's label back verbatim -- match on what the item is about, not on
whether the user's phrasing echoes an example above. Only classify as
dispatch instead when the channel named has no open item below, or the
message is unambiguously about something else the open item isn't (a new,
unrelated ask for that same channel)."""

_NO_OPEN_RECAP_NOTE = """

Conversation state: this thread has no open recap right now -- nothing has
been recapped yet, or too much has happened since for one to still apply.
recap_detail, recap_list, recap_action, recap_close, and recap_relay all
require an existing recap to reference, so don't classify as any of those
here; a message asking about a channel's status is a plain recap instead, and a
message meant for a channel/agent is a plain dispatch instead."""

# Mirrors _OPEN_RECAP_NOTE/_NO_OPEN_RECAP_NOTE's own reasoning, for the same
# underlying problem: classification sees only the current message, with no
# memory of what the agent itself just said, so a bare "confirm" or "never
# mind" has nothing in its own wording to distinguish "answering the
# proposal you just saw" from a random aside -- confirm/cancel's own bullet
# above states the *condition* ("only when the agent has already proposed a
# specific dispatch"), but the classifier has no way to know whether that
# condition currently holds without being told. Observed live: a lone
# "confirm" sent right after the agent's own "Confirm to send, or cancel."
# came back chit_chat instead of confirm, most likely because nothing in
# that one word signals a pending proposal on its own. Appended independently
# of the open/no-open-recap note above -- a thread can have both an open
# recap and a pending dispatch proposal at once (e.g. mid recap_action).
_PENDING_DISPATCH_NOTE = """

Conversation state: this thread has a dispatch proposal awaiting confirm or
cancel right now -- the agent's last message asked the user to "Confirm to
send, or cancel." A short reply like "yes", "confirmed", "do it", "go
ahead", "no", "cancel", or "never mind" is answering that specific
question, so classify it as confirm or cancel accordingly, even though the
reply's own wording carries no other content to classify from."""

_NO_PENDING_DISPATCH_NOTE = """

Conversation state: this thread has no dispatch proposal awaiting confirm
or cancel right now -- nothing has been proposed, or it was already
resolved. Don't classify a message as confirm or cancel here even if it
would look like an affirmation/rejection in isolation (e.g. "yes", "never
mind") -- classify it as chit_chat, or whatever its own wording otherwise
matches, since there's nothing pending for it to confirm or cancel."""


class RouterError(Exception):
    """Raised when the LLM's response can't be trusted as a classification."""


@dataclass(frozen=True)
class Intent:
    kind: str
    channel: str | None = None
    target_agent: str | None = None
    message: str | None = None
    detail: str = "concise"
    # Only set by the router's own LLM classification for a `recap_relay`
    # intent -- the user's reference to which recap item `message` is about
    # (see router module docstring). `recap_action`/`recap_detail`/
    # `recap_list` instead put their own reference straight into `message`,
    # since they have no separate content to relay alongside it.
    item_reference: str | None = None
    # Never set by the router's own LLM classification -- only by
    # `daemon._recap_action`/`daemon._recap_relay`, to thread a recap
    # follow-up's dispatch back to the message the recap grounded it in (see
    # `RecapItem.source_event_id`).
    reply_to: str | None = None


class IntentRouter:
    def __init__(
        self, llm: LLMClient, config: Config, audit: AuditLog | None = None
    ) -> None:
        self._llm = llm
        self._audit = audit
        self._system_prompt = _SYSTEM_PROMPT.format(
            channel_list=_format_channel_list(config)
        )

    def route(
        self,
        text: str,
        thread_id: str | None = None,
        has_open_recap: bool = False,
        has_pending_dispatch: bool = False,
        open_recap_items: tuple[tuple[str, str], ...] = (),
    ) -> Intent:
        """Classify `text` into an `Intent`, calling the configured LLM.

        Every call is recorded as a `transcript_in` audit entry (if an
        `AuditLog` was configured) before the LLM call, so the audit trail
        covers what came in even if classification itself fails.

        `has_open_recap` -- whether the calling thread has a stored recap it
        can still reference -- is appended to the system prompt as an
        explicit conversation-state note (see module docstring) rather than
        left for the model to guess from the message text alone.
        `has_pending_dispatch` -- whether the thread has a proposed dispatch
        still awaiting confirm/cancel -- gets the same treatment (see
        `_PENDING_DISPATCH_NOTE`), for the same reason: a bare "confirm" or
        "never mind" carries no signal of its own that a proposal is
        actually pending.

        `open_recap_items` -- (channel, label) pairs for the open recap's own
        items, when there is one -- is appended to `_OPEN_RECAP_NOTE` so the
        classifier can check a message against what the open items actually
        are, not just their wording shape. Without this, a message that's
        clearly *about* an open item but doesn't echo `_OPEN_RECAP_NOTE`'s
        own example phrasing (e.g. "for dripbird F4, ...") reliably fell back
        to `dispatch` instead of `recap_relay` -- e.g. "on the dripbird
        default model, is Kimi named after anyone specific?" when the open
        recap had a dripbird item literally about the default model choice --
        because the classifier had no item list to check the message against,
        only its own phrasing to pattern-match. Empty when `has_open_recap`
        is `False`, or the caller has no items for this thread.

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
        open_recap_note = _OPEN_RECAP_NOTE if has_open_recap else _NO_OPEN_RECAP_NOTE
        if has_open_recap and open_recap_items:
            open_recap_note += "\n\n" + _format_open_recap_items(open_recap_items)
        system_prompt = (
            self._system_prompt
            + open_recap_note
            + (
                _PENDING_DISPATCH_NOTE
                if has_pending_dispatch
                else _NO_PENDING_DISPATCH_NOTE
            )
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        intent = _parse_intent(self._llm.complete_json(messages))
        if intent.kind == "chit_chat":
            intent = _parse_intent(self._llm.complete_json(messages))
        return intent


def _format_open_recap_items(items: tuple[tuple[str, str], ...]) -> str:
    lines = [f"- {channel}: {label}" for channel, label in items]
    return "Open recap items:\n" + "\n".join(lines)


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
        item_reference=response.get("item_reference"),
    )
