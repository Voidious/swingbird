"""Daemon: wires the inbound relay client to the router, pending-action
store, and audit log, and enforces the safety invariants from §5.

Only events from `config.owner.pubkey`, posted in the daemon's own DM with
the owner, are ever routed as a command. Project channels are subscribed to
for recap purposes only -- even a message from the owner there (e.g. an
@mention) is ignored as a command, so ordinary conversation in a dev channel
can never be misread as a dispatch/confirm/cancel. (Reacting to explicit
@mentions in project channels may be worth adding later; for now the DM is
the only interaction surface.) Everyone else's traffic in a subscribed
channel (including a coding agent's own replies) is ignored here too. A
single bad or unexpected event never kills the daemon: known, safety-relevant
failures (an ambiguous target, a bad LLM response, an unwritable channel,
...) become a reply to the sender instead of a crash, and anything truly
unexpected is logged and the loop continues.

Alongside the configured project channels, the daemon always subscribes to
its own 1:1 DM with the owner (resolved via `outbound.open_dm`) -- Buzz DMs
turn out to be ordinary #h-tagged channel events under the hood, so this
needed no new wire format, just one more channel id in the subscription.

On startup the daemon also joins every configured project channel it isn't
already a member of (`_join_project_channels`) -- good bot manners, since
membership isn't actually required to read or write an *open* channel (the
relay allows either membership or open visibility), so posting into
channels it never joined would otherwise look odd in the member list. A
join only fails for a private channel the identity isn't already in; that's
logged and never fatal. The same failure mode can also surface later, from
`relay_dispatch`/a recap fetch against a channel configured as private
without the daemon in it -- `_ACTIONABLE_ERRORS` including `RelayError`
turns that into a "Couldn't do that" reply to the owner (asking them to add
the bot) instead of a silently swallowed exception.

The daemon also publishes its own presence (online while the event loop is
running, offline on exit) so its availability dot in Buzz Desktop reflects
whether it's actually up -- a deployed daemon that isn't running should
never look identical to one that is.

After a confirmed dispatch, the daemon waits (up to `config.dispatch.
reply_wait_seconds`) for a reply to the relayed instruction, then
summarizes it back into the owner's DM -- see `_watch_for_reply` and
`_resolve_reply_watch`. This runs as a background asyncio task alongside
the main event loop rather than blocking it, since the reply (if any)
arrives as just another event on the same subscription; matching it to
the right wait is done by NIP-10 `e`-tag, not by agent identity, since
the daemon doesn't track individual coding agents' pubkeys.

A coding agent's own reply frequently doesn't `e`-tag the relayed message
directly: an async status update often threads to the whole conversation's
root instead of the specific message that carried the instruction, and a
reply nested two or more levels deep NIP-10-tags both a "root" and a
"reply" id, in either order. `_reply_watch_id` resolves the id actually
worth watching up front (the thread's true root for a grounded dispatch,
the relayed event's own id for a fresh top-level one), and
`_reply_target_ids` checks every `e`-tag id on an inbound event against
it, so either shape of reply is caught -- see both for why.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import replace

from swingbird import outbound
from swingbird.audit import AuditLog
from swingbird.config import Config, load_config
from swingbird.dispatch_phrasing import rephrase_for_dispatch
from swingbird.history import fetch_thread_root
from swingbird.inbound import InboundClient, InboundError
from swingbird.llm import LLMClient, LLMError
from swingbird.pending_actions import (
    DispatchProposal,
    PendingActionError,
    PendingActionStore,
    cancel_dispatch,
    confirm_dispatch,
    propose_dispatch,
)
from swingbird.recap import RecapError, RecapItem, build_recap
from swingbird.recap_actions import (
    RecapActionError,
    RecapActionStore,
    resolve_reference,
)
from swingbird.reply_summary import summarize_reply
from swingbird.router import Intent, IntentRouter, RouterError

DEFAULT_CONFIG_PATH = "swingbird.toml"
DEFAULT_AUDIT_LOG_PATH = "audit.jsonl"

_CHIT_CHAT_REPLY = (
    "That's outside what I handle -- ask me for a recap, or to dispatch an "
    "instruction to a project channel."
)
_ACTIONABLE_ERRORS = (
    RouterError,
    PendingActionError,
    RecapError,
    RecapActionError,
    LLMError,
    outbound.RelayError,
)


class DaemonError(Exception):
    """Raised when an inbound event can't be handled (e.g. malformed tags)."""


class Daemon:
    """Routes verified inbound events from `config.owner.pubkey` to actions."""

    def __init__(
        self,
        config: Config,
        inbound: InboundClient,
        router: IntentRouter,
        store: PendingActionStore,
        llm: LLMClient,
        audit: AuditLog,
        dm_id: str | None = None,
        recap_store: RecapActionStore | None = None,
    ) -> None:
        self._config = config
        self._inbound = inbound
        self._router = router
        self._store = store
        self._llm = llm
        self._audit = audit
        # The last recap's structured items per thread, so a follow-up DM
        # ("go ahead with F4") can resolve "F4" back to a real channel and
        # instruction -- see recap_actions.py.
        self._recap_store = (
            recap_store if recap_store is not None else RecapActionStore()
        )
        # Resolved fresh in run() via outbound.open_dm(); only events posted
        # in this channel are ever routed as a command (see module
        # docstring). None until run() sets it, which means no channel can
        # match and every event is ignored -- the safe default for a daemon
        # that hasn't finished starting up.
        self._dm_id = dm_id
        # Relayed-instruction event id -> a Future resolved with whatever
        # event replies to it, so a confirm's background wait-and-summarize
        # task (see _watch_for_reply) can be woken from _handle_event.
        self._reply_watches: dict[str, asyncio.Future] = {}

    async def run(self) -> None:
        # Captured before open_dm()/connect() so the backlog cutoff covers
        # the whole startup handshake -- both involve real network
        # round-trips (buzz-cli subprocess, then WebSocket + NIP-42 auth),
        # and a `since` taken only once subscribe() itself runs would
        # silently drop any message the owner sends while the daemon is
        # still coming up.
        since = int(time.time())
        self._sync_display_name()
        self._join_project_channels()
        # Resolved here rather than in build_daemon() so construction stays
        # side-effect-free; this is the daemon's own DM with the owner,
        # opened (or resurfaced) fresh each run via the same buzz-cli path
        # outbound.py already uses for every other write.
        self._dm_id = outbound.open_dm(self._config.owner.pubkey)
        channel_ids = [channel.id for channel in self._config.channels] + [self._dm_id]
        print(
            f"swingbird: resolved DM channel {self._dm_id!r}; "
            f"subscribing to {channel_ids}"
        )
        await self._inbound.connect()
        await self._inbound.subscribe(channel_ids, since=since)
        self._set_presence("online")
        print("swingbird: connected and listening")
        try:
            async for event in self._inbound.events():
                await self._safe_handle(event)
        finally:
            self._set_presence("offline")

    def _set_presence(self, status: str) -> None:
        # Best-effort: a presence hiccup is cosmetic (it only drives the
        # availability dot in Buzz Desktop) and must never take the daemon
        # down or block it from processing real events.
        try:
            outbound.set_presence(status)
        except outbound.RelayError as exc:
            print(f"swingbird: failed to set presence to {status!r}: {exc}")

    def _join_project_channels(self) -> None:
        # Best-effort, like presence/display-name sync: joining a channel
        # the identity already belongs to (or an open one) always succeeds
        # on the relay, so a failure here means a real, reportable problem
        # (a private channel the identity isn't in) -- never a reason to
        # block startup. See module docstring.
        for channel in self._config.channels:
            try:
                outbound.join_channel(channel.id)
            except outbound.RelayError as exc:
                print(
                    f"swingbird: couldn't join channel {channel.name!r} "
                    f"({channel.id}) -- if it's private, add me to it: {exc}"
                )

    def _sync_display_name(self) -> None:
        # Keeps a fresh identity (or a renamed deployment) from showing up
        # under a stale/default profile name. Best-effort like presence --
        # a lookup/update hiccup here is cosmetic and must never block
        # startup or take the daemon down.
        wanted = self._config.identity.name
        try:
            current = outbound.get_own_display_name()
            if current != wanted:
                outbound.set_display_name(wanted)
                print(f"swingbird: updated display name {current!r} -> {wanted!r}")
        except outbound.RelayError as exc:
            print(f"swingbird: failed to sync display name: {exc}")

    async def _safe_handle(self, event: dict) -> None:
        try:
            await self._handle_event(event)
        except Exception as exc:  # noqa: BLE001 - last-resort net, see module docstring
            print(f"swingbird: failed to handle event {event.get('id')}: {exc}")

    async def _handle_event(self, event: dict) -> None:
        print(f"swingbird: event {event.get('id')} from {event.get('pubkey')}")
        if self._resolve_reply_watch(event):
            return
        if event["pubkey"] != self._config.owner.pubkey:
            print(
                f"swingbird: ignoring -- not from owner ({self._config.owner.pubkey})"
            )
            return
        channel_id = _channel_of(event)
        if channel_id != self._dm_id:
            print(
                f"swingbird: ignoring -- not the owner's DM channel ({self._dm_id!r})"
            )
            return
        reply = self._process(event, channel_id)
        try:
            outbound.send_message(channel_id, reply, reply_to=event["id"])
        except outbound.RelayError as exc:
            print(f"swingbird: failed to send reply to {event['id']}: {exc}")

    def _resolve_reply_watch(self, event: dict) -> bool:
        """If `event` replies anywhere within a thread we're waiting on,
        fulfill that wait and report it as handled.

        Checked against every NIP-10 `e`-tag id on `event`
        (`_reply_target_ids`), not just one -- see `_reply_watch_id` for why
        a reply can legitimately carry a different id than the one that was
        actually watched while still belonging to the same watched thread.
        Matched by tag id alone, not by who posted it -- a reply is never
        itself treated as an owner command (§5), regardless of its author.
        """
        for target_id in _reply_target_ids(event):
            future = self._reply_watches.pop(target_id, None)
            if future is None:
                continue
            if future.done():
                return False
            future.set_result(event)
            return True
        return False

    def _watch_for_reply(
        self, relayed_event_id: str, dm_channel_id: str, dm_reply_to: str
    ) -> None:
        future = asyncio.get_running_loop().create_future()
        self._reply_watches[relayed_event_id] = future
        asyncio.create_task(
            self._safe_await_and_summarize(
                relayed_event_id, dm_channel_id, dm_reply_to, future
            )
        )

    async def _safe_await_and_summarize(
        self,
        relayed_event_id: str,
        dm_channel_id: str,
        dm_reply_to: str,
        future: asyncio.Future,
    ) -> None:
        # This runs detached (via asyncio.create_task, never awaited by the
        # main loop), so it needs its own safety net -- same rationale as
        # _safe_handle wrapping _handle_event.
        try:
            await self._await_and_summarize(
                relayed_event_id, dm_channel_id, dm_reply_to, future
            )
        except Exception as exc:  # noqa: BLE001 - background task, must never propagate
            print(f"swingbird: failed to summarize reply to {relayed_event_id}: {exc}")

    async def _await_and_summarize(
        self,
        relayed_event_id: str,
        dm_channel_id: str,
        dm_reply_to: str,
        future: asyncio.Future,
    ) -> None:
        try:
            reply_event = await asyncio.wait_for(
                future, timeout=self._config.dispatch.reply_wait_seconds
            )
        except TimeoutError:
            self._reply_watches.pop(relayed_event_id, None)
            return
        summary = summarize_reply(self._llm, reply_event["content"])
        outbound.send_message(dm_channel_id, summary, reply_to=dm_reply_to)

    def _process(self, event: dict, thread_id: str) -> str:
        try:
            intent = self._router.route(event["content"], thread_id=thread_id)
            return self._act(intent, thread_id, event["id"])
        except _ACTIONABLE_ERRORS as exc:
            return f"Couldn't do that: {exc}"

    def _act(self, intent: Intent, thread_id: str, event_id: str) -> str:
        if intent.kind == "recap":
            channel_names = [intent.channel] if intent.channel else None
            built_recap = build_recap(
                self._llm,
                self._config,
                channel_names=channel_names,
                detail=intent.detail,
            )
            self._recap_store.set(thread_id, built_recap.items)
            return built_recap.text
        if intent.kind in ("dispatch", "clarify_response"):
            return self._dispatch_or_ask(intent, thread_id)
        if intent.kind == "confirm":
            return self._confirm(thread_id, event_id)
        if intent.kind == "cancel":
            cancel_dispatch(self._store, thread_id, audit=self._audit)
            return "Cancelled -- nothing was sent."
        if intent.kind == "recap_action":
            return self._recap_action(intent, thread_id)
        if intent.kind == "recap_detail":
            return self._recap_detail(intent, thread_id)
        return _CHIT_CHAT_REPLY

    def _recap_action(self, intent: Intent, thread_id: str) -> str:
        item = self._resolve_single_recap_item(thread_id, intent.message)
        self._audit.log_recap_reference(thread_id, "recap_action", intent.message, item)
        instruction = rephrase_for_dispatch(self._llm, item, intent.message)
        dispatch_intent = Intent(
            kind="dispatch",
            channel=item.channel,
            message=instruction,
            reply_to=item.source_event_id,
        )
        return self._dispatch_or_ask(dispatch_intent, thread_id)

    def _recap_detail(self, intent: Intent, thread_id: str) -> str:
        item = self._resolve_single_recap_item(thread_id, intent.message)
        self._audit.log_recap_reference(thread_id, "recap_detail", intent.message, item)
        return f"{item.channel} -- {item.summary}\n\n{item.instruction}"

    def _resolve_single_recap_item(
        self, thread_id: str, reference: str | None
    ) -> RecapItem:
        """Resolve `reference` against the last recap's items for `thread_id`,
        raising `RecapActionError` (caught centrally, see `_ACTIONABLE_ERRORS`)
        for anything that isn't exactly one match.

        v1 scope: a reference matching more than one item (including an
        explicit "all") is treated the same as an ambiguous single-item
        reference -- ask which one, rather than acting on a batch. See
        recap_actions.py's module docstring for why that policy lives here
        rather than in `resolve_reference` itself.
        """
        items = self._recap_store.get(thread_id)
        if not items:
            raise RecapActionError(
                "I don't have a recent recap to reference here -- ask for a "
                "recap first."
            )
        matched = resolve_reference(items, reference)
        if len(matched) > 1:
            labels = ", ".join(f"{item.channel}/{item.label}" for item in matched)
            raise RecapActionError(
                "That matches more than one item, and I can only act on one "
                f"at a time for now -- which did you mean: {labels}?"
            )
        return matched[0]

    def _dispatch_or_ask(self, intent: Intent, thread_id: str) -> str:
        if intent.channel is None or intent.message is None:
            return (
                "I didn't catch which channel or what to relay -- which "
                "project channel should I dispatch to, and what should the "
                "message say?"
            )
        if intent.kind != "dispatch":
            intent = replace(intent, kind="dispatch")
        proposal = propose_dispatch(
            self._store, self._config, thread_id, intent, audit=self._audit
        )
        target = f" (for {proposal.target_agent})" if proposal.target_agent else ""
        return (
            f"About to relay to {intent.channel}{target}: "
            f"{proposal.instruction!r}. Confirm to send, or cancel."
        )

    def _confirm(self, thread_id: str, dm_reply_to: str) -> str:
        event_id, proposal = confirm_dispatch(
            self._store, thread_id, self._config.owner, audit=self._audit
        )
        self._watch_for_reply(
            self._reply_watch_id(event_id, proposal),
            dm_channel_id=thread_id,
            dm_reply_to=dm_reply_to,
        )
        return f"Confirmed and relayed (event {event_id})."

    def _reply_watch_id(self, event_id: str, proposal: DispatchProposal) -> str:
        """Return the id to register a reply-wait against for this dispatch.

        A fresh (ungrounded) dispatch posts top-level, so it's already its
        own thread root and `event_id` is exactly right. A recap-action
        dispatch instead threads to `proposal.reply_to` (see
        `pending_actions.py`) -- a message inside an existing thread whose
        real root can be several messages further up, and a coding agent's
        own status-update reply often threads to *that* root rather than to
        the specific relayed message (see module docstring). Resolving the
        true root here, once, up front, is what lets a single dict lookup in
        `_resolve_reply_watch` match either shape of reply. Best-effort: a
        lookup failure falls back to `event_id` (matching a direct reply
        only, the pre-existing behavior) rather than losing the
        wait-and-summarize outright.
        """
        if proposal.reply_to is None:
            return event_id
        try:
            return fetch_thread_root(proposal.channel_id, proposal.reply_to)
        except outbound.RelayError as exc:
            print(f"swingbird: failed to resolve thread root for {event_id}: {exc}")
            return event_id


def _channel_of(event: dict) -> str:
    """Return the channel (#h tag) an event was posted in.

    Also doubles as the pending-action store's "thread id": the channel a
    message arrives in *is* the conversation, so a bare "yes" typed as a new
    message in that channel resolves the proposal made there without needing
    to reply-thread it, and confirm/cancel can never cross channels.
    """
    for tag in event.get("tags", []):
        if tag and tag[0] == "h":
            return tag[1]
    raise DaemonError(f"event {event.get('id')} has no channel (#h) tag")


def _reply_target_ids(event: dict) -> list[str]:
    """Return every event id referenced by an `e` tag on `event` (NIP-10).

    A reply can carry a "root" tag, a "reply" tag, or both depending on
    thread depth, in either order -- see `Daemon._reply_watch_id` -- so
    matching a reply-wait needs to check all of them, not just the first.
    """
    return [tag[1] for tag in event.get("tags", []) if len(tag) >= 2 and tag[0] == "e"]


def build_daemon(config_path: str, audit_log_path: str) -> Daemon:
    """Load config and wire up a `Daemon`, reading the relay key from env."""
    config = load_config(config_path)
    private_key = _read_private_key(config.relay.private_key_env)
    audit = AuditLog(audit_log_path)
    llm = LLMClient(config.llm)
    router = IntentRouter(llm, config, audit=audit)
    inbound = InboundClient(config.relay.url, private_key)
    return Daemon(config, inbound, router, PendingActionStore(), llm, audit)


def _read_private_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise InboundError(f"environment variable {env_var} is not set")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the swingbird TPM agent daemon.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG_PATH)
    args = parser.parse_args()

    daemon = build_daemon(args.config, args.audit_log)
    asyncio.run(daemon.run())
