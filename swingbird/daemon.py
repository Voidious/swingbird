"""Daemon: wires the inbound relay client to the router, pending-action
store, and audit log, and enforces the safety invariants from §5.

Only events from `config.owner.pubkey` are ever routed as a command --
everyone else's traffic in a subscribed channel (including a coding agent's
own replies) is ignored here, so it can never be misread as a
dispatch/confirm/cancel. A single bad or unexpected event never kills the
daemon: known, safety-relevant failures (an ambiguous target, a bad LLM
response, an unwritable channel, ...) become a reply to the sender instead
of a crash, and anything truly unexpected is logged and the loop continues.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import replace

from swingbird import outbound
from swingbird.audit import AuditLog
from swingbird.config import Config, load_config
from swingbird.inbound import InboundClient, InboundError
from swingbird.llm import LLMClient, LLMError
from swingbird.pending_actions import (
    PendingActionError,
    PendingActionStore,
    cancel_dispatch,
    confirm_dispatch,
    propose_dispatch,
)
from swingbird.recap import RecapError, build_recap
from swingbird.router import Intent, IntentRouter, RouterError

DEFAULT_CONFIG_PATH = "swingbird.toml"
DEFAULT_AUDIT_LOG_PATH = "audit.jsonl"

_CHIT_CHAT_REPLY = (
    "That's outside what I handle -- ask me for a recap, or to dispatch an "
    "instruction to a project channel."
)
_ACTIONABLE_ERRORS = (RouterError, PendingActionError, RecapError, LLMError)


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
    ) -> None:
        self._config = config
        self._inbound = inbound
        self._router = router
        self._store = store
        self._llm = llm
        self._audit = audit

    async def run(self) -> None:
        channel_ids = [channel.id for channel in self._config.channels]
        await self._inbound.connect()
        await self._inbound.subscribe(channel_ids)
        async for event in self._inbound.events():
            await self._safe_handle(event)

    async def _safe_handle(self, event: dict) -> None:
        try:
            await self._handle_event(event)
        except Exception as exc:  # noqa: BLE001 - last-resort net, see module docstring
            print(f"swingbird: failed to handle event {event.get('id')}: {exc}")

    async def _handle_event(self, event: dict) -> None:
        if event["pubkey"] != self._config.owner.pubkey:
            return
        channel_id = _channel_of(event)
        reply = self._process(event, channel_id)
        try:
            outbound.send_message(channel_id, reply, reply_to=event["id"])
        except outbound.RelayError as exc:
            print(f"swingbird: failed to send reply to {event['id']}: {exc}")

    def _process(self, event: dict, thread_id: str) -> str:
        try:
            intent = self._router.route(event["content"], thread_id=thread_id)
            return self._act(intent, thread_id)
        except _ACTIONABLE_ERRORS as exc:
            return f"Couldn't do that: {exc}"

    def _act(self, intent: Intent, thread_id: str) -> str:
        if intent.kind == "recap":
            channel_names = [intent.channel] if intent.channel else None
            return build_recap(self._llm, self._config, channel_names=channel_names)
        if intent.kind in ("dispatch", "clarify_response"):
            return self._dispatch_or_ask(intent, thread_id)
        if intent.kind == "confirm":
            return self._confirm(thread_id)
        if intent.kind == "cancel":
            cancel_dispatch(self._store, thread_id, audit=self._audit)
            return "Cancelled -- nothing was sent."
        return _CHIT_CHAT_REPLY

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

    def _confirm(self, thread_id: str) -> str:
        event_id = confirm_dispatch(
            self._store, thread_id, self._config.owner.name, audit=self._audit
        )
        return f"Confirmed and relayed (event {event_id})."


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
