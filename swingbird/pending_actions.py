"""Pending-action store and confirm/cancel flow, per §4.2/§5.

A `dispatch` intent from the router (step 5) doesn't post anything by
itself -- it becomes a `DispatchProposal` that sits in the store until
the user confirms or cancels it. This is where "confirm-before-dispatch
is mandatory for anything that writes" (§5) actually lives: nothing
reaches `outbound.relay_dispatch` except through `confirm_dispatch`.

The store also enforces "no guessing on ambiguity" for confirm/cancel
themselves: a bare "yes" that isn't a reply to one specific proposal only
resolves cleanly if there's exactly one pending action anywhere; two or
more (or zero) is treated as ambiguous rather than picking one.
"""

from __future__ import annotations

from dataclasses import dataclass

from swingbird import outbound
from swingbird.config import Config
from swingbird.router import Intent


class PendingActionError(Exception):
    """Raised when a dispatch can't be proposed, or a confirm/cancel can't
    be resolved unambiguously."""


@dataclass(frozen=True)
class DispatchProposal:
    channel_id: str
    instruction: str
    target_agent: str | None


class PendingActionStore:
    """One pending dispatch proposal per conversation thread."""

    def __init__(self) -> None:
        self._pending: dict[str, DispatchProposal] = {}

    def propose(self, thread_id: str, proposal: DispatchProposal) -> None:
        self._pending[thread_id] = proposal

    def get(self, thread_id: str) -> DispatchProposal | None:
        return self._pending.get(thread_id)

    def resolve(self, thread_id: str | None) -> DispatchProposal:
        """Pop and return the proposal a confirm/cancel applies to.

        If `thread_id` is a reply to a specific proposal, resolve that
        one directly. If `thread_id` is None, resolve the single pending
        proposal across all threads -- raising if that's ambiguous (zero
        or more than one candidate).
        """
        if thread_id is not None:
            try:
                return self._pending.pop(thread_id)
            except KeyError:
                raise PendingActionError(
                    f"no pending action for thread {thread_id!r}"
                ) from None
        if not self._pending:
            raise PendingActionError("no pending action to confirm or cancel")
        if len(self._pending) > 1:
            raise PendingActionError(
                "more than one pending action; ambiguous which one"
            )
        (only_thread,) = self._pending
        return self._pending.pop(only_thread)


def propose_dispatch(
    store: PendingActionStore, config: Config, thread_id: str, intent: Intent
) -> DispatchProposal:
    """Build a `DispatchProposal` from a router `dispatch` intent and store it.

    Raises if the channel is missing, unknown, or not writable -- per §5,
    an unresolved or disallowed target must be asked about, never turned
    into a proposal.
    """
    if intent.kind != "dispatch":
        raise PendingActionError(f"not a dispatch intent: {intent.kind!r}")
    if intent.channel is None or intent.message is None:
        raise PendingActionError("dispatch intent is missing a channel or message")
    channel = config.channel_by_name(intent.channel)
    if channel is None:
        raise PendingActionError(f"unknown channel: {intent.channel!r}")
    if not channel.write:
        raise PendingActionError(f"channel is not writable: {intent.channel!r}")

    proposal = DispatchProposal(
        channel_id=channel.id,
        instruction=intent.message,
        target_agent=intent.target_agent,
    )
    store.propose(thread_id, proposal)
    return proposal


def confirm_dispatch(
    store: PendingActionStore, thread_id: str | None, requested_by: str
) -> str:
    """Resolve the pending dispatch and post it; return the new event id."""
    proposal = store.resolve(thread_id)
    return outbound.relay_dispatch(
        proposal.channel_id, proposal.instruction, requested_by
    )


def cancel_dispatch(
    store: PendingActionStore, thread_id: str | None
) -> DispatchProposal:
    """Resolve (discard) the pending dispatch without posting it."""
    return store.resolve(thread_id)
