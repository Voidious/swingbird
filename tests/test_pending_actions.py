import pytest

from swingbird import outbound
from swingbird.config import ChannelConfig, Config, LLMConfig, RelayConfig
from swingbird.pending_actions import (
    DispatchProposal,
    PendingActionError,
    PendingActionStore,
    cancel_dispatch,
    confirm_dispatch,
    propose_dispatch,
)
from swingbird.router import Intent

CONFIG = Config(
    llm=LLMConfig(base_url="https://x", model="m", api_key_env="X_KEY"),
    relay=RelayConfig(url="wss://relay.example", private_key_env="SWINGBIRD_KEY"),
    channels=(
        ChannelConfig(id="chan-1", name="backend", write=True, agents=("Codex",)),
        ChannelConfig(id="chan-2", name="frontend", write=False, agents=("Goose",)),
    ),
)

DISPATCH_INTENT = Intent(
    kind="dispatch",
    channel="backend",
    target_agent="Codex",
    message="fix the login timeout bug",
)


def test_propose_dispatch_stores_resolved_channel_id():
    store = PendingActionStore()

    proposal = propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)

    assert proposal == DispatchProposal(
        channel_id="chan-1",
        instruction="fix the login timeout bug",
        target_agent="Codex",
    )
    assert store.get("thread-1") == proposal


def test_propose_dispatch_rejects_non_dispatch_intent():
    store = PendingActionStore()

    with pytest.raises(PendingActionError, match="not a dispatch intent"):
        propose_dispatch(store, CONFIG, "thread-1", Intent(kind="recap"))


def test_propose_dispatch_rejects_missing_channel():
    store = PendingActionStore()
    intent = Intent(kind="dispatch", channel=None, message="do it")

    with pytest.raises(PendingActionError, match="missing a channel or message"):
        propose_dispatch(store, CONFIG, "thread-1", intent)


def test_propose_dispatch_rejects_unknown_channel():
    store = PendingActionStore()
    intent = Intent(kind="dispatch", channel="nonexistent", message="do it")

    with pytest.raises(PendingActionError, match="unknown channel"):
        propose_dispatch(store, CONFIG, "thread-1", intent)


def test_propose_dispatch_rejects_non_writable_channel():
    store = PendingActionStore()
    intent = Intent(kind="dispatch", channel="frontend", message="do it")

    with pytest.raises(PendingActionError, match="not writable"):
        propose_dispatch(store, CONFIG, "thread-1", intent)


def test_confirm_dispatch_posts_and_clears(monkeypatch):
    calls = []
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: calls.append(a) or "evt-1"
    )
    store = PendingActionStore()
    propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)

    event_id = confirm_dispatch(store, "thread-1", "Voidious")

    assert event_id == "evt-1"
    assert calls == [("chan-1", "fix the login timeout bug", "Voidious")]
    assert store.get("thread-1") is None


def test_cancel_dispatch_clears_without_posting(monkeypatch):
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: pytest.fail("should not dispatch")
    )
    store = PendingActionStore()
    propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)

    proposal = cancel_dispatch(store, "thread-1")

    assert proposal.channel_id == "chan-1"
    assert store.get("thread-1") is None


def test_resolve_by_thread_id_raises_if_none_pending():
    store = PendingActionStore()

    with pytest.raises(PendingActionError, match="no pending action for thread"):
        cancel_dispatch(store, "thread-1")


def test_resolve_without_thread_id_raises_if_none_pending():
    store = PendingActionStore()

    with pytest.raises(PendingActionError, match="no pending action to confirm"):
        cancel_dispatch(store, None)


def test_resolve_without_thread_id_raises_if_ambiguous():
    store = PendingActionStore()
    propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)
    propose_dispatch(store, CONFIG, "thread-2", DISPATCH_INTENT)

    with pytest.raises(PendingActionError, match="ambiguous which one"):
        cancel_dispatch(store, None)


def test_resolve_without_thread_id_succeeds_if_exactly_one_pending():
    store = PendingActionStore()
    propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)

    proposal = cancel_dispatch(store, None)

    assert proposal.channel_id == "chan-1"
    assert store.get("thread-1") is None
