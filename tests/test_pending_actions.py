import pytest

from swingbird import outbound
from swingbird.config import (
    ChannelConfig,
    Config,
    LLMConfig,
    OwnerConfig,
    RelayConfig,
)
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
    owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
)

DISPATCH_INTENT = Intent(
    kind="dispatch",
    channel="backend",
    target_agent="Codex",
    message="fix the login timeout bug",
)


class FakeAuditLog:
    def __init__(self):
        self.proposed: list[tuple] = []
        self.decisions: list[tuple] = []

    def log_proposed_action(self, thread_id, proposal):
        self.proposed.append((thread_id, proposal))

    def log_decision(self, thread_id, decision, proposal, event_id=None):
        self.decisions.append((thread_id, decision, proposal, event_id))


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


def test_propose_dispatch_logs_proposed_action_when_audit_configured():
    store = PendingActionStore()
    audit = FakeAuditLog()

    proposal = propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT, audit=audit)

    assert audit.proposed == [("thread-1", proposal)]


def test_propose_dispatch_without_audit_does_not_raise():
    store = PendingActionStore()

    propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)


def _setup_audit_store(thread_id="thread-1"):
    store = PendingActionStore()
    audit = FakeAuditLog()
    propose_dispatch(store, CONFIG, thread_id, DISPATCH_INTENT, audit=audit)
    return store, audit


def _assert_single_audit_decision(audit, expected_thread_id):
    (entry,) = audit.decisions
    thread_id, decision, proposal, event_id = entry
    assert thread_id == expected_thread_id
    return decision, proposal, event_id


def test_confirm_dispatch_logs_decision_with_event_id_when_audit_configured(
    monkeypatch,
):
    monkeypatch.setattr(outbound, "relay_dispatch", lambda *a: "evt-1")
    store, audit = _setup_audit_store()

    confirm_dispatch(store, "thread-1", "Voidious", audit=audit)

    decision, proposal, event_id = _assert_single_audit_decision(audit, "thread-1")
    assert decision == "confirmed"
    assert proposal.channel_id == "chan-1"
    assert event_id == "evt-1"


def test_confirm_dispatch_without_audit_does_not_raise(monkeypatch):
    monkeypatch.setattr(outbound, "relay_dispatch", lambda *a: "evt-1")
    store = PendingActionStore()
    propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)

    confirm_dispatch(store, "thread-1", "Voidious")


def test_cancel_dispatch_logs_decision_when_audit_configured():
    store, audit = _setup_audit_store()

    cancel_dispatch(store, "thread-1", audit=audit)

    decision, proposal, event_id = _assert_single_audit_decision(audit, "thread-1")
    assert decision == "cancelled"
    assert proposal.channel_id == "chan-1"
    assert event_id is None


def test_cancel_dispatch_without_audit_does_not_raise():
    store = PendingActionStore()
    propose_dispatch(store, CONFIG, "thread-1", DISPATCH_INTENT)

    cancel_dispatch(store, "thread-1")


def test_resolve_without_thread_id_logs_the_actual_resolved_thread_id():
    store, audit = _setup_audit_store()

    cancel_dispatch(store, None, audit=audit)

    (proposed_thread_id, _) = audit.proposed[0]
    (decision_thread_id, _, _, _) = audit.decisions[0]
    assert proposed_thread_id == decision_thread_id == "thread-1"
