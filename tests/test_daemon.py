import asyncio
import sys

import pytest

from swingbird import daemon, outbound
from swingbird.audit import AuditLog
from swingbird.config import ChannelConfig, Config, LLMConfig, OwnerConfig, RelayConfig
from swingbird.daemon import (
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_CONFIG_PATH,
    Daemon,
    build_daemon,
)
from swingbird.inbound import InboundError
from swingbird.pending_actions import PendingActionStore
from swingbird.router import IntentRouter

OWNER_PUBKEY = "owner-pubkey"
OTHER_PUBKEY = "someone-else"

CONFIG = Config(
    llm=LLMConfig(base_url="https://x", model="m", api_key_env="X_KEY"),
    relay=RelayConfig(url="wss://relay.example", private_key_env="RELAY_KEY"),
    channels=(
        ChannelConfig(id="chan-1", name="backend", write=True, agents=("Codex",)),
        ChannelConfig(id="chan-2", name="frontend", write=False, agents=("Goose",)),
    ),
    owner=OwnerConfig(pubkey=OWNER_PUBKEY, name="Voidious"),
)


class FakeLLM:
    """Duck-types `LLMClient`: a canned classification and/or recap reply."""

    def __init__(self, json_response=None, text_response=""):
        self._json_response = json_response
        self._text_response = text_response
        self.calls: list[list[dict]] = []

    def complete_json(self, messages):
        self.calls.append(messages)
        return self._json_response

    def complete(self, messages):
        self.calls.append(messages)
        return self._text_response


class FakeInbound:
    """Duck-types `InboundClient`: replays a canned list of events."""

    def __init__(self, events):
        self._events = events
        self.connected = False
        self.subscribed = None

    async def connect(self):
        self.connected = True

    async def subscribe(self, channel_ids):
        self.subscribed = channel_ids

    async def events(self):
        for event in self._events:
            yield event


def _event(pubkey=OWNER_PUBKEY, content="hi", tags=None, event_id="evt-1"):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "created_at": 1000,
        "kind": 9,
        "tags": [["p", "some-pubkey"], ["h", "chan-1"]] if tags is None else tags,
        "content": content,
        "sig": "sig",
    }


def _daemon(tmp_path, llm, inbound=None, store=None):
    audit = AuditLog(tmp_path / "audit.jsonl")
    router = IntentRouter(llm, CONFIG, audit=audit)
    return Daemon(
        CONFIG,
        inbound or FakeInbound([]),
        router,
        store or PendingActionStore(),
        llm,
        audit,
    )


def _sent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        outbound, "send_message", lambda *a, **k: calls.append((a, k)) or "reply-evt"
    )
    return calls


def _handle_event_and_get_first_sent(tmp_path, llm, sent, event=None, store=None):
    bot = _daemon(tmp_path, llm, store=store)
    asyncio.run(bot._handle_event(event or _event()))
    return sent[0]


def test_ignores_events_not_from_owner(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot._handle_event(_event(pubkey=OTHER_PUBKEY)))

    assert sent == []
    assert llm.calls == []


def test_recap_reply_is_posted_back_to_the_source_channel(tmp_path, monkeypatch):
    from swingbird import recap

    monkeypatch.setattr(recap, "fetch_recent_messages", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "recap"}, text_response="here's the recap")
    (args, kwargs) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert args == ("chan-1", "here's the recap")
    assert kwargs == {"reply_to": "evt-1"}


def test_dispatch_proposes_and_asks_for_confirmation(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    store = PendingActionStore()
    llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix the login timeout bug",
        }
    )
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent, store=store)
    expected_reply = (
        "About to relay to backend (for Codex): 'fix the login timeout bug'. "
        "Confirm to send, or cancel."
    )
    assert args == ("chan-1", expected_reply)
    assert store.get("chan-1") is not None


def test_dispatch_missing_target_asks_a_clarifying_question(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    store = PendingActionStore()
    llm = FakeLLM(
        json_response={"intent": "dispatch", "channel": None, "message": None}
    )
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent, store=store)
    assert "which project channel" in args[1]
    assert store.get("chan-1") is None


def test_clarify_response_reuses_the_dispatch_path(tmp_path, monkeypatch):
    _sent(monkeypatch)
    store = PendingActionStore()
    llm = FakeLLM(
        json_response={
            "intent": "clarify_response",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix the bug",
        }
    )
    bot = _daemon(tmp_path, llm, store=store)

    asyncio.run(bot._handle_event(_event()))

    assert store.get("chan-1") is not None


def test_confirm_posts_and_replies(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    relayed = []
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: relayed.append(a) or "posted-evt"
    )
    store = PendingActionStore()
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        }
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)
    asyncio.run(bot._handle_event(_event(event_id="evt-1")))

    # A bare "yes" typed as a new message, not a reply -- confirming still
    # resolves because it lands in the same channel the proposal was made in.
    bot._llm._json_response = {"intent": "confirm"}
    asyncio.run(bot._handle_event(_event(event_id="evt-2")))

    assert relayed == [("chan-1", "fix it", "Voidious")]
    (args, _) = sent[-1]
    assert args == ("chan-1", "Confirmed and relayed (event posted-evt).")
    assert store.get("chan-1") is None


def test_confirm_without_a_pending_action_is_safe(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: pytest.fail("should not dispatch")
    )
    llm = FakeLLM(json_response={"intent": "confirm"})
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert args == (
        "chan-1",
        "Couldn't do that: no pending action for thread 'chan-1'",
    )


def test_cancel_discards_without_posting(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: pytest.fail("should not dispatch")
    )
    store = PendingActionStore()
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": None,
            "message": "fix it",
        }
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)
    asyncio.run(bot._handle_event(_event(event_id="evt-1")))

    bot._llm._json_response = {"intent": "cancel"}
    asyncio.run(bot._handle_event(_event(event_id="evt-2")))

    (args, _) = sent[-1]
    assert args == ("chan-1", "Cancelled -- nothing was sent.")
    assert store.get("chan-1") is None


def test_chit_chat_reply(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert "outside what I handle" in args[1]


def test_router_error_becomes_a_reply_not_a_crash(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "not-a-real-intent"})
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert args[1].startswith("Couldn't do that:")


def test_reply_send_failure_is_logged_not_raised(tmp_path, monkeypatch, capsys):
    def _fail(*a, **k):
        raise outbound.RelayError("buzz cli not found")

    monkeypatch.setattr(outbound, "send_message", _fail)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot._handle_event(_event()))

    assert "buzz cli not found" in capsys.readouterr().out


def test_run_connects_subscribes_and_survives_a_malformed_event(
    tmp_path, monkeypatch, capsys
):
    sent = _sent(monkeypatch)
    monkeypatch.setattr(outbound, "open_dm", lambda pubkey: "dm-chan")
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    malformed = _event(event_id="bad-1", tags=[])
    good = _event(event_id="good-1")
    inbound = FakeInbound([malformed, good])
    bot = _daemon(tmp_path, llm, inbound=inbound)

    asyncio.run(bot.run())

    expected_reply = (
        "That's outside what I handle -- ask me for a recap, or to dispatch "
        "an instruction to a project channel."
    )
    assert inbound.connected is True
    assert inbound.subscribed == ["chan-1", "chan-2", "dm-chan"]
    assert sent == [(("chan-1", expected_reply), {"reply_to": "good-1"})]
    assert "bad-1" in capsys.readouterr().out


def test_run_subscribes_to_the_owners_dm_resolved_for_this_run(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    calls = []
    monkeypatch.setattr(
        outbound, "open_dm", lambda pubkey: calls.append(pubkey) or "dm-chan"
    )
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    dm_event = _event(tags=[["h", "dm-chan"]], event_id="dm-evt")
    inbound = FakeInbound([dm_event])
    bot = _daemon(tmp_path, llm, inbound=inbound)

    asyncio.run(bot.run())

    assert calls == [OWNER_PUBKEY]
    (args, kwargs) = sent[0]
    assert args[0] == "dm-chan"
    assert kwargs == {"reply_to": "dm-evt"}


def test_build_daemon_wires_config_llm_and_inbound(tmp_path, monkeypatch):
    monkeypatch.setenv("X_API_KEY", "key")
    monkeypatch.setenv("RELAY_KEY", "1" * 64)
    config_path = tmp_path / "swingbird.toml"
    config_path.write_text(f"""
[llm]
base_url = "https://x"
model = "m"
api_key_env = "X_API_KEY"

[relay]
url = "wss://relay.example"
private_key_env = "RELAY_KEY"

[owner]
pubkey = "{OWNER_PUBKEY}"
name = "Voidious"

[[channels]]
id = "chan-1"
name = "backend"
write = true
agents = ["Codex"]
""")

    bot = build_daemon(str(config_path), str(tmp_path / "audit.jsonl"))

    assert isinstance(bot, Daemon)
    assert bot._config.owner.pubkey == OWNER_PUBKEY


def test_build_daemon_raises_if_relay_key_env_is_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("X_API_KEY", "key")
    monkeypatch.delenv("RELAY_KEY", raising=False)
    config_path = tmp_path / "swingbird.toml"
    config_path.write_text(f"""
[llm]
base_url = "https://x"
model = "m"
api_key_env = "X_API_KEY"

[relay]
url = "wss://relay.example"
private_key_env = "RELAY_KEY"

[owner]
pubkey = "{OWNER_PUBKEY}"
name = "Voidious"

[[channels]]
id = "chan-1"
name = "backend"
write = true
agents = ["Codex"]
""")

    with pytest.raises(InboundError, match="RELAY_KEY"):
        build_daemon(str(config_path), str(tmp_path / "audit.jsonl"))


class _FakeBuiltDaemon:
    def __init__(self):
        self.ran = False

    async def run(self):
        self.ran = True


def test_main_uses_default_paths_and_runs_the_daemon(monkeypatch):
    built = _FakeBuiltDaemon()
    calls = {}

    def fake_build_daemon(config_path, audit_log_path):
        calls["config_path"] = config_path
        calls["audit_log_path"] = audit_log_path
        return built

    monkeypatch.setattr(daemon, "build_daemon", fake_build_daemon)
    monkeypatch.setattr(sys, "argv", ["swingbird"])

    daemon.main()

    assert calls["config_path"] == DEFAULT_CONFIG_PATH
    assert calls["audit_log_path"] == DEFAULT_AUDIT_LOG_PATH
    assert built.ran is True


def test_main_passes_through_custom_paths(monkeypatch):
    built = _FakeBuiltDaemon()
    calls = {}

    def fake_build_daemon(config_path, audit_log_path):
        calls["config_path"] = config_path
        calls["audit_log_path"] = audit_log_path
        return built

    monkeypatch.setattr(daemon, "build_daemon", fake_build_daemon)
    monkeypatch.setattr(
        sys,
        "argv",
        ["swingbird", "--config", "other.toml", "--audit-log", "other.jsonl"],
    )

    daemon.main()

    assert calls["config_path"] == "other.toml"
    assert calls["audit_log_path"] == "other.jsonl"
