import pytest

from swingbird import recap
from swingbird.config import (
    ChannelConfig,
    Config,
    LLMConfig,
    OwnerConfig,
    RelayConfig,
)
from swingbird.llm import LLMClient
from swingbird.recap import RecapError, build_recap

LLM_CONFIG = LLMConfig(base_url="https://x", model="m", api_key_env="X_KEY")
RELAY_CONFIG = RelayConfig(url="wss://relay.example", private_key_env="SWINGBIRD_KEY")
CONFIG = Config(
    llm=LLM_CONFIG,
    relay=RELAY_CONFIG,
    channels=(
        ChannelConfig(id="chan-1", name="backend", write=True, agents=("Codex",)),
        ChannelConfig(id="chan-2", name="frontend", write=False, agents=("Goose",)),
    ),
    owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, content):
        self.chat = FakeChat(FakeCompletions(content))


def _llm(content: str) -> tuple[LLMClient, FakeOpenAI]:
    fake = FakeOpenAI(content)
    return LLMClient(LLM_CONFIG, client=fake), fake


def test_build_recap_summarizes_all_channels(monkeypatch):
    def fake_fetch(channel_id, limit=None):
        return {
            "chan-1": [{"created_at": 1, "content": "backend msg"}],
            "chan-2": [{"created_at": 2, "content": "frontend msg"}],
        }[channel_id]

    monkeypatch.setattr(recap, "fetch_recent_messages", fake_fetch)
    llm, fake = _llm("here's the recap")

    result = build_recap(llm, CONFIG)

    assert result == "here's the recap"
    transcript = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "## backend" in transcript
    assert "backend msg" in transcript
    assert "## frontend" in transcript
    assert "frontend msg" in transcript


def test_build_recap_restricts_to_named_channels(monkeypatch):
    calls = []

    def fake_fetch(channel_id, limit=None):
        calls.append(channel_id)
        return []

    monkeypatch.setattr(recap, "fetch_recent_messages", fake_fetch)
    llm, _ = _llm("recap")

    build_recap(llm, CONFIG, channel_names=["backend"])

    assert calls == ["chan-1"]


def test_build_recap_rejects_unknown_channel_name(monkeypatch):
    llm, _ = _llm("recap")

    with pytest.raises(RecapError, match="unknown channel"):
        build_recap(llm, CONFIG, channel_names=["nonexistent"])


def test_build_recap_notes_empty_channel(monkeypatch):
    monkeypatch.setattr(
        recap, "fetch_recent_messages", lambda channel_id, limit=None: []
    )
    llm, fake = _llm("recap")

    build_recap(llm, CONFIG, channel_names=["backend"])

    transcript = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "(no recent activity)" in transcript


def test_build_recap_passes_limit_through(monkeypatch):
    seen_limits = []

    def fake_fetch(channel_id, limit=None):
        seen_limits.append(limit)
        return []

    monkeypatch.setattr(recap, "fetch_recent_messages", fake_fetch)
    llm, _ = _llm("recap")

    build_recap(llm, CONFIG, channel_names=["backend"], limit=5)

    assert seen_limits == [5]
