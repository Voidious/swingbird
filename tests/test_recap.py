import json

import pytest

from swingbird import recap
from swingbird.config import (
    ChannelConfig,
    Config,
    LLMConfig,
    OwnerConfig,
    RecapConfig,
    RelayConfig,
)
from swingbird.llm import LLMClient
from swingbird.recap import (
    _CONCISE_SYSTEM_PROMPT,
    _DETAILED_SYSTEM_PROMPT,
    RecapError,
    RecapItem,
    build_recap,
)

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

NOW = 1_700_000_000
FRESH = NOW - 1_000
STALE = NOW - 40 * 86400  # 40 days ago, outside the default 30-day window


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


def _llm(
    text: str = "recap", items: list[dict] | None = None
) -> tuple[LLMClient, FakeOpenAI]:
    content = json.dumps({"text": text, "items": items or []})
    fake = FakeOpenAI(content)
    return LLMClient(LLM_CONFIG, client=fake), fake


def _raw_llm(content: str) -> tuple[LLMClient, FakeOpenAI]:
    """For responses that don't match the normal {"text": ...} shape."""
    fake = FakeOpenAI(content)
    return LLMClient(LLM_CONFIG, client=fake), fake


@pytest.fixture(autouse=True)
def _freeze_time(monkeypatch):
    monkeypatch.setattr(recap.time, "time", lambda: NOW)


def _transcript(fake) -> str:
    return fake.chat.completions.calls[0]["messages"][1]["content"]


def _system_prompt(fake) -> str:
    return fake.chat.completions.calls[0]["messages"][0]["content"]


def test_build_recap_summarizes_all_channels(monkeypatch):
    def fake_fetch(channel_id, since_ts, max_messages=None):
        return {
            "chan-1": [{"created_at": FRESH, "content": "backend msg"}],
            "chan-2": [{"created_at": FRESH, "content": "frontend msg"}],
        }[channel_id]

    monkeypatch.setattr(recap, "fetch_messages_since", fake_fetch)
    llm, fake = _llm("here's the recap")

    result = build_recap(llm, CONFIG)

    assert result.text == "here's the recap"
    transcript = _transcript(fake)
    assert "## backend" in transcript
    assert "backend msg" in transcript
    assert "## frontend" in transcript
    assert "frontend msg" in transcript


def test_build_recap_parses_structured_items(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "here's the recap",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "unused-ignore propagation",
                "instruction": "Fix the deterministic directive trip-check.",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items == (
        RecapItem(
            channel="backend",
            label="F4",
            summary="unused-ignore propagation",
            instruction="Fix the deterministic directive trip-check.",
        ),
    )


def test_build_recap_drops_items_without_an_instruction(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "here's the recap",
        items=[
            {"channel": "backend", "label": "F4", "summary": "no next step yet"},
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items == ()


def test_build_recap_rejects_response_missing_text(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _raw_llm(json.dumps({"items": []}))

    with pytest.raises(RecapError, match="missing recap text"):
        build_recap(llm, CONFIG)


def test_build_recap_restricts_to_named_channels(monkeypatch):
    calls = []

    def fake_fetch(channel_id, limit=None):
        calls.append((channel_id, limit))
        return []

    monkeypatch.setattr(recap, "fetch_recent_messages", fake_fetch)
    llm, _ = _llm()

    build_recap(llm, CONFIG, channel_names=["backend"])

    assert calls == [("chan-1", recap.EXPLICIT_CHANNEL_MESSAGE_LIMIT)]


def test_build_recap_rejects_unknown_channel_name(monkeypatch):
    llm, _ = _llm()

    with pytest.raises(RecapError, match="unknown channel"):
        build_recap(llm, CONFIG, channel_names=["nonexistent"])


def _setup_empty_channel_recap(monkeypatch, channel_names=None):
    if channel_names is None:
        channel_names = ["backend"]
    monkeypatch.setattr(
        recap, "fetch_recent_messages", lambda channel_id, limit=None: []
    )
    llm, fake = _llm()

    build_recap(llm, CONFIG, channel_names=channel_names)
    return fake


def test_build_recap_notes_empty_channel(monkeypatch):
    fake = _setup_empty_channel_recap(monkeypatch)

    assert "(no recent activity)" in _transcript(fake)


def _setup_and_build_recap(monkeypatch, fake_fetch, config):
    monkeypatch.setattr(recap, "fetch_messages_since", fake_fetch)
    llm, fake = _llm()

    build_recap(llm, config)
    return fake


def test_build_recap_passes_configured_max_messages_through(monkeypatch):
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_messages_per_channel=42),
    )
    seen_max = []

    def fake_fetch(channel_id, since_ts, max_messages=None):
        seen_max.append(max_messages)
        return []

    _setup_and_build_recap(monkeypatch, fake_fetch, config)

    assert seen_max == [42, 42]


def _build_recap_and_get_transcript(monkeypatch, fake_fetch, config):
    fake = _setup_and_build_recap(monkeypatch, fake_fetch, config)

    return _transcript(fake)


def test_build_recap_omits_empty_channel_from_all_channels_recap(monkeypatch):
    def fake_fetch(channel_id, since_ts, max_messages=None):
        return {
            "chan-1": [],
            "chan-2": [{"created_at": FRESH, "content": "fresh frontend msg"}],
        }[channel_id]

    transcript = _build_recap_and_get_transcript(monkeypatch, fake_fetch, CONFIG)
    assert "backend" not in transcript
    assert "## frontend" in transcript
    assert "fresh frontend msg" in transcript


def test_build_recap_includes_stale_channel_when_named_explicitly(monkeypatch):
    monkeypatch.setattr(
        recap,
        "fetch_recent_messages",
        lambda channel_id, limit=None: [{"created_at": STALE, "content": "old msg"}],
    )
    llm, fake = _llm()

    build_recap(llm, CONFIG, channel_names=["backend"])

    transcript = _transcript(fake)
    assert "## backend" in transcript
    assert "old msg" in transcript


def test_build_recap_all_channels_stale_yields_placeholder_transcript(monkeypatch):
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [],
    )
    llm, fake = _llm()

    build_recap(llm, CONFIG)

    assert _transcript(fake) == "(no channels with recent activity)"


def test_build_recap_includes_goal_in_channel_header(monkeypatch):
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=(
            ChannelConfig(
                id="chan-1",
                name="backend",
                write=True,
                agents=("Codex",),
                goal="Preparing the 0.8.0 release",
            ),
        ),
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
    )
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [
            {"created_at": FRESH, "content": "msg"}
        ],
    )
    llm, fake = _llm()

    build_recap(llm, config)

    assert "## backend (goal: Preparing the 0.8.0 release)" in _transcript(fake)


def test_build_recap_respects_configured_stale_after_days(monkeypatch):
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(stale_after_days=60),
    )

    def fake_fetch(channel_id, since_ts, max_messages=None):
        events = {
            "chan-1": [{"created_at": STALE, "content": "old backend msg"}],
            "chan-2": [{"created_at": FRESH, "content": "fresh frontend msg"}],
        }[channel_id]
        return [event for event in events if event["created_at"] >= since_ts]

    # STALE is 40 days ago, inside a 60-day window -- both channels included.
    transcript = _build_recap_and_get_transcript(monkeypatch, fake_fetch, config)
    assert "## backend" in transcript
    assert "## frontend" in transcript


def test_build_recap_defaults_to_concise_prompt(monkeypatch):
    fake = _setup_empty_channel_recap(monkeypatch)

    assert _system_prompt(fake) == _CONCISE_SYSTEM_PROMPT


def test_build_recap_uses_detailed_prompt_when_requested(monkeypatch):
    monkeypatch.setattr(
        recap, "fetch_recent_messages", lambda channel_id, limit=None: []
    )
    llm, fake = _llm()

    build_recap(llm, CONFIG, channel_names=["backend"], detail="detailed")

    assert _system_prompt(fake) == _DETAILED_SYSTEM_PROMPT


def test_build_recap_unknown_detail_falls_back_to_concise(monkeypatch):
    monkeypatch.setattr(
        recap, "fetch_recent_messages", lambda channel_id, limit=None: []
    )
    llm, fake = _llm()

    build_recap(llm, CONFIG, channel_names=["backend"], detail="bogus")

    assert _system_prompt(fake) == _CONCISE_SYSTEM_PROMPT
