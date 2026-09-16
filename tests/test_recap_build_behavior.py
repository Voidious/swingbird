import json

from swingbird import recap, recap_transcript
from swingbird.config import (
    ChannelConfig,
    Config,
    LLMConfig,
    OwnerConfig,
    RecapConfig,
    RelayConfig,
)
from swingbird.llm import LLMClient
from swingbird.recap import _CONCISE_SYSTEM_PROMPT, _detailed_system_prompt, build_recap

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


class FakeSequentialCompletions:
    """Returns each of `responses` in turn (one per call), repeating the
    last one for any call beyond the given list -- for tests exercising
    `_backfill_missing_sources`'s own follow-up call, which needs a second,
    independently-scripted response distinct from the main extraction
    call's. A `response` that's an exception instance is raised instead of
    returned, so a test can simulate the backfill call itself failing."""

    def __init__(self, responses: list):
        self._responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, BaseException):
            raise response
        return FakeResponse(response)


class FakeSequentialOpenAI:
    def __init__(self, responses: list):
        self.chat = FakeChat(FakeSequentialCompletions(responses))


def _llm_sequence(*responses: str) -> tuple[LLMClient, FakeSequentialOpenAI]:
    """Like `_llm`, but for tests needing distinct scripted responses across
    more than one `complete_json` call -- pass each response already JSON-
    encoded (or an exception instance to simulate that call failing)."""
    fake = FakeSequentialOpenAI(list(responses))
    return LLMClient(LLM_CONFIG, client=fake), fake


def _llm(
    text: str = "recap", items: list[dict] | None = None
) -> tuple[LLMClient, FakeOpenAI]:
    content = json.dumps({"text": text, "items": items or []})
    fake = FakeOpenAI(content)
    return LLMClient(LLM_CONFIG, client=fake), fake


def _system_prompt(fake) -> str:
    return fake.chat.completions.calls[0]["messages"][0]["content"]


def _setup_detailed_recap(monkeypatch, config):
    monkeypatch.setattr(
        recap_transcript,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [],
    )
    llm, fake = _llm()
    build_recap(llm, config, channel_names=["backend"], detail="detailed")
    return llm, fake


def test_build_recap_appends_a_link_per_shown_item_in_detailed_mode(monkeypatch):
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=2),
    )
    monkeypatch.setattr(
        recap_transcript,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [
                {"created_at": FRESH, "content": "first", "id": "evt-a"},
                {"created_at": FRESH, "content": "second", "id": "evt-b"},
            ]
            if channel_id == "chan-1"
            else []
        ),
    )
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- F5:** prose covering the second item.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
                "source_id": "m1",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
                "source_id": "m2",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n"
        "buzz://message?channel=chan-1&id=evt-a\n\n"
        "**backend -- F5:** prose covering the second item.\n"
        "buzz://message?channel=chan-1&id=evt-b"
    )


def test_build_recap_detailed_mode_shows_up_to_configured_items_as_primary(
    monkeypatch,
):
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=2),
    )
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- F5:** prose covering the second item.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
            {
                "channel": "backend",
                "label": "F6",
                "summary": "tertiary",
                "instruction": "do the tertiary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert [item.is_primary for item in result.items] == [True, True, False]
    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- F5:** prose covering the second item.\n\n"
        "**backend -- 1 additional open item:** F6"
    )


def test_build_recap_strips_llm_narrated_plus_prefixed_count(monkeypatch):
    # Observed live: the LLM narrated "(+1 more open item.)" -- a leading
    # "+" the original regex's plain \d+ alternative didn't match, so the
    # bogus note survived uncorrected right where the deterministic count
    # should have landed instead.
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=1),
    )
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item. (+1 more open item.)",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- 1 additional open item:** F5"
    )


def test_build_recap_names_multiple_additional_items_in_detailed_mode(monkeypatch):
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=1),
    )
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
            {
                "channel": "backend",
                "label": "F6",
                "summary": "tertiary",
                "instruction": "do the tertiary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- 2 additional open items:** F5, F6"
    )


def test_build_recap_drops_stray_standalone_count_paragraph_in_detailed_mode(
    monkeypatch,
):
    # Observed live: for a channel already narrated across multiple item
    # paragraphs, the LLM narrated the fold count as its own separate,
    # bare "**channel**: (+1 more open item.)" paragraph (the concise
    # "no open item" shape) instead of appending it to the last shown
    # item's own paragraph. Once its bogus count note is stripped, that
    # paragraph is a content-free "**channel**:" header and should be
    # dropped entirely -- the real count still belongs on the last shown
    # item's own paragraph.
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=2),
    )
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- F5:** prose covering the second item.\n\n"
        "**backend**: (+1 more open item.)",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
            {
                "channel": "backend",
                "label": "F6",
                "summary": "tertiary",
                "instruction": "do the tertiary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- F5:** prose covering the second item.\n\n"
        "**backend -- 1 additional open item:** F6"
    )


def test_build_recap_concise_mode_only_shows_one_item_as_primary(monkeypatch):
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend**: primary item text.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert [item.is_primary for item in result.items] == [True, False]


def test_build_recap_uses_detailed_prompt_when_requested(monkeypatch):
    (_, fake) = _setup_detailed_recap(monkeypatch, CONFIG)

    assert _system_prompt(fake) == _detailed_system_prompt(
        CONFIG.recap.max_detailed_items
    )


def test_build_recap_detailed_prompt_reflects_configured_max_items(monkeypatch):
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=6),
    )
    (_, fake) = _setup_detailed_recap(monkeypatch, config)

    assert _system_prompt(fake) == _detailed_system_prompt(6)
    assert "up to 6" in _system_prompt(fake)


def test_detailed_prompt_asks_for_richer_item_summaries_than_concise():
    # recap_list.render_items builds "the other items" straight from each
    # RecapItem's stored summary/instruction with no LLM call of its own,
    # so those fields need to already carry detailed-mode richness -- the
    # detailed prompt must ask for it, and the concise one must not.
    assert "same richness" in _detailed_system_prompt(1)
    assert "same richness" not in _CONCISE_SYSTEM_PROMPT


def test_build_recap_unknown_detail_falls_back_to_concise(monkeypatch):
    monkeypatch.setattr(
        recap_transcript,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [],
    )
    llm, fake = _llm()

    build_recap(llm, CONFIG, channel_names=["backend"], detail="bogus")

    assert _system_prompt(fake) == _CONCISE_SYSTEM_PROMPT


def _user_content(fake) -> str:
    return fake.chat.completions.calls[0]["messages"][1]["content"]


def _closed_item(**overrides):
    from swingbird.recap import RecapItem

    fields = {
        "channel": "backend",
        "label": "F1",
        "summary": "old bug",
        "instruction": "Fix the old bug.",
        "source_event_id": None,
        "keywords": (),
    }
    fields.update(overrides)
    return RecapItem(**fields)


def _setup_empty_closed_items_recap(monkeypatch, recap):
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    return _llm()


def _setup_closed_items_recap(monkeypatch, tmp_path, recap):
    from swingbird.closed_items import ClosedItemStore

    llm, fake = _setup_empty_closed_items_recap(monkeypatch, recap)
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    return llm, fake, closed_items


def test_build_recap_omits_closed_items_guard_when_nothing_is_closed(
    tmp_path, monkeypatch
):
    llm, fake, closed_items = _setup_closed_items_recap(monkeypatch, tmp_path, recap)

    build_recap(llm, CONFIG, closed_items=closed_items)

    assert _system_prompt(fake) == _CONCISE_SYSTEM_PROMPT
    assert "already been marked closed" not in _system_prompt(fake)


def test_build_recap_appends_closed_items_guard_when_something_is_closed(
    tmp_path, monkeypatch
):
    llm, fake, closed_items = _setup_closed_items_recap(monkeypatch, tmp_path, recap)
    closed_items.close(_closed_item())

    build_recap(llm, CONFIG, closed_items=closed_items)

    system_content = _system_prompt(fake)
    assert "already been marked closed" in system_content
    assert "backend: F1 -- Fix the old bug." in system_content


def test_build_recap_closed_items_guard_scoped_to_recapped_channels(
    tmp_path, monkeypatch
):
    llm, fake, closed_items = _setup_closed_items_recap(monkeypatch, tmp_path, recap)
    closed_items.close(_closed_item(channel="frontend", label="F2"))

    build_recap(llm, CONFIG, channel_names=["backend"], closed_items=closed_items)

    assert "already been marked closed" not in _system_prompt(fake)


def test_build_recap_tags_the_closed_items_source_message_in_the_transcript(
    tmp_path, monkeypatch
):
    from swingbird.closed_items import ClosedItemStore

    monkeypatch.setattr(
        recap_transcript,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [
            {"id": "evt-1", "created_at": 1000, "content": "fixed the old bug"},
            {"id": "evt-2", "created_at": 1001, "content": "something unrelated"},
        ],
    )
    llm, fake = _llm()
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    closed_items.close(_closed_item(source_event_id="evt-1"))

    build_recap(llm, CONFIG, channel_names=["backend"], closed_items=closed_items)

    transcript = _user_content(fake)
    assert "fixed the old bug [closed]" in transcript
    assert "something unrelated [closed]" not in transcript
    assert "something unrelated" in transcript


def test_build_recap_closed_items_window_is_floored_at_stale_after_days(
    tmp_path, monkeypatch
):
    """A `closed_item_window_days` shorter than the recap's own
    `stale_after_days` message window must never let a still-young closed
    item (one whose restatement could still appear in the transcript being
    recapped) fall outside the closed-items guard -- see
    `RecapConfig.closed_item_window_days`'s own docstring."""
    from swingbird.closed_items import ClosedItemStore

    llm, fake = _setup_empty_closed_items_recap(monkeypatch, recap)
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(stale_after_days=100, closed_item_window_days=30),
    )
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    monkeypatch.setattr(
        recap.time, "time", lambda: NOW
    )  # pin "now" for a deterministic closed_at
    closed_items.close(_closed_item())
    # closed_at is NOW; move the clock forward 60 days so the item is
    # older than closed_item_window_days (30) but still within
    # stale_after_days (100) -- only the floor keeps it in scope.
    monkeypatch.setattr(recap.time, "time", lambda: NOW + 60 * 86400)

    build_recap(llm, config, closed_items=closed_items)

    assert "already been marked closed" in _system_prompt(fake)
