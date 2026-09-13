import json

import pytest

from swingbird import recap
from swingbird.config import ChannelConfig, Config, OwnerConfig, RecapConfig
from swingbird.llm import LLMClient
from swingbird.recap import _CONCISE_SYSTEM_PROMPT, RecapError, RecapItem, build_recap
from tests.test_recap_build_behavior import (
    CONFIG,
    FRESH,
    LLM_CONFIG,
    NOW,
    RELAY_CONFIG,
    STALE,
    FakeOpenAI,
    _llm,
    _system_prompt,
)


def _raw_llm(content: str) -> tuple[LLMClient, FakeOpenAI]:
    """For responses that don't match the normal {"text": ...} shape."""
    fake = FakeOpenAI(content)
    return LLMClient(LLM_CONFIG, client=fake), fake


@pytest.fixture(autouse=True)
def _freeze_time(monkeypatch):
    monkeypatch.setattr(recap.time, "time", lambda: NOW)


def _transcript(fake) -> str:
    return fake.chat.completions.calls[0]["messages"][1]["content"]


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


def test_build_recap_tags_transcript_messages_with_a_local_id(monkeypatch):
    monkeypatch.setattr(
        recap,
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
    llm, fake = _llm()

    build_recap(llm, CONFIG)

    transcript = _transcript(fake)
    assert "[m1] " in transcript
    assert "[m2] " in transcript


def _setup_channel_with_messages_and_build_recap(monkeypatch, text="recap", items=None):
    monkeypatch.setattr(
        recap,
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
        text,
        items=items
        or [
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
                "source_id": "m2",
            }
        ],
    )

    result = build_recap(llm, CONFIG)
    return result


def test_build_recap_resolves_source_id_to_the_real_event_id(monkeypatch):
    result = _setup_channel_with_messages_and_build_recap(monkeypatch)

    assert result.items[0].source_event_id == "evt-b"


def test_build_recap_resolves_source_id_to_the_message_content(monkeypatch):
    result = _setup_channel_with_messages_and_build_recap(monkeypatch)

    assert result.items[0].source_content == "second"


def _build_empty_channel_recap(monkeypatch, items=None):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "recap",
        items=items
        or [
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
            }
        ],
    )

    result = build_recap(llm, CONFIG)
    return result


def test_build_recap_leaves_source_content_none_without_a_source_id(monkeypatch):
    result = _build_empty_channel_recap(monkeypatch)

    assert result.items[0].source_content is None


def test_build_recap_leaves_source_event_id_none_without_a_source_id(monkeypatch):
    result = _build_empty_channel_recap(monkeypatch)

    assert result.items[0].source_event_id is None


def test_build_recap_leaves_source_event_id_none_for_an_unknown_tag(monkeypatch):
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [{"created_at": FRESH, "content": "first", "id": "evt-a"}]
            if channel_id == "chan-1"
            else []
        ),
    )
    llm, _ = _llm(
        "recap",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
                "source_id": "m9",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].source_event_id is None


def test_build_recap_leaves_source_event_id_none_for_an_unmapped_channel(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "recap",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
                "source_id": "m1",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].source_event_id is None


def test_build_recap_skips_map_entry_for_messages_without_an_id(monkeypatch):
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [{"created_at": FRESH, "content": "no id here"}]
            if channel_id == "chan-1"
            else []
        ),
    )
    llm, _ = _llm(
        "recap",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
                "source_id": "m1",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].source_event_id is None


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


def test_build_recap_parses_keywords(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "here's the recap",
        items=[
            {
                "channel": "backend",
                "label": "F6",
                "summary": "lint residue",
                "instruction": "fix the prefer-const finding",
                "keywords": ["lint issue", "prefer-const finding"],
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].keywords == ("lint issue", "prefer-const finding")


def test_build_recap_defaults_keywords_to_empty_tuple_when_omitted(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "here's the recap",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].keywords == ()


def test_build_recap_drops_non_string_and_blank_keyword_entries(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "here's the recap",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
                "keywords": ["real one", "  ", 7, ""],
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].keywords == ("real one",)


def test_build_recap_ignores_a_non_list_keywords_value(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "here's the recap",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
                "keywords": "not a list",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].keywords == ()


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


def test_build_recap_appends_additional_item_count_to_concise_text(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend**: primary item text.\n\n**frontend**: nothing new.",
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

    assert result.text == (
        "**backend**: primary item text. (1 additional open item.)\n\n"
        "**frontend**: nothing new."
    )


def test_build_recap_pluralizes_additional_item_count(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
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
            {
                "channel": "backend",
                "label": "F6",
                "summary": "tertiary",
                "instruction": "do the tertiary thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == "**backend**: primary item text. (2 additional open items.)"


def test_build_recap_strips_llm_narrated_count_before_appending_real_one(monkeypatch):
    # Observed live: the LLM narrated its own "(1 more open item.)" despite
    # the prompt telling it not to, right where our deterministic count
    # would land -- without stripping first, the two would stack.
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**swingbird**: primary item text. (1 more open item.)",
        items=[
            {
                "channel": "swingbird",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "swingbird",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == (
        "**swingbird**: primary item text. (1 additional open item.)"
    )


def test_build_recap_strips_llm_narrated_zero_count_with_nothing_to_append(
    monkeypatch,
):
    # Observed live: the LLM narrated "(0 more open items.)" for a channel
    # that really does have zero additional items -- there's no real count
    # to append afterward, but the bogus note must still be removed rather
    # than left standing uncorrected.
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**crispen**: primary item text. (0 more open items.)",
        items=[
            {
                "channel": "crispen",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == "**crispen**: primary item text."


def test_build_recap_strips_llm_narrated_countless_remain_note(monkeypatch):
    # Observed live: the LLM narrated "(Additional open items remain.)" --
    # no leading number/no/zero, and "remain" instead of "remaining" -- which
    # the original regex (requiring a leading count word) didn't catch,
    # leaving a bogus, uncorrected note standing on its own.
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**swingbird**: primary item text. (Additional open items remain.)",
        items=[
            {
                "channel": "swingbird",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "swingbird",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == (
        "**swingbird**: primary item text. (1 additional open item.)"
    )


def test_build_recap_strips_llm_narrated_count_case_insensitively(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend**: primary item text. (2 More Open Items.)",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == "**backend**: primary item text."


def test_build_recap_leaves_unrelated_trailing_parenthetical_alone(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend**: primary item text. (recommended)",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == "**backend**: primary item text. (recommended)"


def test_build_recap_appends_source_link_to_primary_items_paragraph(monkeypatch):
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [{"created_at": FRESH, "content": "first", "id": "evt-a"}]
            if channel_id == "chan-1"
            else []
        ),
    )
    llm, _ = _llm(
        "**backend**: primary item text.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
                "source_id": "m1",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == (
        "**backend**: primary item text.\nbuzz://message?channel=chan-1&id=evt-a"
    )


def test_build_recap_omits_source_link_when_item_not_grounded(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend**: primary item text.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == "**backend**: primary item text."


def test_build_recap_omits_source_link_for_a_non_primary_item(monkeypatch):
    monkeypatch.setattr(
        recap,
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
        "**backend**: primary item text.",
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

    result = build_recap(llm, CONFIG)

    assert result.text == (
        "**backend**: primary item text. (1 additional open item.)\n"
        "buzz://message?channel=chan-1&id=evt-a"
    )


def test_build_recap_appends_source_link_in_detailed_mode(monkeypatch):
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [{"created_at": FRESH, "content": "first", "id": "evt-a"}]
            if channel_id == "chan-1"
            else []
        ),
    )
    llm, _ = _llm(
        "**backend -- F4:** prose covering the one item already.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
                "source_id": "m1",
            }
        ],
    )

    result = build_recap(llm, CONFIG, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the one item already.\n"
        "buzz://message?channel=chan-1&id=evt-a"
    )


def test_build_recap_rejects_response_missing_text(monkeypatch):
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _raw_llm(json.dumps({"items": []}))

    with pytest.raises(RecapError, match="missing recap text"):
        build_recap(llm, CONFIG)


def _build_recap_with_fake_fetch(
    monkeypatch, fake_fetch, config, channel_names=("backend",)
):
    monkeypatch.setattr(recap, "fetch_messages_since", fake_fetch)
    llm, _ = _llm()
    build_recap(llm, config, channel_names=channel_names)
    return llm


def test_build_recap_restricts_to_named_channels(monkeypatch):
    calls = []

    def fake_fetch(channel_id, since_ts, max_messages=None):
        calls.append(channel_id)
        return []

    _build_recap_with_fake_fetch(monkeypatch, fake_fetch, CONFIG)

    assert calls == ["chan-1"]


def test_build_recap_named_channel_uses_same_window_as_all_channels(monkeypatch):
    """A named channel is no longer special-cased to "most recent N
    regardless of age" -- it goes through the exact same
    `fetch_messages_since(cutoff, max_messages)` call as an all-channels
    recap, just restricted to that one channel."""
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_messages_per_channel=42),
    )
    seen = []

    def fake_fetch(channel_id, since_ts, max_messages=None):
        seen.append((since_ts, max_messages))
        return []

    llm = _build_recap_with_fake_fetch(monkeypatch, fake_fetch, config)
    build_recap(llm, config)

    assert seen[0] == seen[1]


def _build_backend_recap_transcript() -> str:
    llm, fake = _llm()
    build_recap(llm, CONFIG, channel_names=["backend"])
    transcript = _transcript(fake)
    assert "## backend" in transcript
    return transcript


def test_build_recap_includes_fresh_content_for_named_channel(monkeypatch):
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [
            {"created_at": FRESH, "content": "fresh backend msg"}
        ],
    )
    transcript = _build_backend_recap_transcript()
    assert "fresh backend msg" in transcript


def test_build_recap_rejects_unknown_channel_name(monkeypatch):
    llm, _ = _llm()

    with pytest.raises(RecapError, match="unknown channel"):
        build_recap(llm, CONFIG, channel_names=["nonexistent"])


def _setup_empty_channel_recap(monkeypatch, channel_names=None):
    if channel_names is None:
        channel_names = ["backend"]
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [],
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


def test_build_recap_shows_placeholder_for_named_channel_outside_the_window(
    monkeypatch,
):
    """Unlike the old "explicit channels bypass staleness" behavior, a
    named channel now uses the same time-windowed fetch as an all-channels
    recap -- so a message older than the window doesn't show up here
    either. What's still different from the all-channels case is that the
    channel isn't dropped outright: it gets a "(no recent activity)"
    paragraph instead of being omitted (see the omitted-channel test
    below)."""
    monkeypatch.setattr(
        recap,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: [],
    )
    transcript = _build_backend_recap_transcript()
    assert "(no recent activity)" in transcript


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
