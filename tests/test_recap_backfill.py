import json

import openai

from swingbird import recap
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, FRESH, _llm, _llm_sequence


def _two_message_backend(monkeypatch):
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


def test_build_recap_backfills_a_missing_source_id(monkeypatch):
    _two_message_backend(monkeypatch)
    main_response = json.dumps(
        {
            "text": "recap",
            "items": [
                {
                    "channel": "backend",
                    "label": "F4",
                    "summary": "s",
                    "instruction": "do it",
                }
            ],
        }
    )
    backfill_response = json.dumps({"groundings": [{"index": 0, "source_id": "m2"}]})
    llm, fake = _llm_sequence(main_response, backfill_response)

    result = build_recap(llm, CONFIG)

    assert len(fake.chat.completions.calls) == 2
    assert result.items[0].source_event_id == "evt-b"
    assert result.items[0].source_content == "second"


def test_build_recap_leaves_item_ungrounded_when_backfill_finds_nothing(monkeypatch):
    _two_message_backend(monkeypatch)
    main_response = json.dumps(
        {
            "text": "recap",
            "items": [
                {
                    "channel": "backend",
                    "label": "F4",
                    "summary": "s",
                    "instruction": "do it",
                }
            ],
        }
    )
    backfill_response = json.dumps({"groundings": [{"index": 0, "source_id": None}]})
    llm, _ = _llm_sequence(main_response, backfill_response)

    result = build_recap(llm, CONFIG)

    assert result.items[0].source_event_id is None


def test_build_recap_survives_a_failing_backfill_call(monkeypatch):
    _two_message_backend(monkeypatch)
    main_response = json.dumps(
        {
            "text": "recap",
            "items": [
                {
                    "channel": "backend",
                    "label": "F4",
                    "summary": "s",
                    "instruction": "do it",
                }
            ],
        }
    )
    llm, _ = _llm_sequence(main_response, openai.OpenAIError("boom"))

    result = build_recap(llm, CONFIG)

    assert result.items[0].source_event_id is None


def test_build_recap_skips_backfill_when_nothing_is_missing(monkeypatch):
    _two_message_backend(monkeypatch)
    llm, fake = _llm(
        "recap",
        items=[
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

    assert len(fake.chat.completions.calls) == 1
    assert result.items[0].source_event_id == "evt-b"
