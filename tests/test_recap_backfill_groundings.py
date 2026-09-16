import json

from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, FRESH, _llm_sequence


def test_build_recap_ignores_malformed_backfill_groundings(monkeypatch):
    monkeypatch.setattr(
        recap_transcript,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [{"created_at": FRESH, "content": "first", "id": "evt-a"}]
            if channel_id == "chan-1"
            else [{"created_at": FRESH, "content": "second", "id": "evt-c"}]
        ),
    )
    main_response = json.dumps(
        {
            "text": "recap",
            "items": [
                {
                    "channel": "backend",
                    "label": "F4",
                    "summary": "s",
                    "instruction": "do it",
                },
                {
                    "channel": "frontend",
                    "label": "F5",
                    "summary": "s2",
                    "instruction": "do it too",
                },
            ],
        }
    )
    backfill_response = json.dumps(
        {
            "groundings": [
                "not a dict",
                {"index": "zero", "source_id": "m1"},
                {"index": 5, "source_id": "m1"},
                {"index": 1, "source_id": "unknown-tag"},
                {"index": 0, "source_id": "m1"},
            ]
        }
    )
    llm, _ = _llm_sequence(main_response, backfill_response)

    result = build_recap(llm, CONFIG)

    assert result.items[0].source_event_id == "evt-a"
    assert result.items[1].source_event_id is None
