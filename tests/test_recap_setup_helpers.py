from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, FRESH, _llm


def _setup_channel_with_messages_and_build_recap(monkeypatch, text="recap", items=None):
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
