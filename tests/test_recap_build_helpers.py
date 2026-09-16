from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, _llm


def _build_recap_empty_channel_with_item(monkeypatch):
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
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
    return result


def test_build_recap_leaves_a_non_hyphenated_label_unchanged(monkeypatch):
    result = _build_recap_empty_channel_with_item(monkeypatch)

    assert result.items[0].label == "F4"
