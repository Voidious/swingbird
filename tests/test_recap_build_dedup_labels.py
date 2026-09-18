from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, _llm


def test_build_recap_dedupes_items_with_the_same_normalized_label(monkeypatch):
    # Live bug: a detailed recap extracted two "items" entries for the same
    # underlying work, one labeled "recap-close" (hyphenated slug) and the
    # other "recap close" (plain words) -- _normalize_label already treats
    # those as identical, but nothing had previously used that to collapse
    # a duplicate "items" entry.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "recap",
        items=[
            {
                "channel": "backend",
                "label": "recap-close",
                "summary": "first summary",
                "instruction": "do it",
            },
            {
                "channel": "backend",
                "label": "recap close",
                "summary": "second summary",
                "instruction": "do it too",
            },
            {
                "channel": "frontend",
                "label": "F9",
                "summary": "dropdown regression",
                "instruction": "fix the dropdown",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert len(result.items) == 2
    assert result.items[0].channel == "backend"
    assert result.items[0].label == "recap close"
    assert result.items[0].summary == "first summary"
    assert result.items[1].channel == "frontend"


def test_build_recap_never_dedupes_items_with_empty_labels(monkeypatch):
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "recap",
        items=[
            {
                "channel": "backend",
                "label": "",
                "summary": "first",
                "instruction": "do it",
            },
            {
                "channel": "backend",
                "label": "",
                "summary": "second",
                "instruction": "do it too",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert len(result.items) == 2
