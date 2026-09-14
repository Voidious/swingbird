from swingbird import recap
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, _llm


def test_build_recap_flattens_a_hyphenated_label_to_spaces(monkeypatch):
    # Observed live: the LLM sometimes echoes a git-branch-shaped slug from
    # the transcript as an item's label (e.g. "fix-message-id-reliability")
    # instead of "a few words naming the item" per
    # _ITEM_EXTRACTION_INSTRUCTIONS -- inconsistently, since a different
    # recap call for the same underlying work came back with plain words.
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "here's the recap",
        items=[
            {
                "channel": "backend",
                "label": "fix-message-id-reliability",
                "summary": "s",
                "instruction": "do it",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].label == "fix message id reliability"
