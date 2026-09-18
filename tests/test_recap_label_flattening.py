from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, _llm


def test_build_recap_flattens_a_hyphenated_label_to_spaces(monkeypatch):
    # Observed live: the LLM sometimes echoes a git-branch-shaped slug from
    # the transcript as an item's label (e.g. "fix-message-id-reliability")
    # instead of "a few words naming the item" per
    # _ITEM_EXTRACTION_INSTRUCTIONS -- inconsistently, since a different
    # recap call for the same underlying work came back with plain words.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
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


def test_build_recap_leaves_a_phrase_with_an_internal_hyphen_unchanged(monkeypatch):
    # Live bug: a blanket hyphen-to-space replace also flattened a normal
    # phrase's own legitimately hyphenated compound word (e.g. "follow-up"),
    # turning "Integrate recap follow-up reliability" into "...follow up
    # reliability" -- a string the LLM's own "text" narration never wrote,
    # since nothing renormalizes that copy. `_ensure_channel_paragraphs`
    # then couldn't find the already-narrated paragraph by this changed
    # label and appended a duplicate fallback for it. Only a label with no
    # spaces at all (a bare slug, like the sibling test above) should be
    # flattened.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        items=[
            {
                "channel": "backend",
                "label": "Integrate recap follow-up reliability",
                "summary": "s",
                "instruction": "do it",
            }
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].label == "Integrate recap follow-up reliability"
