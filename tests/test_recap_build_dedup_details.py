from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, _llm


def test_build_recap_dedupes_duplicate_detailed_paragraphs(monkeypatch):
    # Same live bug as the "items"-level dedup above, but for "text" itself
    # -- "text" is the LLM's own free-form prose, not reconstructed from
    # "items", so a duplicate paragraph there isn't guaranteed to disappear
    # just because "items" no longer has a matching duplicate entry.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- recap-close:** first paragraph.\n\n"
        "**backend -- recap close:** second, restated paragraph.",
        items=[
            {
                "channel": "backend",
                "label": "recap close",
                "summary": "recap close summary",
                "instruction": "do it",
            },
        ],
    )

    result = build_recap(llm, CONFIG, detail="detailed")

    assert result.text == "**backend -- recap close:** first paragraph."


def test_build_recap_dedupes_detailed_paragraphs_with_no_matching_item(monkeypatch):
    # A duplicate paragraph header with no corresponding "items" entry (the
    # LLM named a channel/label combination "items" doesn't have) still
    # gets deduped by its own normalized header -- there's just no
    # canonical item prefix to rewrite the survivor to, so it's kept as
    # the LLM wrote it.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** first paragraph.\n\n"
        "**backend -- f4:** second, restated paragraph.",
        items=[],
    )

    result = build_recap(llm, CONFIG, detail="detailed")

    assert result.text == "**backend -- F4:** first paragraph."


def test_build_recap_does_not_dedupe_detailed_paragraphs_in_concise_mode(monkeypatch):
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    text = (
        "**backend -- recap-close:** first paragraph.\n\n"
        "**backend -- recap close:** second, restated paragraph."
    )
    llm, _ = _llm(text, items=[])

    result = build_recap(llm, CONFIG)

    assert result.text == text
