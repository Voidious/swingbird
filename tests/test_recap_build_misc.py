from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, _llm


def test_build_recap_backfills_a_missing_item_when_its_channel_has_others(
    monkeypatch,
):
    # Live bug: a channel-level check ("does backend have *a* paragraph at
    # all?") wrongly treated a channel as fully covered once *any* of its
    # primary items got narrated, silently dropping the rest -- observed
    # live with max_detailed_items=3 and only the first two of three
    # primary items narrated in "text". This also broke the "N additional
    # open items" fold note downstream, since it anchors to the last
    # primary item's own paragraph, which never existed.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F1:** first item.\n\n**backend -- F2:** second item.",
        items=[
            {
                "channel": "backend",
                "label": "F1",
                "summary": "first item",
                "instruction": "do the first thing",
            },
            {
                "channel": "backend",
                "label": "F2",
                "summary": "second item",
                "instruction": "do the second thing",
            },
            {
                "channel": "backend",
                "label": "F3",
                "summary": "third item",
                "instruction": "do the third thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG, detail="detailed")

    assert result.text == (
        "**backend -- F1:** first item.\n\n"
        "**backend -- F2:** second item.\n\n"
        "**backend -- F3:** third item"
    )


def test_build_recap_backfills_a_missing_item_next_to_its_own_channel(monkeypatch):
    # Live bug: a missing primary item's fallback paragraph was always
    # appended at the very end of the whole "text", so a missing item for
    # an earlier channel landed *after* a later channel's own paragraph
    # once that one had already been narrated -- breaking the "grouped by
    # channel" contract (_FORMAT_GUARD/_DETAILED_FORMAT_GUARD) instead of
    # just restating the missing item. It should land right after its own
    # channel's last existing paragraph instead.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F1:** first item.\n\n**frontend -- G1:** unrelated fix.",
        items=[
            {
                "channel": "backend",
                "label": "F1",
                "summary": "first item",
                "instruction": "do the first thing",
            },
            {
                "channel": "backend",
                "label": "F2",
                "summary": "second item",
                "instruction": "do the second thing",
            },
            {
                "channel": "frontend",
                "label": "G1",
                "summary": "unrelated fix",
                "instruction": "do the other thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG, detail="detailed")

    assert result.text == (
        "**backend -- F1:** first item.\n\n"
        "**backend -- F2:** second item\n\n"
        "**frontend -- G1:** unrelated fix."
    )
