from swingbird import recap
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
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
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
