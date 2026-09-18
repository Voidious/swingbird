from swingbird import recap_transcript
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, FRESH, _llm


def _transcript(fake) -> str:
    return fake.chat.completions.calls[0]["messages"][1]["content"]


def _build_recap_and_assert_message_tags():
    llm, fake = _llm()
    build_recap(llm, CONFIG)
    transcript = _transcript(fake)
    assert "[m1] " in transcript
    assert "[m2] " in transcript
    return transcript


def test_build_recap_tags_transcript_messages_with_a_local_id(monkeypatch):
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
    _build_recap_and_assert_message_tags()


def test_build_recap_numbers_tags_globally_not_per_channel(monkeypatch):
    monkeypatch.setattr(
        recap_transcript,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [{"created_at": FRESH, "content": "backend msg", "id": "evt-a"}]
            if channel_id == "chan-1"
            else [{"created_at": FRESH, "content": "frontend msg", "id": "evt-b"}]
        ),
    )
    transcript = _build_recap_and_assert_message_tags()
    # Frontend's own message must not be re-numbered "m1" just because
    # backend's section (listed first) already used it -- see
    # `_build_transcript`'s own docstring for why a repeated tag across
    # channels is exactly the bug this guards against.
    assert transcript.count("[m1] ") == 1


def test_build_recap_does_not_resolve_a_tag_against_the_wrong_channel(monkeypatch):
    monkeypatch.setattr(
        recap_transcript,
        "fetch_messages_since",
        lambda channel_id, since_ts, max_messages=None: (
            [{"created_at": FRESH, "content": "backend msg", "id": "evt-backend"}]
            if channel_id == "chan-1"
            else [
                {"created_at": FRESH, "content": "frontend msg", "id": "evt-frontend"}
            ]
        ),
    )
    # Backend's own message is "m1"; frontend's is "m2" (global numbering).
    # An item that claims channel "backend" but cites frontend's "m2" is
    # exactly the extraction failure `_build_transcript` documents -- it
    # must resolve to no source at all, never to backend's own "m1" or to
    # frontend's real "evt-frontend" mislabeled as backend's.
    llm, _ = _llm(
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "s",
                "instruction": "do it",
                "source_id": "m2",
            }
        ]
    )

    result = build_recap(llm, CONFIG)

    assert result.items[0].source_event_id is None
