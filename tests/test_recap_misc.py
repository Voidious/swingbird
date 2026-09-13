from swingbird import recap
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, _llm


def test_build_recap_strips_llm_narrated_omitted_note(monkeypatch):
    # Observed live: the LLM narrated "(Additional open items omitted.)" --
    # same countless shape as its "(Additional open items remain.)" cousin
    # (see test_recap.py), but naming the fold itself as an omission instead
    # of describing what's left over. The original regex only accepted a
    # trailing "remain(s/ing)?", so this phrasing sailed through unstripped
    # and stacked next to our own real note.
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**swingbird**: primary item text. (Additional open items omitted.)",
        items=[
            {
                "channel": "swingbird",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "swingbird",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == (
        "**swingbird**: primary item text. (1 additional open item.)"
    )
