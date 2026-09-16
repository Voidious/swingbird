import json

from swingbird import recap_transcript
from swingbird.llm import LLMClient
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, LLM_CONFIG, FakeOpenAI


def _raw_llm(content: str) -> tuple[LLMClient, FakeOpenAI]:
    """For responses that don't match the normal {"text": ...} shape."""
    fake = FakeOpenAI(content)
    return LLMClient(LLM_CONFIG, client=fake), fake


def test_build_recap_reconstructs_text_when_only_text_is_missing(monkeypatch):
    # Live bug: "items" came back fully grounded but the "text" key was
    # missing from the response entirely (not malformed JSON, not an empty
    # string -- genuinely absent), which used to fail the whole recap even
    # though there was enough already-grounded content to build one from.
    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _raw_llm(
        json.dumps(
            {
                "items": [
                    {
                        "channel": "backend",
                        "label": "F4",
                        "summary": "login fix",
                        "instruction": "ship it",
                    }
                ]
            }
        )
    )

    result = build_recap(llm, CONFIG)

    assert result.text == "**backend**: login fix"
    assert result.items[0].label == "F4"
