import json

import pytest

from swingbird.config import (
    ChannelConfig,
    Config,
    LLMConfig,
    OwnerConfig,
    RelayConfig,
)
from swingbird.llm import LLMClient
from swingbird.router import Intent, IntentRouter, RouterError

LLM_CONFIG = LLMConfig(
    base_url="https://api.moonshot.ai/v1",
    model="kimi-k2.6",
    api_key_env="MOONSHOT_API_KEY",
)
RELAY_CONFIG = RelayConfig(url="wss://relay.example", private_key_env="SWINGBIRD_KEY")
CONFIG = Config(
    llm=LLM_CONFIG,
    relay=RELAY_CONFIG,
    channels=(
        ChannelConfig(id="c1", name="backend", write=True, agents=("Codex",)),
        ChannelConfig(id="c2", name="frontend", write=False, agents=("Goose",)),
    ),
    owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self._contents = content if isinstance(content, list) else [content]
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._contents) - 1)
        return FakeResponse(self._contents[index])


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, content):
        self.chat = FakeChat(FakeCompletions(content))


class FakeAuditLog:
    def __init__(self):
        self.transcripts: list[tuple[str | None, str]] = []

    def log_transcript_in(self, thread_id, text):
        self.transcripts.append((thread_id, text))


def _router(content: str) -> tuple[IntentRouter, FakeOpenAI]:
    fake = FakeOpenAI(content)
    llm = LLMClient(LLM_CONFIG, client=fake)
    return IntentRouter(llm, CONFIG), fake


def test_route_recap():
    router, _ = _router('{"intent": "recap"}')

    intent = router.route("what's going on?")

    assert intent == Intent(kind="recap")
    assert intent.detail == "concise"


def test_route_recap_detailed():
    router, _ = _router('{"intent": "recap", "detail": "detailed"}')

    intent = router.route("give me a detailed recap")

    assert intent == Intent(kind="recap", detail="detailed")


def test_route_recap_unrecognized_detail_defaults_to_concise():
    router, _ = _router('{"intent": "recap", "detail": "extremely thorough"}')

    intent = router.route("what's going on?")

    assert intent.detail == "concise"


def test_route_dispatch_with_known_channel_and_agent():
    router, _ = _router(
        json.dumps(
            {
                "intent": "dispatch",
                "channel": "backend",
                "target_agent": "Codex",
                "message": "fix the login timeout bug",
            }
        )
    )

    intent = router.route("tell backend to fix the login timeout bug")

    assert intent == Intent(
        kind="dispatch",
        channel="backend",
        target_agent="Codex",
        message="fix the login timeout bug",
    )


def test_route_dispatch_for_a_question_addressed_to_a_named_channel():
    router, _ = _router(
        json.dumps(
            {
                "intent": "dispatch",
                "channel": "backend",
                "target_agent": "Codex",
                "message": "verify backend is written in Python",
            }
        )
    )

    intent = router.route("verify backend is written in Python")

    assert intent == Intent(
        kind="dispatch",
        channel="backend",
        target_agent="Codex",
        message="verify backend is written in Python",
    )


def test_route_dispatch_with_ambiguous_target_leaves_fields_null():
    router, _ = _router(
        json.dumps({"intent": "dispatch", "message": "fix the bug", "channel": None})
    )

    intent = router.route("tell someone to fix the bug")

    assert intent.channel is None
    assert intent.target_agent is None


def test_route_recap_action_extracts_reference_verbatim():
    router, _ = _router(json.dumps({"intent": "recap_action", "message": "F4"}))

    intent = router.route("go ahead with F4 for dripbird")

    assert intent == Intent(kind="recap_action", message="F4")


def test_route_recap_detail_extracts_reference_verbatim():
    router, _ = _router(
        json.dumps({"intent": "recap_detail", "message": "the duplicate extractor fix"})
    )

    intent = router.route("tell me more about the duplicate extractor fix")

    assert intent == Intent(kind="recap_detail", message="the duplicate extractor fix")


def test_route_recap_relay_extracts_item_reference_and_message_separately():
    router, _ = _router(
        json.dumps(
            {
                "intent": "recap_relay",
                "item_reference": "F4",
                "message": "couldn't we just pre-compile it?",
            }
        )
    )

    intent = router.route("for dripbird F4, couldn't we just pre-compile it?")

    assert intent == Intent(
        kind="recap_relay",
        item_reference="F4",
        message="couldn't we just pre-compile it?",
    )


def test_route_recap_relay_without_item_reference_leaves_it_null():
    router, _ = _router(
        json.dumps({"intent": "recap_relay", "message": "what about caching instead?"})
    )

    intent = router.route("what about caching instead?")

    assert intent.item_reference is None
    assert intent.message == "what about caching instead?"


def test_route_confirm_and_cancel():
    router, _ = _router('{"intent": "confirm"}')
    assert router.route("yes, do it").kind == "confirm"

    router, _ = _router('{"intent": "cancel"}')
    assert router.route("never mind").kind == "cancel"


def test_route_chit_chat():
    router, _ = _router('{"intent": "chit_chat"}')

    assert router.route("how's the weather?").kind == "chit_chat"


def test_route_retries_once_after_chit_chat_and_returns_second_result():
    router, fake = _router(['{"intent": "chit_chat"}', '{"intent": "recap"}'])

    intent = router.route("what's going on with backend?")

    assert intent.kind == "recap"
    assert len(fake.chat.completions.calls) == 2


def test_route_does_not_retry_more_than_once():
    router, fake = _router(['{"intent": "chit_chat"}', '{"intent": "chit_chat"}'])

    intent = router.route("how's the weather?")

    assert intent.kind == "chit_chat"
    assert len(fake.chat.completions.calls) == 2


def test_route_raises_on_unknown_intent():
    router, _ = _router('{"intent": "not-a-real-intent"}')

    with pytest.raises(RouterError, match="unknown intent"):
        router.route("???")


def _route_and_get_system_content():
    router, fake = _router('{"intent": "recap"}')
    router.route("what's going on?")
    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    return router, system_content


def test_route_notes_no_open_recap_by_default():
    (_, system_content) = _route_and_get_system_content()
    assert "no open recap" in system_content


def test_route_notes_open_recap_when_flagged():
    router, fake = _router('{"intent": "recap_detail", "message": "F4"}')

    router.route("tell me more about F4", has_open_recap=True)

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "already has an open recap" in system_content


def test_route_open_recap_note_mentions_recap_relay():
    router, fake = _router('{"intent": "recap_relay", "message": "F4"}')

    router.route(
        "for dripbird F4, couldn't we just pre-compile it?", has_open_recap=True
    )

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "recap_relay" in system_content


def test_route_open_recap_items_listed_when_provided():
    router, fake = _router('{"intent": "recap_relay", "item_reference": "dripbird"}')

    router.route(
        "on the dripbird default model, is Kimi named after anyone specific?",
        has_open_recap=True,
        open_recap_items=(("dripbird", "Kimi 2.6 default model"),),
    )

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "Open recap items:" in system_content
    assert "dripbird: Kimi 2.6 default model" in system_content


def test_route_open_recap_items_omitted_without_open_recap():
    router, fake = _router('{"intent": "dispatch"}')

    router.route(
        "on the dripbird default model, is Kimi named after anyone specific?",
        has_open_recap=False,
        open_recap_items=(("dripbird", "Kimi 2.6 default model"),),
    )

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "Open recap items:" not in system_content


def test_route_open_recap_items_block_omitted_when_empty():
    router, fake = _router('{"intent": "recap_detail", "message": "F4"}')

    router.route("tell me more about F4", has_open_recap=True, open_recap_items=())

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "Open recap items:" not in system_content


def test_route_notes_no_pending_dispatch_by_default():
    (_, system_content) = _route_and_get_system_content()
    assert "no dispatch proposal awaiting confirm" in system_content


def test_route_notes_pending_dispatch_when_flagged():
    router, fake = _router('{"intent": "confirm"}')

    router.route("confirm", has_pending_dispatch=True)

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "has a dispatch proposal awaiting confirm or" in system_content


def test_route_open_recap_and_pending_dispatch_notes_both_appended():
    router, fake = _router('{"intent": "confirm"}')

    router.route("confirm", has_open_recap=True, has_pending_dispatch=True)

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "already has an open recap" in system_content
    assert "has a dispatch proposal awaiting confirm or" in system_content


def test_system_prompt_includes_known_channels_and_agents():
    (_, system_content) = _route_and_get_system_content()
    assert "backend" in system_content
    assert "Codex" in system_content
    assert "frontend" in system_content
    assert "Goose" in system_content


def test_forces_json_mode():
    router, fake = _router('{"intent": "recap"}')

    router.route("what's going on?")

    assert fake.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_route_logs_transcript_in_when_audit_configured():
    fake = FakeOpenAI('{"intent": "recap"}')
    llm = LLMClient(LLM_CONFIG, client=fake)
    audit = FakeAuditLog()
    router = IntentRouter(llm, CONFIG, audit=audit)

    router.route("what's going on?", thread_id="thread-1")

    assert audit.transcripts == [("thread-1", "what's going on?")]


def test_route_without_audit_configured_does_not_raise():
    router, _ = _router('{"intent": "recap"}')

    router.route("what's going on?")
