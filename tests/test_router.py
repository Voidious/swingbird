import json

import pytest

from swingbird.config import ChannelConfig, Config, LLMConfig, RelayConfig
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
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, content):
        self.chat = FakeChat(FakeCompletions(content))


def _router(content: str) -> tuple[IntentRouter, FakeOpenAI]:
    fake = FakeOpenAI(content)
    llm = LLMClient(LLM_CONFIG, client=fake)
    return IntentRouter(llm, CONFIG), fake


def test_route_recap():
    router, _ = _router('{"intent": "recap"}')

    intent = router.route("what's going on?")

    assert intent == Intent(kind="recap")


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


def test_route_dispatch_with_ambiguous_target_leaves_fields_null():
    router, _ = _router(
        json.dumps({"intent": "dispatch", "message": "fix the bug", "channel": None})
    )

    intent = router.route("tell someone to fix the bug")

    assert intent.channel is None
    assert intent.target_agent is None


def test_route_confirm_and_cancel():
    router, _ = _router('{"intent": "confirm"}')
    assert router.route("yes, do it").kind == "confirm"

    router, _ = _router('{"intent": "cancel"}')
    assert router.route("never mind").kind == "cancel"


def test_route_chit_chat():
    router, _ = _router('{"intent": "chit_chat"}')

    assert router.route("how's the weather?").kind == "chit_chat"


def test_route_raises_on_unknown_intent():
    router, _ = _router('{"intent": "not-a-real-intent"}')

    with pytest.raises(RouterError, match="unknown intent"):
        router.route("???")


def test_system_prompt_includes_known_channels_and_agents():
    router, fake = _router('{"intent": "recap"}')

    router.route("what's going on?")

    system_content = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert "backend" in system_content
    assert "Codex" in system_content
    assert "frontend" in system_content
    assert "Goose" in system_content


def test_forces_json_mode():
    router, fake = _router('{"intent": "recap"}')

    router.route("what's going on?")

    assert fake.chat.completions.calls[0]["response_format"] == {"type": "json_object"}
