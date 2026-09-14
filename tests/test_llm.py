import httpx2
import openai
import pytest

from swingbird.config import LLMConfig
from swingbird.llm import LLMClient, LLMError

CONFIG = LLMConfig(
    base_url="https://api.moonshot.ai/v1",
    model="kimi-k2.6",
    api_key_env="MOONSHOT_API_KEY",
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
    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeResponse(self._content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    def __init__(self, content=None, error=None):
        self.chat = FakeChat(FakeCompletions(content=content, error=error))


def test_complete_returns_content():
    fake = FakeOpenAI(content="hello there")
    client = LLMClient(CONFIG, client=fake)

    assert client.complete([{"role": "user", "content": "hi"}]) == "hello there"


def test_complete_handles_none_content():
    fake = FakeOpenAI(content=None)
    client = LLMClient(CONFIG, client=fake)

    assert client.complete([{"role": "user", "content": "hi"}]) == ""


def _complete_json_test(fake, message=None, expected=None):
    if message is None:
        message = [{"role": "user", "content": "what's up?"}]
    if expected is None:
        expected = {"intent": "recap"}
    client = LLMClient(CONFIG, client=fake)

    result = client.complete_json(message)

    assert result == expected


def test_complete_json_parses_response():
    fake = FakeOpenAI(content='{"intent": "recap"}')
    _complete_json_test(fake)
    assert fake.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_json_raises_on_invalid_json():
    fake = FakeOpenAI(content="not json")
    client = LLMClient(CONFIG, client=fake)

    with pytest.raises(LLMError, match="did not return valid JSON"):
        client.complete_json([{"role": "user", "content": "what's up?"}])


def test_complete_json_strips_labeled_code_fence():
    fake = FakeOpenAI(content='```json\n{"intent": "recap"}\n```')
    _complete_json_test(fake)


def test_complete_json_strips_unlabeled_code_fence():
    fake = FakeOpenAI(content='```\n{"intent": "recap"}\n```')
    _complete_json_test(fake)


def test_chat_wraps_openai_errors():
    request = httpx2.Request("POST", "https://api.moonshot.ai/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    fake = FakeOpenAI(error=error)
    client = LLMClient(CONFIG, client=fake)

    with pytest.raises(LLMError, match="LLM request failed"):
        client.complete([{"role": "user", "content": "hi"}])


def test_missing_api_key_env_raises(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    with pytest.raises(LLMError, match="MOONSHOT_API_KEY"):
        LLMClient(CONFIG)


def test_builds_real_client_when_api_key_present(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    client = LLMClient(CONFIG)

    assert client._client.api_key == "test-key"
