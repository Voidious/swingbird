import json

from swingbird.llm_mock import MockLLMClient


class FakeLLMClient:
    def __init__(self):
        self.complete_calls: list[list[dict[str, str]]] = []
        self.complete_json_calls: list[list[dict[str, str]]] = []
        self._complete_responses = iter(["first reply", "second reply"])
        self._complete_json_responses = iter(
            [{"intent": "chit_chat"}, {"intent": "recap"}]
        )

    def complete(self, messages):
        self.complete_calls.append(messages)
        return next(self._complete_responses)

    def complete_json(self, messages):
        self.complete_json_calls.append(messages)
        return next(self._complete_json_responses)


def test_complete_calls_real_client_on_first_prompt(tmp_path):
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, tmp_path / "cache.json")
    messages = [{"role": "user", "content": "hi"}]

    result = mock.complete(messages)

    assert result == "first reply"
    assert len(fake.complete_calls) == 1


def test_complete_replays_cached_response_for_identical_prompt(tmp_path):
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, tmp_path / "cache.json")
    messages = [{"role": "user", "content": "hi"}]

    first = mock.complete(messages)
    second = mock.complete(messages)

    assert first == second == "first reply"
    assert len(fake.complete_calls) == 1


def test_complete_calls_real_client_again_for_a_new_prompt(tmp_path):
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, tmp_path / "cache.json")

    first = mock.complete([{"role": "user", "content": "hi"}])
    second = mock.complete([{"role": "user", "content": "bye"}])

    assert first == "first reply"
    assert second == "second reply"
    assert len(fake.complete_calls) == 2


def test_complete_json_replays_cached_response(tmp_path):
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, tmp_path / "cache.json")
    messages = [{"role": "user", "content": "recap please"}]

    first = mock.complete_json(messages)
    second = mock.complete_json(messages)

    assert first == second == {"intent": "chit_chat"}
    assert len(fake.complete_json_calls) == 1


def test_complete_and_complete_json_do_not_share_a_cache_entry(tmp_path):
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, tmp_path / "cache.json")
    messages = [{"role": "user", "content": "same content"}]

    text_result = mock.complete(messages)
    json_result = mock.complete_json(messages)

    assert text_result == "first reply"
    assert json_result == {"intent": "chit_chat"}
    assert len(fake.complete_calls) == 1
    assert len(fake.complete_json_calls) == 1


def test_cache_persists_to_disk_and_is_reloaded(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeLLMClient()
    messages = [{"role": "user", "content": "hi"}]
    MockLLMClient(fake, cache_path).complete(messages)

    assert cache_path.is_file()

    second_fake = FakeLLMClient()
    reloaded = MockLLMClient(second_fake, cache_path)
    result = reloaded.complete(messages)

    assert result == "first reply"
    assert len(second_fake.complete_calls) == 0


def _create_mock_llm_client(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, cache_path)
    return cache_path, fake, mock


def test_cache_key_is_stable_regardless_of_dict_key_order(tmp_path):
    (_, fake, mock) = _create_mock_llm_client(tmp_path)

    mock.complete([{"role": "user", "content": "hi"}])
    result = mock.complete([{"content": "hi", "role": "user"}])

    assert result == "first reply"
    assert len(fake.complete_calls) == 1


def test_cache_key_normalizes_case_and_punctuation_in_user_messages(tmp_path):
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, tmp_path / "cache.json")

    mock.complete([{"role": "user", "content": "Recap."}])
    result = mock.complete([{"role": "user", "content": "recap"}])

    assert result == "first reply"
    assert len(fake.complete_calls) == 1


def test_cache_key_does_not_normalize_system_messages(tmp_path):
    fake = FakeLLMClient()
    mock = MockLLMClient(fake, tmp_path / "cache.json")

    system_message = {"role": "system", "content": "Be Concise."}
    other_system_message = {"role": "system", "content": "be concise"}
    user_message = {"role": "user", "content": "hi"}

    mock.complete([system_message, user_message])
    result = mock.complete([other_system_message, user_message])

    assert result == "second reply"
    assert len(fake.complete_calls) == 2


def test_cache_file_contents_are_json_serializable(tmp_path):
    (cache_path, _, mock) = _create_mock_llm_client(tmp_path)

    mock.complete_json([{"role": "user", "content": "recap please"}])

    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
