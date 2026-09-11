from swingbird.recap import RecapItem
from swingbird.recap_relay import relay_with_context

ITEM = RecapItem(
    channel="dripbird",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
)


class FakeLLM:
    def __init__(self, text_response=""):
        self._text_response = text_response
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self._text_response


def test_relay_with_context_returns_the_llm_completion():
    llm = FakeLLM(text_response="couldn't we just pre-compile it?")

    result = relay_with_context(llm, ITEM, "F4", "couldn't we just pre-compile it?")

    assert result == "couldn't we just pre-compile it?"


def _run_relay_and_get_user_content(item, reference="F4", message="couldn't we?"):
    llm = FakeLLM()
    relay_with_context(llm, item, reference, message)
    user_content = llm.calls[0][1]["content"]
    return user_content


def test_relay_with_context_includes_summary_instruction_reference_and_message():
    user_content = _run_relay_and_get_user_content(ITEM)
    assert ITEM.summary in user_content
    assert ITEM.instruction in user_content
    assert "F4" in user_content
    assert "couldn't we?" in user_content


def test_relay_with_context_includes_source_content_when_present():
    grounded_item = ITEM.__class__(
        channel=ITEM.channel,
        label=ITEM.label,
        summary=ITEM.summary,
        instruction=ITEM.instruction,
        source_content="the actual message the recommendation came from",
    )
    user_content = _run_relay_and_get_user_content(grounded_item)
    assert "the actual message the recommendation came from" in user_content


def test_relay_with_context_omits_source_content_when_absent():
    user_content = _run_relay_and_get_user_content(ITEM)
    assert "Original message" not in user_content


def test_relay_with_context_defaults_reference_to_the_items_label():
    llm = FakeLLM()

    relay_with_context(llm, ITEM, None, "couldn't we just pre-compile it?")

    user_content = llm.calls[0][1]["content"]
    assert "User's reference to the item: F4" in user_content
