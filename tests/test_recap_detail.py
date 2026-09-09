from swingbird.recap import RecapItem
from swingbird.recap_detail import elaborate

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


def test_elaborate_returns_the_llm_completion():
    llm = FakeLLM(text_response="Here's more on F4: it's blocked on a design call.")

    result = elaborate(llm, ITEM, [], [], "F4")

    assert result == "Here's more on F4: it's blocked on a design call."


def _user_content(item=ITEM, thread_messages=(), dm_messages=(), reference="F4"):
    llm = FakeLLM()
    elaborate(llm, item, list(thread_messages), list(dm_messages), reference)
    return llm.calls[0][1]["content"]


def test_elaborate_includes_summary_instruction_and_reference():
    content = _user_content()
    assert ITEM.summary in content
    assert ITEM.instruction in content
    assert "F4" in content


def test_elaborate_includes_thread_messages_when_present():
    content = _user_content(
        thread_messages=[{"created_at": 1, "content": "the original message"}]
    )
    assert "Full thread this was grounded in" in content
    assert "the original message" in content


def test_elaborate_omits_thread_section_when_absent():
    content = _user_content()
    assert "Full thread" not in content


def test_elaborate_includes_dm_messages_when_present():
    content = _user_content(
        dm_messages=[{"created_at": 2, "content": "the recap conversation so far"}]
    )
    assert "DM conversation so far" in content
    assert "the recap conversation so far" in content


def test_elaborate_omits_dm_section_when_absent():
    content = _user_content()
    assert "DM conversation" not in content


def test_elaborate_defaults_reference_to_all():
    llm = FakeLLM()

    elaborate(llm, ITEM, [], [], None)

    content = llm.calls[0][1]["content"]
    assert "User's reference: all" in content
