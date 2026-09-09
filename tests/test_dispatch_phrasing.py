from swingbird.dispatch_phrasing import rephrase_for_dispatch
from swingbird.recap import RecapItem

ITEM = RecapItem(
    channel="dripbird",
    label="F4",
    summary="unused-ignore propagation",
    instruction=(
        "Decide which issue to fix first -- F4 (trip-check), F5 (design "
        "call), F6 (lower priority); recommend starting with F4."
    ),
)


class FakeLLM:
    def __init__(self, text_response=""):
        self._text_response = text_response
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self._text_response


def test_rephrase_for_dispatch_returns_the_llm_completion():
    llm = FakeLLM(text_response="Go ahead and implement F4: add the trip-check.")

    result = rephrase_for_dispatch(llm, ITEM, "F4")

    assert result == "Go ahead and implement F4: add the trip-check."


def _run_rephrase_and_get_user_content(item, reference="F4"):
    llm = FakeLLM()
    rephrase_for_dispatch(llm, item, reference)
    user_content = llm.calls[0][1]["content"]
    return user_content


def test_rephrase_for_dispatch_includes_summary_instruction_and_reference():
    user_content = _run_rephrase_and_get_user_content(ITEM)
    assert ITEM.summary in user_content
    assert ITEM.instruction in user_content
    assert "F4" in user_content


def test_rephrase_for_dispatch_includes_source_content_when_present():
    grounded_item = ITEM.__class__(
        channel=ITEM.channel,
        label=ITEM.label,
        summary=ITEM.summary,
        instruction=ITEM.instruction,
        source_content="the actual message the recommendation came from",
    )
    user_content = _run_rephrase_and_get_user_content(grounded_item)
    assert "the actual message the recommendation came from" in user_content


def test_rephrase_for_dispatch_omits_source_content_when_absent():
    user_content = _run_rephrase_and_get_user_content(ITEM)
    assert "Original message" not in user_content


def test_rephrase_for_dispatch_defaults_reference_to_all():
    llm = FakeLLM()

    rephrase_for_dispatch(llm, ITEM, None)

    user_content = llm.calls[0][1]["content"]
    assert "User's reference: all" in user_content
