import pytest

from swingbird.recap import RecapItem
from swingbird.recap_actions import RecapActionError
from swingbird.recap_close_selection import select_items_to_close

F4_ITEM = RecapItem(
    channel="swingbird",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
)
F5_ITEM = RecapItem(
    channel="swingbird",
    label="F5",
    summary="s",
    instruction="i",
    is_primary=False,
    keywords=("lint issue",),
)
DRIPBIRD_ITEM = RecapItem(
    channel="dripbird",
    label="F2",
    summary="s",
    instruction="i",
)


class FakeLLM:
    def __init__(self, json_response=None):
        self._json_response = json_response
        self.calls: list[list[dict]] = []

    def complete_json(self, messages):
        self.calls.append(messages)
        return self._json_response


def test_select_items_to_close_returns_the_matched_indices():
    llm = FakeLLM(json_response={"indices": [1]})

    result = select_items_to_close(llm, (F4_ITEM, F5_ITEM), "F4")

    assert result == (F4_ITEM,)


def test_select_items_to_close_returns_multiple_items_in_index_order():
    llm = FakeLLM(json_response={"indices": [2, 1]})

    result = select_items_to_close(llm, (F4_ITEM, F5_ITEM), "all swingbird items")

    assert result == (F5_ITEM, F4_ITEM)


def test_select_items_to_close_across_channels():
    llm = FakeLLM(json_response={"indices": [1, 3]})

    result = select_items_to_close(
        llm, (F4_ITEM, F5_ITEM, DRIPBIRD_ITEM), "F4 for swingbird and F2 for dripbird"
    )

    assert result == (F4_ITEM, DRIPBIRD_ITEM)


def test_select_items_to_close_dedupes_repeated_indices():
    llm = FakeLLM(json_response={"indices": [1, 1]})

    result = select_items_to_close(llm, (F4_ITEM, F5_ITEM), "F4")

    assert result == (F4_ITEM,)


def test_select_items_to_close_ignores_out_of_range_indices():
    llm = FakeLLM(json_response={"indices": [1, 99, 0]})

    result = select_items_to_close(llm, (F4_ITEM, F5_ITEM), "F4")

    assert result == (F4_ITEM,)


def test_select_items_to_close_ignores_non_integer_entries():
    llm = FakeLLM(json_response={"indices": [1, "2", None]})

    result = select_items_to_close(llm, (F4_ITEM, F5_ITEM), "F4")

    assert result == (F4_ITEM,)


def test_select_items_to_close_raises_on_empty_selection():
    llm = FakeLLM(json_response={"indices": []})

    with pytest.raises(RecapActionError, match="no recap item matches 'F9'"):
        select_items_to_close(llm, (F4_ITEM, F5_ITEM), "F9")


def test_select_items_to_close_raises_when_indices_is_not_a_list():
    llm = FakeLLM(json_response={"indices": "1"})

    with pytest.raises(RecapActionError):
        select_items_to_close(llm, (F4_ITEM, F5_ITEM), "F4")


def test_select_items_to_close_raises_when_indices_key_is_missing():
    llm = FakeLLM(json_response={})

    with pytest.raises(RecapActionError):
        select_items_to_close(llm, (F4_ITEM, F5_ITEM), "F4")


def test_select_items_to_close_includes_item_listing_and_request_in_the_prompt():
    llm = FakeLLM(json_response={"indices": [1]})

    select_items_to_close(llm, (F4_ITEM, F5_ITEM, DRIPBIRD_ITEM), "close F4")

    user_content = llm.calls[0][1]["content"]
    assert "1. [swingbird] PRIMARY F4" in user_content
    assert "2. [swingbird] ADDITIONAL F5 (keywords: lint issue)" in user_content
    assert "3. [dripbird] PRIMARY F2" in user_content
    assert "Request: close F4" in user_content


def test_select_items_to_close_handles_an_unspecified_request():
    llm = FakeLLM(json_response={"indices": [1]})

    select_items_to_close(llm, (F4_ITEM,), None)

    user_content = llm.calls[0][1]["content"]
    assert "Request: (unspecified)" in user_content
