import dataclasses

from swingbird.config import ChannelConfig, Config, LLMConfig, OwnerConfig, RelayConfig
from swingbird.recap import RecapItem
from swingbird.recap_detail import elaborate

CONFIG = Config(
    llm=LLMConfig(base_url="https://x", model="m", api_key_env="X_KEY"),
    relay=RelayConfig(url="wss://relay.example", private_key_env="SWINGBIRD_KEY"),
    channels=(
        ChannelConfig(id="chan-1", name="dripbird", write=True, agents=("Codex",)),
    ),
    owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
)

ITEM = RecapItem(
    channel="dripbird",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
)
OTHER_ITEM = RecapItem(
    channel="dripbird",
    label="F5",
    summary="undefined-sentinel cloneDeep split",
    instruction="Design a fix for the cloneDeep split.",
    is_primary=False,
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

    result = elaborate(llm, [ITEM], [[]], [], "F4", CONFIG)

    assert result == "Here's more on F4: it's blocked on a design call."


def _user_content(
    items=(ITEM,), threads=None, dm_messages=(), reference="F4", no_other_items=False
):
    llm = FakeLLM()
    elaborate(
        llm,
        list(items),
        threads if threads is not None else [[] for _ in items],
        list(dm_messages),
        reference,
        CONFIG,
        no_other_items,
    )
    return llm.calls[0][1]["content"]


def test_elaborate_includes_summary_instruction_and_reference():
    content = _user_content()
    assert ITEM.summary in content
    assert ITEM.instruction in content
    assert "F4" in content


def test_elaborate_gives_the_llm_each_items_exact_label():
    # Regression: the LLM never saw an item's stored label, only its
    # summary/instruction, so it sometimes invented its own paraphrased
    # heading (e.g. "Integrate OpenAI support" for a stored label of
    # "OpenAI support"). _backfill_missing_paragraphs then found no
    # paragraph matching the real label and appended a duplicate, so the
    # same item appeared twice under two different headings.
    content = _user_content(items=(ITEM, OTHER_ITEM), threads=[[], []])

    assert f"({ITEM.label})" in content
    assert f"({OTHER_ITEM.label})" in content


def test_elaborate_includes_thread_messages_when_present():
    content = _user_content(
        threads=[[{"created_at": 1, "content": "the original message"}]]
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

    elaborate(llm, [ITEM], [[]], [], None, CONFIG)

    content = llm.calls[0][1]["content"]
    assert "User's reference: all" in content


def test_elaborate_single_item_omits_item_numbering():
    content = _user_content()
    assert "Item 1/1" not in content


def test_elaborate_multiple_items_numbers_and_includes_both():
    content = _user_content(items=(ITEM, OTHER_ITEM), threads=[[], []])

    assert "Item 1/2" in content
    assert "Item 2/2" in content
    assert ITEM.summary in content
    assert OTHER_ITEM.summary in content


def test_elaborate_no_other_items_tells_the_llm_there_is_nothing_else():
    content = _user_content(no_other_items=True)

    assert "no items beyond the one below" in content


def test_elaborate_omits_no_other_items_note_by_default():
    content = _user_content()

    assert "no items beyond" not in content


def test_elaborate_appends_source_link_to_the_grounded_items_paragraph():
    grounded = dataclasses.replace(ITEM, source_event_id="evt-a")
    llm = FakeLLM(text_response="**dripbird -- F4:** it's blocked on a design call.")

    result = elaborate(llm, [grounded], [[]], [], "F4", CONFIG)

    assert result == (
        "**dripbird -- F4:** it's blocked on a design call.\n"
        "buzz://message?channel=chan-1&id=evt-a"
    )


def test_elaborate_omits_source_link_when_item_not_grounded():
    llm = FakeLLM(text_response="**dripbird -- F4:** it's blocked on a design call.")

    result = elaborate(llm, [ITEM], [[]], [], "F4", CONFIG)

    assert result == "**dripbird -- F4:** it's blocked on a design call."


def test_elaborate_omits_source_link_for_an_unmapped_channel():
    grounded = dataclasses.replace(
        ITEM, channel="ghost-channel", source_event_id="evt-a"
    )
    llm = FakeLLM(text_response="**ghost-channel -- F4:** it's blocked.")

    result = elaborate(llm, [grounded], [[]], [], "F4", CONFIG)

    assert result == "**ghost-channel -- F4:** it's blocked."


def test_elaborate_backfills_every_item_the_llm_collapsed_into_one_answer():
    # Live bug: given multiple items, the LLM sometimes ignores
    # _FORMAT_GUARD and folds them into one combined answer instead of
    # giving each its own paragraph -- silently dropping every item but
    # the one it narrated from what the user sees.
    llm = FakeLLM(text_response="Both are still blocked on the same design call.")

    result = elaborate(llm, [ITEM, OTHER_ITEM], [[], []], [], "all", CONFIG)

    assert result == (
        "Both are still blocked on the same design call.\n\n"
        "**dripbird -- F4:** unused-ignore propagation "
        "Next: Fix the deterministic directive trip-check.\n\n"
        "**dripbird -- F5:** undefined-sentinel cloneDeep split "
        "Next: Design a fix for the cloneDeep split."
    )


def test_elaborate_backfills_only_the_item_missing_its_own_paragraph():
    # One item got its own paragraph, the other didn't -- only the missing
    # one should be backfilled, not both.
    llm = FakeLLM(text_response="**dripbird -- F4:** it's blocked on a design call.")

    result = elaborate(llm, [ITEM, OTHER_ITEM], [[], []], [], "all", CONFIG)

    assert result == (
        "**dripbird -- F4:** it's blocked on a design call.\n\n"
        "**dripbird -- F5:** undefined-sentinel cloneDeep split "
        "Next: Design a fix for the cloneDeep split."
    )


def test_elaborate_does_not_backfill_when_every_item_has_its_own_paragraph():
    text_response = (
        "**dripbird -- F4:** it's blocked.\n\n"
        "**dripbird -- F5:** it's still being designed."
    )
    llm = FakeLLM(text_response=text_response)

    result = elaborate(llm, [ITEM, OTHER_ITEM], [[], []], [], "all", CONFIG)

    assert result == text_response


def test_elaborate_does_not_backfill_a_single_item_response():
    # The backfill only exists to catch a multi-item collapse -- a single
    # item that ignores the format guard is a pre-existing, separate
    # concern (its link just won't attach, same as always).
    llm = FakeLLM(text_response="it's blocked on a design call, no bold prefix.")

    result = elaborate(llm, [ITEM], [[]], [], "F4", CONFIG)

    assert result == "it's blocked on a design call, no bold prefix."
