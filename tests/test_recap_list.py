import dataclasses

from swingbird.config import ChannelConfig, Config, LLMConfig, OwnerConfig, RelayConfig
from swingbird.recap import RecapItem
from swingbird.recap_list import render_items

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


def test_render_items_never_calls_the_llm_just_formats_stored_fields():
    text = render_items([ITEM], False, CONFIG)

    assert text == (
        "**dripbird -- F4:** unused-ignore propagation "
        "Fix the deterministic directive trip-check."
    )


def test_render_items_formats_multiple_items_as_separate_paragraphs():
    text = render_items([ITEM, OTHER_ITEM], False, CONFIG)

    assert text == (
        "**dripbird -- F4:** unused-ignore propagation "
        "Fix the deterministic directive trip-check.\n\n"
        "**dripbird -- F5:** undefined-sentinel cloneDeep split "
        "Design a fix for the cloneDeep split."
    )


def test_render_items_no_other_items_prepends_a_plain_note():
    text = render_items([ITEM], True, CONFIG)

    assert text.startswith("There are no other open items for that channel")
    assert "unused-ignore propagation" in text


def test_render_items_omits_the_note_by_default():
    text = render_items([ITEM], False, CONFIG)

    assert "no other open items" not in text


def test_render_items_appends_source_link_to_the_grounded_items_paragraph():
    grounded = dataclasses.replace(ITEM, source_event_id="evt-a")

    text = render_items([grounded], False, CONFIG)

    assert text == (
        "**dripbird -- F4:** unused-ignore propagation "
        "Fix the deterministic directive trip-check.\n"
        "buzz://message?channel=chan-1&id=evt-a"
    )


def test_render_items_omits_source_link_when_item_not_grounded():
    text = render_items([ITEM], False, CONFIG)

    assert "buzz://message" not in text


def test_render_items_omits_source_link_for_an_unmapped_channel():
    grounded = dataclasses.replace(
        ITEM, channel="ghost-channel", source_event_id="evt-a"
    )

    text = render_items([grounded], False, CONFIG)

    assert "buzz://message" not in text
