import pytest

from swingbird.recap import RecapItem
from swingbird.recap_actions import (
    RecapActionError,
    RecapActionStore,
    resolve_reference,
)

ITEM_F4 = RecapItem(
    channel="dripbird",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
)
ITEM_F5 = RecapItem(
    channel="dripbird",
    label="F5",
    summary="undefined-sentinel cloneDeep split",
    instruction="Design a fix for the cloneDeep split.",
)
ITEM_BACKEND = RecapItem(
    channel="backend",
    label="login bug",
    summary="login times out",
    instruction="Fix the login timeout bug.",
)


def test_store_returns_none_for_unknown_thread():
    store = RecapActionStore()

    assert store.get("thread-1") is None


def test_store_set_and_get_roundtrip():
    store = RecapActionStore()
    items = (ITEM_F4, ITEM_F5)

    store.set("thread-1", items)

    assert store.get("thread-1") == items


def test_store_set_replaces_previous_items_for_the_same_thread():
    store = RecapActionStore()
    store.set("thread-1", (ITEM_F4,))

    store.set("thread-1", (ITEM_BACKEND,))

    assert store.get("thread-1") == (ITEM_BACKEND,)


def test_resolve_reference_by_label():
    matches = resolve_reference((ITEM_F4, ITEM_F5), "F4")

    assert matches == [ITEM_F4]


def test_resolve_reference_by_label_is_case_insensitive():
    matches = resolve_reference((ITEM_F4,), "f4")

    assert matches == [ITEM_F4]


def test_resolve_reference_by_channel_when_it_uniquely_matches():
    matches = resolve_reference((ITEM_F4, ITEM_BACKEND), "backend")

    assert matches == [ITEM_BACKEND]


def test_resolve_reference_empty_string_means_all():
    matches = resolve_reference((ITEM_F4, ITEM_F5), "")

    assert matches == [ITEM_F4, ITEM_F5]


@pytest.mark.parametrize("reference", ["all", "ALL", "everything", "  all  "])
def test_resolve_reference_all_markers_mean_all(reference):
    matches = resolve_reference((ITEM_F4, ITEM_F5), reference)

    assert matches == [ITEM_F4, ITEM_F5]


def test_resolve_reference_all_with_no_items_raises():
    with pytest.raises(RecapActionError, match="no items in the last recap"):
        resolve_reference((), "all")


def test_resolve_reference_no_match_raises():
    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((ITEM_F4, ITEM_F5), "F9")


def test_resolve_reference_returns_every_item_that_matches():
    # Both items are in the "dripbird" channel, so a channel-name reference
    # matches both -- returned as-is; it's the caller's call whether more
    # than one match is acceptable (see module docstring).
    matches = resolve_reference((ITEM_F4, ITEM_F5), "dripbird")

    assert matches == [ITEM_F4, ITEM_F5]
