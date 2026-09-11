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
    resolved = resolve_reference((ITEM_F4, ITEM_F5), "F4")

    assert resolved.items == [ITEM_F4]
    assert resolved.degraded is False


def test_resolve_reference_by_label_is_case_insensitive():
    resolved = resolve_reference((ITEM_F4,), "f4")

    assert resolved.items == [ITEM_F4]


def test_resolve_reference_by_channel_when_it_uniquely_matches():
    resolved = resolve_reference((ITEM_F4, ITEM_BACKEND), "backend")

    assert resolved.items == [ITEM_BACKEND]


def test_resolve_reference_empty_string_means_all():
    resolved = resolve_reference((ITEM_F4, ITEM_F5), "")

    assert resolved.items == [ITEM_F4, ITEM_F5]
    assert resolved.degraded is False


@pytest.mark.parametrize("reference", ["all", "ALL", "everything", "  all  "])
def test_resolve_reference_all_markers_mean_all(reference):
    resolved = resolve_reference((ITEM_F4, ITEM_F5), reference)

    assert resolved.items == [ITEM_F4, ITEM_F5]


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
    resolved = resolve_reference((ITEM_F4, ITEM_F5), "dripbird")

    assert resolved.items == [ITEM_F4, ITEM_F5]


def test_resolve_reference_by_label_qualified_with_its_channel():
    # "F4 for dripbird" is longer than either "F4" or "dripbird" alone, so
    # it can't be a substring of either -- the channel narrows the
    # candidates first, then the label matches within them.
    resolved = resolve_reference((ITEM_F4, ITEM_F5, ITEM_BACKEND), "F4 for dripbird")

    assert resolved.items == [ITEM_F4]


def test_resolve_reference_by_label_qualified_with_a_channel_that_recurs():
    other_channel_f4 = RecapItem(
        channel="crispen",
        label="F4",
        summary="a different F4",
        instruction="do the crispen thing",
    )

    resolved = resolve_reference((ITEM_F4, other_channel_f4), "F4 for crispen")

    assert resolved.items == [other_channel_f4]


def test_resolve_reference_channel_qualifier_that_matches_no_label_raises():
    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((ITEM_F4, ITEM_F5), "F9 for dripbird")


def test_resolve_reference_by_word_within_a_bundled_multi_option_label():
    # A recap item covering a multi-way decision (e.g. "F4 or F5 or F6, pick
    # one") gets a compound label -- neither the whole label nor the whole
    # channel-qualified reference is a substring of the other, so only a
    # word-level fallback resolves "F4 for dripbird" against it.
    bundled = RecapItem(
        channel="dripbird",
        label="F4/F5/F6",
        summary="three issues found",
        instruction="pick one to fix first",
    )

    resolved = resolve_reference((bundled, ITEM_BACKEND), "F4 for dripbird")

    assert resolved.items == [bundled]


def test_resolve_reference_word_fallback_ignores_stopwords():
    # "for" and "dripbird" are both words in the reference, but only "F4"
    # should be able to identify the item -- a stray label containing a
    # common connector word must not spuriously match via "for" alone.
    decoy = RecapItem(
        channel="crispen",
        label="wait for CI",
        summary="s",
        instruction="i",
    )

    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((decoy,), "F4 for dripbird")


def test_resolve_reference_does_not_crash_on_an_item_with_an_empty_label():
    # An empty label must never reverse-match as a substring of everything
    # -- otherwise every reference would spuriously match an unlabeled item.
    # Two dripbird items here (rather than one) so the single-candidate
    # channel fallback below doesn't mask this: it only kicks in when
    # channel-narrowing leaves exactly one item.
    unlabeled = RecapItem(
        channel="dripbird",
        label="",
        summary="s",
        instruction="i",
    )

    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((unlabeled, ITEM_F5), "F4 for dripbird")


def test_resolve_reference_by_explicit_channel_narrows_even_without_label_words():
    # `channel` (the router's own `Intent.channel` classification) narrows
    # candidates even when the reference text itself doesn't name the
    # channel or the item's label at all -- "the open items" never mentions
    # "dripbird" or "F4", so only the explicit channel makes this resolvable.
    # "open" is itself a plural-intent word, and dripbird has no non-primary
    # item here, so this also exercises the degraded single-item fallback.
    resolved = resolve_reference(
        (ITEM_F4, ITEM_BACKEND), "the open items", channel="dripbird"
    )

    assert resolved.items == [ITEM_F4]
    assert resolved.degraded is True


def test_resolve_reference_explicit_channel_takes_priority_over_message_text():
    # A reference that happens to name a different channel in its own text
    # must not override the router's authoritative `channel` classification.
    resolved = resolve_reference(
        (ITEM_F4, ITEM_BACKEND),
        "the backend issue",
        channel="dripbird",
    )

    assert resolved.items == [ITEM_F4]
    assert resolved.degraded is False


def test_resolve_reference_generic_reference_resolves_to_the_channels_one_item():
    # A generic reference ("the first one for dripbird") never names any
    # label, but channel-narrowing (derived from the reference text itself
    # here, no explicit `channel` needed) leaves exactly one dripbird item,
    # so it resolves rather than raising -- no plural-intent wording here,
    # so this is *not* the degraded fallback (see the "open items" test
    # above for that case).
    resolved = resolve_reference((ITEM_F4, ITEM_BACKEND), "the first one for dripbird")

    assert resolved.items == [ITEM_F4]
    assert resolved.degraded is False


def test_resolve_reference_generic_reference_without_a_channel_still_raises():
    # No channel signal at all (neither explicit `channel` nor a channel
    # name in the reference text) -- a generic reference must not guess
    # across channels just because there happen to be few items.
    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((ITEM_F4, ITEM_BACKEND), "the first one")


def test_resolve_reference_generic_reference_with_ambiguous_channel_still_raises():
    # Two items share the "dripbird" channel here, and both are (by
    # fixture default) marked `is_primary=True` -- realistically only one
    # item per channel is ever primary (see `recap.py`'s `_parse_recap`),
    # but this exercises that the primary-item fallback below refuses to
    # guess between multiple primaries rather than assuming the first one.
    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((ITEM_F4, ITEM_F5), "the first one for dripbird")


def test_resolve_reference_generic_reference_resolves_to_the_primary_item():
    # A generic, non-plural reference to a channel that's already
    # unambiguous should default to that channel's primary item -- the one
    # "text" itself narrated -- even when other (non-primary) items exist
    # for it, rather than only working when it's the channel's sole item
    # (see the single-item case above).
    non_primary_f5 = RecapItem(
        channel="dripbird",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )

    resolved = resolve_reference(
        (ITEM_F4, non_primary_f5, ITEM_BACKEND),
        "tell me more about dripbird",
        channel="dripbird",
    )

    assert resolved.items == [ITEM_F4]
    assert resolved.degraded is False


def test_resolve_reference_plural_intent_returns_every_non_primary_item():
    # ITEM_F4 is the (default) primary item for dripbird; a non-primary F5
    # is what "the additional items" should resolve to -- not the primary
    # one "text" already narrated.
    non_primary_f5 = RecapItem(
        channel="dripbird",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )

    resolved = resolve_reference(
        (ITEM_F4, non_primary_f5), "the additional items", channel="dripbird"
    )

    assert resolved.items == [non_primary_f5]
    assert resolved.degraded is False


def test_resolve_reference_plural_intent_without_a_channel_signal_still_raises():
    # "other(s)"/"additional"/etc only resolves once the channel is
    # unambiguous -- same "no guessing across channels" rule as the generic
    # single-item fallback.
    non_primary_f5 = RecapItem(
        channel="dripbird",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )

    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((ITEM_F4, non_primary_f5, ITEM_BACKEND), "the other items")


def test_resolve_reference_matches_by_keyword():
    item_with_keywords = RecapItem(
        channel="dripbird",
        label="F6",
        summary="lint residue",
        instruction="fix the prefer-const finding",
        keywords=("lint issue", "prefer-const finding"),
    )

    resolved = resolve_reference(
        (item_with_keywords, ITEM_BACKEND), "the lint issue for dripbird"
    )

    assert resolved.items == [item_with_keywords]
    assert resolved.degraded is False


def test_resolve_reference_matches_by_keyword_word_fallback():
    # Same word-level fallback label matching gets ("F4 for dripbird" isn't a
    # substring of a longer keyword and vice versa) applies to keywords too.
    item_with_keywords = RecapItem(
        channel="dripbird",
        label="F6",
        summary="lint residue",
        instruction="fix the prefer-const finding",
        keywords=("the prefer-const lint finding",),
    )

    resolved = resolve_reference(
        (item_with_keywords, ITEM_BACKEND), "prefer-const for dripbird"
    )

    assert resolved.items == [item_with_keywords]


def test_resolve_reference_does_not_crash_on_an_item_with_no_keywords():
    resolved = resolve_reference((ITEM_F4, ITEM_BACKEND), "F4")

    assert resolved.items == [ITEM_F4]


def test_resolve_reference_plural_intent_with_no_non_primary_items_falls_back():
    # A generic plural reference for a channel that only ever had one open
    # item has nothing non-primary to return -- falls through to the same
    # single-item fallback a non-plural generic reference would use, but
    # flagged `degraded=True` since "other items" didn't actually resolve to
    # anything beyond the one item already in the recap.
    resolved = resolve_reference(
        (ITEM_F4,), "what are the other items", channel="dripbird"
    )

    assert resolved.items == [ITEM_F4]
    assert resolved.degraded is True
