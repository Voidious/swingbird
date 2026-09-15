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


def test_store_channel_for_defaults_to_none():
    store = RecapActionStore()
    store.set("thread-1", (ITEM_F4,))

    assert store.channel_for("thread-1") is None


def test_store_channel_for_returns_the_scoped_channel():
    store = RecapActionStore()
    store.set("thread-1", (ITEM_F4,), channel="dripbird")

    assert store.channel_for("thread-1") == "dripbird"


def test_store_channel_for_unknown_thread_returns_none():
    store = RecapActionStore()

    assert store.channel_for("thread-1") is None


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


def test_resolve_reference_empty_string_with_explicit_channel_narrows_to_it():
    # Live bug: the router can reduce "what are the additional open items
    # for swingbird?" to an empty leftover reference plus an explicit
    # `channel="swingbird"` classification. An empty/"all" reference must
    # still be scoped to that channel, not returned as every item across
    # every channel in the thread's whole store (which would leak other
    # channels' items into the answer after a global, all-channels recap).
    resolved = resolve_reference(
        (ITEM_F4, ITEM_F5, ITEM_BACKEND), "", channel="dripbird"
    )

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


def test_resolve_reference_exact_label_match_is_not_swallowed_by_a_longer_fuzzy_match():
    # Live bug: giving the exact, full label of one item still came back
    # ambiguous. The old fuzzy pass is bidirectional -- it also matches when
    # a *candidate's* label is a substring of the reference -- so a full
    # title that happens to contain a second item's short label as a
    # fragment (here, "F4") matched both items instead of recognizing the
    # first was named exactly.
    full_title = RecapItem(
        channel="dripbird",
        label="close the duplicate extractor F4 login fix",
        summary="s",
        instruction="i",
    )
    short_label = RecapItem(
        channel="dripbird",
        label="F4",
        summary="s",
        instruction="i",
        is_primary=False,
    )

    resolved = resolve_reference(
        (full_title, short_label), "close the duplicate extractor F4 login fix"
    )

    assert resolved.items == [full_title]
    assert resolved.degraded is False


def test_resolve_reference_exact_label_match_is_case_insensitive():
    other = RecapItem(
        channel="dripbird", label="F4 login fix", summary="s", instruction="i"
    )

    resolved = resolve_reference((ITEM_F4, other), "F4 LOGIN FIX")

    assert resolved.items == [other]
    assert resolved.degraded is False


def test_resolve_reference_multiple_exact_label_matches_still_fall_through():
    # Two different items can legitimately share the exact same label text
    # (e.g. the same label recurring in two channels with no channel
    # narrowing available) -- that's genuine ambiguity by identity, not
    # something the exact-match shortcut can resolve, so it must fall
    # through to the ordinary fuzzy pass (which, for a bare shared label
    # with no channel signal, returns both as multiple matches).
    other_channel_f4 = RecapItem(
        channel="crispen", label="F4", summary="a different F4", instruction="i"
    )

    resolved = resolve_reference((ITEM_F4, other_channel_f4), "F4")

    assert resolved.items == [ITEM_F4, other_channel_f4]


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


def test_resolve_reference_word_fallback_requires_a_word_boundary_match():
    # Live bug: "what are the other swingbird items?" hijacked a completely
    # unrelated item whose keyword was "bare confirm" -- "are" (from "what
    # are") isn't a stopword, and a plain substring check finds it inside
    # "bare". Requiring a word-boundary match prevents this false positive
    # while still falling through to the plural-intent "every non-primary
    # item" fallback the reference actually asked for.
    decoy = RecapItem(
        channel="swingbird",
        label="no-source-ID confirmation fallback",
        summary="s",
        instruction="i",
        keywords=("bare confirm", "pending dispatch"),
    )
    other = RecapItem(
        channel="swingbird",
        label="message-ID reliability",
        summary="s",
        instruction="i",
        is_primary=False,
    )

    resolved = resolve_reference(
        (decoy, other), "what are the other swingbird items?", channel="swingbird"
    )

    assert resolved.items == [other]
    assert resolved.degraded is False


def test_resolve_reference_word_fallback_still_matches_a_bundled_label_at_the_edge():
    # The word-boundary requirement must not lose the case it exists for:
    # "F4" bounded by "/" on both sides in a compound label is still a
    # legitimate whole-word match.
    bundled = RecapItem(
        channel="dripbird",
        label="F4/F5/F6",
        summary="three issues found",
        instruction="pick one to fix first",
    )

    resolved = resolve_reference((bundled, ITEM_BACKEND), "F6 for dripbird")

    assert resolved.items == [bundled]


def test_resolve_reference_word_fallback_ignores_generic_words():
    # Live bug: "the swingbird fix" matched both a disambiguation fix and an
    # async freeze fix, since both labels end in "...fix" and "fix" isn't a
    # stopword. Excluding generic words from the word-level fallback should
    # leave no word match at all, falling through to the channel's primary
    # item instead of returning both.
    primary_fix = RecapItem(
        channel="swingbird",
        label="recap-follow-up disambiguation fix",
        summary="s",
        instruction="i",
    )
    other_fix = RecapItem(
        channel="swingbird",
        label="async event-loop freeze fix",
        summary="s",
        instruction="i",
        is_primary=False,
    )

    resolved = resolve_reference(
        (primary_fix, other_fix), "the swingbird fix", channel="swingbird"
    )

    assert resolved.items == [primary_fix]
    assert resolved.degraded is False


def test_resolve_reference_plural_intent_word_in_a_labels_own_text_does_not_hijack_it():
    # Live bug: a recap said "(5 additional open items)" for swingbird, but
    # every phrasing of "tell me about the other/additional/open items"
    # returned just one item -- the one literally labeled
    # "recap-additional-items-followup". Its own label contains "items"
    # (a _GENERIC_WORDS entry) and, in other phrasings, "additional"/"open"
    # (both _PLURAL_INTENT_WORDS), so the word-level fallback matched it
    # directly and never reached the "give me every non-primary item"
    # fallback below. Excluding plural-intent words from that fallback too
    # (like generic words) should let a genuine "other items" request reach
    # every non-primary item instead of being hijacked by this one.
    primary = RecapItem(channel="swingbird", label="F1", summary="s", instruction="i")
    about_items_feature = RecapItem(
        channel="swingbird",
        label="recap-additional-items-followup",
        summary="s",
        instruction="i",
        is_primary=False,
    )
    another = RecapItem(
        channel="swingbird", label="F3", summary="s", instruction="i", is_primary=False
    )

    for reference in (
        "tell me about the other swingbird items",
        "tell me more about the additional swingbird open items",
        "tell me about the open swingbird items",
    ):
        resolved = resolve_reference(
            (primary, about_items_feature, another), reference, channel="swingbird"
        )
        assert resolved.items == [about_items_feature, another]
        assert resolved.degraded is False


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
    # fixture default) marked `is_primary=True` -- a concise recap only
    # ever marks one item per channel primary, but a detailed recap can
    # genuinely mark several (see `recap.py`'s `_parse_recap`), so this
    # exercises that the primary-item fallback below refuses to guess
    # between multiple primaries rather than assuming the first one.
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
    # single-item fallback. Unlike the single-channel cases above, `items`
    # here genuinely spans two channels (dripbird and backend), so there's
    # real ambiguity left even with no explicit channel signal -- this must
    # still raise rather than guess.
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


def test_resolve_reference_plural_intent_infers_channel_from_single_channel_items():
    # Live bug (2026-09-12): after a project-scoped recap ("recap dripbird"),
    # every stored item already belongs to dripbird -- but "tell me about
    # the additional items" names no channel, and the router's classifier
    # only sees this one message, so it has no way to report `channel`
    # either. Requiring an explicit channel signal before the plural-intent
    # fallback could run meant this raised "no recap item matches" even
    # though there was nothing actually ambiguous: everything on the table
    # was already dripbird's.
    non_primary_f5 = RecapItem(
        channel="dripbird",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )

    resolved = resolve_reference((ITEM_F4, non_primary_f5), "the additional items")

    assert resolved.items == [non_primary_f5]
    assert resolved.degraded is False


def test_resolve_reference_plural_intent_single_channel_degrades_without_non_primary():
    # Same single-channel-items inference, but the channel's only item is
    # the primary one -- "the rest" still resolves (to that one item) rather
    # than raising, same as the explicit-channel case already covered by
    # test_resolve_reference_plural_intent_with_no_non_primary_items_falls_back,
    # just without needing an explicit channel signal to get there.
    resolved = resolve_reference((ITEM_F4,), "what are the other items")

    assert resolved.items == [ITEM_F4]
    assert resolved.degraded is True


def test_resolve_reference_unmatched_label_still_raises_with_single_channel_items():
    # Deliberately narrower than the plural-intent case above: a reference
    # that looks like it's naming a specific item (no plural-intent wording)
    # but doesn't match anything must still raise, even when the store's
    # items happen to span only one channel -- inferring the channel from a
    # single-channel item set is only safe for an explicit "give me the
    # rest," not for guessing that an unmatched label-like reference must
    # have meant the store's one channel.
    with pytest.raises(RecapActionError, match="no recap item matches"):
        resolve_reference((ITEM_F4,), "tell me about F9")


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
