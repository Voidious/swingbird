from swingbird.recap import RecapItem
from swingbird.recap_disambiguation import (
    AmbiguousRecapReference,
    DisambiguationStore,
    PendingDisambiguation,
    format_choices,
    resolve_choice,
)
from swingbird.router import Intent

ITEM_F4 = RecapItem(
    channel="backend",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
)
ITEM_F5 = RecapItem(
    channel="frontend",
    label="F5",
    summary="undefined-sentinel cloneDeep split",
    instruction="Design a fix for the cloneDeep split.",
)
CANDIDATES = (ITEM_F4, ITEM_F5)


def test_ambiguous_recap_reference_carries_its_candidates():
    exc = AmbiguousRecapReference("which did you mean?", CANDIDATES)

    assert str(exc) == "which did you mean?"
    assert exc.candidates == CANDIDATES


def test_store_returns_none_for_unknown_thread():
    store = DisambiguationStore()

    assert store.get("thread-1") is None


def test_store_set_and_get_roundtrip():
    store = DisambiguationStore()
    pending = PendingDisambiguation(
        "recap_action", CANDIDATES, Intent(kind="recap_action", message="all")
    )

    store.set("thread-1", pending)

    assert store.get("thread-1") is pending


def test_store_clear_removes_the_pending_entry():
    store = DisambiguationStore()
    store.set(
        "thread-1",
        PendingDisambiguation(
            "recap_action", CANDIDATES, Intent(kind="recap_action", message="all")
        ),
    )

    store.clear("thread-1")

    assert store.get("thread-1") is None


def test_store_clear_is_a_noop_for_an_unknown_thread():
    store = DisambiguationStore()

    store.clear("thread-1")

    assert store.get("thread-1") is None


def test_format_choices_numbers_candidates_in_order():
    assert format_choices(CANDIDATES) == "1. backend/F4, 2. frontend/F5"


def test_resolve_choice_by_bare_digit():
    assert resolve_choice(CANDIDATES, "2") is ITEM_F5


def test_resolve_choice_by_decorated_digit():
    assert resolve_choice(CANDIDATES, "1.") is ITEM_F4
    assert resolve_choice(CANDIDATES, "#1") is ITEM_F4
    assert resolve_choice(CANDIDATES, "option 2") is ITEM_F5


def test_resolve_choice_by_number_word():
    assert resolve_choice(CANDIDATES, "one") is ITEM_F4
    assert resolve_choice(CANDIDATES, "the first one") is ITEM_F4
    assert resolve_choice(CANDIDATES, "number two") is ITEM_F5


def test_resolve_choice_index_out_of_range_falls_back_to_label_match():
    assert resolve_choice(CANDIDATES, "9") is None


def test_resolve_choice_by_exact_label():
    assert resolve_choice(CANDIDATES, "F4") is ITEM_F4
    assert resolve_choice(CANDIDATES, "f4") is ITEM_F4


def test_resolve_choice_by_exact_channel_and_label():
    assert resolve_choice(CANDIDATES, "frontend/F5") is ITEM_F5


def test_resolve_choice_returns_none_for_unrelated_text():
    assert resolve_choice(CANDIDATES, "will the chipmunks ever reunite?") is None


def test_resolve_choice_returns_none_for_empty_text():
    assert resolve_choice(CANDIDATES, "   ") is None
