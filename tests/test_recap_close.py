from swingbird.recap import RecapItem
from swingbird.recap_close import PendingClose, PendingCloseStore, resolve_close_reply

ITEM = RecapItem(channel="dripbird", label="F4", summary="s", instruction="Fix it.")


def test_store_returns_none_for_an_unset_thread():
    store = PendingCloseStore()

    assert store.get("thread-1") is None


def test_store_set_then_get_round_trips():
    store = PendingCloseStore()
    pending = PendingClose((ITEM,))

    store.set("thread-1", pending)

    assert store.get("thread-1") is pending


def test_store_set_replaces_an_existing_pending_close():
    store = PendingCloseStore()
    store.set("thread-1", PendingClose((ITEM,)))
    replacement = PendingClose(())

    store.set("thread-1", replacement)

    assert store.get("thread-1") is replacement


def test_store_clear_removes_the_pending_close():
    store = PendingCloseStore()
    store.set("thread-1", PendingClose((ITEM,)))

    store.clear("thread-1")

    assert store.get("thread-1") is None


def test_store_clear_on_an_unset_thread_is_a_no_op():
    store = PendingCloseStore()

    store.clear("thread-1")  # doesn't raise

    assert store.get("thread-1") is None


def test_store_is_scoped_per_thread():
    store = PendingCloseStore()
    store.set("thread-1", PendingClose((ITEM,)))

    assert store.get("thread-2") is None


def test_resolve_close_reply_confirm_words():
    for text in ("yes", "Yes", "confirm", "confirmed", "do it", "close it", "go ahead"):
        assert resolve_close_reply(text) is True


def test_resolve_close_reply_cancel_words():
    for text in ("no", "No", "cancel", "never mind", "nevermind", "don't"):
        assert resolve_close_reply(text) is False


def test_resolve_close_reply_strips_surrounding_whitespace_and_punctuation():
    assert resolve_close_reply("  yes.  ") is True
    assert resolve_close_reply("cancel!") is False


def test_resolve_close_reply_unrecognized_text_returns_none():
    assert resolve_close_reply("what's the status on F5?") is None


def test_resolve_close_reply_empty_text_returns_none():
    assert resolve_close_reply("") is None
