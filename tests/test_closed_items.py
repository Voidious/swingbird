import json

from swingbird.closed_items import ClosedItemStore
from swingbird.recap import RecapItem

ITEM = RecapItem(
    channel="dripbird",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
    source_event_id="evt-1",
    keywords=("lint", "F4"),
)


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "closed_items.jsonl"

    ClosedItemStore(path)

    assert path.parent.exists()


def test_close_appends_a_snapshot_to_disk(tmp_path):
    path = tmp_path / "closed_items.jsonl"

    ClosedItemStore(path).close(ITEM)

    (line,) = path.read_text().splitlines()
    record = json.loads(line)
    assert record["channel"] == "dripbird"
    assert record["label"] == "F4"
    assert record["summary"] == "unused-ignore propagation"
    assert record["instruction"] == "Fix the deterministic directive trip-check."
    assert record["source_event_id"] == "evt-1"
    assert record["keywords"] == ["lint", "F4"]
    assert isinstance(record["closed_at"], float)


def _assert_dripbird_f4_item(item):
    assert item.channel == "dripbird", (
        f"expected channel 'dripbird', got {item.channel!r}"
    )
    assert item.label == "F4", f"expected label 'F4', got {item.label!r}"
    assert item.source_event_id == "evt-1", (
        f"expected source_event_id 'evt-1', got {item.source_event_id!r}"
    )


def test_close_returns_the_closed_item(tmp_path):
    store = ClosedItemStore(tmp_path / "closed_items.jsonl")

    closed = store.close(ITEM)

    _assert_dripbird_f4_item(closed)


def test_close_is_visible_in_memory_without_reloading(tmp_path):
    store = ClosedItemStore(tmp_path / "closed_items.jsonl")

    store.close(ITEM)

    assert len(store.for_channels(None, since=0)) == 1


def test_reloads_previously_closed_items_from_disk(tmp_path):
    path = tmp_path / "closed_items.jsonl"
    ClosedItemStore(path).close(ITEM)

    reloaded = ClosedItemStore(path)

    (item,) = reloaded.for_channels(None, since=0)
    _assert_dripbird_f4_item(item)
    assert item.keywords == ("lint", "F4")


def test_missing_file_starts_empty(tmp_path):
    store = ClosedItemStore(tmp_path / "closed_items.jsonl")

    assert store.for_channels(None, since=0) == ()


def test_load_skips_blank_lines(tmp_path):
    path = tmp_path / "closed_items.jsonl"
    path.write_text(
        json.dumps(
            {
                "channel": "dripbird",
                "label": "F4",
                "summary": "s",
                "instruction": "i",
                "source_event_id": None,
                "keywords": [],
                "closed_at": 100.0,
            }
        )
        + "\n\n"
    )

    store = ClosedItemStore(path)

    assert len(store.for_channels(None, since=0)) == 1


def test_load_defaults_missing_optional_fields(tmp_path):
    path = tmp_path / "closed_items.jsonl"
    path.write_text(
        json.dumps({"channel": "dripbird", "label": "F4", "closed_at": 100.0}) + "\n"
    )

    (item,) = ClosedItemStore(path).for_channels(None, since=0)

    assert item.summary == ""
    assert item.instruction == ""
    assert item.source_event_id is None
    assert item.keywords == ()


def _populate_closed_item_store(tmp_path):
    store = ClosedItemStore(tmp_path / "closed_items.jsonl")
    store.close(ITEM)
    store.close(
        RecapItem(
            channel="swingbird",
            label="F5",
            summary="s",
            instruction="i",
        )
    )
    return store


def test_for_channels_narrows_by_channel_name(tmp_path):
    store = _populate_closed_item_store(tmp_path)

    matched = store.for_channels({"dripbird"}, since=0)

    assert [item.channel for item in matched] == ["dripbird"]


def test_for_channels_none_returns_every_channel(tmp_path):
    store = _populate_closed_item_store(tmp_path)

    matched = store.for_channels(None, since=0)

    assert {item.channel for item in matched} == {"dripbird", "swingbird"}


def test_for_channels_excludes_items_closed_before_since(tmp_path):
    store = ClosedItemStore(tmp_path / "closed_items.jsonl")
    store.close(ITEM)

    assert store.for_channels(None, since=1e15) == ()


def test_for_channels_includes_items_closed_at_or_after_since(tmp_path):
    store = ClosedItemStore(tmp_path / "closed_items.jsonl")
    closed = store.close(ITEM)

    assert store.for_channels(None, since=closed.closed_at) == (closed,)


def test_reset_scoped_to_channel_clears_only_that_channel(tmp_path):
    store = _populate_closed_item_store(tmp_path)

    cleared = store.reset("dripbird")

    assert cleared == 1
    assert [item.channel for item in store.for_channels(None, since=0)] == ["swingbird"]


def test_reset_none_clears_every_channel(tmp_path):
    store = _populate_closed_item_store(tmp_path)

    cleared = store.reset(None)

    assert cleared == 2
    assert store.for_channels(None, since=0) == ()


def _make_store_with_one_closed_item(tmp_path):
    path = tmp_path / "closed_items.jsonl"
    store = ClosedItemStore(path)
    store.close(ITEM)
    return store, path


def test_reset_rewrites_the_file_on_disk(tmp_path):
    store, path = _make_store_with_one_closed_item(tmp_path)
    store.close(
        RecapItem(channel="swingbird", label="F5", summary="s", instruction="i")
    )

    store.reset("dripbird")

    reloaded = ClosedItemStore(path)
    assert [item.channel for item in reloaded.for_channels(None, since=0)] == [
        "swingbird"
    ]


def test_reset_with_nothing_to_clear_returns_zero_and_leaves_file_untouched(tmp_path):
    store, path = _make_store_with_one_closed_item(tmp_path)
    before = path.read_text()

    cleared = store.reset("swingbird")

    assert cleared == 0
    assert path.read_text() == before
