import json
from datetime import datetime

from swingbird.audit import AuditLog
from swingbird.pending_actions import DispatchProposal
from swingbird.recap import RecapItem

PROPOSAL = DispatchProposal(
    channel_id="chan-1", instruction="fix the login timeout bug", target_agent="Codex"
)
RECAP_ITEM = RecapItem(
    channel="dripbird",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
)


def _read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "audit.jsonl"

    AuditLog(path).log_transcript_in("thread-1", "hi")

    assert path.exists()


def test_log_transcript_in_writes_record(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_transcript_in("thread-1", "what's going on?")

    (record,) = _read_records(path)
    assert record["kind"] == "transcript_in"
    assert record["thread_id"] == "thread-1"
    assert record["text"] == "what's going on?"
    datetime.fromisoformat(record["timestamp"])


def test_log_proposed_action_writes_proposal_fields(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_proposed_action("thread-1", PROPOSAL)

    (record,) = _read_records(path)
    assert record["kind"] == "proposed_action"
    assert record["thread_id"] == "thread-1"
    assert record["channel_id"] == "chan-1"
    assert record["instruction"] == "fix the login timeout bug"
    assert record["target_agent"] == "Codex"
    assert record["reply_to"] is None


def test_log_decision_without_event_id_omits_event_id(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_decision("thread-1", "cancelled", PROPOSAL)

    (record,) = _read_records(path)
    assert record["kind"] == "decision"
    assert record["decision"] == "cancelled"
    assert "event_id" not in record


def test_log_decision_with_event_id_includes_it(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_decision("thread-1", "confirmed", PROPOSAL, event_id="evt-1")

    (record,) = _read_records(path)
    assert record["decision"] == "confirmed"
    assert record["event_id"] == "evt-1"


def test_log_recap_reference_writes_item_fields(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_recap_reference("thread-1", "recap_action", "F4", RECAP_ITEM)

    (record,) = _read_records(path)
    assert record["kind"] == "recap_reference"
    assert record["thread_id"] == "thread-1"
    assert record["recap_kind"] == "recap_action"
    assert record["reference"] == "F4"
    assert record["channel"] == "dripbird"
    assert record["label"] == "F4"
    assert record["source_event_id"] is None


def test_log_recap_built_writes_every_item(tmp_path):
    path = tmp_path / "audit.jsonl"
    other_item = RecapItem(
        channel="dripbird",
        label="F6",
        summary="lint residue",
        instruction="Fix the F6 lint residue.",
        is_primary=False,
        source_event_id="evt-9",
        keywords=("lint", "F6"),
    )

    AuditLog(path).log_recap_built("thread-1", (RECAP_ITEM, other_item))

    (record,) = _read_records(path)
    assert record["kind"] == "recap_built"
    assert record["thread_id"] == "thread-1"
    assert record["items"] == [
        {
            "channel": "dripbird",
            "label": "F4",
            "keywords": [],
            "is_primary": True,
            "source_event_id": None,
        },
        {
            "channel": "dripbird",
            "label": "F6",
            "keywords": ["lint", "F6"],
            "is_primary": False,
            "source_event_id": "evt-9",
        },
    ]


def test_log_recap_built_with_no_items_writes_an_empty_list(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_recap_built("thread-1", ())

    (record,) = _read_records(path)
    assert record["items"] == []


def test_log_closed_items_writes_channel_label_and_source(tmp_path):
    path = tmp_path / "audit.jsonl"
    other_item = RecapItem(
        channel="dripbird",
        label="F6",
        summary="lint residue",
        instruction="Fix the F6 lint residue.",
        source_event_id="evt-9",
    )

    AuditLog(path).log_closed_items("thread-1", (RECAP_ITEM, other_item))

    (record,) = _read_records(path)
    assert record["kind"] == "closed_items"
    assert record["thread_id"] == "thread-1"
    assert record["items"] == [
        {"channel": "dripbird", "label": "F4", "source_event_id": None},
        {"channel": "dripbird", "label": "F6", "source_event_id": "evt-9"},
    ]


def test_log_closed_items_reset_with_channel_writes_channel_and_count(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_closed_items_reset("thread-1", "dripbird", 3)

    (record,) = _read_records(path)
    assert record["kind"] == "closed_items_reset"
    assert record["thread_id"] == "thread-1"
    assert record["channel"] == "dripbird"
    assert record["count"] == 3


def test_log_closed_items_reset_without_channel_writes_null(tmp_path):
    path = tmp_path / "audit.jsonl"

    AuditLog(path).log_closed_items_reset("thread-1", None, 0)

    (record,) = _read_records(path)
    assert record["channel"] is None
    assert record["count"] == 0


def test_appends_records_across_calls(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLog(path)

    audit.log_transcript_in("thread-1", "first")
    audit.log_transcript_in("thread-1", "second")

    records = _read_records(path)
    assert [r["text"] for r in records] == ["first", "second"]
