import json
from datetime import datetime

from swingbird.audit import AuditLog
from swingbird.pending_actions import DispatchProposal

PROPOSAL = DispatchProposal(
    channel_id="chan-1", instruction="fix the login timeout bug", target_agent="Codex"
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


def test_appends_records_across_calls(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLog(path)

    audit.log_transcript_in("thread-1", "first")
    audit.log_transcript_in("thread-1", "second")

    records = _read_records(path)
    assert [r["text"] for r in records] == ["first", "second"]
