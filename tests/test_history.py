import subprocess

from swingbird import outbound
from swingbird.history import fetch_recent_messages


class FakeRun:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[dict] = []

    def __call__(self, args, input=None, capture_output=None, text=None, check=None):
        self.calls.append({"args": args, "input": input})
        return subprocess.CompletedProcess(
            args=args,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def test_fetch_recent_messages_returns_events(monkeypatch):
    fake = FakeRun(
        stdout='[{"id": "evt-1", "content": "hi", "created_at": 100}]',
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    events = fetch_recent_messages("chan-1")

    assert events == [{"id": "evt-1", "content": "hi", "created_at": 100}]
    assert fake.calls[0]["args"] == ["buzz", "messages", "get", "--channel", "chan-1"]


def test_fetch_recent_messages_passes_limit(monkeypatch):
    fake = FakeRun(stdout="[]")
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    fetch_recent_messages("chan-1", limit=10)

    assert fake.calls[0]["args"] == [
        "buzz",
        "messages",
        "get",
        "--channel",
        "chan-1",
        "--limit",
        "10",
    ]
