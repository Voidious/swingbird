import json
import subprocess

from swingbird import outbound
from swingbird.history import (
    fetch_messages_since,
    fetch_recent_messages,
    fetch_thread_root,
)


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


class FakeRunSequence:
    """Returns a different canned page on each successive call, for testing
    `fetch_messages_since`'s paging loop."""

    def __init__(self, pages: list[list[dict]]):
        self._pages = [json.dumps(page) for page in pages]
        self.calls: list[dict] = []

    def __call__(self, args, input=None, capture_output=None, text=None, check=None):
        self.calls.append({"args": args, "input": input})
        stdout = self._pages[len(self.calls) - 1]
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=stdout, stderr=""
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


def test_fetch_recent_messages_passes_before(monkeypatch):
    fake = FakeRun(stdout="[]")
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    fetch_recent_messages("chan-1", limit=10, before=12345)

    assert fake.calls[0]["args"] == [
        "buzz",
        "messages",
        "get",
        "--channel",
        "chan-1",
        "--limit",
        "10",
        "--before",
        "12345",
    ]


def test_fetch_thread_root_returns_the_event_with_no_e_tag(monkeypatch):
    fake = FakeRun(
        stdout=json.dumps(
            [
                {"id": "root-evt", "content": "root", "tags": [["h", "chan-1"]]},
                {
                    "id": "mid-evt",
                    "content": "mid",
                    "tags": [["h", "chan-1"], ["e", "root-evt", "", "reply"]],
                },
                {
                    "id": "leaf-evt",
                    "content": "leaf",
                    "tags": [
                        ["h", "chan-1"],
                        ["e", "root-evt", "", "root"],
                        ["e", "mid-evt", "", "reply"],
                    ],
                },
            ]
        ),
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    root = fetch_thread_root("chan-1", "leaf-evt")

    assert root == "root-evt"
    assert fake.calls[0]["args"] == [
        "buzz",
        "messages",
        "thread",
        "--channel",
        "chan-1",
        "--event",
        "leaf-evt",
    ]


def test_fetch_thread_root_falls_back_to_event_id_without_a_rootless_event(
    monkeypatch,
):
    """Defensive: if the thread lookup doesn't contain an event with no `e`
    tag at all (shouldn't happen for a real thread), fall back to the
    event id that was asked about rather than guessing at a wrong root."""
    fake = FakeRun(
        stdout=json.dumps(
            [
                {
                    "id": "evt-1",
                    "content": "hi",
                    "tags": [["e", "missing-root", "", "reply"]],
                }
            ]
        ),
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    root = fetch_thread_root("chan-1", "evt-1")

    assert root == "evt-1"


def test_fetch_messages_since_stops_at_a_single_page(monkeypatch):
    """A page whose oldest message already reaches `since_ts` needs no
    further paging."""
    fake = FakeRunSequence(
        [[{"id": "e1", "content": "a", "created_at": 50}]],
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    events = fetch_messages_since("chan-1", since_ts=50)

    assert events == [{"id": "e1", "content": "a", "created_at": 50}]
    assert len(fake.calls) == 1
    assert "--before" not in fake.calls[0]["args"]


def test_fetch_messages_since_pages_backwards_past_the_first_page(monkeypatch):
    fake = FakeRunSequence(
        [
            [{"id": "e2", "content": "b", "created_at": 200}],
            [{"id": "e1", "content": "a", "created_at": 100}],
        ],
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    events = fetch_messages_since("chan-1", since_ts=100, page_size=1)

    # oldest-to-newest across both pages, no gaps or duplicates.
    assert events == [
        {"id": "e1", "content": "a", "created_at": 100},
        {"id": "e2", "content": "b", "created_at": 200},
    ]
    assert len(fake.calls) == 2
    assert fake.calls[1]["args"][-1] == "200"  # --before <page-1's oldest>


def test_fetch_messages_since_stops_on_an_empty_page(monkeypatch):
    """An exhausted channel (fewer messages than the window) must not loop
    forever waiting for a page that never crosses `since_ts`."""
    fake = FakeRunSequence(
        [
            [{"id": "e1", "content": "a", "created_at": 200}],
            [],
        ],
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    events = fetch_messages_since("chan-1", since_ts=50, page_size=1)

    assert events == [{"id": "e1", "content": "a", "created_at": 200}]
    assert len(fake.calls) == 2


def test_fetch_messages_since_filters_events_older_than_since_ts(monkeypatch):
    """The final page can contain events older than `since_ts` -- only the
    ones at or after it belong in the window."""
    fake = FakeRunSequence(
        [
            [
                {"id": "e1", "content": "old", "created_at": 40},
                {"id": "e2", "content": "new", "created_at": 100},
            ]
        ],
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    events = fetch_messages_since("chan-1", since_ts=50)

    assert events == [{"id": "e2", "content": "new", "created_at": 100}]


def test_fetch_messages_since_stops_if_a_page_makes_no_progress(monkeypatch):
    """Defensive guard: if a `--before`-bounded page's oldest message isn't
    actually older than the boundary requested (a stuck relay response),
    stop instead of looping forever re-requesting the same boundary."""
    fake = FakeRunSequence(
        [
            [{"id": "e2", "content": "b", "created_at": 200}],
            [{"id": "e2", "content": "b", "created_at": 200}],
        ],
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    events = fetch_messages_since("chan-1", since_ts=0, page_size=1)

    assert len(fake.calls) == 2
    assert events == [{"id": "e2", "content": "b", "created_at": 200}]


def test_fetch_messages_since_respects_max_messages_cap(monkeypatch):
    """A safety cap stops paging even if `since_ts` hasn't been reached
    yet, so one very chatty channel can't page indefinitely."""
    fake = FakeRunSequence(
        [
            [{"id": "e2", "content": "b", "created_at": 200}],
            [{"id": "e1", "content": "a", "created_at": 100}],
        ],
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    events = fetch_messages_since("chan-1", since_ts=0, page_size=1, max_messages=1)

    assert len(fake.calls) == 1
    assert events == [{"id": "e2", "content": "b", "created_at": 200}]
