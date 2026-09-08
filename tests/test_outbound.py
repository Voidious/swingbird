import subprocess

import pytest

from swingbird import outbound
from swingbird.outbound import (
    RelayError,
    get_own_display_name,
    join_channel,
    open_dm,
    relay_dispatch,
    send_message,
    set_display_name,
    set_presence,
)


class FakeRun:
    def __init__(self, stdout="", returncode=0, stderr="", error=None):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, args, input=None, capture_output=None, text=None, check=None):
        self.calls.append({"args": args, "input": input})
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(
            args=args,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def test_send_message_posts_content_via_stdin(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-1", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    event_id = send_message("chan-1", "hello there")

    assert event_id == "evt-1"
    assert fake.calls[0]["args"] == [
        "buzz",
        "messages",
        "send",
        "--channel",
        "chan-1",
        "--content",
        "-",
    ]
    assert fake.calls[0]["input"] == "hello there"


def test_send_message_includes_reply_to(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-2", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    send_message("chan-1", "hello", reply_to="root-evt")

    assert "--reply-to" in fake.calls[0]["args"]
    assert fake.calls[0]["args"][fake.calls[0]["args"].index("--reply-to") + 1] == (
        "root-evt"
    )


def test_send_message_escapes_stray_at_word_when_no_mentions(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-2c", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    send_message("chan-1", "no one @mentioned you, so @here is moot")

    assert fake.calls[0]["input"] == (
        "no one @\u200bmentioned you, so @\u200bhere is moot"
    )


def test_send_message_leaves_email_like_text_alone(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-2d", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    send_message("chan-1", "reach user@host.example for details")

    assert fake.calls[0]["input"] == "reach user@host.example for details"


def test_send_message_does_not_escape_when_mentions_given(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-2e", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    send_message("chan-1", "hey @someone", mentions=["pubkey-a"])

    assert fake.calls[0]["input"] == "hey @someone"


def test_send_message_includes_mentions(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-2b", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    send_message("chan-1", "hello", mentions=["pubkey-a", "pubkey-b"])

    args = fake.calls[0]["args"]
    assert args.count("--mention") == 2
    assert args[args.index("--mention") + 1] == "pubkey-a"
    assert args[-1] == "pubkey-b"


def test_open_dm_returns_dm_id(monkeypatch):
    fake = FakeRun(
        stdout='{"event_id": "evt-3", "accepted": true, "message": "", '
        '"dm_id": "dm-chan-1"}'
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    assert open_dm("deadbeef") == "dm-chan-1"
    assert fake.calls[0]["args"] == ["buzz", "dms", "open", "--pubkey", "deadbeef"]


def test_join_channel_posts_join(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-7", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    join_channel("chan-1")

    assert fake.calls[0]["args"] == ["buzz", "channels", "join", "--channel", "chan-1"]


def test_join_channel_raises_on_a_private_channel_rejection(monkeypatch):
    fake = FakeRun(
        returncode=1,
        stderr='{"error": "relay_error", '
        '"message": "relay error 403: restricted: channel is private"}',
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    with pytest.raises(RelayError, match="restricted: channel is private"):
        join_channel("chan-1")


def test_set_presence_posts_status(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-5", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    set_presence("online")

    assert fake.calls[0]["args"] == [
        "buzz",
        "users",
        "set-presence",
        "--status",
        "online",
    ]


def test_get_own_display_name_returns_the_callers_profile(monkeypatch):
    fake = FakeRun(
        stdout='[{"display_name": "swingbird", "pubkey": "abc"}]',
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    assert get_own_display_name() == "swingbird"
    assert fake.calls[0]["args"] == ["buzz", "users", "get"]


def test_get_own_display_name_returns_none_when_no_profile(monkeypatch):
    fake = FakeRun(stdout="[]")
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    assert get_own_display_name() is None


def test_set_display_name_posts_name(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-6", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    set_display_name("swingbird")

    assert fake.calls[0]["args"] == [
        "buzz",
        "users",
        "set-profile",
        "--name",
        "swingbird",
    ]


def test_relay_dispatch_prefixes_attribution(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-4", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    event_id = relay_dispatch(
        "chan-1", "fix the login timeout bug", "Voidious", "owner-pubkey"
    )

    assert event_id == "evt-4"
    assert fake.calls[0]["input"] == (
        "Relaying instruction from @Voidious: fix the login timeout bug"
    )


def test_relay_dispatch_mentions_the_requester(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-4b", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    relay_dispatch("chan-1", "fix the login timeout bug", "Voidious", "owner-pubkey")

    args = fake.calls[0]["args"]
    assert args[args.index("--mention") + 1] == "owner-pubkey"


def test_relay_dispatch_mentions_target_agent(monkeypatch):
    fake = FakeRun(stdout='{"event_id": "evt-5", "accepted": true, "message": ""}')
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    event_id = relay_dispatch(
        "chan-1",
        "fix the login timeout bug",
        "Voidious",
        "owner-pubkey",
        target_agent="Codex",
    )

    assert event_id == "evt-5"
    assert fake.calls[0]["input"] == (
        "@Codex Relaying instruction from @Voidious: fix the login timeout bug"
    )


def test_buzz_cli_not_found(monkeypatch):
    fake = FakeRun(error=FileNotFoundError())
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    with pytest.raises(RelayError, match="not found on PATH"):
        send_message("chan-1", "hello")


def test_nonzero_exit_raises_with_stderr(monkeypatch):
    fake = FakeRun(
        returncode=1, stderr='{"error": "not_found", "message": "no such channel"}'
    )
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    with pytest.raises(RelayError, match="no such channel"):
        send_message("chan-1", "hello")


def test_unparseable_output_raises(monkeypatch):
    fake = FakeRun(stdout="not json")
    monkeypatch.setattr(outbound.subprocess, "run", fake)

    with pytest.raises(RelayError, match="unparseable output"):
        send_message("chan-1", "hello")
