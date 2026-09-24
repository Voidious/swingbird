import pytest

from swingbird import voice_cues
from swingbird.config import VoiceOutputConfig
from swingbird.voice_cues import (
    LISTENING_STARTED_HZ,
    LISTENING_STOPPED_HZ,
    CueError,
    play_listening_started,
    play_listening_stopped,
)


class FakeStdin:
    def __init__(self):
        self.written = bytearray()
        self.closed = False

    def write(self, data):
        self.written.extend(data)

    def close(self):
        self.closed = True


class FakePopen:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdin = FakeStdin()
        self.args = None

    def wait(self):
        pass


def _mock_popen(monkeypatch, returncode=0):
    fake_popen = FakePopen(returncode=returncode)

    def fake_ctor(args, stdin=None):
        fake_popen.args = args
        return fake_popen

    monkeypatch.setattr(voice_cues.subprocess, "Popen", fake_ctor)
    return fake_popen


def test_play_listening_started_targets_onboard_device_by_default(monkeypatch):
    fake_popen = _mock_popen(monkeypatch)

    play_listening_started(VoiceOutputConfig())

    assert fake_popen.args == [
        "aplay",
        "-D",
        "default",
        "-r",
        str(voice_cues.SAMPLE_RATE),
        "-f",
        "S16_LE",
        "-t",
        "raw",
        "-c",
        "1",
        "-",
    ]
    assert len(fake_popen.stdin.written) > 0
    assert fake_popen.stdin.closed


def test_play_listening_started_targets_configured_device(monkeypatch):
    fake_popen = _mock_popen(monkeypatch)

    play_listening_started(VoiceOutputConfig(device="plughw:CARD=ArrayUAC10,DEV=0"))

    assert (
        fake_popen.args[fake_popen.args.index("-D") + 1]
        == "plughw:CARD=ArrayUAC10,DEV=0"
    )


def test_listening_started_and_stopped_tones_differ(monkeypatch):
    started = voice_cues._tone(LISTENING_STARTED_HZ)
    stopped = voice_cues._tone(LISTENING_STOPPED_HZ)

    assert started != stopped
    assert LISTENING_STARTED_HZ != LISTENING_STOPPED_HZ


def test_play_raises_when_aplay_not_found(monkeypatch):
    def raise_not_found(args, stdin=None):
        raise FileNotFoundError()

    monkeypatch.setattr(voice_cues.subprocess, "Popen", raise_not_found)

    with pytest.raises(CueError, match="aplay not found on PATH"):
        play_listening_started(VoiceOutputConfig())


def test_play_raises_when_aplay_exits_nonzero(monkeypatch):
    _mock_popen(monkeypatch, returncode=1)

    with pytest.raises(CueError, match="aplay exited with code 1"):
        play_listening_stopped(VoiceOutputConfig())
