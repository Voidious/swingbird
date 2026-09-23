import numpy as np
import pytest

from swingbird import voice_audio
from swingbird.config import VoiceMicConfig
from swingbird.voice_audio import (
    MicStreamError,
    call_translating_stream_error,
    close_mic_stream,
    open_mic_stream,
    read_frame,
)


class _CustomError(Exception):
    pass


class FakeStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, n):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class FakeProcess:
    def __init__(self, chunks=()):
        self.stdout = FakeStdout(chunks)


def test_open_mic_stream_onboard_targets_default_device(monkeypatch):
    popen_calls = []
    monkeypatch.setattr(
        voice_audio.subprocess,
        "Popen",
        lambda args, stdout=None: popen_calls.append(args) or FakeProcess(),
    )

    open_mic_stream(VoiceMicConfig(type="onboard"))

    assert popen_calls == [
        [
            "arecord",
            "-D",
            "default",
            "-r",
            "16000",
            "-f",
            "S16_LE",
            "-t",
            "raw",
            "-c",
            "1",
            "-",
        ]
    ]


def test_open_mic_stream_usb_targets_usb_device(monkeypatch):
    popen_calls = []
    monkeypatch.setattr(
        voice_audio.subprocess,
        "Popen",
        lambda args, stdout=None: popen_calls.append(args) or FakeProcess(),
    )

    open_mic_stream(VoiceMicConfig(type="usb"))

    assert popen_calls[0][popen_calls[0].index("-D") + 1] == "usb"


def test_open_mic_stream_raises_when_arecord_not_found(monkeypatch):
    def raise_not_found(args, stdout=None):
        raise FileNotFoundError()

    monkeypatch.setattr(voice_audio.subprocess, "Popen", raise_not_found)

    with pytest.raises(MicStreamError, match="arecord not found on PATH"):
        open_mic_stream(VoiceMicConfig())


def test_read_frame_returns_int16_array():
    frame = np.array([1, -2, 3, -4], dtype=np.int16)
    padded = frame.tobytes() + b"\x00" * (
        voice_audio.CHUNK_BYTES - len(frame.tobytes())
    )
    process = FakeProcess([padded])

    result = read_frame(process)

    assert result.dtype == np.int16
    assert len(result) == voice_audio.FRAME_SAMPLES


def test_read_frame_raises_when_stream_ends_unexpectedly():
    process = FakeProcess([b"\x00\x01"])  # shorter than one frame

    with pytest.raises(MicStreamError, match="arecord stream ended unexpectedly"):
        read_frame(process)


def test_close_mic_stream_terminates_then_waits():
    calls = []

    class FakeProcess:
        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))

    close_mic_stream(FakeProcess())

    assert calls == ["terminate", ("wait", voice_audio._TERMINATE_TIMEOUT_SECONDS)]


def test_close_mic_stream_kills_when_terminate_does_not_finish_in_time():
    calls = []

    class FakeProcess:
        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            if timeout is not None:
                raise voice_audio.subprocess.TimeoutExpired(
                    cmd="arecord", timeout=timeout
                )
            calls.append("wait-after-kill")

        def kill(self):
            calls.append("kill")

    close_mic_stream(FakeProcess())

    assert calls == ["terminate", "kill", "wait-after-kill"]


def test_call_translating_stream_error_returns_value_on_success():
    result = call_translating_stream_error(_CustomError, lambda x, y: x + y, 1, y=2)

    assert result == 3


def test_call_translating_stream_error_translates_mic_stream_error():
    def raise_mic_stream_error():
        raise MicStreamError("boom")

    with pytest.raises(_CustomError, match="boom"):
        call_translating_stream_error(_CustomError, raise_mic_stream_error)
