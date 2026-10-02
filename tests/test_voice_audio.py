import sys
import types

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


def test_open_mic_stream_defaults_to_default_device(monkeypatch):
    popen_calls = []
    monkeypatch.setattr(
        voice_audio.subprocess,
        "Popen",
        lambda args, stdout=None: popen_calls.append(args) or FakeProcess(),
    )

    open_mic_stream(VoiceMicConfig())

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
            "--buffer-time",
            "4000000",
            "--period-time",
            "100000",
            "-",
        ]
    ]


def test_open_mic_stream_targets_configured_device(monkeypatch):
    popen_calls = []
    monkeypatch.setattr(
        voice_audio.subprocess,
        "Popen",
        lambda args, stdout=None: popen_calls.append(args) or FakeProcess(),
    )

    open_mic_stream(VoiceMicConfig(device="plughw:CARD=ArrayUAC10,DEV=0"))

    assert (
        popen_calls[0][popen_calls[0].index("-D") + 1] == "plughw:CARD=ArrayUAC10,DEV=0"
    )


def test_open_mic_stream_records_the_configured_channel_count(monkeypatch):
    popen_calls = []
    monkeypatch.setattr(
        voice_audio.subprocess,
        "Popen",
        lambda args, stdout=None: popen_calls.append(args) or FakeProcess(),
    )

    process = open_mic_stream(VoiceMicConfig(channels=2))

    assert popen_calls[0][popen_calls[0].index("-c") + 1] == "2"
    assert process.mic_channels == 2


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


def test_read_frame_returns_only_the_first_channel_of_a_multichannel_stream():
    mic = np.arange(1, voice_audio.FRAME_SAMPLES + 1, dtype=np.int16)
    loopback = np.full(voice_audio.FRAME_SAMPLES, 999, dtype=np.int16)
    interleaved = np.stack([mic, loopback], axis=1).tobytes()
    process = FakeProcess([interleaved])
    process.mic_channels = 2

    result = read_frame(process)

    assert result.dtype == np.int16
    assert np.array_equal(result, mic)


class FakeCanceller:
    def __init__(self):
        self.calls = []

    def cancel_echo(self, mic, reference):
        self.calls.append((mic, reference))
        return [m - r for m, r in zip(mic, reference)]


def test_read_frame_runs_the_echo_canceller_on_mic_and_reference_chunks():
    mic = np.arange(1, voice_audio.FRAME_SAMPLES + 1, dtype=np.int16)
    reference = np.full(voice_audio.FRAME_SAMPLES, 7, dtype=np.int16)
    process = FakeProcess([np.stack([mic, reference], axis=1).tobytes()])
    process.mic_channels = 2
    process.echo_canceller = FakeCanceller()

    result = read_frame(process)

    assert result.dtype == np.int16
    assert np.array_equal(result, mic - 7)
    calls = process.echo_canceller.calls
    assert len(calls) == voice_audio.FRAME_SAMPLES // voice_audio._AEC_FRAME_SAMPLES
    assert all(len(m) == len(r) == voice_audio._AEC_FRAME_SAMPLES for m, r in calls)
    assert calls[0][0] == mic[: voice_audio._AEC_FRAME_SAMPLES].tolist()
    assert set(calls[0][1]) == {7}


def test_open_mic_stream_attaches_a_speex_canceller_only_when_enabled(monkeypatch):
    created = []

    class FakeAec:
        def __init__(self, *args):
            created.append(args)

    monkeypatch.setitem(sys.modules, "pyaec", types.SimpleNamespace(Aec=FakeAec))
    monkeypatch.setattr(
        voice_audio.subprocess, "Popen", lambda args, stdout=None: FakeProcess()
    )

    plain = open_mic_stream(VoiceMicConfig(channels=2))
    cancelling = open_mic_stream(VoiceMicConfig(channels=2, echo_cancel=True))

    assert plain.echo_canceller is None
    assert isinstance(cancelling.echo_canceller, FakeAec)
    assert created == [
        (
            voice_audio._AEC_FRAME_SAMPLES,
            voice_audio._AEC_FILTER_SAMPLES,
            voice_audio.SAMPLE_RATE,
            False,
        )
    ]


def test_open_mic_stream_raises_when_echo_cancellation_cannot_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyaec", None)
    monkeypatch.setattr(
        voice_audio.subprocess, "Popen", lambda args, stdout=None: FakeProcess()
    )

    with pytest.raises(MicStreamError, match="echo cancellation is unavailable"):
        open_mic_stream(VoiceMicConfig(channels=2, echo_cancel=True))


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
