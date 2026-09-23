import numpy as np
import pytest

from swingbird import voice_stt
from swingbird.config import VoiceMicConfig, VoiceSTTConfig
from swingbird.voice_audio import FRAME_SAMPLES, MicStreamError
from swingbird.voice_stt import (
    STTError,
    Transcript,
    record_and_transcribe,
    record_utterance,
    transcribe,
)


class FakeSegment:
    def __init__(self, text, avg_logprob=-0.1):
        self.text = text
        self.avg_logprob = avg_logprob


class FakeWhisperModel:
    def __init__(self, model_size, device, compute_type):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = []
        self.segments = [FakeSegment("hello"), FakeSegment("world")]

    def transcribe(self, audio, beam_size=5):
        self.transcribe_calls.append((audio, beam_size))
        return self.segments, object()


class FakeVAD:
    def __init__(self):
        self.scores = []

    def predict(self, frame):
        return self.scores.pop(0)


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True


def _frame():
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


def test_load_model_uses_cpu_and_configured_size(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        voice_stt,
        "WhisperModel",
        lambda model_size, device, compute_type: captured.update(
            model_size=model_size, device=device, compute_type=compute_type
        ),
    )

    voice_stt.load_model(VoiceSTTConfig(model="small.en"))

    assert captured == {
        "model_size": "small.en",
        "device": "cpu",
        "compute_type": "int8",
    }


def test_transcribe_joins_segment_text_and_normalizes_audio():
    model = FakeWhisperModel("small", "cpu", "int8")
    audio = np.array([32767, -32768, 0], dtype=np.int16)

    result = transcribe(model, audio)

    assert result == Transcript(text="hello world", is_confident=True)
    normalized, beam_size = model.transcribe_calls[0]
    assert beam_size == 5
    assert normalized.dtype == np.float32
    assert normalized[0] == pytest.approx(32767 / 32768.0)
    assert normalized[1] == pytest.approx(-1.0)


def test_transcribe_is_not_confident_when_any_segment_avg_logprob_is_low():
    model = FakeWhisperModel("small", "cpu", "int8")
    model.segments = [
        FakeSegment("hello", avg_logprob=-0.1),
        FakeSegment("world", avg_logprob=voice_stt.MIN_AVG_LOGPROB - 0.01),
    ]

    result = transcribe(model, np.zeros(1, dtype=np.int16))

    assert result == Transcript(text="hello world", is_confident=False)


def test_transcribe_avg_logprob_at_threshold_is_confident():
    model = FakeWhisperModel("small", "cpu", "int8")
    model.segments = [FakeSegment("hello", avg_logprob=voice_stt.MIN_AVG_LOGPROB)]

    result = transcribe(model, np.zeros(1, dtype=np.int16))

    assert result.is_confident is True


def test_transcribe_is_not_confident_with_no_segments():
    model = FakeWhisperModel("small", "cpu", "int8")
    model.segments = []

    result = transcribe(model, np.zeros(1, dtype=np.int16))

    assert result == Transcript(text="", is_confident=False)


def test_record_utterance_stops_after_silence_follows_speech(monkeypatch):
    fake_process = FakeProcess()
    monkeypatch.setattr(voice_stt, "open_mic_stream", lambda mic: fake_process)
    frames = iter([_frame()] * 20)
    monkeypatch.setattr(voice_stt, "read_frame", lambda process: next(frames))

    fake_vad = FakeVAD()
    # Silence before speech starts (shouldn't count toward the stop
    # condition), then one speech frame, then enough silence to trigger it.
    fake_vad.scores = [0.0, 0.0, 0.9] + [0.0] * voice_stt.SILENCE_FRAMES_TO_STOP
    monkeypatch.setattr(voice_stt, "VAD", lambda: fake_vad)

    audio = record_utterance(VoiceMicConfig())

    assert len(audio) == FRAME_SAMPLES * (3 + voice_stt.SILENCE_FRAMES_TO_STOP)
    assert fake_process.terminated
    assert fake_process.waited


def _setup_record_utterance(monkeypatch):
    """Stub a mic stream good for up to 10,000 frames and a `VAD` whose
    scores the caller fills in -- shared setup for every `record_utterance`
    test below that isn't exercising the exact short frame sequence
    `test_record_utterance_stops_after_silence_follows_speech` hand-builds.
    """
    fake_process = FakeProcess()
    monkeypatch.setattr(voice_stt, "open_mic_stream", lambda mic: fake_process)
    frames = iter([_frame()] * 10_000)
    monkeypatch.setattr(voice_stt, "read_frame", lambda process: next(frames))
    fake_vad = FakeVAD()
    monkeypatch.setattr(voice_stt, "VAD", lambda: fake_vad)
    return fake_process, fake_vad


def test_record_utterance_stops_at_max_seconds_if_never_silent(monkeypatch):
    fake_process, fake_vad = _setup_record_utterance(monkeypatch)
    max_frames = int(voice_stt.MAX_UTTERANCE_SECONDS * 16000 / FRAME_SAMPLES)
    fake_vad.scores = [0.9] * max_frames

    audio = record_utterance(VoiceMicConfig())

    assert len(audio) == FRAME_SAMPLES * max_frames
    assert fake_process.terminated
    assert fake_process.waited


def test_record_utterance_returns_none_if_speech_never_starts_within_wait(
    monkeypatch,
):
    fake_process, fake_vad = _setup_record_utterance(monkeypatch)
    wait_frames = int(1.0 * 16000 / FRAME_SAMPLES)
    fake_vad.scores = [0.0] * wait_frames

    audio = record_utterance(VoiceMicConfig(), max_wait_seconds=1.0)

    assert audio is None
    assert fake_process.terminated
    assert fake_process.waited


def test_record_utterance_ignores_max_wait_once_speech_has_started(monkeypatch):
    _, fake_vad = _setup_record_utterance(monkeypatch)
    wait_frames = int(1.0 * 16000 / FRAME_SAMPLES)
    # Speech starts on the very last frame the wait window allows, then
    # stops -- shouldn't be treated as a timeout just because it cut it
    # close.
    fake_vad.scores = (
        [0.0] * (wait_frames - 1) + [0.9] + [0.0] * voice_stt.SILENCE_FRAMES_TO_STOP
    )

    audio = record_utterance(VoiceMicConfig(), max_wait_seconds=1.0)

    assert audio is not None
    assert len(audio) == FRAME_SAMPLES * (
        wait_frames + voice_stt.SILENCE_FRAMES_TO_STOP
    )


def test_record_utterance_raises_when_mic_stream_wont_open(monkeypatch):
    def raise_not_found(mic):
        raise MicStreamError("arecord not found on PATH (install alsa-utils)")

    monkeypatch.setattr(voice_stt, "open_mic_stream", raise_not_found)

    with pytest.raises(STTError, match="arecord not found on PATH"):
        record_utterance(VoiceMicConfig())


def test_record_utterance_raises_when_stream_ends_unexpectedly(monkeypatch):
    fake_process = FakeProcess()
    monkeypatch.setattr(voice_stt, "open_mic_stream", lambda mic: fake_process)

    def raise_ended(process):
        raise MicStreamError("arecord stream ended unexpectedly")

    monkeypatch.setattr(voice_stt, "read_frame", raise_ended)

    with pytest.raises(STTError, match="arecord stream ended unexpectedly"):
        record_utterance(VoiceMicConfig())

    assert fake_process.terminated
    assert fake_process.waited


def test_record_and_transcribe_records_then_loads_model_then_transcribes(
    monkeypatch,
):
    calls = []
    expected = Transcript(text="hi", is_confident=True)
    monkeypatch.setattr(
        voice_stt, "load_model", lambda stt: calls.append(("load", stt)) or "model"
    )
    monkeypatch.setattr(
        voice_stt,
        "record_utterance",
        lambda mic, max_wait_seconds: (
            calls.append(("record", mic, max_wait_seconds))
            or np.zeros(1, dtype=np.int16)
        ),
    )
    monkeypatch.setattr(
        voice_stt,
        "transcribe",
        lambda model, audio: calls.append(("transcribe", model, audio)) or expected,
    )

    mic = VoiceMicConfig()
    stt = VoiceSTTConfig(model="small")
    result = record_and_transcribe(mic, stt, max_wait_seconds=5.0)

    assert result == expected
    assert calls[0] == ("record", mic, 5.0)
    assert calls[1] == ("load", stt)
    assert calls[2][0] == "transcribe"
    assert calls[2][1] == "model"


def test_record_and_transcribe_returns_none_without_transcribing_if_no_speech(
    monkeypatch,
):
    load_calls = []
    monkeypatch.setattr(
        voice_stt, "load_model", lambda stt: load_calls.append(stt) or "model"
    )
    monkeypatch.setattr(
        voice_stt, "record_utterance", lambda mic, max_wait_seconds: None
    )
    transcribe_calls = []
    monkeypatch.setattr(
        voice_stt,
        "transcribe",
        lambda model, audio: transcribe_calls.append((model, audio)) or "hi",
    )

    result = record_and_transcribe(VoiceMicConfig(), VoiceSTTConfig(model="small"))

    assert result is None
    assert transcribe_calls == []
    # Recording first (see `record_and_transcribe`'s docstring) means a
    # timed-out wait with no speech should skip the model load entirely,
    # not just the transcription -- no point paying for it unused.
    assert load_calls == []
