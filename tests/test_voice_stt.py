import numpy as np
import pytest

from swingbird import voice_stt
from swingbird.config import VoiceMicConfig, VoiceSTTConfig
from swingbird.voice_audio import FRAME_SAMPLES, MicStreamError
from swingbird.voice_stt import (
    STTError,
    record_and_transcribe,
    record_utterance,
    transcribe,
)


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    def __init__(self, model_size, device, compute_type):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = []

    def transcribe(self, audio, beam_size=5):
        self.transcribe_calls.append((audio, beam_size))
        return [FakeSegment("hello"), FakeSegment("world")], object()


class FakeVAD:
    def __init__(self):
        self.scores = []

    def predict(self, frame):
        return self.scores.pop(0)


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


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

    assert result == "hello world"
    normalized, beam_size = model.transcribe_calls[0]
    assert beam_size == 5
    assert normalized.dtype == np.float32
    assert normalized[0] == pytest.approx(32767 / 32768.0)
    assert normalized[1] == pytest.approx(-1.0)


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


def test_record_utterance_stops_at_max_seconds_if_never_silent(monkeypatch):
    fake_process = FakeProcess()
    monkeypatch.setattr(voice_stt, "open_mic_stream", lambda mic: fake_process)
    frames = iter([_frame()] * 10_000)
    monkeypatch.setattr(voice_stt, "read_frame", lambda process: next(frames))

    fake_vad = FakeVAD()
    max_frames = int(voice_stt.MAX_UTTERANCE_SECONDS * 16000 / FRAME_SAMPLES)
    fake_vad.scores = [0.9] * max_frames
    monkeypatch.setattr(voice_stt, "VAD", lambda: fake_vad)

    audio = record_utterance(VoiceMicConfig())

    assert len(audio) == FRAME_SAMPLES * max_frames
    assert fake_process.terminated


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


def test_record_and_transcribe_loads_model_records_then_transcribes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        voice_stt, "load_model", lambda stt: calls.append(("load", stt)) or "model"
    )
    monkeypatch.setattr(
        voice_stt,
        "record_utterance",
        lambda mic: calls.append(("record", mic)) or np.zeros(1, dtype=np.int16),
    )
    monkeypatch.setattr(
        voice_stt,
        "transcribe",
        lambda model, audio: calls.append(("transcribe", model, audio)) or "hi",
    )

    mic = VoiceMicConfig()
    stt = VoiceSTTConfig(model="small")
    result = record_and_transcribe(mic, stt)

    assert result == "hi"
    assert calls[0] == ("load", stt)
    assert calls[1][0] == "record"
    assert calls[1][1] is mic
    assert calls[2][0] == "transcribe"
    assert calls[2][1] == "model"
