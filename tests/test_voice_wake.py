import numpy as np
import pytest

from swingbird import voice_wake
from swingbird.config import VoiceMicConfig
from swingbird.voice_audio import MicStreamError
from swingbird.voice_wake import WakeWordError, listen_for_wake_word, load_model


class FakeModel:
    def __init__(self, wakeword_model_paths):
        self.paths = wakeword_model_paths
        # Mirrors openWakeWord's real key derivation: the model name plus
        # a version suffix, not the plain config value.
        self.models = {"hey_jarvis_v0.1": object()}
        self.scores = []

    def predict(self, frame):
        return self.scores.pop(0)


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        self.waited = True


def test_pretrained_model_path_raises_for_unknown_wake_word():
    with pytest.raises(WakeWordError, match="no pretrained openWakeWord model"):
        voice_wake._pretrained_model_path("swingbird")


def test_load_model_returns_model_and_score_key(monkeypatch):
    captured = {}

    def fake_ctor(wakeword_model_paths):
        captured["paths"] = wakeword_model_paths
        return FakeModel(wakeword_model_paths)

    monkeypatch.setattr(voice_wake, "Model", fake_ctor)

    _model, score_key = load_model("hey_jarvis")

    assert score_key == "hey_jarvis_v0.1"
    assert captured["paths"] == [voice_wake._pretrained_model_path("hey_jarvis")]


def test_listen_for_wake_word_returns_once_threshold_met(monkeypatch):
    fake_model = FakeModel([])
    fake_model.scores = [
        {"hey_jarvis_v0.1": 0.1},
        {"hey_jarvis_v0.1": 0.9},
    ]
    monkeypatch.setattr(voice_wake, "Model", lambda wakeword_model_paths: fake_model)

    fake_process = FakeProcess()
    mic_calls = []
    monkeypatch.setattr(
        voice_wake,
        "open_mic_stream",
        lambda mic: mic_calls.append(mic) or fake_process,
    )
    frame = np.zeros(1280, dtype=np.int16)
    frames = iter([frame, frame])
    monkeypatch.setattr(voice_wake, "read_frame", lambda process: next(frames))

    mic = VoiceMicConfig(type="usb")
    listen_for_wake_word("hey_jarvis", mic)

    assert mic_calls == [mic]
    assert fake_process.terminated
    assert fake_process.waited


def test_listen_for_wake_word_raises_when_mic_stream_wont_open(monkeypatch):
    fake_model = FakeModel([])
    monkeypatch.setattr(voice_wake, "Model", lambda wakeword_model_paths: fake_model)

    def raise_not_found(mic):
        raise MicStreamError("arecord not found on PATH (install alsa-utils)")

    monkeypatch.setattr(voice_wake, "open_mic_stream", raise_not_found)

    with pytest.raises(WakeWordError, match="arecord not found on PATH"):
        listen_for_wake_word("hey_jarvis", VoiceMicConfig())


def test_listen_for_wake_word_raises_when_stream_ends_unexpectedly(monkeypatch):
    fake_model = FakeModel([])
    monkeypatch.setattr(voice_wake, "Model", lambda wakeword_model_paths: fake_model)

    fake_process = FakeProcess()
    monkeypatch.setattr(voice_wake, "open_mic_stream", lambda mic: fake_process)

    def raise_ended(process):
        raise MicStreamError("arecord stream ended unexpectedly")

    monkeypatch.setattr(voice_wake, "read_frame", raise_ended)

    with pytest.raises(WakeWordError, match="arecord stream ended unexpectedly"):
        listen_for_wake_word("hey_jarvis", VoiceMicConfig())

    assert fake_process.terminated
    assert fake_process.waited
