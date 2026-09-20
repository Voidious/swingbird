import numpy as np
import pytest

from swingbird import voice_wake
from swingbird.config import VoiceMicConfig
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


class FakeStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, n):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class FakeRecordProcess:
    def __init__(self, chunks):
        self.stdout = FakeStdout(chunks)
        self.terminated = False

    def terminate(self):
        self.terminated = True


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

    frame_bytes = np.zeros(voice_wake.FRAME_SAMPLES, dtype=np.int16).tobytes()
    fake_process = FakeRecordProcess([frame_bytes, frame_bytes])
    popen_calls = []
    monkeypatch.setattr(
        voice_wake.subprocess,
        "Popen",
        lambda args, stdout=None: popen_calls.append(args) or fake_process,
    )

    listen_for_wake_word("hey_jarvis", VoiceMicConfig(type="onboard"))

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
    assert fake_process.terminated


def test_listen_for_wake_word_usb_mic_targets_usb_device(monkeypatch):
    fake_model = FakeModel([])
    fake_model.scores = [{"hey_jarvis_v0.1": 0.9}]
    monkeypatch.setattr(voice_wake, "Model", lambda wakeword_model_paths: fake_model)

    frame_bytes = np.zeros(voice_wake.FRAME_SAMPLES, dtype=np.int16).tobytes()
    fake_process = FakeRecordProcess([frame_bytes])
    popen_calls = []
    monkeypatch.setattr(
        voice_wake.subprocess,
        "Popen",
        lambda args, stdout=None: popen_calls.append(args) or fake_process,
    )

    listen_for_wake_word("hey_jarvis", VoiceMicConfig(type="usb"))

    assert popen_calls[0][popen_calls[0].index("-D") + 1] == "usb"


def test_listen_for_wake_word_raises_when_arecord_not_found(monkeypatch):
    fake_model = FakeModel([])
    monkeypatch.setattr(voice_wake, "Model", lambda wakeword_model_paths: fake_model)

    def raise_not_found(args, stdout=None):
        raise FileNotFoundError()

    monkeypatch.setattr(voice_wake.subprocess, "Popen", raise_not_found)

    with pytest.raises(WakeWordError, match="arecord not found on PATH"):
        listen_for_wake_word("hey_jarvis", VoiceMicConfig())


def test_listen_for_wake_word_raises_when_stream_ends_unexpectedly(monkeypatch):
    fake_model = FakeModel([])
    monkeypatch.setattr(voice_wake, "Model", lambda wakeword_model_paths: fake_model)

    fake_process = FakeRecordProcess([b"\x00\x01"])  # shorter than one frame
    monkeypatch.setattr(
        voice_wake.subprocess, "Popen", lambda args, stdout=None: fake_process
    )

    with pytest.raises(WakeWordError, match="arecord stream ended unexpectedly"):
        listen_for_wake_word("hey_jarvis", VoiceMicConfig())

    assert fake_process.terminated
