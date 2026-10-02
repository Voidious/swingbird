import threading

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

    def wait(self, timeout=None):
        self.waited = True


def test_model_path_prefers_a_checked_in_custom_model(tmp_path):
    (tmp_path / "hey_jarvis.onnx").write_bytes(b"")

    path = voice_wake._model_path("hey_jarvis", tmp_path)

    assert path == str(tmp_path / "hey_jarvis.onnx")


def test_model_path_falls_back_to_a_pretrained_model(tmp_path):
    path = voice_wake._model_path("hey_jarvis", tmp_path)

    assert path == voice_wake.openwakeword.models["hey_jarvis"]["model_path"]


def test_model_path_raises_for_unknown_wake_word_listing_custom_models(tmp_path):
    (tmp_path / "hey_swingbird.onnx").write_bytes(b"")

    with pytest.raises(WakeWordError, match="no wake word model") as excinfo:
        voice_wake._model_path("nonsense", tmp_path)

    assert "hey_swingbird" in str(excinfo.value)
    assert "hey_jarvis" in str(excinfo.value)


def test_load_model_returns_model_and_score_key(monkeypatch, tmp_path):
    captured = {}

    def fake_ctor(wakeword_model_paths):
        captured["paths"] = wakeword_model_paths
        return FakeModel(wakeword_model_paths)

    monkeypatch.setattr(voice_wake, "Model", fake_ctor)

    _model, score_key = load_model("hey_jarvis", tmp_path)

    assert score_key == "hey_jarvis_v0.1"
    assert captured["paths"] == [voice_wake._model_path("hey_jarvis", tmp_path)]


def test_load_model_loads_a_custom_model_from_models_dir(monkeypatch, tmp_path):
    (tmp_path / "hey_swingbird.onnx").write_bytes(b"")
    captured = {}

    def fake_ctor(wakeword_model_paths):
        captured["paths"] = wakeword_model_paths
        return FakeModel(wakeword_model_paths)

    monkeypatch.setattr(voice_wake, "Model", fake_ctor)

    load_model("hey_swingbird", tmp_path)

    assert captured["paths"] == [str(tmp_path / "hey_swingbird.onnx")]


def test_checked_in_hey_swingbird_model_loads_with_the_real_runtime():
    """The shipped default wake word must actually load through the real
    openWakeWord runtime, not just a fake `Model` -- catches a model file
    that's missing, corrupt, or exported in a format this openWakeWord
    version can't run."""
    _model, score_key = load_model("hey_swingbird")

    assert score_key == "hey_swingbird"


def _configure_two_frame_read(monkeypatch, frame_length, dtype):
    frame = np.zeros(frame_length, dtype=dtype)
    frames = iter([frame, frame])
    monkeypatch.setattr(voice_wake, "read_frame", lambda process: next(frames))
    return frames


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
    _configure_two_frame_read(monkeypatch, 1280, np.int16)

    mic = VoiceMicConfig(device="plughw:CARD=ArrayUAC10,DEV=0")
    detected = listen_for_wake_word("hey_jarvis", mic)

    assert detected is True
    assert mic_calls == [mic]
    assert fake_process.terminated
    assert fake_process.waited


def _setup_fake_model_and_mic_stream(monkeypatch):
    fake_model = FakeModel([])
    monkeypatch.setattr(voice_wake, "Model", lambda wakeword_model_paths: fake_model)

    fake_process = FakeProcess()
    monkeypatch.setattr(voice_wake, "open_mic_stream", lambda mic: fake_process)
    return fake_model, fake_process


def test_listen_for_wake_word_honors_custom_threshold(monkeypatch):
    fake_model, _ = _setup_fake_model_and_mic_stream(monkeypatch)
    fake_model.scores = [
        {"hey_jarvis_v0.1": 0.3},
        {"hey_jarvis_v0.1": 0.45},
    ]
    frames = _configure_two_frame_read(monkeypatch, 1280, np.int16)

    detected = listen_for_wake_word("hey_jarvis", VoiceMicConfig(), threshold=0.4)

    assert detected is True
    assert next(frames, None) is None


def test_listen_for_wake_word_returns_false_when_stop_event_set(monkeypatch):
    """Voice Mode design doc §V.9: `daemon.py`'s `_wait_for_wake_word` sets
    `stop_event` to interrupt an otherwise-idle wait and speak a proactive
    agent-reply summary instead of blocking on the wake word indefinitely."""
    (_, fake_process) = _setup_fake_model_and_mic_stream(monkeypatch)

    def _unexpected_read(process):
        raise AssertionError("read_frame should not run once stop_event is set")

    monkeypatch.setattr(voice_wake, "read_frame", _unexpected_read)

    stop_event = threading.Event()
    stop_event.set()

    detected = listen_for_wake_word("hey_jarvis", VoiceMicConfig(), stop_event)

    assert detected is False
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
    (_, fake_process) = _setup_fake_model_and_mic_stream(monkeypatch)

    def raise_ended(process):
        raise MicStreamError("arecord stream ended unexpectedly")

    monkeypatch.setattr(voice_wake, "read_frame", raise_ended)

    with pytest.raises(WakeWordError, match="arecord stream ended unexpectedly"):
        listen_for_wake_word("hey_jarvis", VoiceMicConfig())

    assert fake_process.terminated
    assert fake_process.waited
