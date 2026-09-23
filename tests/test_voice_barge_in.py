import threading
import time

import numpy as np
import pytest

from swingbird import voice_barge_in
from swingbird.config import (
    VoiceMicConfig,
    VoiceOutputConfig,
    VoiceSTTConfig,
    VoiceTTSConfig,
)
from swingbird.voice_audio import MicStreamError
from swingbird.voice_stt import STTError, Transcript


class ConstantVAD:
    """Unlike `test_voice_stt.py`'s `FakeVAD` (a fixed-length list of
    scores, popped one per call), `speak_with_barge_in`'s monitoring loop
    polls indefinitely until playback finishes on its own or this trips --
    a constant score keeps every test deterministic regardless of exactly
    how many frames get read before that happens.
    """

    def __init__(self, score: float):
        self.score = score

    def predict(self, frame):
        return self.score


def _patch_barge_in_speak_and_stream(monkeypatch, fake_speak):
    monkeypatch.setattr(voice_barge_in, "speak", fake_speak)
    monkeypatch.setattr(voice_barge_in, "open_mic_stream", lambda mic: "the-record")
    close_calls = []
    monkeypatch.setattr(
        voice_barge_in, "close_mic_stream", lambda record: close_calls.append(record)
    )
    return close_calls


def _patch_barge_in_monitoring(monkeypatch, fake_speak):
    close_calls = _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)
    monkeypatch.setattr(voice_barge_in, "read_frame", lambda record: np.zeros(1))
    monkeypatch.setattr(voice_barge_in, "VAD", lambda: ConstantVAD(0.0))
    return close_calls


def test_speak_with_barge_in_returns_none_when_playback_finishes_uninterrupted(
    monkeypatch,
):
    speak_calls = []

    def fake_speak(text, tts, output, stop_event=None):
        # Give the monitoring loop real time to poll a few frames (all
        # non-speech, via ConstantVAD(0.0) below) before playback "finishes"
        # on its own -- proving this path isn't just winning a race against
        # a monitor loop that never got to run at all.
        time.sleep(0.05)
        speak_calls.append((text, tts, output, stop_event))

    close_calls = _patch_barge_in_monitoring(monkeypatch, fake_speak)

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
    )

    assert result is None
    assert len(speak_calls) == 1
    text, _tts, _output, stop_event = speak_calls[0]
    assert text == "reply text"
    assert isinstance(stop_event, threading.Event)
    assert close_calls == ["the-record"]


def test_speak_with_barge_in_stops_playback_and_transcribes_interruption(
    monkeypatch, capsys
):
    stop_events_seen = []

    def fake_speak(text, tts, output, stop_event=None):
        stop_events_seen.append(stop_event)
        # Blocks like a real interruptible `speak()` would, until the
        # monitoring loop below signals a barge-in -- proves the barge-in
        # is what stopped playback, not an unrelated natural finish.
        stop_event.wait(timeout=1.0)

    close_calls = _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)
    frame = np.array([1, 2, 3])
    monkeypatch.setattr(voice_barge_in, "read_frame", lambda record: frame)
    monkeypatch.setattr(voice_barge_in, "VAD", lambda: ConstantVAD(0.9))

    capture_calls = []
    canned_audio = np.array([9, 9, 9])

    def fake_capture_until_silence(record, vad, frames, speech_started, wait_frames):
        capture_calls.append((record, frames, speech_started, wait_frames))
        return canned_audio

    monkeypatch.setattr(
        voice_barge_in, "capture_until_silence", fake_capture_until_silence
    )
    monkeypatch.setattr(voice_barge_in, "load_model", lambda stt: "the-model")
    expected = Transcript(text="stop", is_confident=True)
    transcribe_calls = []

    def fake_transcribe(model, audio):
        transcribe_calls.append((model, audio))
        return expected

    monkeypatch.setattr(voice_barge_in, "transcribe", fake_transcribe)

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
    )

    assert result == expected
    assert stop_events_seen[0].is_set() is True
    assert capture_calls == [("the-record", [frame], True, 0)]
    assert transcribe_calls == [("the-model", canned_audio)]
    assert close_calls == ["the-record"]
    out = capsys.readouterr().out
    assert "barge-in detected" in out
    assert "barge-in captured" in out
    assert "confident=True" in out
    assert "text='stop'" in out


def test_speak_with_barge_in_seeds_capture_with_pre_roll_frames(monkeypatch):
    """A real utterance's first 1-3 frames often score below
    `VAD_SPEECH_THRESHOLD` before it climbs high enough to trigger --
    live-tested 2026-09-23, where that clipped "what are" off the front of
    an interruption. This proves the frames read *before* the triggering
    one are still included in what gets captured, not discarded.
    """

    def fake_speak(text, tts, output, stop_event=None):
        stop_event.wait(timeout=1.0)

    _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)

    quiet_frames = [np.array([i]) for i in range(voice_barge_in.PRE_ROLL_FRAMES + 2)]
    trigger_frame = np.array([99])
    frames_read = iter(quiet_frames + [trigger_frame])
    monkeypatch.setattr(voice_barge_in, "read_frame", lambda record: next(frames_read))
    # Scores below threshold for every quiet frame, then one at/above it.
    scores = iter([0.0] * len(quiet_frames) + [0.9])

    class ScriptedVAD:
        def predict(self, frame):
            return next(scores)

    monkeypatch.setattr(voice_barge_in, "VAD", ScriptedVAD)

    capture_calls = []
    monkeypatch.setattr(
        voice_barge_in,
        "capture_until_silence",
        lambda record, vad, frames, speech_started, wait_frames: (
            capture_calls.append(frames) or np.array([0])
        ),
    )
    monkeypatch.setattr(voice_barge_in, "load_model", lambda stt: "the-model")
    monkeypatch.setattr(
        voice_barge_in,
        "transcribe",
        lambda model, audio: Transcript(text="", is_confident=False),
    )

    voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
    )

    # Only the most recent PRE_ROLL_FRAMES quiet frames are kept (the
    # buffer is bounded), immediately followed by the triggering frame.
    expected = quiet_frames[-voice_barge_in.PRE_ROLL_FRAMES :] + [trigger_frame]
    assert len(capture_calls) == 1
    assert [f.tolist() for f in capture_calls[0]] == [f.tolist() for f in expected]


def test_speak_with_barge_in_reraises_playback_exception(monkeypatch):
    def fake_speak(text, tts, output, stop_event=None):
        raise RuntimeError("boom")

    close_calls = _patch_barge_in_monitoring(monkeypatch, fake_speak)

    with pytest.raises(RuntimeError, match="boom"):
        voice_barge_in.speak_with_barge_in(
            "hi",
            VoiceTTSConfig(voice="v"),
            VoiceOutputConfig(),
            VoiceMicConfig(),
            VoiceSTTConfig(),
        )

    # Torn down despite the error, same as the happy paths.
    assert close_calls == ["the-record"]


def test_speak_with_barge_in_raises_when_mic_wont_open(monkeypatch):
    speak_calls = []

    def fake_speak(text, tts, output, stop_event=None):
        speak_calls.append(stop_event)
        stop_event.wait(timeout=1.0)

    monkeypatch.setattr(voice_barge_in, "speak", fake_speak)

    def raise_not_found(mic):
        raise MicStreamError("arecord not found on PATH (install alsa-utils)")

    monkeypatch.setattr(voice_barge_in, "open_mic_stream", raise_not_found)
    close_calls = []
    monkeypatch.setattr(
        voice_barge_in, "close_mic_stream", lambda record: close_calls.append(record)
    )

    with pytest.raises(STTError, match="arecord not found on PATH"):
        voice_barge_in.speak_with_barge_in(
            "hi",
            VoiceTTSConfig(voice="v"),
            VoiceOutputConfig(),
            VoiceMicConfig(),
            VoiceSTTConfig(),
        )

    # Nothing to close_mic_stream since the mic never opened, but playback
    # must still have been signalled to stop rather than left running.
    assert close_calls == []
    assert speak_calls[0].is_set() is True


def test_speak_with_barge_in_raises_when_stream_ends_unexpectedly(monkeypatch):
    def fake_speak(text, tts, output, stop_event=None):
        stop_event.wait(timeout=1.0)

    close_calls = _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)

    def raise_ended(record):
        raise MicStreamError("arecord stream ended unexpectedly")

    monkeypatch.setattr(voice_barge_in, "read_frame", raise_ended)
    monkeypatch.setattr(voice_barge_in, "VAD", lambda: ConstantVAD(0.0))

    with pytest.raises(STTError, match="arecord stream ended unexpectedly"):
        voice_barge_in.speak_with_barge_in(
            "hi",
            VoiceTTSConfig(voice="v"),
            VoiceOutputConfig(),
            VoiceMicConfig(),
            VoiceSTTConfig(),
        )

    assert close_calls == ["the-record"]
