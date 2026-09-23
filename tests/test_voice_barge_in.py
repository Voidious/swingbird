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

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        # Give the monitoring loop real time to poll a few frames (all
        # non-speech, via ConstantVAD(0.0) below) before playback "finishes"
        # on its own -- proving this path isn't just winning a race against
        # a monitor loop that never got to run at all.
        time.sleep(0.05)
        speak_calls.append((text, tts, output, stop_event, start_chunk))

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
    text, _tts, _output, stop_event, start_chunk = speak_calls[0]
    assert text == "reply text"
    assert isinstance(stop_event, threading.Event)
    assert start_chunk == 0
    assert close_calls == ["the-record"]


def test_speak_with_barge_in_passes_start_chunk_through_to_speak(monkeypatch):
    """A caller resuming a reply that was already interrupted once before
    (`daemon.py`, after a chit_chat-classified barge-in -- see
    `BargeInResult`'s own docstring) passes its own `start_chunk` in; the
    very first playback attempt must honor it, not always start at 0.
    """
    speak_calls = []

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        speak_calls.append(start_chunk)
        time.sleep(0.05)

    _patch_barge_in_monitoring(monkeypatch, fake_speak)

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        start_chunk=5,
    )

    assert result is None
    assert speak_calls == [5]


def test_speak_with_barge_in_stops_playback_and_transcribes_interruption(
    monkeypatch, capsys
):
    stop_events_seen = []

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        stop_events_seen.append(stop_event)
        # Blocks like a real interruptible `speak()` would, until the
        # monitoring loop below signals a barge-in -- proves the barge-in
        # is what stopped playback, not an unrelated natural finish.
        stop_event.wait(timeout=1.0)
        # Simulates `speak()` reporting exactly where it stopped, so this
        # test also proves that index reaches the returned `BargeInResult`
        # rather than being lost or hardcoded to 0.
        return 3

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

    # trigger_frames=1 keeps this test focused on the stop/capture/
    # transcribe flow -- the consecutive-frame debounce itself has its own
    # dedicated test below.
    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        trigger_frames=1,
    )

    assert result == voice_barge_in.BargeInResult(transcript=expected, resume_chunk=3)
    assert stop_events_seen[0].is_set() is True
    assert capture_calls == [("the-record", [frame], True, 0)]
    assert transcribe_calls == [("the-model", canned_audio)]
    assert close_calls == ["the-record"]
    out = capsys.readouterr().out
    assert "barge-in detected" in out
    assert "barge-in captured" in out
    assert "confident=True" in out
    assert "text='stop'" in out


def _patch_barge_in_for_vad_tests(monkeypatch, fake_speak):
    _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)
    monkeypatch.setattr(voice_barge_in, "read_frame", lambda record: np.zeros(1))
    monkeypatch.setattr(voice_barge_in, "VAD", lambda: ConstantVAD(0.6))


def test_speak_with_barge_in_uses_vad_threshold_for_the_trigger_decision(monkeypatch):
    """A frame scoring below `vad_threshold` (even if above
    `voice_stt.VAD_SPEECH_THRESHOLD`) must not trigger a barge-in -- this is
    the confidence knob `config.VoiceConfig.barge_in_vad_threshold` tunes,
    independent of `trigger_frames`' duration requirement.
    """

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        time.sleep(0.05)

    _patch_barge_in_for_vad_tests(monkeypatch, fake_speak)

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        trigger_frames=1,
        vad_threshold=0.8,
    )

    # 0.6 never crosses 0.8, so playback just runs to completion -- no
    # trigger, no capture, no transcript.
    assert result is None


def test_speak_with_barge_in_lowering_vad_threshold_makes_it_more_sensitive(
    monkeypatch,
):
    """The same 0.6-scoring signal that `vad_threshold=0.8` ignores above
    does trigger once `vad_threshold` is lowered to admit it -- proving
    this is a real, live knob, not a value that's read and ignored.
    """

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        stop_event.wait(timeout=1.0)

    _patch_barge_in_for_vad_tests(monkeypatch, fake_speak)
    monkeypatch.setattr(voice_barge_in, "load_model", lambda stt: "the-model")
    monkeypatch.setattr(
        voice_barge_in,
        "capture_until_silence",
        lambda record, vad, frames, speech_started, wait_frames: np.array([0]),
    )
    expected = Transcript(text="stop", is_confident=True)
    monkeypatch.setattr(voice_barge_in, "transcribe", lambda model, audio: expected)

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        trigger_frames=1,
        vad_threshold=0.5,
    )

    assert result == voice_barge_in.BargeInResult(transcript=expected, resume_chunk=0)


def _patch_barge_in_capture(monkeypatch, vad_class):
    monkeypatch.setattr(voice_barge_in, "VAD", vad_class)

    capture_calls = []
    monkeypatch.setattr(
        voice_barge_in,
        "capture_until_silence",
        lambda record, vad, frames, speech_started, wait_frames: (
            capture_calls.append(frames) or np.array([0])
        ),
    )
    monkeypatch.setattr(voice_barge_in, "load_model", lambda stt: "the-model")
    return capture_calls


def test_speak_with_barge_in_seeds_capture_with_pre_roll_frames(monkeypatch):
    """A real utterance's first 1-3 frames often score below
    `VAD_SPEECH_THRESHOLD` before it climbs high enough to trigger --
    live-tested 2026-09-23, where that clipped "what are" off the front of
    an interruption. This proves the frames read *before* the triggering
    one are still included in what gets captured, not discarded.
    """

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
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

    capture_calls = _patch_barge_in_capture(monkeypatch, ScriptedVAD)
    monkeypatch.setattr(
        voice_barge_in,
        "transcribe",
        lambda model, audio: Transcript(text="stop", is_confident=True),
    )

    # trigger_frames=1 preserves the original single-frame trigger for
    # this test -- it's about pre-roll seeding, not the consecutive-frame
    # debounce, which has its own dedicated test below. A confident,
    # non-dismissal transcript keeps this to one attempt: a low-confidence
    # one would make `speak_with_barge_in` resume (see the resume tests
    # below), calling `speak` (and exhausting this same one-shot
    # VAD/read_frame scripting) a second time.
    voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        trigger_frames=1,
    )

    # Only the most recent PRE_ROLL_FRAMES quiet frames are kept (the
    # buffer is bounded), immediately followed by the triggering frame.
    expected = quiet_frames[-voice_barge_in.PRE_ROLL_FRAMES :] + [trigger_frame]
    assert len(capture_calls) == 1
    assert [f.tolist() for f in capture_calls[0]] == [f.tolist() for f in expected]


def test_speak_with_barge_in_requires_consecutive_trigger_frames(monkeypatch):
    """A two-frame blip that drops back below threshold shouldn't stop
    playback -- only `trigger_frames` *consecutive* speech-scoring frames
    should. Live-tested 2026-09-23: a single mouse click, or sitting up in
    a chair, was enough to interrupt playback under the original
    one-frame trigger.
    """

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        stop_event.wait(timeout=1.0)

    _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)

    frames = [np.array([i]) for i in range(6)]
    frames_read = iter(frames)
    monkeypatch.setattr(voice_barge_in, "read_frame", lambda record: next(frames_read))
    # A two-frame blip (below trigger_frames=3), then three consecutive
    # frames that do cross it.
    scores = iter([0.9, 0.9, 0.0, 0.9, 0.9, 0.9])

    class ScriptedVAD:
        def predict(self, frame):
            return next(scores)

    capture_calls = _patch_barge_in_capture(monkeypatch, ScriptedVAD)
    expected_transcript = Transcript(text="stop", is_confident=True)
    monkeypatch.setattr(
        voice_barge_in, "transcribe", lambda model, audio: expected_transcript
    )

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        trigger_frames=3,
    )

    assert result.transcript == expected_transcript
    # The blip's two frames reset the consecutive count -- capture only
    # happens once, seeded with every frame read (the blip plus the three
    # that actually triggered it).
    assert len(capture_calls) == 1
    assert [f.tolist() for f in capture_calls[0]] == [f.tolist() for f in frames]


def _patch_barge_in_speak_stream_and_vad(monkeypatch, fake_speak):
    _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)
    vad_instances = iter([ConstantVAD(0.9), ConstantVAD(0.0)])
    monkeypatch.setattr(voice_barge_in, "VAD", lambda: next(vad_instances))


def _patch_barge_in_mic_and_capture_stubs(monkeypatch, voice_barge_in):
    monkeypatch.setattr(voice_barge_in, "read_frame", lambda record: np.zeros(1))
    monkeypatch.setattr(voice_barge_in, "load_model", lambda stt: "the-model")
    monkeypatch.setattr(
        voice_barge_in,
        "capture_until_silence",
        lambda record, vad, frames, speech_started, wait_frames: np.array([0]),
    )


def _run_false_trigger_barge_in(monkeypatch, voice_barge_in):
    _patch_barge_in_mic_and_capture_stubs(monkeypatch, voice_barge_in)
    monkeypatch.setattr(
        voice_barge_in,
        "transcribe",
        lambda model, audio: Transcript(text="", is_confident=False),
    )

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        trigger_frames=1,
    )

    assert result is None


def test_speak_with_barge_in_resumes_after_a_false_trigger(monkeypatch, capsys):
    """A captured interruption that STT doesn't trust (empty/low-
    confidence, per `voice_stt.transcribe`'s `vad_filter`) isn't real
    speech -- `text` should be resumed from wherever it stopped rather
    than lost or restarted from the beginning, matching the accidental-
    interruption reports from 2026-09-23.
    """
    speak_calls = []
    start_chunks = []

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        speak_calls.append(text)
        start_chunks.append(start_chunk)
        if len(speak_calls) == 1:
            # First attempt: blocks until barged into below, then reports
            # (as a real interrupted `speak()` would) exactly which chunk
            # it stopped on.
            stop_event.wait(timeout=1.0)
            return 2
        # Resumed attempt: finishes on its own, uninterrupted.
        time.sleep(0.05)
        return None

    _patch_barge_in_speak_stream_and_vad(monkeypatch, fake_speak)
    _run_false_trigger_barge_in(monkeypatch, voice_barge_in)
    assert speak_calls == ["reply text", "reply text"]
    # The resumed attempt picks up from chunk 2 -- where the first attempt
    # stopped -- not chunk 0.
    assert start_chunks == [0, 2]
    assert "resuming reply" in capsys.readouterr().out


def test_speak_with_barge_in_resumes_on_a_dismiss_phrase(monkeypatch):
    """A confident "never mind"/"continue"/etc. means "that wasn't a real
    command, keep going" -- resume `text`, don't route it as a command.
    """
    speak_calls = []

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        speak_calls.append(text)
        if len(speak_calls) == 1:
            stop_event.wait(timeout=1.0)
        else:
            time.sleep(0.05)

    _patch_barge_in_speak_stream_and_vad(monkeypatch, fake_speak)
    _patch_barge_in_mic_and_capture_stubs(monkeypatch, voice_barge_in)
    monkeypatch.setattr(
        voice_barge_in,
        "transcribe",
        lambda model, audio: Transcript(text="Never mind.", is_confident=True),
    )

    result = voice_barge_in.speak_with_barge_in(
        "reply text",
        VoiceTTSConfig(voice="v"),
        VoiceOutputConfig(),
        VoiceMicConfig(),
        VoiceSTTConfig(),
        trigger_frames=1,
    )

    assert result is None
    assert speak_calls == ["reply text", "reply text"]


def test_speak_with_barge_in_gives_up_after_max_resume_attempts(monkeypatch):
    """A persistently noisy room shouldn't keep swingbird re-speaking the
    same reply forever -- after `MAX_RESUME_ATTEMPTS` straight false
    triggers, this gives up and returns `None`, same as an uninterrupted
    reply.
    """
    speak_calls = []

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
        speak_calls.append(text)
        stop_event.wait(timeout=1.0)

    _patch_barge_in_speak_and_stream(monkeypatch, fake_speak)
    monkeypatch.setattr(voice_barge_in, "VAD", lambda: ConstantVAD(0.9))
    _run_false_trigger_barge_in(monkeypatch, voice_barge_in)
    assert len(speak_calls) == voice_barge_in.MAX_RESUME_ATTEMPTS + 1


def test_speak_with_barge_in_reraises_playback_exception(monkeypatch):
    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
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

    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
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
    def fake_speak(text, tts, output, stop_event=None, start_chunk=0):
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
