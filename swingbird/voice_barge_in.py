"""Mid-speech barge-in (Voice Mode design doc §V.12).

The wake-word/STT pipeline (`voice_wake.py`, `voice_stt.py`) only ever
listens *between* turns -- nothing monitors the mic while `voice_tts.speak`
is talking, so today the only way to say something while swingbird is
still mid-reply is to wait it out. This module runs a second `arecord`
stream concurrently with playback, purely to detect that the user has
started talking; once that's sustained long enough to trust
(`_speak_once_with_barge_in`'s `trigger_frames`), playback is stopped (via
`voice_tts.speak`'s `stop_event`) and the *same* mic stream keeps
recording -- reusing `voice_stt.capture_until_silence` -- to capture the
rest of what the user is saying, exactly as if they'd said it after a
normal wake-word/follow-up cue. `daemon.py` calls `speak_with_barge_in`
everywhere it used to call `voice_tts.speak` directly for a voice reply.

No debounce against swingbird's own voice is applied *during* playback
here, unlike after it: real acoustic echo cancellation is out of scope,
so on a setup where the speaker audibly leaks into the mic (e.g. WSLg's
bridge, without a headset), this VAD can in principle mistake swingbird's
own voice for a barge-in. Untested on hardware where that's actually a
problem (§V.14) -- worth revisiting if a live run shows false triggers.

A captured interruption isn't always real speech, though: a brief noise
(a mouse click, sitting up in a chair, even breathing on a sensitive
headset mic) can cross `VAD_SPEECH_THRESHOLD` for a moment with nothing
real ever following it, and `voice_stt.transcribe`'s `vad_filter` then
comes back not confident (or empty) -- live-tested 2026-09-23, three
separate sessions, each losing a mid-reply reply outright once that
happened, since the old behavior spoke a low-confidence apology over
*nothing*, discarding `text` for good. `speak_with_barge_in` (the public
entry point below) now treats that outcome, and an explicit "never
mind"/"continue"/"resume" from the user, as *not* a real interruption --
it resumes `text` from the start instead, up to `MAX_RESUME_ATTEMPTS`
times, rather than losing it. Only a confident transcript that isn't one
of those dismissal phrases is returned to `daemon.py` as a genuine
barge-in.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from openwakeword.vad import VAD

from swingbird.config import (
    VoiceMicConfig,
    VoiceOutputConfig,
    VoiceSTTConfig,
    VoiceTTSConfig,
)
from swingbird.voice_audio import (
    SAMPLE_RATE,
    call_translating_stream_error,
    close_mic_stream,
    open_mic_stream,
    read_frame,
)
from swingbird.voice_stt import (
    VAD_SPEECH_THRESHOLD,
    STTError,
    Transcript,
    capture_until_silence,
    load_model,
    transcribe,
)
from swingbird.voice_tts import speak

# VAD doesn't confidently score a frame as speech from its very first
# 80ms -- a quiet-onset word (live-tested: "what are the other items?"
# came back as just "the other items.", 2026-09-23) can take 1-3 frames
# before crossing VAD_SPEECH_THRESHOLD, and those frames were never kept
# anywhere before the trigger. Buffering this many frames *before* a
# trigger and seeding the capture with all of them (not just the
# triggering frame) recovers that lead-in instead of clipping it.
PRE_ROLL_FRAMES = 4

# Fallback for a caller with no `Config` around (e.g. a test, or a future
# standalone smoke-test CLI) -- daemon.py always threads through the real
# config.py-driven `voice.barge_in_trigger_frames` instead. Mirrors
# `voice_stt.MAX_UTTERANCE_SECONDS`'s role as its own module's
# non-config-driven default, same reasoning.
DEFAULT_TRIGGER_FRAMES = 4

# Said (or heard, mistakenly) after a barge-in, these mean "that wasn't a
# real command, keep going" rather than a new instruction -- matched
# case-insensitively with trailing punctuation stripped, not as a
# substring, so a real command that happens to end in "...and continue
# with the rest" isn't misread as a dismissal.
_DISMISS_PHRASES = frozenset(
    {"never mind", "nevermind", "continue", "resume", "keep going", "go on"}
)

# How many times `speak_with_barge_in` will resume the same reply from the
# start after a false trigger or dismissal before giving up and letting
# the turn end normally, as if the reply had played through uninterrupted.
# Bounds how long a persistently noisy room can keep swingbird re-speaking
# one reply instead of ever finishing a turn -- live-tested 2026-09-23,
# where a chair creak and a mouse click each falsely triggered a barge-in
# within the same session.
MAX_RESUME_ATTEMPTS = 3


def _is_dismiss_phrase(text: str) -> bool:
    return text.strip().lower().rstrip(".!?,") in _DISMISS_PHRASES


def speak_with_barge_in(
    text: str,
    tts: VoiceTTSConfig,
    output: VoiceOutputConfig,
    mic: VoiceMicConfig,
    stt: VoiceSTTConfig,
    trigger_frames: int = DEFAULT_TRIGGER_FRAMES,
) -> Transcript | None:
    """Speak `text` aloud, listening on `mic` at the same time for the
    user talking over it -- resuming `text` from the start, rather than
    losing it, if what gets captured turns out not to be a real
    interruption (see module docstring).

    Runs `_speak_once_with_barge_in` for one playback-plus-listen pass. If
    that returns `None` (nothing interrupted playback), this returns
    `None` too. If it returns a confident `Transcript` that isn't a
    dismissal phrase, that's a genuine barge-in -- returned as-is for
    `daemon.py` to route as the next thing said. Otherwise (not
    confident, or confident but a dismissal like "never mind") the
    interruption wasn't real: `text` is spoken again from the beginning,
    up to `MAX_RESUME_ATTEMPTS` times, before giving up and returning
    `None` as if the reply had simply finished.

    `trigger_frames` (Voice Mode design doc §V.12) is how many consecutive
    VAD-positive frames a mid-playback signal must sustain before it's
    trusted as the start of real speech rather than a transient -- see
    `config.VoiceConfig.barge_in_trigger_frames`'s own docstring.
    """
    for _attempt in range(MAX_RESUME_ATTEMPTS + 1):
        transcript = _speak_once_with_barge_in(
            text, tts, output, mic, stt, trigger_frames
        )
        if transcript is None:
            return None
        if transcript.is_confident and not _is_dismiss_phrase(transcript.text):
            return transcript
        print(
            "swingbird: barge-in wasn't a real interruption "
            f"(confident={transcript.is_confident}, text={transcript.text!r}), "
            "resuming reply..."
        )
    return None


def _speak_once_with_barge_in(
    text: str,
    tts: VoiceTTSConfig,
    output: VoiceOutputConfig,
    mic: VoiceMicConfig,
    stt: VoiceSTTConfig,
    trigger_frames: int,
) -> Transcript | None:
    """One playback-plus-listen pass of `speak_with_barge_in` -- speaks
    `text` on a background thread while this thread reads `mic`'s own
    `arecord` stream frame by frame through the same VAD threshold
    `voice_stt.record_utterance` uses.

    A frame only counts as the start of real speech once `trigger_frames`
    *consecutive* frames have scored at or above `VAD_SPEECH_THRESHOLD` --
    a single frame doing so (the original §V.12 behavior) was too quick to
    trip on a transient like a mouse click. Every frame read, speech-
    scoring or not, is kept in a rolling window sized to also hold
    `PRE_ROLL_FRAMES` frames *before* that run starts, so a quiet-onset
    word (VAD needs 1-3 frames to confidently score speech at all) isn't
    clipped from the front of what gets captured once the trigger does
    fire. Once it does, playback is stopped immediately (via
    `stop_event`) and the *same* mic stream keeps recording
    (`capture_until_silence`, seeded with that whole window) until the
    interruption ends, then transcribes and returns it. Returns `None` if
    playback finished with nothing said over it.

    Both `aplay` (via `speak`'s own teardown) and this function's own
    `arecord` (via `close_mic_stream`) are always fully torn down before
    returning, whichever path is taken -- no stale subprocess left holding
    either device for whatever opens next.

    Prints when a barge-in triggers, and again once the interruption is
    captured and transcribed (with its duration/confidence/text) -- this
    module otherwise runs silently, so a live run's console log couldn't
    previously distinguish "never triggered," "triggered but stuck inside
    `capture_until_silence`" (e.g. VAD never seeing enough silence to stop,
    on hardware where playback bleeds into the mic), and "triggered,
    captured, but the daemon did something unexpected with the result."
    """
    stop_playback = threading.Event()
    playback_done = threading.Event()
    playback_errors: list[Exception] = []

    def _run_playback() -> None:
        try:
            speak(text, tts, output, stop_event=stop_playback)
        except Exception as exc:  # noqa: BLE001 - re-raised on this thread below
            playback_errors.append(exc)
        finally:
            playback_done.set()

    playback_thread = threading.Thread(target=_run_playback, daemon=True)
    playback_thread.start()

    record = None
    transcript: Transcript | None = None
    try:
        record = call_translating_stream_error(STTError, open_mic_stream, mic)
        vad = VAD()
        window: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES + trigger_frames)
        consecutive_speech_frames = 0
        while not playback_done.is_set():
            frame = call_translating_stream_error(STTError, read_frame, record)
            window.append(frame)
            if vad.predict(frame) < VAD_SPEECH_THRESHOLD:
                consecutive_speech_frames = 0
                continue
            consecutive_speech_frames += 1
            if consecutive_speech_frames < trigger_frames:
                continue
            print("swingbird: barge-in detected, capturing interruption...")
            stop_playback.set()
            audio = capture_until_silence(
                record,
                vad,
                frames=list(window),
                speech_started=True,
                wait_frames=0,
            )
            model = load_model(stt)
            transcript = transcribe(model, audio)
            print(
                f"swingbird: barge-in captured {len(audio) / SAMPLE_RATE:.1f}s, "
                f"confident={transcript.is_confident}, text={transcript.text!r}"
            )
            break
    finally:
        # Set unconditionally, not just on the barge-in path above: if the
        # mic never even opened, or a stream error hit mid-listen, playback
        # would otherwise keep talking in the background with nothing left
        # to stop it, orphaned past this function returning (or raising).
        stop_playback.set()
        if record is not None:
            close_mic_stream(record)
        playback_thread.join()

    if playback_errors:
        raise playback_errors[0]
    return transcript
