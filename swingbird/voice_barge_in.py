"""Mid-speech barge-in (Voice Mode design doc §V.12).

The wake-word/STT pipeline (`voice_wake.py`, `voice_stt.py`) only ever
listens *between* turns -- nothing monitors the mic while `voice_tts.speak`
is talking, so today the only way to say something while swingbird is
still mid-reply is to wait it out. This module runs a second `arecord`
stream concurrently with playback, purely to detect that the user has
started talking; once that's sustained long enough to trust
(`_speak_once_with_barge_in`'s `trigger_energy` accumulator), playback is stopped (via
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

Wake word barge-in (`WakeBargeIn`, `[voice].barge_in_requires_wake_word`)
swaps that VAD trigger for the wake word model: playback only stops when
the wake phrase itself is heard, then the command that follows is captured
on the same stream with no second wake word -- how Alexa/Siri behave. VAD
asks "is anything speech-like happening?", which swingbird's own leaked
voice answers yes to; the wake model asks "was this phrase said?", which
leakage of ordinary reply text doesn't. The capture/transcribe/resume
handling below is shared by both triggers.

A captured interruption isn't always real speech, though: a brief noise
(a mouse click, sitting up in a chair, even breathing on a sensitive
headset mic) can cross `vad_threshold` for a moment with nothing real
ever following it, and `voice_stt.transcribe`'s `vad_filter` then comes
back not confident (or empty) -- live-tested 2026-09-23, three separate
sessions, each losing a mid-reply reply outright once that happened,
since the old behavior spoke a low-confidence apology over *nothing*,
discarding `text` for good. `speak_with_barge_in` (the public entry point
below) now treats that outcome, and an explicit "never mind"/"continue"/
"resume" from the user, as *not* a real interruption -- it resumes `text`
from the chunk (`voice_tts.speak`'s own numbering) playback had reached,
not the very beginning, up to `MAX_RESUME_ATTEMPTS` times, rather than
losing it or making the user re-hear a long reply's already-spoken start.
A confident transcript that *is* one of a separate set of stop phrases
("stop"/"cancel"/etc., Voidious 2026-09-24) is also not routed as a
command, but for the opposite reason -- it's a real interruption, just one
that means "I'm done listening to this," so `text` isn't resumed either;
the reply simply ends there, same as if it had finished playing on its
own. Only a confident transcript that's neither a dismissal nor a stop
phrase is returned to `daemon.py` (wrapped in a `BargeInResult`, carrying
that same resume point) as a genuine barge-in -- which still might not be
a real command once `daemon.py` routes it; see `BargeInResult`'s own
docstring.
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from openwakeword.model import Model
from openwakeword.vad import VAD

from swingbird.config import (
    VoiceMicConfig,
    VoiceOutputConfig,
    VoiceSTTConfig,
    VoiceTTSConfig,
)
from swingbird.voice_audio import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    call_translating_stream_error,
    close_mic_stream,
    open_mic_stream,
    read_frame,
)
from swingbird.voice_stt import (
    STTError,
    Transcript,
    capture_until_silence,
    load_model,
    transcribe,
)
from swingbird.voice_tts import speak
from swingbird.voice_wake import load_model as load_wake_model

# VAD doesn't confidently score a frame as speech from its very first
# 80ms -- a quiet-onset word (live-tested: "what are the other items?"
# came back as just "the other items.", 2026-09-23) can take 1-3 frames
# before crossing VAD_SPEECH_THRESHOLD, and those frames were never kept
# anywhere before the trigger. Buffering this many frames *before* a
# trigger and seeding the capture with all of them (not just the
# triggering frame) recovers that lead-in instead of clipping it.
PRE_ROLL_FRAMES = 4

# No longer a hard "this many consecutive frames" requirement (see
# DEFAULT_TRIGGER_ENERGY below) -- now only sizes the rolling pre-roll
# buffer alongside PRE_ROLL_FRAMES, so a slow-to-accumulate trigger doesn't
# lose more of its own lead-in than this many frames back. Fallback for a
# caller with no `Config` around (e.g. a test, or a future standalone
# smoke-test CLI) -- daemon.py always threads through the real
# config.py-driven `voice.barge_in_trigger_frames` instead. Mirrors
# `voice_stt.MAX_UTTERANCE_SECONDS`'s role as its own module's
# non-config-driven default, same reasoning.
DEFAULT_TRIGGER_FRAMES = 4

# Per-frame VAD confidence a barge-in frame must clear to count as any
# evidence at all -- separate from, and stricter than,
# `voice_stt.VAD_SPEECH_THRESHOLD` (0.5), which this same interruption's own
# `capture_until_silence` still uses once triggered (endpointing an
# already-confirmed utterance is a different, lower-stakes decision than
# confirming one in the first place). A false trigger during playback is
# costlier than a missed endpoint mid-capture -- it stops the reply and, per
# `daemon.py`'s chit_chat/resume handling, either routes the noise as a
# command or has to resume the reply -- so the *start* of a barge-in is held
# to a stricter bar than everything after it. Independent of
# `trigger_energy`: this is the floor each frame's score is measured
# against, that knob is how much cumulative margin above (or below) it is
# needed to trust the result -- live-tested background noise (2026-09-23)
# crossed `VAD_SPEECH_THRESHOLD` for enough *consecutive* frames to trigger
# the old duration-only model even with a raised frame count, which a purely
# duration-based fix can't help. Not itself live-tested yet -- a considered
# starting point, same as `trigger_energy`'s own initial default, worth
# retuning once real hardware (§V.14) is in the loop. Fallback for a caller
# with no `Config` around, same reasoning as `DEFAULT_TRIGGER_FRAMES`.
DEFAULT_TRIGGER_VAD_THRESHOLD = 0.8

# How much cumulative evidence a mid-playback signal must build up before
# it's trusted as the start of real speech (Voidious, 2026-09-26, replacing
# the original "N consecutive frames" model above): each frame adds its own
# `vad.predict(frame) - vad_threshold` to a running total, floored at 0.0
# rather than allowed to go negative, and a barge-in fires once that total
# reaches `trigger_energy`. A frame scoring well clear of `vad_threshold`
# builds credit faster than one that barely clears it, and a frame that dips
# back below only drains the total by its own shortfall instead of wiping
# out everything accumulated so far the way a single sub-threshold frame did
# under the old consecutive-frame count. Calibrated against the defaults
# above so a solidly-scoring voice (~1.0) still triggers in roughly
# `DEFAULT_TRIGGER_FRAMES` frames: 4 * (1.0 - 0.8) = 0.8, rounded down
# slightly since real speech rarely scores a clean 1.0 every frame.
DEFAULT_TRIGGER_ENERGY = 0.6

# Frames discarded right after a wake word barge-in fires, before capturing
# the command that follows it. The model scores the phrase as it finishes, so
# the last syllable or two of "swingbird" is still arriving when it trips;
# without this, VAD sees that tail as the start of the utterance and
# transcription picks up a stray "bird" in front of the real command (which
# would also defeat the exact-match stop/dismiss phrases). Not yet tuned on
# real hardware -- raise it if a live run still shows the tail leaking in,
# lower it if the first word of a command gets clipped.
WAKE_SETTLE_FRAMES = 2

_wake_models: dict[str, tuple[Model, str]] = {}


def _wake_model_for(wake_word: str) -> tuple[Model, str]:
    """The wake model for `wake_word`, loaded once per process and reset
    before each reuse -- a per-reply ONNX load is slow on the Orange Pi, and
    that's time the reply is playing with nothing listening for the wake word.
    """
    if wake_word not in _wake_models:
        _wake_models[wake_word] = load_wake_model(wake_word)
    model, key = _wake_models[wake_word]
    model.reset()
    return model, key


@dataclass(frozen=True)
class WakeBargeIn:
    """Settings for wake-word barge-in (`[voice].barge_in_requires_wake_word`):
    a reply is interrupted only by `wake_word` scoring at least `threshold`
    on a mid-playback frame, instead of the VAD accumulator. After it fires,
    the command is awaited for up to `window_seconds` (the same
    `wake_word_window_seconds` a normal wake word turn waits) with no second
    wake word required -- the same flow as Alexa/Siri.
    """

    wake_word: str
    threshold: float
    window_seconds: int


@dataclass(frozen=True)
class BargeInResult:
    """A genuine (confident, non-dismissal) interruption of a spoken reply,
    together with the chunk index (`voice_tts.speak`'s own numbering)
    playback had reached when it stopped.

    `daemon.py` needs `resume_chunk` for a reason `speak_with_barge_in`
    itself doesn't: its own dismiss-phrase/false-trigger resume (see
    `speak_with_barge_in`'s docstring) is fully internal, but a transcript
    that *is* confident and not a dismissal still might not be a real
    command once routed through the daemon's intent machinery -- a
    chit_chat classification there means resuming this same reply from
    `resume_chunk`, not losing it to the "that's outside what I handle"
    reply. See `daemon._run_voice_exchange`.
    """

    transcript: Transcript
    resume_chunk: int


# Said (or heard, mistakenly) after a barge-in, these mean "that wasn't a
# real command, keep going" rather than a new instruction -- matched
# case-insensitively with trailing punctuation stripped, not as a
# substring, so a real command that happens to end in "...and continue
# with the rest" isn't misread as a dismissal.
_DISMISS_PHRASES = frozenset(
    {"never mind", "nevermind", "continue", "resume", "keep going", "go on"}
)

# Said after a barge-in, these mean "stop talking, but don't resume and
# don't treat this as a command either" (Voidious, 2026-09-24: "the most
# likely thing you're going to be barging in about is just to stop
# talking"). Matched the same way as `_DISMISS_PHRASES` -- case-
# insensitively, trailing punctuation stripped, exact phrase not
# substring. Unlike a dismissal, which resumes `text` on the assumption
# the interruption wasn't real, a stop phrase *is* real -- it just isn't
# routed through `_process` as a command, and the reply it cut off is
# simply over, same as if it had finished playing on its own.
_STOP_PHRASES = frozenset({"stop", "stop talking", "cancel", "abort"})

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


def _is_stop_phrase(text: str) -> bool:
    return text.strip().lower().rstrip(".!?,") in _STOP_PHRASES


def speak_with_barge_in(
    text: str,
    tts: VoiceTTSConfig,
    output: VoiceOutputConfig,
    mic: VoiceMicConfig,
    stt: VoiceSTTConfig,
    trigger_frames: int = DEFAULT_TRIGGER_FRAMES,
    vad_threshold: float = DEFAULT_TRIGGER_VAD_THRESHOLD,
    trigger_energy: float = DEFAULT_TRIGGER_ENERGY,
    start_chunk: int = 0,
    wake_barge_in: WakeBargeIn | None = None,
) -> BargeInResult | None:
    """Speak `text` aloud, listening on `mic` at the same time for the
    user talking over it -- resuming `text` from wherever it stopped,
    rather than losing it, if what gets captured turns out not to be a
    real interruption (see module docstring).

    Runs `_speak_once_with_barge_in` for one playback-plus-listen pass. If
    that returns `None` (nothing interrupted playback), this returns
    `None` too. A confident transcript is one of three things: a stop
    phrase like "stop"/"cancel" (real interruption, but `text` simply ends
    here -- this returns `None`, exactly as if playback had finished on
    its own, so the caller doesn't resume it or route it as a command); a
    dismissal phrase like "never mind" (not treated as real -- `text`
    resumes from that same chunk); or anything else, a genuine barge-in --
    wrapped in a `BargeInResult` (with where in `text` it stopped) for
    `daemon.py` to route as the next thing said. A non-confident
    transcript is treated the same as a dismissal -- resumed, not routed.
    Resuming happens up to `MAX_RESUME_ATTEMPTS` times before giving up and
    returning `None` as if the reply had simply finished.

    `start_chunk` lets a caller resume `text` from partway through --
    `daemon.py` passes the `resume_chunk` off a previous `BargeInResult`
    back in when a barge-in's own transcript turns out, once routed, not
    to be a real command either (see `daemon._run_voice_exchange`).

    `vad_threshold`/`trigger_energy` (Voice Mode design doc §V.12) are the
    per-frame confidence floor a mid-playback signal's score is measured
    against, and how much cumulative margin above that floor must build up,
    before it's trusted as the start of real speech rather than a transient
    -- see `config.VoiceConfig.barge_in_vad_threshold`/
    `barge_in_trigger_energy`'s own docstrings. `trigger_frames` no longer
    gates the trigger itself; it only sizes the rolling pre-roll buffer
    alongside `PRE_ROLL_FRAMES`.

    `wake_barge_in`, when given, replaces that VAD trigger entirely: only
    the wake word interrupts, and `vad_threshold`/`trigger_energy`/
    `trigger_frames` are unused. A wake word that's followed by no speech
    within its window is treated like any other false trigger -- the reply
    resumes.
    """
    chunk = start_chunk
    for _attempt in range(MAX_RESUME_ATTEMPTS + 1):
        transcript, resume_chunk = _speak_once_with_barge_in(
            text,
            tts,
            output,
            mic,
            stt,
            trigger_frames,
            vad_threshold,
            trigger_energy,
            chunk,
            wake_barge_in,
        )
        if transcript is None:
            return None
        if transcript.is_confident:
            if _is_stop_phrase(transcript.text):
                print("swingbird: barge-in was a stop request, ending reply...")
                return None
            if not _is_dismiss_phrase(transcript.text):
                return BargeInResult(
                    transcript=transcript, resume_chunk=resume_chunk or 0
                )
        chunk = resume_chunk or 0
        print(
            "swingbird: barge-in wasn't a real interruption "
            f"(confident={transcript.is_confident}, text={transcript.text!r}), "
            "resuming reply..."
        )
    return None


def _capture_after_wake_word(
    record: subprocess.Popen, vad: VAD, window_seconds: int
) -> np.ndarray | None:
    """Capture the command following a wake word barge-in on the same mic
    stream, after discarding `WAKE_SETTLE_FRAMES` of the wake phrase's own
    tail. Returns `None` if no speech starts within `window_seconds`."""
    for _ in range(WAKE_SETTLE_FRAMES):
        call_translating_stream_error(STTError, read_frame, record)
    return capture_until_silence(
        record,
        vad,
        frames=[],
        speech_started=False,
        wait_frames=int(window_seconds * SAMPLE_RATE / FRAME_SAMPLES),
    )


def _speak_once_with_barge_in(
    text: str,
    tts: VoiceTTSConfig,
    output: VoiceOutputConfig,
    mic: VoiceMicConfig,
    stt: VoiceSTTConfig,
    trigger_frames: int,
    vad_threshold: float,
    trigger_energy: float,
    start_chunk: int,
    wake_barge_in: WakeBargeIn | None,
) -> tuple[Transcript | None, int | None]:
    """One playback-plus-listen pass of `speak_with_barge_in` -- speaks
    `text` (from `start_chunk` on) on a background thread while this
    thread reads `mic`'s own `arecord` stream frame by frame, and returns
    `(transcript, resume_chunk)`: `resume_chunk` is wherever `speak`
    stopped (`None` if it played through), independent of whether a
    barge-in happened to be what stopped it.

    A frame only counts as the start of real speech once a leaky
    accumulator -- incremented every frame by `vad.predict(frame) -
    vad_threshold` and floored at 0.0 -- reaches `trigger_energy`. A
    single frame scoring at or above `vad_threshold` (the original §V.12
    behavior) was too quick to trip on a transient like a mouse click;
    requiring a fixed run of consecutive qualifying frames fixed that but
    reset all progress on any single dip, even a brief, mild one. The
    accumulator keeps a stronger signal building credit faster and a dip
    draining it only by its own shortfall, rather than wiping it out
    outright. Every frame read, speech-scoring or not, is kept in a
    rolling window sized to also hold `PRE_ROLL_FRAMES` frames *before*
    the accumulator starts climbing, so a quiet-onset word (VAD needs 1-3
    frames to confidently score speech at all) isn't clipped from the
    front of what gets captured once the trigger does fire. Once it does,
    playback is stopped immediately (via `stop_event`) and the *same* mic
    stream keeps recording
    (`capture_until_silence`, seeded with that whole window) until the
    interruption ends, then transcribes and returns it. Returns `(None,
    resume_chunk)` if playback finished with nothing said over it --
    `resume_chunk` will be `None` too unless `speak` itself was cut off by
    something other than a barge-in (see `finally`'s own comment).

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
    resume_chunk: int | None = None

    def _run_playback() -> None:
        nonlocal resume_chunk
        try:
            resume_chunk = speak(
                text, tts, output, stop_event=stop_playback, start_chunk=start_chunk
            )
        except Exception as exc:  # noqa: BLE001 - re-raised on this thread below
            playback_errors.append(exc)
        finally:
            playback_done.set()

    playback_thread = threading.Thread(target=_run_playback, daemon=True)
    playback_thread.start()

    record = None
    transcript: Transcript | None = None
    frames_monitored = 0
    peak_score = 0.0
    try:
        # Models load *before* the mic opens: an `arecord` left running
        # through a multi-second ONNX load fills its pipe, overruns, and
        # drops audio -- so a wake word said early in the reply was never
        # heard (live-observed as "overrun!!!" right after the model load).
        vad = VAD()
        if wake_barge_in is not None:
            wake_model, wake_key = _wake_model_for(wake_barge_in.wake_word)
        record = call_translating_stream_error(STTError, open_mic_stream, mic)
        window: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES + trigger_frames)
        energy = 0.0
        while not playback_done.is_set():
            frame = call_translating_stream_error(STTError, read_frame, record)
            window.append(frame)
            if wake_barge_in is not None:
                score = wake_model.predict(frame)[wake_key]
                frames_monitored += 1
                peak_score = max(peak_score, score)
                triggered = score >= wake_barge_in.threshold
            else:
                energy = max(0.0, energy + vad.predict(frame) - vad_threshold)
                triggered = energy >= trigger_energy
            if not triggered:
                continue
            print("swingbird: barge-in detected, capturing interruption...")
            stop_playback.set()
            if wake_barge_in is not None:
                audio = _capture_after_wake_word(
                    record, vad, wake_barge_in.window_seconds
                )
            else:
                audio = capture_until_silence(
                    record,
                    vad,
                    frames=list(window),
                    speech_started=True,
                    wait_frames=0,
                )
            if audio is None:
                # Wake word, but nothing said after it: same outcome as any
                # other false trigger, so the reply resumes.
                transcript = Transcript(text="", is_confident=False)
                print("swingbird: wake word barge-in heard no command")
                break
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
        if wake_barge_in is not None:
            # The number to tune `barge_in_wake_threshold` against: a peak
            # near 0 means the model never heard the phrase at all (audio
            # path), a peak just under the threshold means it's the threshold.
            print(
                f"swingbird: wake barge-in monitored {frames_monitored} frames, "
                f"peak wake score {peak_score:.3f}"
            )

    if playback_errors:
        raise playback_errors[0]
    return transcript, resume_chunk
