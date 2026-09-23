"""Mid-speech barge-in (Voice Mode design doc §V.12).

The wake-word/STT pipeline (`voice_wake.py`, `voice_stt.py`) only ever
listens *between* turns -- nothing monitors the mic while `voice_tts.speak`
is talking, so today the only way to say something while swingbird is
still mid-reply is to wait it out. This module runs a second `arecord`
stream concurrently with playback, purely to detect that the user has
started talking; the instant it does, playback is stopped (via
`voice_tts.speak`'s `stop_event`) and the *same* mic stream keeps
recording -- reusing `voice_stt.capture_until_silence` -- to capture the
rest of what the user is saying, exactly as if they'd said it after a
normal wake-word/follow-up cue. `daemon.py` calls `speak_with_barge_in`
everywhere it used to call `voice_tts.speak` directly for a voice reply.

Post-TTS debounce (also §V.12) is a separate, simpler concern handled in
`daemon.py` itself, not here: it only matters once playback has actually
finished with no interruption, to keep its own trailing audio (or the
"listening started" cue right after it) from being misread as the start
of the *next* listen -- see `VoiceConfig.debounce_seconds`'s own
docstring. This module only ever runs while playback is still live.

No debounce against swingbird's own voice is applied *during* playback
here, unlike after it: real acoustic echo cancellation is out of scope,
so on a setup where the speaker audibly leaks into the mic (e.g. WSLg's
bridge, without a headset), this VAD can in principle mistake swingbird's
own voice for a barge-in. Untested on hardware where that's actually a
problem (§V.14) -- worth revisiting if a live run shows false triggers.
"""

from __future__ import annotations

import threading

from openwakeword.vad import VAD

from swingbird.config import (
    VoiceMicConfig,
    VoiceOutputConfig,
    VoiceSTTConfig,
    VoiceTTSConfig,
)
from swingbird.voice_audio import (
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


def speak_with_barge_in(
    text: str,
    tts: VoiceTTSConfig,
    output: VoiceOutputConfig,
    mic: VoiceMicConfig,
    stt: VoiceSTTConfig,
) -> Transcript | None:
    """Speak `text` aloud, listening on `mic` at the same time for the
    user talking over it.

    Runs `voice_tts.speak` on a background thread while this thread reads
    `mic`'s own `arecord` stream frame by frame through the same VAD
    threshold `voice_stt.record_utterance` uses. If a frame ever looks
    like speech before playback finishes on its own, this stops playback
    immediately (via `stop_event`) and keeps recording on the *same*
    stream until that utterance ends (`capture_until_silence`, seeded with
    the triggering frame so nothing between "speech detected" and "started
    capturing" is lost), then transcribes and returns it. Returns `None`
    if playback finished with nothing said over it -- the normal case,
    telling `daemon.py` to fall back to its usual cue-then-listen flow for
    the next turn instead.

    Both `aplay` (via `speak`'s own teardown) and this function's own
    `arecord` (via `close_mic_stream`) are always fully torn down before
    returning, whichever path is taken -- no stale subprocess left holding
    either device for whatever opens next.
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
        while not playback_done.is_set():
            frame = call_translating_stream_error(STTError, read_frame, record)
            if vad.predict(frame) < VAD_SPEECH_THRESHOLD:
                continue
            stop_playback.set()
            audio = capture_until_silence(
                record, vad, frames=[frame], speech_started=True, wait_frames=0
            )
            model = load_model(stt)
            transcript = transcribe(model, audio)
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
