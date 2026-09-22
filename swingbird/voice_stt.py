"""faster-whisper speech-to-text (Voice Mode design doc §V.5, §V.16 step 4).

This module is deliberately capture-and-transcribe only, proving STT
accuracy against real speech in the room it'll live in before anything
downstream of it exists. `daemon.py`'s `_run_voice_turn` now calls
`record_and_transcribe` right after the wake word fires (§V.16 step 5);
this module's own `__main__` is still there as a standalone smoke-test
CLI.

Endpointing (deciding when the user has stopped talking) reuses
openWakeWord's own bundled Silero VAD (`openwakeword.vad.VAD`) rather than
adding a second VAD dependency -- §V.5 lists VAD as Silero-backed
regardless of which package ships the ONNX weights, and `voice_wake.py`
already pulls in `openwakeword` (which bundles `silero_vad.onnx`), so this
is reuse, not a new moving part.

Unlike openWakeWord's pretrained models, faster-whisper's "small"/
"small.en" weights aren't bundled -- they download from Hugging Face on
first use and are cached under the default HF cache dir after that (see
AGENTS.md).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel
from openwakeword.vad import VAD

from swingbird.config import VoiceMicConfig, VoiceSTTConfig, load_config
from swingbird.voice_audio import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    call_translating_stream_error,
    close_mic_stream,
    open_mic_stream,
    read_frame,
)

# int8 on CPU is what faster-whisper's own benchmarks assume for "small"
# to run ~2x real-time on the Orange Pi's CPU (§V.5) -- full-precision
# float32 would give up that margin for accuracy most users won't notice.
COMPUTE_TYPE = "int8"

VAD_SPEECH_THRESHOLD = 0.5
# Silence this long after speech has started ends the utterance -- 80ms
# per frame * 15 frames = 1.2s, a middle ground between cutting off a
# mid-sentence pause and making every turn wait needlessly long after the
# user actually stops talking. Revisit once §V.16 step 4's real-room
# testing shows whether it needs tuning.
SILENCE_FRAMES_TO_STOP = 15
MAX_UTTERANCE_SECONDS = 15

# `Segment.avg_logprob` is faster-whisper's own per-segment confidence
# proxy: the average log probability of that segment's tokens, more
# negative meaning the model was less sure what it heard (as opposed to
# `no_speech_prob`, which is about whether there was speech at all --
# already handled upstream by VAD gating `record_utterance`). Confidence
# is the *minimum* across segments, not an average, so one garbled segment
# in an otherwise-clear utterance still rejects the whole thing -- per
# AGENTS.md's "never guess" stance elsewhere in this codebase (grounding
# recap items, resolving ambiguous references), a misheard command
# shouldn't get a chance to reach dispatch confirmation just because the
# rest of the sentence transcribed cleanly.
MIN_AVG_LOGPROB = -1.0

LOW_CONFIDENCE_REPLY = "Sorry, I didn't catch that."


@dataclass(frozen=True)
class Transcript:
    """One recorded-and-transcribed utterance. `is_confident` is `False`
    when Whisper's own segment-level `avg_logprob` suggests it guessed
    rather than actually heard the words -- see `transcribe`'s docstring.
    `text` is still populated either way; callers that only want to act on
    trustworthy input must check `is_confident` themselves rather than
    treating a low-confidence `Transcript` as equivalent to a `None`
    result from `record_and_transcribe` (that still means "no speech was
    ever detected," a different condition).
    """

    text: str
    is_confident: bool


class STTError(Exception):
    """Raised when `arecord` fails during utterance capture."""


def load_model(stt: VoiceSTTConfig) -> WhisperModel:
    return WhisperModel(stt.model, device="cpu", compute_type=COMPUTE_TYPE)


def record_utterance(
    mic: VoiceMicConfig, max_wait_seconds: float = MAX_UTTERANCE_SECONDS
) -> np.ndarray | None:
    """Record one utterance from `mic`, via `voice_audio`'s `arecord`
    stream, stopping at the first silence that follows detected speech (or
    `MAX_UTTERANCE_SECONDS` of total speech, whichever comes first).

    `max_wait_seconds` only bounds how long this waits for speech to
    *start* -- once it has, the utterance always runs to
    `SILENCE_FRAMES_TO_STOP`/`MAX_UTTERANCE_SECONDS` regardless of
    `max_wait_seconds`. Defaulting it to `MAX_UTTERANCE_SECONDS` matches
    this function's original (pre-§V.11) single-timer behavior exactly;
    `daemon.py`'s follow-up-window listen (§V.11, no wake word required)
    passes a longer value since the user might pause well past 15s before
    speaking again. Returns `None`, not a silent/empty array, if
    `max_wait_seconds` elapses with no speech ever detected -- lets a
    follow-up turn's caller tell "gave up waiting" apart from "captured a
    real (if quiet) utterance," which it needs to fall back to requiring
    the wake word again.
    """
    record = call_translating_stream_error(STTError, open_mic_stream, mic)

    vad = VAD()
    frames: list[np.ndarray] = []
    speech_started = False
    silent_frame_count = 0
    speech_frame_count = 0
    wait_frames = int(max_wait_seconds * SAMPLE_RATE / FRAME_SAMPLES)
    max_speech_frames = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)

    try:
        while True:
            frame = call_translating_stream_error(STTError, read_frame, record)
            frames.append(frame)
            if vad.predict(frame) >= VAD_SPEECH_THRESHOLD:
                speech_started = True
                silent_frame_count = 0
            elif speech_started:
                silent_frame_count += 1
                if silent_frame_count >= SILENCE_FRAMES_TO_STOP:
                    break

            if speech_started:
                speech_frame_count += 1
                if speech_frame_count >= max_speech_frames:
                    break
            elif len(frames) >= wait_frames:
                return None
    finally:
        close_mic_stream(record)

    return np.concatenate(frames)


def transcribe(model: WhisperModel, audio: np.ndarray) -> Transcript:
    """Transcribe `audio` (int16 PCM, 16kHz mono) with a loaded model.

    A transcript with zero segments (Whisper decided there was nothing to
    transcribe, despite VAD having triggered `record_utterance` in the
    first place) is treated as not confident rather than crashing on an
    empty `min()` -- same "reject rather than guess" outcome as a real
    low-`avg_logprob` segment, just a different way of getting there.
    """
    normalized = audio.astype(np.float32) / 32768.0
    segments, _info = model.transcribe(normalized, beam_size=5)
    texts = []
    avg_logprobs = []
    for segment in segments:
        texts.append(segment.text.strip())
        avg_logprobs.append(segment.avg_logprob)
    text = " ".join(texts).strip()
    is_confident = bool(avg_logprobs) and min(avg_logprobs) >= MIN_AVG_LOGPROB
    return Transcript(text=text, is_confident=is_confident)


def record_and_transcribe(
    mic: VoiceMicConfig,
    stt: VoiceSTTConfig,
    max_wait_seconds: float = MAX_UTTERANCE_SECONDS,
) -> Transcript | None:
    """Record one utterance and transcribe it, or return `None` (skipping
    transcription) if `record_utterance` gave up waiting for speech to
    start -- see its own docstring for `max_wait_seconds`. A captured
    utterance always yields a `Transcript`, confident or not; callers
    decide what to do with a low-confidence one (see `Transcript`'s own
    docstring) -- this function only captures and transcribes.

    Records *before* loading the whisper model, not after -- only
    `transcribe()` below needs the model, but constructing a fresh
    `WhisperModel` takes long enough (~0.8s on the Orange Pi, live-tested
    2026-09-22) that loading it first left the mic not actually open yet
    right after `voice_cues.play_listening_started`'s "go ahead" chime,
    silently swallowing the user's first word or two. Recording first
    means the mic is live the instant the cue finishes, and the model-load
    cost lands after the utterance is already captured instead of before.
    """
    audio = record_utterance(mic, max_wait_seconds)
    if audio is None:
        return None
    model = load_model(stt)
    return transcribe(model, audio)


def _main() -> None:  # pragma: no cover -- manual smoke test, see §V.16 step 4
    parser = argparse.ArgumentParser(
        description="Record and transcribe one utterance, for manual smoke testing."
    )
    parser.add_argument("--config", default="swingbird.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Recording -- speak now (stops after a pause)...")
    transcript = record_and_transcribe(config.voice.mic, config.voice.stt)
    if transcript is None:
        print("No speech detected.")
    else:
        confidence = "confident" if transcript.is_confident else "LOW CONFIDENCE"
        print(f"Transcript ({confidence}): {transcript.text!r}")


if __name__ == "__main__":
    _main()
