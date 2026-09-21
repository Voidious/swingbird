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


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    """Transcribe `audio` (int16 PCM, 16kHz mono) with a loaded model."""
    normalized = audio.astype(np.float32) / 32768.0
    segments, _info = model.transcribe(normalized, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()


def record_and_transcribe(
    mic: VoiceMicConfig,
    stt: VoiceSTTConfig,
    max_wait_seconds: float = MAX_UTTERANCE_SECONDS,
) -> str | None:
    """Record one utterance and transcribe it, or return `None` (skipping
    transcription) if `record_utterance` gave up waiting for speech to
    start -- see its own docstring for `max_wait_seconds`.
    """
    model = load_model(stt)
    audio = record_utterance(mic, max_wait_seconds)
    if audio is None:
        return None
    return transcribe(model, audio)


def _main() -> None:  # pragma: no cover -- manual smoke test, see §V.16 step 4
    parser = argparse.ArgumentParser(
        description="Record and transcribe one utterance, for manual smoke testing."
    )
    parser.add_argument("--config", default="swingbird.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Recording -- speak now (stops after a pause)...")
    text = record_and_transcribe(config.voice.mic, config.voice.stt)
    print(f"Transcript: {text!r}")


if __name__ == "__main__":
    _main()
