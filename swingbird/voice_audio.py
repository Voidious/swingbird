"""Shared microphone-capture plumbing for wake-word listening and STT
(`voice_wake.py`, `voice_stt.py`) -- both read the same `arecord` stream
at the same frame size off the same device mapping, so this is factored
out here rather than duplicated (and left free to drift) between the two.

Mirrors `voice_tts.py`'s choice of `aplay` for playback: `arecord` is one
less native audio dependency to get working on both WSL and the Orange Pi
(§V.14), via the same ALSA/Pulse bridge story.
"""

from __future__ import annotations

import subprocess

import numpy as np

from swingbird.config import VoiceMicConfig

# openWakeWord's own frame size: its predict() wants multiples of 80ms
# (1280 samples) at 16kHz mono (see Model.predict's docstring) -- STT
# capture uses the same frames so both modules read one shared format.
FRAME_SAMPLES = 1280
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # S16_LE
CHUNK_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE

# Mirrors voice_tts.py's `_ALSA_DEVICE_BY_OUTPUT_TYPE` for the input side:
# "onboard" is the ALSA/Pulse default input (the WSL dev machine today,
# the Orange Pi's onboard mic later); "usb" is the reSpeaker XVF3800
# array, a placeholder device name now so wiring in the real one later is
# a one-line change.
_ALSA_DEVICE_BY_MIC_TYPE = {
    "onboard": "default",
    "usb": "usb",
}


class MicStreamError(Exception):
    """Raised when `arecord` can't start, or its stream ends unexpectedly."""


def open_mic_stream(mic: VoiceMicConfig) -> subprocess.Popen:
    """Start `arecord` capturing raw PCM from `mic`'s device to stdout."""
    device = _ALSA_DEVICE_BY_MIC_TYPE[mic.type]
    try:
        return subprocess.Popen(
            [
                "arecord",
                "-D",
                device,
                "-r",
                str(SAMPLE_RATE),
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "-c",
                "1",
                "-",
            ],
            stdout=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MicStreamError("arecord not found on PATH (install alsa-utils)") from exc


def read_frame(process: subprocess.Popen) -> np.ndarray:
    """Read one `FRAME_SAMPLES`-sample frame from `process`'s stdout."""
    raw = process.stdout.read(CHUNK_BYTES)
    if len(raw) < CHUNK_BYTES:
        raise MicStreamError("arecord stream ended unexpectedly")
    return np.frombuffer(raw, dtype=np.int16)


def call_translating_stream_error(error_cls: type[Exception], func, *args, **kwargs):
    """Call `func(*args, **kwargs)`, re-raising a `MicStreamError` as
    `error_cls` instead. `voice_wake.py` and `voice_stt.py` both wrap
    `open_mic_stream`/`read_frame` this way, just into their own
    module-specific error type, so the translation itself lives here once.
    """
    try:
        return func(*args, **kwargs)
    except MicStreamError as exc:
        raise error_cls(str(exc)) from exc
