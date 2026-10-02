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

# Speex echo canceller (via `pyaec`) settings: 320-sample (20ms) frames
# divide `FRAME_SAMPLES` evenly, and a 3200-sample (200ms) filter covers the
# speaker-to-mic path. Swept 160/320-sample frames and 1600-8000-sample
# filters on a real XVF3800 recording; every setting recovered the wake word.
_AEC_FRAME_SAMPLES = 320
_AEC_FILTER_SAMPLES = 3200

# How long `close_mic_stream` gives a SIGTERM'd `arecord` to actually exit
# before escalating to SIGKILL -- see its own docstring for why a bare
# `wait()` with no timeout is unsafe here: an `arecord` that gets stuck
# mid-teardown (observed live, 2026-09-23 -- SIGTERM's own "Aborted by
# signal Terminated" line printed, but the process never followed up with
# its usual second `pcm_read` error line or actually exited) hangs
# `process.wait()` forever, which hangs everything downstream of it --
# the whole voice loop, silently, with no exception and nothing further
# printed. Short enough that a real hang doesn't stall a turn for long,
# long enough that ordinary teardown (order-of-milliseconds per every
# prior live test) never comes close to it.
_TERMINATE_TIMEOUT_SECONDS = 2.0


class MicStreamError(Exception):
    """Raised when `arecord` can't start, or its stream ends unexpectedly."""


def open_mic_stream(mic: VoiceMicConfig) -> subprocess.Popen:
    """Start `arecord` capturing raw PCM from `mic.device` to stdout.
    `mic.channels` channels are recorded and `read_frame` returns the first,
    with the second subtracted from it when `mic.echo_cancel` is set.
    """
    device = mic.device
    try:
        process = subprocess.Popen(
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
                str(mic.channels),
                "-",
            ],
            stdout=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MicStreamError("arecord not found on PATH (install alsa-utils)") from exc
    process.mic_channels = mic.channels
    process.echo_canceller = _new_echo_canceller() if mic.echo_cancel else None
    return process


def _new_echo_canceller():
    # Imported here: `pyaec` loads a native library at import time, and most
    # setups never enable echo cancellation.
    try:
        from pyaec import Aec

        return Aec(_AEC_FRAME_SAMPLES, _AEC_FILTER_SAMPLES, SAMPLE_RATE, False)
    except Exception as exc:
        raise MicStreamError(f"echo cancellation is unavailable: {exc}") from exc


def close_mic_stream(process: subprocess.Popen) -> None:
    """Terminate `process` and block until it has actually exited (or force
    it to, past `_TERMINATE_TIMEOUT_SECONDS`).

    `voice_wake.py` and `voice_stt.py` both used to call `process.terminate()`
    alone in their `finally` blocks and return immediately -- but a SIGTERM'd
    `arecord` doesn't release the ALSA/Pulse device the instant the signal is
    sent, especially crossing WSLg's ALSA-to-Pulse bridge. The very next audio
    open (a cue or `voice_tts.speak`'s `aplay`, or another `arecord` for the
    next turn) could then race that teardown -- live-tested against a real
    detailed-recap follow-up (2026-09-21): the log showed `aplay` starting
    right after an `arecord` was "Aborted by signal Terminated", immediately
    followed by a tens-of-seconds ALSA underrun, consistent with the playback
    open stalling on a mic device that hadn't finished releasing. Waiting
    here makes `listen_for_wake_word`/`record_utterance` block until the mic
    device is actually free before their caller can open anything else.

    A bare `wait()` (no timeout) turned out not to be safe, though: a live
    barge-in run (2026-09-23) showed `arecord` printing its "Aborted by
    signal Terminated" line and then never finishing its own teardown (no
    follow-up `pcm_read` error line, no actual exit) -- `wait()` blocked
    forever, which silently hung the entire voice loop with no exception
    and nothing further printed, mimicking a dead mic from the outside.
    SIGKILL after `_TERMINATE_TIMEOUT_SECONDS` bounds that.
    """
    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def read_frame(process: subprocess.Popen) -> np.ndarray:
    """Read one `FRAME_SAMPLES`-sample frame of the first channel from
    `process`'s stdout. The other channels are discarded, except that the
    second is subtracted from the first when the stream has an echo canceller.
    """
    channels = getattr(process, "mic_channels", 1)
    chunk_bytes = CHUNK_BYTES * channels
    raw = process.stdout.read(chunk_bytes)
    if len(raw) < chunk_bytes:
        raise MicStreamError("arecord stream ended unexpectedly")
    samples = np.frombuffer(raw, dtype=np.int16)
    mic = samples[::channels] if channels > 1 else samples
    canceller = getattr(process, "echo_canceller", None)
    if canceller is None:
        return mic
    reference = samples[1::channels]
    cleaned = []
    for start in range(0, len(mic), _AEC_FRAME_SAMPLES):
        end = start + _AEC_FRAME_SAMPLES
        cleaned += canceller.cancel_echo(
            mic[start:end].tolist(), reference[start:end].tolist()
        )
    return np.array(cleaned, dtype=np.int16)


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
