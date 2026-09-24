"""Short audio cues at voice-turn listening boundaries, per Voidious's
request (swingbird-dev, 2026-09-20): a light chime confirms the mic just
opened -- right after the wake word registers, or right after a reply
finishes while the open-mic follow-up window (§V.11) is still listening --
so the user doesn't have to guess whether it's safe to start talking. A
second, lower chime marks the moment listening actually stops and the wake
word will be required again.

Synthesized as an in-memory sine tone rather than a bundled audio asset --
no new dependency (numpy is already pulled in via `voice_audio.py`, `aplay`
via `voice_tts.py`) and nothing to ship or gitignore. Voidious asked for
these to be "very light and short", hence the low amplitude and ~80ms
duration -- tune the constants below directly once his live listening
feedback comes in; no config knob yet since the exact feel isn't settled.
"""

from __future__ import annotations

import subprocess

import numpy as np

from swingbird.config import VoiceOutputConfig

SAMPLE_RATE = 16000
_DURATION_SECONDS = 0.08
# Quiet by design -- a startle-volume beep defeats "light". Both cues share
# this one constant, so they're always equally loud as each other; lowered
# 0.2 -> 0.15 (~25%) per Voidious's live-listening feedback (2026-09-21)
# that both the wake-word-ack and follow-up-timeout cues felt too loud --
# the "go ahead" cue after a reply already used this exact same tone/
# amplitude, so there was nothing to align it *to*, just an across-the-
# board volume cut.
_AMPLITUDE = 0.15
_FADE_SECONDS = 0.005  # avoids an audible click at tone start/end

# A rising tone reads as "go ahead"; the stop cue is a perfect fifth lower,
# not just quieter or shorter, so the two are trivially distinguishable
# without paying close attention.
LISTENING_STARTED_HZ = 880.0
LISTENING_STOPPED_HZ = 587.0


class CueError(Exception):
    """Raised when `aplay` can't play a cue tone."""


def _tone(frequency: float) -> bytes:
    sample_count = int(SAMPLE_RATE * _DURATION_SECONDS)
    t = np.arange(sample_count) / SAMPLE_RATE
    wave = _AMPLITUDE * np.sin(2 * np.pi * frequency * t)

    fade_samples = min(int(SAMPLE_RATE * _FADE_SECONDS), sample_count // 2)
    envelope = np.ones(sample_count)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    return (wave * envelope * 32767).astype(np.int16).tobytes()


def _play(pcm: bytes, output: VoiceOutputConfig) -> None:
    device = output.device
    try:
        play = subprocess.Popen(
            [
                "aplay",
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
            stdin=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CueError("aplay not found on PATH (install alsa-utils)") from exc
    play.stdin.write(pcm)
    play.stdin.close()
    play.wait()
    if play.returncode != 0:
        raise CueError(f"aplay exited with code {play.returncode}")


def play_listening_started(output: VoiceOutputConfig) -> None:
    """Play right after the wake word registers, or after finishing a
    reply while the follow-up window (§V.11) is still open -- signals it's
    safe to start talking.
    """
    _play(_tone(LISTENING_STARTED_HZ), output)


def play_listening_stopped(output: VoiceOutputConfig) -> None:
    """Play once the follow-up window (§V.11) elapses with nothing said --
    signals the wake word is required again before the next command.
    """
    _play(_tone(LISTENING_STOPPED_HZ), output)
