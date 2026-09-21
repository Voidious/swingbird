"""Piper text-to-speech playback (Voice Mode design doc §V.5, §V.16 step 2).

Wake word, VAD, and STT are later build-order steps (§V.16 steps 3-5) --
this module is deliberately TTS-only, proving Piper's own output pipeline
end to end on real audio hardware before anything upstream of it exists.
`daemon.py`'s `_run_voice_turn` calls `speak` to speak each turn's reply
(§V.16 step 6); this module's own `__main__` is still there as a
standalone smoke-test CLI.

Piper synthesizes raw PCM directly, so there's no on-disk .wav step;
`aplay` plays it. Subprocess rather than a Python audio library
(sounddevice/PyAudio) because it's one less native dependency to get
working on both WSL (prototyping, x86_64) and the Orange Pi (aarch64,
§V.14) -- `aplay` ships with `alsa-utils` on both, and an ALSA-to-Pulse
bridge (WSLg's own on the dev machine; whatever the Orange Pi's audio
stack uses later) makes "the default device" just work in either
environment without this module knowing which is which.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from piper import PiperVoice
from piper.download_voices import download_voice

from swingbird.config import VoiceOutputConfig, VoiceTTSConfig, load_config

DEFAULT_MODELS_DIR = Path("voice_models")

# §V.6's output.type is a config-selected device, not just documentation:
# VoiceOutputConfig's docstring calls both "onboard" and "usb" permanent
# code paths, even though only "onboard" (the ALSA/Pulse default device)
# is exercised for real before the Orange Pi + XVF3800 arrives (§V.14).
# "usb" is a distinct placeholder now so wiring in the real device name
# later is a one-line change here, not a reshape of this module's
# interface.
_ALSA_DEVICE_BY_OUTPUT_TYPE = {
    "onboard": "default",
    "usb": "usb",
}

# A brief silence between recap items/paragraphs reads as more natural at
# the pace this codebase otherwise keeps -- Voidious asked for "even like
# .2 or .3 seconds" (2026-09-21) after finding back-to-back items in a
# detailed recap ran together with almost no pause. Split on the same
# blank-line ("\n\n") paragraph boundary recap.py's own prompts already use
# to separate items, so this needs no new text markup.
_PARAGRAPH_PAUSE_SECONDS = 0.25


class TTSError(Exception):
    """Raised when a Piper voice model can't be loaded or `aplay` fails."""


def _voice_model_path(voice_name: str, models_dir: Path) -> Path:
    """Return `models_dir`'s `.onnx` file for `voice_name`, fetching it (and
    its sidecar `.onnx.json`) from Piper's own voice repository first if it
    isn't already there -- same first-use-fetches-and-caches shape as
    `voice_stt.py`'s faster-whisper weights, rather than requiring a manual
    `piper.download_voices` run before the first real voice turn.
    """
    path = models_dir / f"{voice_name}.onnx"
    if not path.is_file():
        models_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_voice(voice_name, models_dir)
        except (ValueError, OSError) as exc:
            raise TTSError(
                f"couldn't download Piper voice {voice_name!r}: {exc}"
            ) from exc
    return path


def _silence_pcm(duration_seconds: float, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(sample_rate * duration_seconds)


def speak(
    text: str,
    tts: VoiceTTSConfig,
    output: VoiceOutputConfig,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> None:
    """Synthesize `text` with Piper and play it on `output`'s device.

    Blocking and synchronous on purpose -- nothing calls this from an
    event loop yet (see module docstring), so there's no responsiveness
    constraint to design around before §V.16 step 5 exists.

    Synthesizes the whole of `text` into one in-memory PCM buffer before
    starting `aplay`, rather than streaming Piper's chunks straight into
    its stdin as they're produced. Live-tested against a real detailed
    recap reply (2026-09-21): streaming synthesis fell behind real-time
    partway through a long reply, and each `stdin.write` call blocked on
    `aplay` draining a pipe synthesis wasn't refilling fast enough --
    audible as speech breaking into progressively longer silent gaps
    between shrinking blips, never finishing cleanly. Buffering first
    fully decouples synthesis speed from playback timing, at the cost of
    a longer delay before playback starts on a long reply.

    `text` is split into paragraphs on blank lines (the same "\\n\\n"
    boundary recap.py's own prompts use to separate items) and a short
    `_PARAGRAPH_PAUSE_SECONDS` silence is inserted between them, so
    consecutive recap items don't run together with no breathing room.
    """
    voice = PiperVoice.load(str(_voice_model_path(tts.voice, models_dir)))
    device = _ALSA_DEVICE_BY_OUTPUT_TYPE[output.type]
    sample_rate = voice.config.sample_rate

    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
    silence = _silence_pcm(_PARAGRAPH_PAUSE_SECONDS, sample_rate)
    audio_parts = []
    for index, paragraph in enumerate(paragraphs):
        for chunk in voice.synthesize(paragraph):
            audio_parts.append(chunk.audio_int16_bytes)
        if index < len(paragraphs) - 1:
            audio_parts.append(silence)
    audio = b"".join(audio_parts)

    try:
        play = subprocess.Popen(
            [
                "aplay",
                "-D",
                device,
                "-r",
                str(sample_rate),
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
        raise TTSError("aplay not found on PATH (install alsa-utils)") from exc

    play.stdin.write(audio)
    play.stdin.close()
    play.wait()
    if play.returncode != 0:
        raise TTSError(f"aplay exited with code {play.returncode}")


def _main() -> None:  # pragma: no cover -- manual smoke test, see §V.16 step 2
    parser = argparse.ArgumentParser(
        description="Speak text aloud via Piper, for manual smoke testing."
    )
    parser.add_argument("text", help="Text for swingbird to speak aloud")
    parser.add_argument("--config", default="swingbird.toml")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    args = parser.parse_args()

    config = load_config(args.config)
    if config.voice.tts is None:
        raise SystemExit(
            "no [voice.tts] section in config -- add one with a `voice` "
            "field naming a downloaded Piper voice"
        )
    speak(args.text, config.voice.tts, config.voice.output, Path(args.models_dir))


if __name__ == "__main__":
    _main()
