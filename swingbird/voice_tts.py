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


class TTSError(Exception):
    """Raised when a Piper voice model can't be loaded or `aplay` fails."""


def _voice_model_path(voice_name: str, models_dir: Path) -> Path:
    path = models_dir / f"{voice_name}.onnx"
    if not path.is_file():
        raise TTSError(
            f"Piper voice model not found: {path} -- download it first, e.g. "
            f"`uv run python -m piper.download_voices "
            f"--download-dir {models_dir} {voice_name}`"
        )
    return path


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
    """
    voice = PiperVoice.load(str(_voice_model_path(tts.voice, models_dir)))
    device = _ALSA_DEVICE_BY_OUTPUT_TYPE[output.type]

    try:
        play = subprocess.Popen(
            [
                "aplay",
                "-D",
                device,
                "-r",
                str(voice.config.sample_rate),
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

    for chunk in voice.synthesize(text):
        play.stdin.write(chunk.audio_int16_bytes)
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
