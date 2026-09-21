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
import time
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
# to separate items, so this needs no new text markup. A real wall-clock
# `time.sleep` between separate `aplay` invocations now (see `speak`'s
# docstring for why), not PCM silence appended inside one continuous
# stream -- same audible gap either way.
_PARAGRAPH_PAUSE_SECONDS = 0.25

# aplay's own ALSA default buffer/period sizing underran audibly on a long
# continuous stream over WSLg's ALSA-to-Pulse bridge -- see `speak`'s
# docstring. Both in microseconds, aplay's own `--buffer-time`/
# `--period-time` unit.
_ALSA_BUFFER_TIME_MICROSECONDS = 500_000
_ALSA_PERIOD_TIME_MICROSECONDS = 100_000


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


def _play(audio: bytes, sample_rate: int, device: str) -> None:
    """Play one already-synthesized PCM buffer through its own `aplay`
    invocation -- see `speak`'s docstring for why `speak` calls this once
    per paragraph rather than once for a whole reply.
    """
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
                "--buffer-time",
                str(_ALSA_BUFFER_TIME_MICROSECONDS),
                "--period-time",
                str(_ALSA_PERIOD_TIME_MICROSECONDS),
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

    `text` is split into paragraphs on blank lines (the same "\\n\\n"
    boundary recap.py's own prompts use to separate items), and each
    paragraph is fully synthesized and played through its *own* `aplay`
    invocation -- one continuous ALSA stream per paragraph, not one for
    the whole reply -- with a `time.sleep(_PARAGRAPH_PAUSE_SECONDS)` gap
    between them.

    This module's history is three rounds of live-tested fixes against
    the same symptom (2026-09-21), each of which helped without fully
    curing it: (1) buffering a paragraph's whole synthesis before
    starting `aplay`, instead of streaming Piper's chunks straight into
    its stdin as they were produced, after a long reply broke up when
    synthesis fell behind real-time and `stdin.write` blocked on a pipe
    nothing was draining; (2) passing explicit `--buffer-time`/
    `--period-time` to `aplay`, after a fully-buffered but single long
    continuous stream still underran audibly, which pure buffering
    doesn't fix since that's ALSA's own default sizing being too tight
    for WSLg's ALSA-to-Pulse bridge; (3) waiting for the prior turn's
    `arecord` to fully exit (`voice_audio.close_mic_stream`) before
    opening any new device, after a mic-to-speaker handoff race showed up
    in one repro's logs as a multi-second underrun right after an
    "Aborted by signal Terminated" line. A 5-item reply still broke up
    afterwards, now consistently *later* into the reply as items got
    shorter/fewer (item 4 of 5, then item 3 of 5) rather than at any
    particular content -- i.e. the failure tracks elapsed continuous
    playback time on this bridge, not anything about a specific
    paragraph. Splitting playback into one `aplay` per paragraph bounds
    how long any single continuous stream has to survive to the length
    of one item, using the pause between items (already wanted for
    pacing) as a natural point to close and reopen the device instead of
    holding it continuously for an entire multi-item reply.
    """
    voice = PiperVoice.load(str(_voice_model_path(tts.voice, models_dir)))
    device = _ALSA_DEVICE_BY_OUTPUT_TYPE[output.type]
    sample_rate = voice.config.sample_rate

    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
    for index, paragraph in enumerate(paragraphs):
        audio = b"".join(
            chunk.audio_int16_bytes for chunk in voice.synthesize(paragraph)
        )
        _play(audio, sample_rate, device)
        if index < len(paragraphs) - 1:
            time.sleep(_PARAGRAPH_PAUSE_SECONDS)


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
