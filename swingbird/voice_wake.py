"""openWakeWord wake-word listening (Voice Mode design doc §V.5, §V.16 step 3).

STT and wiring a transcript into `_process` are later build-order steps
(§V.16 steps 4-5) -- this module is deliberately detection-only, proving
the wake-word listener fires reliably before anything downstream of it
exists. Nothing in the daemon's own event loop calls `listen_for_wake_word`
yet; this is reachable today only via this module's own `__main__`
smoke-test CLI.

Unlike Piper's TTS voices (`voice_tts.py`), openWakeWord's pretrained
models ship inside the `openwakeword` package itself -- no download step,
no models directory. Only openWakeWord's own pretrained phrases are
supported for now ("alexa", "hey_mycroft", "hey_jarvis", etc.) -- per
Voidious, that's a deliberately temporary/testing choice (§V.13), not a
final answer. Training a custom "swingbird" model (or other custom
phrases) is tracked as separate follow-up work, not built here; this
module's job is just to make `[voice].wake_word` a real, working config
knob against whichever pretrained models exist today, so pointing it at a
custom model later is a one-line change once that model exists.

Mic capture uses `arecord`, mirroring `voice_tts.py`'s choice of `aplay`
for playback -- one less native audio dependency to get working on both
WSL and the Orange Pi (§V.14), via the same ALSA/Pulse bridge story.
"""

from __future__ import annotations

import argparse
import subprocess

import numpy as np
import openwakeword
from openwakeword.model import Model

from swingbird.config import VoiceMicConfig, load_config

# openWakeWord's own frame size: predict() wants multiples of 80ms (1280
# samples) at 16kHz mono -- see Model.predict's docstring.
FRAME_SAMPLES = 1280
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # S16_LE

# A single-frame threshold, no consecutive-frame patience -- keeping this
# simple until step 3's live-room testing (§V.16) shows whether false
# positives need the debounce/patience knobs `Model.predict` supports.
DETECTION_THRESHOLD = 0.5

# Mirrors voice_tts.py's `_ALSA_DEVICE_BY_OUTPUT_TYPE`: "onboard" is the
# ALSA/Pulse default input (the WSL dev machine today, the Orange Pi's
# onboard mic later); "usb" is the reSpeaker XVF3800 array, a placeholder
# device name now so wiring in the real one later is a one-line change.
_ALSA_DEVICE_BY_MIC_TYPE = {
    "onboard": "default",
    "usb": "usb",
}


class WakeWordError(Exception):
    """Raised when `wake_word` has no pretrained model, or `arecord` fails."""


def _pretrained_model_path(wake_word: str) -> str:
    try:
        return openwakeword.models[wake_word]["model_path"]
    except KeyError as exc:
        available = ", ".join(sorted(openwakeword.models))
        raise WakeWordError(
            f"no pretrained openWakeWord model for wake_word {wake_word!r} -- "
            f"available pretrained models: {available} (a custom model, e.g. "
            'for "swingbird" itself, is tracked as separate follow-up work)'
        ) from exc


def load_model(wake_word: str) -> tuple[Model, str]:
    """Load the pretrained openWakeWord model for `wake_word`.

    Returns the model and the score key `predict()` returns it under.
    openWakeWord derives that key from the model file's own name (e.g.
    "hey_jarvis_v0.1", not the plain "hey_jarvis" config value), so
    callers need both rather than reconstructing the key themselves.
    """
    model = Model(wakeword_model_paths=[_pretrained_model_path(wake_word)])
    score_key = next(iter(model.models))
    return model, score_key


def listen_for_wake_word(wake_word: str, mic: VoiceMicConfig) -> None:
    """Block until `wake_word` is detected once on `mic`'s device.

    Blocking and synchronous on purpose -- nothing calls this from an
    event loop yet (see module docstring), so there's no responsiveness
    constraint to design around before §V.16 step 5 exists.
    """
    model, score_key = load_model(wake_word)
    device = _ALSA_DEVICE_BY_MIC_TYPE[mic.type]
    chunk_bytes = FRAME_SAMPLES * BYTES_PER_SAMPLE

    try:
        record = subprocess.Popen(
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
        raise WakeWordError("arecord not found on PATH (install alsa-utils)") from exc

    try:
        while True:
            raw = record.stdout.read(chunk_bytes)
            if len(raw) < chunk_bytes:
                raise WakeWordError("arecord stream ended unexpectedly")
            frame = np.frombuffer(raw, dtype=np.int16)
            if model.predict(frame)[score_key] >= DETECTION_THRESHOLD:
                return
    finally:
        record.terminate()


def _main() -> None:  # pragma: no cover -- manual smoke test, see §V.16 step 3
    parser = argparse.ArgumentParser(
        description="Listen for the configured wake word, for manual smoke testing."
    )
    parser.add_argument("--config", default="swingbird.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Listening for wake word {config.voice.wake_word!r}...")
    listen_for_wake_word(config.voice.wake_word, config.voice.mic)
    print("Wake word detected!")


if __name__ == "__main__":
    _main()
