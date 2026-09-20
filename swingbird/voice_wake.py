"""openWakeWord wake-word listening (Voice Mode design doc §V.5, §V.16 step 3).

STT and wiring a transcript into `_process` are later build-order steps
(§V.16 steps 4-5) -- this module is deliberately detection-only, proving
the wake-word listener fires reliably before anything downstream of it
exists. `daemon.py`'s `_run_voice_turn` now calls `listen_for_wake_word`
to start each turn (§V.16 step 5); this module's own `__main__` is still
there as a standalone smoke-test CLI.

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

Mic capture is `voice_audio.py`'s -- shared with `voice_stt.py` since both
read the same `arecord` stream off the same device.
"""

from __future__ import annotations

import argparse

import openwakeword
from openwakeword.model import Model

from swingbird.config import VoiceMicConfig, load_config
from swingbird.voice_audio import (
    call_translating_stream_error,
    open_mic_stream,
    read_frame,
)

# A single-frame threshold, no consecutive-frame patience -- keeping this
# simple until step 3's live-room testing (§V.16) shows whether false
# positives need the debounce/patience knobs `Model.predict` supports.
DETECTION_THRESHOLD = 0.5


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

    record = call_translating_stream_error(WakeWordError, open_mic_stream, mic)

    try:
        while True:
            frame = call_translating_stream_error(WakeWordError, read_frame, record)
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
