"""openWakeWord wake-word listening (Voice Mode design doc §V.5, §V.16 step 3).

STT and wiring a transcript into `_process` are later build-order steps
(§V.16 steps 4-5) -- this module is deliberately detection-only, proving
the wake-word listener fires reliably before anything downstream of it
exists. `daemon.py`'s `_run_voice_turn` now calls `listen_for_wake_word`
to start each turn (§V.16 step 5); this module's own `__main__` is still
there as a standalone smoke-test CLI.

`[voice].wake_word` names either one of swingbird's own custom-trained
models -- a `<wake_word>.onnx` file checked in under
`voice_models/wake_words/` (§V.13) -- or, as a fallback, one of
openWakeWord's pretrained phrases ("alexa", "hey_mycroft", "hey_jarvis",
etc.), which ship inside the `openwakeword` package itself with no
download step. The custom models are the intended answer; the pretrained
ones are kept for development/debugging for now. A checked-in model wins
if it shares a name with a pretrained one, so swingbird's own is never
silently shadowed by the upstream default.

Mic capture is `voice_audio.py`'s -- shared with `voice_stt.py` since both
read the same `arecord` stream off the same device.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import openwakeword
from openwakeword.model import Model

from swingbird.config import VoiceMicConfig, load_config
from swingbird.voice_audio import (
    call_translating_stream_error,
    close_mic_stream,
    open_mic_stream,
    read_frame,
)

# A single-frame threshold, no consecutive-frame patience -- keeping this
# simple until step 3's live-room testing (§V.16) shows whether false
# positives need the debounce/patience knobs `Model.predict` supports.
DETECTION_THRESHOLD = 0.5

# Checked in, unlike `voice_tts.DEFAULT_MODELS_DIR`'s downloaded (gitignored)
# Piper voices -- these are swingbird's own trained models, small enough
# (a few hundred KB) to live in the repo. Relative to the working
# directory, same as `voice_tts.py`'s models dir.
WAKE_WORD_MODELS_DIR = Path("voice_models/wake_words")


class WakeWordError(Exception):
    """Raised when `wake_word` has no model, or `arecord` fails."""


def _model_path(wake_word: str, models_dir: Path) -> str:
    custom = models_dir / f"{wake_word}.onnx"
    if custom.is_file():
        return str(custom)
    try:
        return openwakeword.models[wake_word]["model_path"]
    except KeyError as exc:
        custom_names = sorted(p.stem for p in models_dir.glob("*.onnx"))
        available = ", ".join(custom_names + sorted(openwakeword.models))
        raise WakeWordError(
            f"no wake word model for wake_word {wake_word!r} -- no "
            f"{custom.as_posix()} and no pretrained openWakeWord model of "
            f"that name; available: {available}"
        ) from exc


def load_model(
    wake_word: str, models_dir: Path = WAKE_WORD_MODELS_DIR
) -> tuple[Model, str]:
    """Load the model for `wake_word` -- a custom `<wake_word>.onnx` under
    `models_dir` if one exists, else openWakeWord's pretrained one.

    Returns the model and the score key `predict()` returns it under.
    openWakeWord derives that key from the model file's own name (e.g.
    "hey_jarvis_v0.1", not the plain "hey_jarvis" config value), so
    callers need both rather than reconstructing the key themselves.
    """
    model = Model(wakeword_model_paths=[_model_path(wake_word, models_dir)])
    score_key = next(iter(model.models))
    return model, score_key


def listen_for_wake_word(
    wake_word: str,
    mic: VoiceMicConfig,
    stop_event: threading.Event | None = None,
    threshold: float = DETECTION_THRESHOLD,
) -> bool:
    """Block until `wake_word` scores at least `threshold` once on `mic`'s
    device, or `stop_event` is set first. Returns `True` for the former, `False` for
    the latter.

    `stop_event` (Voice Mode design doc §V.9), when given, lets
    `daemon.py`'s `_wait_for_wake_word` interrupt an otherwise-idle wait
    the moment a proactive agent-reply summary needs speaking, rather than
    making it sit through the rest of this call's own indefinite wait
    first -- checked once per frame (~80ms), the same cadence
    `voice_barge_in`'s own VAD polling already runs at, so the extra
    latency to notice it's set is negligible. `None` (the default)
    preserves the original "block forever until the wake word fires"
    behavior for callers -- standalone smoke testing (`_main`, below) --
    that have no reason to interrupt it.

    Blocking and synchronous on purpose -- nothing calls this from an
    event loop directly (see module docstring); `daemon.py` always runs it
    via `asyncio.to_thread`, so there's no responsiveness constraint to
    design around inside this function itself.
    """
    model, score_key = load_model(wake_word)

    record = call_translating_stream_error(WakeWordError, open_mic_stream, mic)

    try:
        while stop_event is None or not stop_event.is_set():
            frame = call_translating_stream_error(WakeWordError, read_frame, record)
            if model.predict(frame)[score_key] >= threshold:
                return True
        return False
    finally:
        close_mic_stream(record)


def _main() -> None:  # pragma: no cover -- manual smoke test, see §V.16 step 3
    parser = argparse.ArgumentParser(
        description="Listen for the configured wake word, for manual smoke testing."
    )
    parser.add_argument("--config", default="swingbird.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Listening for wake word {config.voice.wake_word!r}...")
    listen_for_wake_word(
        config.voice.wake_word, config.voice.mic, threshold=config.voice.wake_threshold
    )
    print("Wake word detected!")


if __name__ == "__main__":
    _main()
