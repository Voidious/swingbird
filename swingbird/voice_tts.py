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
import re
import subprocess
import threading
import time
from pathlib import Path

from piper import PiperVoice
from piper.download_voices import download_voice

from swingbird.config import VoiceOutputConfig, VoiceTTSConfig, load_config

DEFAULT_MODELS_DIR = Path("voice_models")

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

# Splitting playback per-paragraph (see `speak`'s docstring) only bounds
# continuous stream length when paragraphs happen to be short -- a single
# `recap_detail` elaboration is one long paragraph with no "\n\n" inside it
# at all, and still underran audibly (2026-09-21) even after that split.
# Sentences within a paragraph are grouped into playback chunks capped at
# this many seconds of audio each, so no single `aplay` invocation has to
# sustain a stream longer than this regardless of paragraph structure.
_MAX_CONTINUOUS_SECONDS = 8.0
_BYTES_PER_SAMPLE = 2  # S16_LE mono

# A duration-capped elaboration *still* underran audibly (2026-09-21) after
# the chunk split above, with no ALSA-reported underrun or mic-teardown race
# in the log this time -- meaning the earlier `close_mic_stream` fix (a
# `terminate()` not implying the device is actually free yet over WSLg's
# ALSA-to-Pulse bridge) has a same-shaped sibling here: chunk boundaries
# open a fresh `aplay` back-to-back with zero gap, since `_play`'s own
# `wait()` only confirms the *process* exited, not that the bridge has
# released the device for the next `Popen` to reacquire cleanly. A small
# real gap between chunks of the same paragraph gives the bridge time to
# settle between rapid same-paragraph reopens, short enough to preserve the
# "no pause within a paragraph" pacing `_PARAGRAPH_PAUSE_SECONDS` is for.
# On real Orange Pi hardware (2026-09-22), 0.05s no longer needed to be that
# short -- there's no more underrun to avoid dwelling in, and Voidious heard
# the boundary as an awkward clip ("like commas from within the previous
# sentence") rather than a clean pause. Widened to read as an actual, if
# brief, break.
_CHUNK_PAUSE_SECONDS = 0.15

# `_play` writes a chunk's audio to `aplay`'s stdin in slices this long,
# rather than one call for the whole chunk, so a `stop_event` (Voice Mode
# design doc §V.12's barge-in) set mid-chunk stops playback within one
# slice instead of only at the next chunk/paragraph boundary -- up to
# `_MAX_CONTINUOUS_SECONDS` (8s) of latency otherwise. Same order of
# magnitude as `_ALSA_PERIOD_TIME_MICROSECONDS` (0.1s) so this doesn't
# change how `aplay` itself is fed relative to what its own buffering
# already assumes.
_STOP_POLL_SECONDS = 0.1

# Simple and imperfect -- splits after `.`/`!`/`?` followed by whitespace,
# so an abbreviation like "Mr." would also split. Matches this codebase's
# existing rule-based approach to spoken text (see `voice_render.py`'s own
# docstring) rather than a general-purpose sentence tokenizer: this
# codebase's own LLM-generated recap/elaboration text doesn't use
# abbreviations like that, and a rare miss here just means one chunk plays
# a little longer, not an error.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


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


def _play(
    audio: bytes,
    sample_rate: int,
    device: str,
    stop_event: threading.Event | None = None,
) -> None:
    """Play one already-synthesized PCM buffer through its own `aplay`
    invocation -- see `speak`'s docstring for why `speak` calls this once
    per paragraph rather than once for a whole reply.

    Writes `audio` to `aplay`'s stdin in `_STOP_POLL_SECONDS`-sized slices
    rather than one call, checking `stop_event` before each one -- when
    it's set (Voice Mode design doc §V.12's barge-in), `aplay` is
    `terminate()`d immediately rather than left to drain the rest of the
    buffer, so an interrupting utterance doesn't have to wait out however
    much of this chunk was left. `stop_event=None` (every caller before
    §V.12) never checks, so playback always runs to completion exactly as
    before -- just written in the same slices either way, which doesn't
    change what's heard.
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

    slice_bytes = max(1, int(_STOP_POLL_SECONDS * sample_rate)) * _BYTES_PER_SAMPLE
    try:
        for offset in range(0, len(audio), slice_bytes):
            if stop_event is not None and stop_event.is_set():
                play.terminate()
                return
            play.stdin.write(audio[offset : offset + slice_bytes])
    finally:
        play.stdin.close()
        play.wait()

    if play.returncode != 0:
        raise TTSError(f"aplay exited with code {play.returncode}")


def _split_sentences(paragraph: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY_RE.split(paragraph) if s.strip()]


def _synthesize_chunks(voice, paragraph: str, sample_rate: int) -> list[bytes]:
    """Synthesize `paragraph` sentence-by-sentence, grouping consecutive
    sentences into playback chunks capped at `_MAX_CONTINUOUS_SECONDS` of
    audio each -- see the constant's own comment for why. A paragraph with
    no sentence-ending punctuation (or one short enough to fit under the
    cap already) comes back as a single chunk, identical to synthesizing
    it whole.
    """
    max_bytes = int(_MAX_CONTINUOUS_SECONDS * sample_rate) * _BYTES_PER_SAMPLE
    chunks = []
    current = bytearray()
    for sentence in _split_sentences(paragraph):
        audio = b"".join(
            chunk.audio_int16_bytes for chunk in voice.synthesize(sentence)
        )
        if current and len(current) + len(audio) > max_bytes:
            chunks.append(bytes(current))
            current = bytearray()
        current.extend(audio)
    if current:
        chunks.append(bytes(current))
    return chunks


def speak(
    text: str,
    tts: VoiceTTSConfig,
    output: VoiceOutputConfig,
    models_dir: Path = DEFAULT_MODELS_DIR,
    stop_event: threading.Event | None = None,
    start_chunk: int = 0,
) -> int | None:
    """Synthesize `text` with Piper and play it on `output`'s device.
    Returns the index of the chunk playback stopped on if `stop_event` cut
    it short, or `None` if it played through to the end.

    `stop_event`, when given (Voice Mode design doc §V.12's barge-in),
    lets a caller -- `voice_barge_in.speak_with_barge_in`, monitoring the
    mic concurrently -- stop an in-progress reply the instant the user
    starts talking over it, rather than waiting for the current chunk, let
    alone the whole reply, to finish. Checked before playing each chunk
    (skipping it entirely if already set) and passed into `_play` itself
    so a chunk already mid-playback stops within one
    `_STOP_POLL_SECONDS` slice; either way, once set, `speak` returns
    without playing any later chunk/paragraph or sleeping between them.

    `start_chunk`/the return value (Voice Mode design doc §V.12): every
    chunk across the whole reply is numbered in one flat sequence,
    regardless of which paragraph it's in, and `start_chunk` skips
    straight to that index without (re-)playing anything before it.
    Resuming a reply after a false barge-in trigger used to always restart
    `text` from the very beginning -- for a long reply, that meant
    re-hearing everything already heard before the interruption (Voidious,
    2026-09-23). `voice_barge_in.speak_with_barge_in` instead passes back
    in whatever index this returned, so a resume picks up from the chunk
    that got cut off, not chunk zero. `text` is still synthesized in full
    on a resume (see below for why up-front synthesis already happens
    regardless) -- only *playback* is skipped ahead.

    Blocking and synchronous on purpose -- `daemon.py`'s `_run_voice_turn`
    runs it via `asyncio.to_thread` rather than awaiting it directly (see
    module docstring), so there's no responsiveness constraint to design
    around inside this function itself.

    `text` is split into paragraphs on blank lines (the same "\\n\\n"
    boundary recap.py's own prompts use to separate items), and each
    paragraph is fully synthesized and played through its *own* `aplay`
    invocation -- one continuous ALSA stream per paragraph, not one for
    the whole reply -- with a `time.sleep(_PARAGRAPH_PAUSE_SECONDS)` gap
    between them.

    This module's history is five rounds of live-tested fixes against the
    same symptom (2026-09-21), each of which helped without fully curing
    it: (1) buffering a paragraph's whole synthesis before starting
    `aplay`, instead of streaming Piper's chunks straight into its stdin
    as they were produced, after a long reply broke up when synthesis fell
    behind real-time and `stdin.write` blocked on a pipe nothing was
    draining; (2) passing explicit `--buffer-time`/`--period-time` to
    `aplay`, after a fully-buffered but single long continuous stream
    still underran audibly, which pure buffering doesn't fix since that's
    ALSA's own default sizing being too tight for WSLg's ALSA-to-Pulse
    bridge; (3) waiting for the prior turn's `arecord` to fully exit
    (`voice_audio.close_mic_stream`) before opening any new device, after
    a mic-to-speaker handoff race showed up in one repro's logs as a
    multi-second underrun right after an "Aborted by signal Terminated"
    line; (4) playing each paragraph through its own `aplay` invocation
    instead of one for the whole reply, after a 5-item reply kept breaking
    up consistently later as items got shorter/fewer -- which fixed that
    5-item reply, but a single `recap_detail` elaboration (one long
    paragraph, no "\n\n" inside it at all) still underran just as badly,
    proving the failure tracks elapsed *continuous stream* duration, not
    paragraph count; (5) generalizing (4): each paragraph's sentences are
    grouped into `_synthesize_chunks` chunks capped at
    `_MAX_CONTINUOUS_SECONDS`, each played through its own `aplay` call --
    which still underran audibly on a long elaboration, but this time with
    *no* ALSA underrun message and no mic-teardown race in the log,
    pointing at the same speaker-to-speaker reopen race (3) fixed for
    mic-to-speaker, since chunks played back-to-back with no gap at all.
    This step adds a small `time.sleep(_CHUNK_PAUSE_SECONDS)` between
    chunks of the *same* paragraph (distinct from, and much shorter than,
    `_PARAGRAPH_PAUSE_SECONDS` between different paragraphs) so the bridge
    gets a moment to settle between rapid reopens without adding an
    audible gap mid-sentence.

    On real Orange Pi hardware (2026-09-22) the breakup this history fixed
    was gone, but the *silence* between paragraphs read as much longer than
    `_PARAGRAPH_PAUSE_SECONDS` (0.25s) -- because every paragraph used to be
    synthesized right before it played, the actual gap a listener heard was
    the sleep *plus* however long Piper took to synthesize the next
    paragraph's audio, not the sleep alone. All paragraphs are synthesized
    up front, before any of them play, so the only thing left between
    paragraphs at playback time is the intended sleep. Trade-off: a long,
    multi-paragraph reply now waits for the *whole* reply to synthesize
    before speaking its first word, instead of only its first paragraph --
    acceptable here since `_process`'s own LLM round trip already leaves a
    silent gap before speech starts at all.
    """
    voice = PiperVoice.load(str(_voice_model_path(tts.voice, models_dir)))
    device = output.device
    sample_rate = voice.config.sample_rate

    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
    paragraph_chunks = [
        _synthesize_chunks(voice, paragraph, sample_rate) for paragraph in paragraphs
    ]
    # Flattened into one (audio, pause-after) sequence, numbered start to
    # finish across paragraph boundaries, so `start_chunk`/the return value
    # can address "how far into this whole reply" with a single index
    # rather than a (paragraph, chunk) pair.
    chunks: list[tuple[bytes, float]] = []
    for index, para_chunks in enumerate(paragraph_chunks):
        for chunk_index, chunk_audio in enumerate(para_chunks):
            if chunk_index < len(para_chunks) - 1:
                pause = _CHUNK_PAUSE_SECONDS
            elif index < len(paragraph_chunks) - 1:
                pause = _PARAGRAPH_PAUSE_SECONDS
            else:
                pause = 0.0
            chunks.append((chunk_audio, pause))

    for chunk_index in range(start_chunk, len(chunks)):
        if stop_event is not None and stop_event.is_set():
            return chunk_index
        chunk_audio, pause = chunks[chunk_index]
        _play(chunk_audio, sample_rate, device, stop_event)
        if stop_event is not None and stop_event.is_set():
            return chunk_index
        if pause:
            time.sleep(pause)
    return None


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
