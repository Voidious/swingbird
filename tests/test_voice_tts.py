import pytest

from swingbird import voice_tts
from swingbird.config import VoiceOutputConfig, VoiceTTSConfig
from swingbird.voice_tts import TTSError, speak


class FakeChunk:
    def __init__(self, data: bytes):
        self.audio_int16_bytes = data


class FakeVoiceConfig:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate


class FakeVoice:
    def __init__(self, chunks, sample_rate=22050, chunks_by_text=None):
        self._chunks = chunks
        self._chunks_by_text = chunks_by_text
        self.config = FakeVoiceConfig(sample_rate)
        self.synthesize_calls = []

    def synthesize(self, text):
        self.synthesize_calls.append(text)
        if self._chunks_by_text is not None:
            return iter(self._chunks_by_text.get(text, []))
        return iter(self._chunks)


class FakeStdin:
    def __init__(self):
        self.written = bytearray()
        self.closed = False

    def write(self, data):
        self.written.extend(data)

    def close(self):
        self.closed = True


class FakePopen:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdin = FakeStdin()
        self.args = None

    def wait(self):
        pass


def _mock_piper_voice_and_popen(monkeypatch, fake_voice):
    load_calls = []
    monkeypatch.setattr(
        voice_tts.PiperVoice,
        "load",
        lambda path: load_calls.append(path) or fake_voice,
    )
    fake_popen = FakePopen(returncode=0)
    return load_calls, fake_popen


def test_split_sentences_splits_on_terminal_punctuation():
    assert voice_tts._split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_sentences_returns_whole_text_with_no_terminal_punctuation():
    assert voice_tts._split_sentences("no punctuation here") == ["no punctuation here"]


def test_speak_downloads_missing_model_then_loads_it(tmp_path, monkeypatch):
    models_dir = tmp_path / "voice_models"
    download_calls = []

    def fake_download_voice(voice_name, download_dir):
        download_calls.append((voice_name, download_dir))
        (download_dir / f"{voice_name}.onnx").write_bytes(b"")
        (download_dir / f"{voice_name}.onnx.json").write_bytes(b"{}")

    monkeypatch.setattr(voice_tts, "download_voice", fake_download_voice)
    fake_voice = FakeVoice([])
    load_calls, fake_popen = _mock_piper_voice_and_popen(monkeypatch, fake_voice)
    monkeypatch.setattr(
        voice_tts.subprocess, "Popen", lambda args, stdin=None: fake_popen
    )

    speak(
        "hi",
        VoiceTTSConfig(voice="missing-voice"),
        VoiceOutputConfig(),
        models_dir=models_dir,
    )

    assert download_calls == [("missing-voice", models_dir)]
    assert load_calls == [str(models_dir / "missing-voice.onnx")]


def test_speak_raises_when_model_download_fails(tmp_path, monkeypatch):
    def fake_download_voice(voice_name, download_dir):
        raise ValueError(f"Voice '{voice_name}' did not match pattern")

    monkeypatch.setattr(voice_tts, "download_voice", fake_download_voice)

    with pytest.raises(TTSError, match="couldn't download Piper voice"):
        speak(
            "missing-voice",
            VoiceTTSConfig(voice="missing-voice"),
            VoiceOutputConfig(),
            models_dir=tmp_path / "voice_models",
        )


def test_speak_happy_path_writes_all_chunks_to_onboard_device(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice([FakeChunk(b"abc"), FakeChunk(b"def")], sample_rate=22050)
    load_calls, fake_popen = _mock_piper_voice_and_popen(monkeypatch, fake_voice)

    def fake_ctor(args, stdin=None):
        fake_popen.args = args
        return fake_popen

    monkeypatch.setattr(voice_tts.subprocess, "Popen", fake_ctor)

    speak(
        "hello",
        VoiceTTSConfig(voice="test-voice"),
        VoiceOutputConfig(type="onboard"),
        models_dir=tmp_path,
    )

    assert load_calls == [str(tmp_path / "test-voice.onnx")]
    assert fake_popen.args == [
        "aplay",
        "-D",
        "default",
        "-r",
        "22050",
        "-f",
        "S16_LE",
        "-t",
        "raw",
        "-c",
        "1",
        "--buffer-time",
        str(voice_tts._ALSA_BUFFER_TIME_MICROSECONDS),
        "--period-time",
        str(voice_tts._ALSA_PERIOD_TIME_MICROSECONDS),
        "-",
    ]
    assert bytes(fake_popen.stdin.written) == b"abcdef"
    assert fake_popen.stdin.closed


def _mock_aplay_and_sleep(monkeypatch, fake_popens):
    remaining_popens = list(fake_popens)
    popen_calls = []
    monkeypatch.setattr(
        voice_tts.subprocess,
        "Popen",
        lambda args, stdin=None: (popen_calls.append(args), remaining_popens.pop(0))[1],
    )
    sleep_calls = []
    monkeypatch.setattr(voice_tts.time, "sleep", sleep_calls.append)
    return popen_calls, sleep_calls


def test_speak_plays_each_paragraph_through_its_own_aplay_with_a_sleep_between(
    tmp_path, monkeypatch
):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice(
        chunks=None,
        sample_rate=4,
        chunks_by_text={
            "first item": [FakeChunk(b"aa")],
            "second item": [FakeChunk(b"bb")],
        },
    )
    monkeypatch.setattr(voice_tts.PiperVoice, "load", lambda path: fake_voice)
    fake_popens = [FakePopen(returncode=0), FakePopen(returncode=0)]
    popen_calls, sleep_calls = _mock_aplay_and_sleep(monkeypatch, fake_popens)

    speak(
        "first item\n\nsecond item",
        VoiceTTSConfig(voice="test-voice"),
        VoiceOutputConfig(),
        models_dir=tmp_path,
    )

    assert fake_voice.synthesize_calls == ["first item", "second item"]
    assert len(popen_calls) == 2
    assert bytes(fake_popens[0].stdin.written) == b"aa"
    assert bytes(fake_popens[1].stdin.written) == b"bb"
    assert sleep_calls == [voice_tts._PARAGRAPH_PAUSE_SECONDS]


def test_speak_splits_a_long_paragraph_into_duration_capped_chunks(
    tmp_path, monkeypatch
):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice(
        chunks=None,
        sample_rate=1,
        chunks_by_text={
            "One.": [FakeChunk(b"aa")],
            "Two.": [FakeChunk(b"bb")],
            "Three.": [FakeChunk(b"cc")],
        },
    )
    monkeypatch.setattr(voice_tts.PiperVoice, "load", lambda path: fake_voice)
    monkeypatch.setattr(voice_tts, "_MAX_CONTINUOUS_SECONDS", 1.0)
    fake_popens = [FakePopen(returncode=0) for _ in range(3)]
    popen_calls, sleep_calls = _mock_aplay_and_sleep(monkeypatch, fake_popens)

    speak(
        "One. Two. Three.",
        VoiceTTSConfig(voice="test-voice"),
        VoiceOutputConfig(),
        models_dir=tmp_path,
    )

    assert fake_voice.synthesize_calls == ["One.", "Two.", "Three."]
    assert len(popen_calls) == 3
    assert bytes(fake_popens[0].stdin.written) == b"aa"
    assert bytes(fake_popens[1].stdin.written) == b"bb"
    assert bytes(fake_popens[2].stdin.written) == b"cc"
    # A short pause between chunks of the same paragraph, not the longer
    # inter-paragraph pause -- and none after the final chunk.
    assert sleep_calls == [
        voice_tts._CHUNK_PAUSE_SECONDS,
        voice_tts._CHUNK_PAUSE_SECONDS,
    ]


def test_speak_single_chunk_paragraph_has_no_pause_inserted(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice(
        chunks=None, sample_rate=1, chunks_by_text={"One.": [FakeChunk(b"aa")]}
    )
    monkeypatch.setattr(voice_tts.PiperVoice, "load", lambda path: fake_voice)
    fake_popens = [FakePopen(returncode=0)]
    popen_calls, sleep_calls = _mock_aplay_and_sleep(monkeypatch, fake_popens)

    speak(
        "One.",
        VoiceTTSConfig(voice="test-voice"),
        VoiceOutputConfig(),
        models_dir=tmp_path,
    )

    assert len(popen_calls) == 1
    assert sleep_calls == []


def test_speak_single_paragraph_has_no_silence_inserted(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice([FakeChunk(b"abc")], sample_rate=22050)
    _, fake_popen = _mock_piper_voice_and_popen(monkeypatch, fake_voice)
    monkeypatch.setattr(
        voice_tts.subprocess, "Popen", lambda args, stdin=None: fake_popen
    )

    speak(
        "just one paragraph",
        VoiceTTSConfig(voice="test-voice"),
        VoiceOutputConfig(),
        models_dir=tmp_path,
    )

    assert fake_voice.synthesize_calls == ["just one paragraph"]
    assert bytes(fake_popen.stdin.written) == b"abc"


def _mock_piper_voice_with_model(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice([FakeChunk(b"x")])
    monkeypatch.setattr(voice_tts.PiperVoice, "load", lambda path: fake_voice)
    return fake_voice


def test_speak_usb_output_targets_usb_device(tmp_path, monkeypatch):
    _mock_piper_voice_with_model(tmp_path, monkeypatch)
    fake_popen = FakePopen(returncode=0)
    monkeypatch.setattr(
        voice_tts.subprocess,
        "Popen",
        lambda args, stdin=None: (setattr(fake_popen, "args", args), fake_popen)[1],
    )

    speak(
        "hello",
        VoiceTTSConfig(voice="test-voice"),
        VoiceOutputConfig(type="usb"),
        models_dir=tmp_path,
    )

    assert fake_popen.args[fake_popen.args.index("-D") + 1] == "usb"
    assert bytes(fake_popen.stdin.written) == b"x"
    assert fake_popen.stdin.closed


def test_speak_raises_when_aplay_not_found(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    monkeypatch.setattr(
        voice_tts.PiperVoice, "load", lambda path: FakeVoice([FakeChunk(b"x")])
    )

    def raise_not_found(args, stdin=None):
        raise FileNotFoundError()

    monkeypatch.setattr(voice_tts.subprocess, "Popen", raise_not_found)

    with pytest.raises(TTSError, match="aplay not found on PATH"):
        speak(
            "hi",
            VoiceTTSConfig(voice="test-voice"),
            VoiceOutputConfig(),
            models_dir=tmp_path,
        )


def test_speak_raises_when_aplay_exits_nonzero(tmp_path, monkeypatch):
    _mock_piper_voice_with_model(tmp_path, monkeypatch)
    fake_popen = FakePopen(returncode=1)
    monkeypatch.setattr(
        voice_tts.subprocess, "Popen", lambda args, stdin=None: fake_popen
    )

    with pytest.raises(TTSError, match="aplay exited with code 1"):
        speak(
            "hi",
            VoiceTTSConfig(voice="test-voice"),
            VoiceOutputConfig(),
            models_dir=tmp_path,
        )
