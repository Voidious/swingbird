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
    def __init__(self, chunks, sample_rate=22050):
        self._chunks = chunks
        self.config = FakeVoiceConfig(sample_rate)

    def synthesize(self, text):
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


def test_speak_raises_when_model_missing(tmp_path):
    with pytest.raises(TTSError, match="Piper voice model not found"):
        speak(
            "hi",
            VoiceTTSConfig(voice="missing-voice"),
            VoiceOutputConfig(),
            models_dir=tmp_path,
        )


def test_speak_happy_path_writes_all_chunks_to_onboard_device(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice([FakeChunk(b"abc"), FakeChunk(b"def")], sample_rate=22050)
    load_calls = []
    monkeypatch.setattr(
        voice_tts.PiperVoice,
        "load",
        lambda path: load_calls.append(path) or fake_voice,
    )
    fake_popen = FakePopen(returncode=0)

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
        "-",
    ]
    assert bytes(fake_popen.stdin.written) == b"abcdef"
    assert fake_popen.stdin.closed


def test_speak_usb_output_targets_usb_device(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice([])
    monkeypatch.setattr(voice_tts.PiperVoice, "load", lambda path: fake_voice)
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
    assert bytes(fake_popen.stdin.written) == b""
    assert fake_popen.stdin.closed


def test_speak_raises_when_aplay_not_found(tmp_path, monkeypatch):
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    monkeypatch.setattr(voice_tts.PiperVoice, "load", lambda path: FakeVoice([]))

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
    (tmp_path / "test-voice.onnx").write_bytes(b"")
    fake_voice = FakeVoice([FakeChunk(b"x")])
    monkeypatch.setattr(voice_tts.PiperVoice, "load", lambda path: fake_voice)
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
