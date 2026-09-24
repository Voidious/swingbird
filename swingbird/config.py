"""Config loading for the swingbird daemon.

The config file lists the LLM backend and the project channels the daemon
is allowed to read from and (optionally) write into, per the least-
privilege design in the TPM agent design doc (§5, §7).

The committed config holds shareable defaults only. A `.swingbird.toml`
override file next to it, gitignored, can supply values that shouldn't be
checked in (e.g. a private relay URL) -- same pattern as crispen's own
pyproject.toml + `.crispen.toml`.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """Raised when the swingbird config file is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key_env: str
    provider: str | None = None


# Known OpenAI-compatible providers' default base URL and API-key env var, so
# a config only needs `provider` + `model` -- `base_url`/`api_key_env` are
# still accepted explicitly to override a default or point at a provider not
# listed here (e.g. a self-hosted OpenAI-compatible endpoint).
LLM_PROVIDERS: dict[str, tuple[str, str]] = {
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


@dataclass(frozen=True)
class ChannelConfig:
    id: str
    name: str
    write: bool
    agents: tuple[str, ...]
    goal: str | None = None


@dataclass(frozen=True)
class RelayConfig:
    url: str
    private_key_env: str


@dataclass(frozen=True)
class OwnerConfig:
    """The one identity allowed to trigger dispatch/confirm/cancel (§5).

    Everyone else's traffic in a subscribed channel -- including a coding
    agent's own replies -- must never be routed as a command.
    """

    pubkey: str
    name: str


DEFAULT_IDENTITY_NAME = "swingbird"

_SUPPORTED_AVATAR_STYLES = ("emoji",)
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass(frozen=True)
class AvatarConfig:
    """A `style`-discriminated avatar descriptor, mirroring Buzz Desktop's
    own `AvatarMode` ("emoji" | "image" | "animated") so adding a second
    style later is an additive change, not a reshape. Only "emoji" is
    supported so far -- `emoji` + `color` is exactly the pair Buzz's own
    emoji-avatar editor round-trips, see `avatar.emoji_avatar_data_url`.
    """

    style: str
    emoji: str
    color: str


@dataclass(frozen=True)
class IdentityConfig:
    """The daemon's own Buzz profile (display name, bio, avatar), kept in
    sync on startup so a fresh identity (or a renamed deployment) shows up
    looking like someone would actually recognize, rather than whatever the
    identity's profile happened to have before.
    """

    name: str = DEFAULT_IDENTITY_NAME
    description: str | None = None
    avatar: AvatarConfig | None = None


DEFAULT_REPLY_WAIT_SECONDS = 180


@dataclass(frozen=True)
class DispatchConfig:
    """How long to wait for a working agent's reply before giving up on
    summarizing it back to the owner (§4.2's "relay a summary" behavior).

    Defaults short on purpose: a reply arriving minutes later, unprompted,
    would be a jarring surprise on a voice-assistant-style deployment --
    someone who wants a longer wait can opt into it explicitly.
    """

    reply_wait_seconds: int = DEFAULT_REPLY_WAIT_SECONDS


DEFAULT_STALE_AFTER_DAYS = 30
DEFAULT_MAX_MESSAGES_PER_CHANNEL = 1000
DEFAULT_MAX_DETAILED_ITEMS = 3
DEFAULT_CLOSED_ITEM_WINDOW_DAYS = 90


@dataclass(frozen=True)
class RecapConfig:
    """How far back "recent activity" reaches before a project is treated
    as stale and omitted from a recap of *all* channels. Never applies when
    the user names a channel explicitly -- see recap.py.

    `max_messages_per_channel` bounds the paging `history.fetch_messages_since`
    does to fill that window -- a safety cap so one very chatty channel can't
    make a recap page through its entire history.

    `max_detailed_items` caps how many of a channel's open items a
    "detailed" recap narrates in its own text before folding the rest into
    a count, same as a concise recap always does for everything past its
    one leading item -- see recap.py's `RecapItem.is_primary`.

    `closed_item_window_days` bounds how far back a closed item (see
    `closed_items.py`) still gets sent to the LLM as "don't re-list this"
    context -- an ever-growing, unbounded closed-items list would keep
    costing tokens on every future recap forever, for items old enough that
    a restatement is vanishingly unlikely anyway. `recap.py`'s `build_recap`
    floors the effective window at `stale_after_days`, so shrinking this
    below the recap's own message window can never make a just-closed item
    (one still young enough for its restatement to appear in the transcript
    being recapped) fall outside the guard that's supposed to suppress it.
    """

    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS
    max_messages_per_channel: int = DEFAULT_MAX_MESSAGES_PER_CHANNEL
    max_detailed_items: int = DEFAULT_MAX_DETAILED_ITEMS
    closed_item_window_days: int = DEFAULT_CLOSED_ITEM_WINDOW_DAYS


_SUPPORTED_MIC_TYPES = ("onboard", "usb")
_SUPPORTED_OUTPUT_TYPES = ("onboard", "usb")
_SUPPORTED_STT_MODELS = ("small", "small.en")

DEFAULT_WAKE_WORD = "swingbird"
DEFAULT_MIC_TYPE = "onboard"
DEFAULT_OUTPUT_TYPE = "onboard"
DEFAULT_STT_MODEL = "small"
DEFAULT_WAKE_WORD_WINDOW_SECONDS = 30
DEFAULT_FOLLOW_UP_WINDOW_SECONDS = 30
DEFAULT_BARGE_IN_TRIGGER_FRAMES = 3
DEFAULT_BARGE_IN_VAD_THRESHOLD = 0.8


@dataclass(frozen=True)
class VoiceMicConfig:
    """Which physical mic input the wake-word/VAD/STT pipeline listens on
    (Voice Mode design doc §V.6). Both types are permanent, tested code
    paths, not a prototype-vs-production split: "onboard" is the WSL dev
    machine's default input today and the Orange Pi's onboard mic later;
    "usb" is the reSpeaker XVF3800 array, the final product's primary
    input on real hardware.
    """

    type: str = DEFAULT_MIC_TYPE


@dataclass(frozen=True)
class VoiceOutputConfig:
    """Which physical audio output Piper's speech plays on (§V.6),
    mirroring `VoiceMicConfig`'s input-side split for the same reason.
    """

    type: str = DEFAULT_OUTPUT_TYPE


@dataclass(frozen=True)
class VoiceSTTConfig:
    """faster-whisper model size (§V.5). Only "small"/"small.en" are
    supported: "base" is a needless accuracy step down for no real gain,
    and "medium" drops below real-time on the Orange Pi 5 Pro's CPU.
    """

    model: str = DEFAULT_STT_MODEL


@dataclass(frozen=True)
class VoiceTTSConfig:
    """Piper voice model name, e.g. "en_US-lessac-medium" -- resolved by
    `voice_tts.py` to `<voice>.onnx`/`<voice>.onnx.json` under its models
    directory. No default: silently picking a voice would be worse than
    failing loudly, since Voidious hasn't chosen one yet (§V.13's wake-word
    equivalent question for the voice itself is still open).
    """

    voice: str


@dataclass(frozen=True)
class VoiceConfig:
    """Voice-mode settings (§V.15). Disabled by default -- an existing
    text-DM-only deployment doesn't need a [voice] section at all, and one
    with the section present but `enabled = false` stays that way until
    it's ready to run against real audio hardware (§V.14).
    """

    enabled: bool = False
    wake_word: str = DEFAULT_WAKE_WORD
    mic: VoiceMicConfig = VoiceMicConfig()
    output: VoiceOutputConfig = VoiceOutputConfig()
    stt: VoiceSTTConfig = VoiceSTTConfig()
    tts: VoiceTTSConfig | None = None
    # How long to wait for the user to start speaking right after the wake
    # word fires before giving up and requiring it again -- separate from
    # voice_stt.py's fixed SILENCE_FRAMES_TO_STOP/MAX_UTTERANCE_SECONDS,
    # which govern *within* one utterance, not whether an utterance starts
    # at all. Previously hardcoded to record_and_transcribe's own default
    # (MAX_UTTERANCE_SECONDS); Voidious asked for it to be a visible,
    # tunable config knob instead.
    wake_word_window_seconds: int = DEFAULT_WAKE_WORD_WINDOW_SECONDS
    # §V.11: how long to keep listening for a follow-up command after a
    # reply, without requiring the wake word again -- same
    # in-utterance-vs-between-utterances distinction as
    # wake_word_window_seconds above.
    follow_up_window_seconds: int = DEFAULT_FOLLOW_UP_WINDOW_SECONDS
    # §V.12: how many consecutive VAD-positive frames (~80ms each) a
    # mid-playback signal must sustain before `voice_barge_in` treats it
    # as a genuine interruption rather than a brief transient -- live
    # testing (2026-09-23) found a single loud frame (a mouse click,
    # sitting up in a chair) was enough to stop playback under the
    # original one-frame trigger. Higher values make barge-in less
    # sensitive (slower to react, fewer false positives from incidental
    # noise); lower values make it more sensitive. A visible, tunable
    # knob for the same reason wake_word_window_seconds is one -- the
    # right default depends on mic/headset/room, not something to lock
    # in from one test session.
    barge_in_trigger_frames: int = DEFAULT_BARGE_IN_TRIGGER_FRAMES
    # §V.12: per-frame VAD confidence a barge-in's own trigger frames must
    # clear before `voice_barge_in` treats it as the start of real speech --
    # separate from, and stricter than, `voice_stt.VAD_SPEECH_THRESHOLD`
    # (0.5), which endpoints a normal utterance (and this same
    # interruption's own capture once triggered). Live testing (2026-09-23)
    # found sustained background noise (a fan) could cross that lower
    # threshold for enough *consecutive* frames to trigger even with
    # barge_in_trigger_frames raised -- a duration-only fix can't help
    # that, since the noise really did sustain. Higher values make barge-in
    # less sensitive to a quiet/ambiguous signal; lower values bring it
    # back down toward voice_stt's own threshold. Independent, visible,
    # tunable knob for the same reason barge_in_trigger_frames is one -- the
    # right default depends on mic/room, not something to lock in from one
    # test session.
    barge_in_vad_threshold: float = DEFAULT_BARGE_IN_VAD_THRESHOLD


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    relay: RelayConfig
    channels: tuple[ChannelConfig, ...]
    owner: OwnerConfig
    dispatch: DispatchConfig = DispatchConfig()
    identity: IdentityConfig = IdentityConfig()
    recap: RecapConfig = RecapConfig()
    voice: VoiceConfig = VoiceConfig()

    def channel_by_name(self, name: str) -> ChannelConfig | None:
        for channel in self.channels:
            if channel.name == name:
                return channel
        return None


def _merge(base: dict, override: dict) -> dict:
    """Overlay `override` onto `base`. Nested dicts merge key by key; any
    other value (including lists, e.g. `[[channels]]`) is replaced outright.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


_RELAY_REQUIRED = ("url", "private_key_env")
_CHANNEL_REQUIRED = ("id", "name", "write", "agents")
_OWNER_REQUIRED = ("pubkey", "name")


def load_config(path: str | Path) -> Config:
    """Load and validate the swingbird TOML config at `path`.

    If a `.swingbird.toml` file exists alongside `path`, its sections are
    merged on top of the base config (see module docstring).
    """
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    override_path = path.parent / ".swingbird.toml"
    if override_path != path and override_path.is_file():
        try:
            raw = _merge(raw, tomllib.loads(override_path.read_text(encoding="utf-8")))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {override_path}: {exc}") from exc

    return Config(
        llm=_parse_llm(raw),
        relay=_parse_relay(raw),
        channels=_parse_channels(raw),
        owner=_parse_owner(raw),
        dispatch=_parse_dispatch(raw),
        identity=_parse_identity(raw),
        recap=_parse_recap(raw),
        voice=_parse_voice(raw),
    )


def _parse_llm(raw: dict) -> LLMConfig:
    section = raw.get("llm")
    if not isinstance(section, dict):
        raise ConfigError("config is missing required [llm] section")
    if not section.get("model"):
        raise ConfigError("[llm] is missing required field: model")

    provider = section.get("provider")
    if provider is not None and provider not in LLM_PROVIDERS:
        raise ConfigError(
            f"[llm].provider must be one of {tuple(LLM_PROVIDERS)!r}, got {provider!r}"
        )
    default_base_url, default_api_key_env = LLM_PROVIDERS.get(provider, (None, None))

    base_url = section.get("base_url") or default_base_url
    if not base_url:
        raise ConfigError(
            "[llm] is missing required field: base_url (or set provider to one "
            f"of {tuple(LLM_PROVIDERS)!r} to use its default)"
        )
    api_key_env = section.get("api_key_env") or default_api_key_env
    if not api_key_env:
        raise ConfigError(
            "[llm] is missing required field: api_key_env (or set provider to "
            f"one of {tuple(LLM_PROVIDERS)!r} to use its default)"
        )

    return LLMConfig(
        base_url=base_url,
        model=section["model"],
        api_key_env=api_key_env,
        provider=provider,
    )


def _parse_relay(raw: dict) -> RelayConfig:
    section = raw.get("relay")
    if not isinstance(section, dict):
        raise ConfigError("config is missing required [relay] section")
    for key in _RELAY_REQUIRED:
        if not section.get(key):
            raise ConfigError(f"[relay] is missing required field: {key}")
    return RelayConfig(
        url=section["url"],
        private_key_env=section["private_key_env"],
    )


def _parse_owner(raw: dict) -> OwnerConfig:
    section = raw.get("owner")
    if not isinstance(section, dict):
        raise ConfigError("config is missing required [owner] section")
    for key in _OWNER_REQUIRED:
        if not section.get(key):
            raise ConfigError(f"[owner] is missing required field: {key}")
    return OwnerConfig(pubkey=section["pubkey"], name=section["name"])


def _parse_dispatch(raw: dict) -> DispatchConfig:
    section = raw.get("dispatch", {})
    if not isinstance(section, dict):
        raise ConfigError("[dispatch] must be a table")
    seconds = section.get("reply_wait_seconds", DEFAULT_REPLY_WAIT_SECONDS)
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
        raise ConfigError("[dispatch].reply_wait_seconds must be a positive integer")
    return DispatchConfig(reply_wait_seconds=seconds)


def _parse_recap(raw: dict) -> RecapConfig:
    section = raw.get("recap", {})
    if not isinstance(section, dict):
        raise ConfigError("[recap] must be a table")
    days = section.get("stale_after_days", DEFAULT_STALE_AFTER_DAYS)
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ConfigError("[recap].stale_after_days must be a positive integer")
    max_messages = section.get(
        "max_messages_per_channel", DEFAULT_MAX_MESSAGES_PER_CHANNEL
    )
    if (
        isinstance(max_messages, bool)
        or not isinstance(max_messages, int)
        or max_messages <= 0
    ):
        raise ConfigError("[recap].max_messages_per_channel must be a positive integer")
    max_detailed_items = section.get("max_detailed_items", DEFAULT_MAX_DETAILED_ITEMS)
    if (
        isinstance(max_detailed_items, bool)
        or not isinstance(max_detailed_items, int)
        or max_detailed_items <= 0
    ):
        raise ConfigError("[recap].max_detailed_items must be a positive integer")
    closed_item_window_days = section.get(
        "closed_item_window_days", DEFAULT_CLOSED_ITEM_WINDOW_DAYS
    )
    if (
        isinstance(closed_item_window_days, bool)
        or not isinstance(closed_item_window_days, int)
        or closed_item_window_days <= 0
    ):
        raise ConfigError("[recap].closed_item_window_days must be a positive integer")
    return RecapConfig(
        stale_after_days=days,
        max_messages_per_channel=max_messages,
        max_detailed_items=max_detailed_items,
        closed_item_window_days=closed_item_window_days,
    )


def _parse_voice(raw: dict) -> VoiceConfig:
    section = raw.get("voice", {})
    if not isinstance(section, dict):
        raise ConfigError("[voice] must be a table")

    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("[voice].enabled must be a boolean")

    wake_word = section.get("wake_word", DEFAULT_WAKE_WORD)
    if not isinstance(wake_word, str) or not wake_word.strip():
        raise ConfigError("[voice].wake_word must be a non-empty string")

    wake_word_window_seconds = section.get(
        "wake_word_window_seconds", DEFAULT_WAKE_WORD_WINDOW_SECONDS
    )
    if (
        isinstance(wake_word_window_seconds, bool)
        or not isinstance(wake_word_window_seconds, int)
        or wake_word_window_seconds <= 0
    ):
        raise ConfigError("[voice].wake_word_window_seconds must be a positive integer")

    follow_up_window_seconds = section.get(
        "follow_up_window_seconds", DEFAULT_FOLLOW_UP_WINDOW_SECONDS
    )
    if (
        isinstance(follow_up_window_seconds, bool)
        or not isinstance(follow_up_window_seconds, int)
        or follow_up_window_seconds <= 0
    ):
        raise ConfigError("[voice].follow_up_window_seconds must be a positive integer")

    barge_in_trigger_frames = section.get(
        "barge_in_trigger_frames", DEFAULT_BARGE_IN_TRIGGER_FRAMES
    )
    if (
        isinstance(barge_in_trigger_frames, bool)
        or not isinstance(barge_in_trigger_frames, int)
        or barge_in_trigger_frames <= 0
    ):
        raise ConfigError("[voice].barge_in_trigger_frames must be a positive integer")

    barge_in_vad_threshold = section.get(
        "barge_in_vad_threshold", DEFAULT_BARGE_IN_VAD_THRESHOLD
    )
    if (
        isinstance(barge_in_vad_threshold, bool)
        or not isinstance(barge_in_vad_threshold, (int, float))
        or not 0.0 < barge_in_vad_threshold <= 1.0
    ):
        raise ConfigError(
            "[voice].barge_in_vad_threshold must be a number between 0 and 1"
        )

    return VoiceConfig(
        enabled=enabled,
        wake_word=wake_word,
        mic=_parse_voice_mic(section.get("mic", {})),
        output=_parse_voice_output(section.get("output", {})),
        stt=_parse_voice_stt(section.get("stt", {})),
        tts=_parse_voice_tts(section.get("tts"), enabled=enabled),
        wake_word_window_seconds=wake_word_window_seconds,
        follow_up_window_seconds=follow_up_window_seconds,
        barge_in_trigger_frames=barge_in_trigger_frames,
        barge_in_vad_threshold=barge_in_vad_threshold,
    )


def _parse_voice_mic(section: object) -> VoiceMicConfig:
    if not isinstance(section, dict):
        raise ConfigError("[voice.mic] must be a table")
    mic_type = section.get("type", DEFAULT_MIC_TYPE)
    if mic_type not in _SUPPORTED_MIC_TYPES:
        raise ConfigError(
            f"[voice.mic].type must be one of {_SUPPORTED_MIC_TYPES!r}, "
            f"got {mic_type!r}"
        )
    return VoiceMicConfig(type=mic_type)


def _parse_voice_output(section: object) -> VoiceOutputConfig:
    if not isinstance(section, dict):
        raise ConfigError("[voice.output] must be a table")
    output_type = section.get("type", DEFAULT_OUTPUT_TYPE)
    if output_type not in _SUPPORTED_OUTPUT_TYPES:
        raise ConfigError(
            f"[voice.output].type must be one of {_SUPPORTED_OUTPUT_TYPES!r}, "
            f"got {output_type!r}"
        )
    return VoiceOutputConfig(type=output_type)


def _parse_voice_stt(section: object) -> VoiceSTTConfig:
    if not isinstance(section, dict):
        raise ConfigError("[voice.stt] must be a table")
    model = section.get("model", DEFAULT_STT_MODEL)
    if model not in _SUPPORTED_STT_MODELS:
        raise ConfigError(
            f"[voice.stt].model must be one of {_SUPPORTED_STT_MODELS!r}, got {model!r}"
        )
    return VoiceSTTConfig(model=model)


def _parse_voice_tts(section: object, *, enabled: bool) -> VoiceTTSConfig | None:
    if section is None:
        if enabled:
            raise ConfigError("[voice.tts] is required when [voice].enabled is true")
        return None
    if not isinstance(section, dict):
        raise ConfigError("[voice.tts] must be a table")
    voice_name = section.get("voice")
    if not isinstance(voice_name, str) or not voice_name.strip():
        raise ConfigError("[voice.tts].voice must be a non-empty string")
    return VoiceTTSConfig(voice=voice_name)


def _parse_identity(raw: dict) -> IdentityConfig:
    section = raw.get("identity", {})
    if not isinstance(section, dict):
        raise ConfigError("[identity] must be a table")
    name = section.get("name", DEFAULT_IDENTITY_NAME)
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("[identity].name must be a non-empty string")
    description = section.get("description")
    if description is not None and (
        not isinstance(description, str) or not description.strip()
    ):
        raise ConfigError("[identity].description must be a non-empty string")
    return IdentityConfig(
        name=name, description=description, avatar=_parse_avatar(section.get("avatar"))
    )


def _parse_avatar(section: object) -> AvatarConfig | None:
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError("[identity.avatar] must be a table")
    style = section.get("style")
    if style not in _SUPPORTED_AVATAR_STYLES:
        raise ConfigError(
            "[identity.avatar].style must be one of "
            f"{_SUPPORTED_AVATAR_STYLES!r}, got {style!r}"
        )
    emoji = section.get("emoji")
    if not isinstance(emoji, str) or not emoji.strip():
        raise ConfigError("[identity.avatar].emoji must be a non-empty string")
    color = section.get("color")
    if not isinstance(color, str) or not _HEX_COLOR_RE.match(color):
        raise ConfigError("[identity.avatar].color must be a hex color like '#3399FF'")
    return AvatarConfig(style=style, emoji=emoji, color=color)


def _parse_channels(raw: dict) -> tuple[ChannelConfig, ...]:
    entries = raw.get("channels")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(
            "config must define at least one [[channels]] entry "
            "(add one to a local .swingbird.toml override, see swingbird.toml)"
        )

    channels: list[ChannelConfig] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for entry in entries:
        for key in _CHANNEL_REQUIRED:
            if key not in entry:
                raise ConfigError(
                    f"[[channels]] entry is missing required field: {key}"
                )
        goal = entry.get("goal")
        if goal is not None and (not isinstance(goal, str) or not goal.strip()):
            raise ConfigError(
                f"[[channels]] entry {entry.get('name')!r} has an invalid goal"
            )
        channel = ChannelConfig(
            id=entry["id"],
            name=entry["name"],
            write=bool(entry["write"]),
            agents=tuple(entry["agents"]),
            goal=goal,
        )
        if channel.id in seen_ids:
            raise ConfigError(f"duplicate channel id in config: {channel.id}")
        if channel.name in seen_names:
            raise ConfigError(f"duplicate channel name in config: {channel.name}")
        seen_ids.add(channel.id)
        seen_names.add(channel.name)
        channels.append(channel)
    return tuple(channels)
