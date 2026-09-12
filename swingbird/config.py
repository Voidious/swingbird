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


DEFAULT_REPLY_WAIT_SECONDS = 90


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


@dataclass(frozen=True)
class RecapConfig:
    """How far back "recent activity" reaches before a project is treated
    as stale and omitted from a recap of *all* channels. Never applies when
    the user names a channel explicitly -- see recap.py.

    `max_messages_per_channel` bounds the paging `history.fetch_messages_since`
    does to fill that window -- a safety cap so one very chatty channel can't
    make a recap page through its entire history.
    """

    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS
    max_messages_per_channel: int = DEFAULT_MAX_MESSAGES_PER_CHANNEL


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    relay: RelayConfig
    channels: tuple[ChannelConfig, ...]
    owner: OwnerConfig
    dispatch: DispatchConfig = DispatchConfig()
    identity: IdentityConfig = IdentityConfig()
    recap: RecapConfig = RecapConfig()

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


_LLM_REQUIRED = ("base_url", "model", "api_key_env")
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
    )


def _parse_llm(raw: dict) -> LLMConfig:
    section = raw.get("llm")
    if not isinstance(section, dict):
        raise ConfigError("config is missing required [llm] section")
    for key in _LLM_REQUIRED:
        if not section.get(key):
            raise ConfigError(f"[llm] is missing required field: {key}")
    return LLMConfig(
        base_url=section["base_url"],
        model=section["model"],
        api_key_env=section["api_key_env"],
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
    return RecapConfig(stale_after_days=days, max_messages_per_channel=max_messages)


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
