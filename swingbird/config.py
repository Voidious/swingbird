"""Config loading for the swingbird daemon.

The config file lists the LLM backend and the project channels the daemon
is allowed to read from and (optionally) write into, per the least-
privilege design in the TPM agent design doc (§5, §7).
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    relay: RelayConfig
    channels: tuple[ChannelConfig, ...]
    owner: OwnerConfig

    def channel_by_name(self, name: str) -> ChannelConfig | None:
        for channel in self.channels:
            if channel.name == name:
                return channel
        return None


_LLM_REQUIRED = ("base_url", "model", "api_key_env")
_RELAY_REQUIRED = ("url", "private_key_env")
_CHANNEL_REQUIRED = ("id", "name", "write", "agents")
_OWNER_REQUIRED = ("pubkey", "name")


def load_config(path: str | Path) -> Config:
    """Load and validate the swingbird TOML config at `path`."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    return Config(
        llm=_parse_llm(raw),
        relay=_parse_relay(raw),
        channels=_parse_channels(raw),
        owner=_parse_owner(raw),
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


def _parse_channels(raw: dict) -> tuple[ChannelConfig, ...]:
    entries = raw.get("channels")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("config must define at least one [[channels]] entry")

    channels: list[ChannelConfig] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for entry in entries:
        for key in _CHANNEL_REQUIRED:
            if key not in entry:
                raise ConfigError(
                    f"[[channels]] entry is missing required field: {key}"
                )
        channel = ChannelConfig(
            id=entry["id"],
            name=entry["name"],
            write=bool(entry["write"]),
            agents=tuple(entry["agents"]),
        )
        if channel.id in seen_ids:
            raise ConfigError(f"duplicate channel id in config: {channel.id}")
        if channel.name in seen_names:
            raise ConfigError(f"duplicate channel name in config: {channel.name}")
        seen_ids.add(channel.id)
        seen_names.add(channel.name)
        channels.append(channel)
    return tuple(channels)
