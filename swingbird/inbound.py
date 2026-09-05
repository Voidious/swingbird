"""Inbound relay client: NIP-42 auth + a persistent WebSocket subscription.

Reads happen over a direct WebSocket connection to the relay -- writes
still go through buzz-cli (see outbound.py) -- because a long-running
daemon can hold a subscription open, unlike the CLI's one-shot process
model. The relay requires NIP-42 AUTH before any subscription and,
beyond plain NIP-42, membership in its own allow-list; both were
confirmed against the live relay while building this module.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import websockets

from swingbird.nostr_crypto import (
    NostrCryptoError,
    parse_private_key,
    sign_event,
    verify_event,
)

CHANNEL_MESSAGE_KIND = 9
AUTH_KIND = 22242
AUTH_TIMEOUT_SECONDS = 20


class InboundError(Exception):
    """Raised when the inbound relay connection or auth fails."""


class InboundClient:
    """A NIP-42-authenticated WebSocket subscription to channel messages."""

    def __init__(self, relay_url: str, private_key_raw: str) -> None:
        self._relay_url = relay_url
        try:
            self._private_key = parse_private_key(private_key_raw)
        except NostrCryptoError as exc:
            raise InboundError(str(exc)) from exc
        self._ws = None
        self._seen_ids: set[str] = set()

    async def connect(self) -> None:
        """Open the WebSocket connection and complete NIP-42 auth."""
        self._ws = await websockets.connect(self._relay_url)
        challenge = await self._wait_for_auth_challenge()
        auth_event = sign_event(
            self._private_key,
            int(time.time()),
            AUTH_KIND,
            [["relay", self._relay_url], ["challenge", challenge]],
            "",
        )
        await self._ws.send(json.dumps(["AUTH", auth_event]))
        await self._wait_for_ok(auth_event["id"])

    async def subscribe(
        self,
        channel_ids: list[str],
        sub_id: str = "swingbird",
        since: int | None = None,
    ) -> None:
        """Subscribe to channel-message events tagged with any of `channel_ids`.

        `since` excludes the relay's stored backlog -- without it, every
        historical message in a channel with history arrives indistinguishable
        from a live one, and the owner-pubkey gate in `daemon.py` would happily
        replay years of old messages as fresh commands on every restart.
        Defaults to "now" so callers get backlog-free behavior by default;
        tests that want deterministic filters can pass an explicit value.
        """
        if self._ws is None:
            raise InboundError("subscribe() called before connect()")
        if since is None:
            since = int(time.time())
        filters = {"kinds": [CHANNEL_MESSAGE_KIND], "#h": channel_ids, "since": since}
        req = ["REQ", sub_id, filters]
        await self._ws.send(json.dumps(req))

    async def events(self) -> AsyncIterator[dict]:
        """Yield verified, de-duplicated events as they arrive on the subscription.

        `NOTICE`/`CLOSED` frames are surfaced (printed) rather than silently
        dropped -- a relay can reject part of a multi-channel subscription
        (e.g. "restricted: not a channel member" for one channel_id) while
        still delivering events for the rest, and that would otherwise look
        identical to "no messages arrived yet" from the caller's side.
        """
        if self._ws is None:
            raise InboundError("events() called before connect()")
        async for raw in self._ws:
            message = json.loads(raw)
            if message[0] in ("NOTICE", "CLOSED"):
                print(f"swingbird: relay sent {message}")
                continue
            if message[0] != "EVENT":
                continue
            event = message[2]
            if event["id"] in self._seen_ids:
                continue
            if not verify_event(event):
                print(
                    f"swingbird: dropping event {event.get('id')} -- failed verification"
                )
                continue
            self._seen_ids.add(event["id"])
            yield event

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def _wait_for_auth_challenge(self) -> str:
        async def _wait() -> str:
            async for raw in self._ws:
                message = json.loads(raw)
                if message[0] == "AUTH":
                    return message[1]
            raise InboundError("connection closed before relay sent an AUTH challenge")

        return await self._with_timeout(_wait())

    async def _wait_for_ok(self, event_id: str) -> None:
        async def _wait() -> None:
            async for raw in self._ws:
                message = json.loads(raw)
                if message[0] == "OK" and message[1] == event_id:
                    if not message[2]:
                        raise InboundError(f"relay rejected AUTH: {message[3]}")
                    return
            raise InboundError("connection closed before relay acknowledged AUTH")

        return await self._with_timeout(_wait())

    @staticmethod
    async def _with_timeout(coro):
        try:
            return await asyncio.wait_for(coro, timeout=AUTH_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            raise InboundError("timed out waiting for the relay during auth") from exc
