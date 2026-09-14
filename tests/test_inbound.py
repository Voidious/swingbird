import asyncio
import json
import time

import pytest

from swingbird import inbound
from swingbird.inbound import InboundClient, InboundError
from swingbird.nostr_crypto import pubkey_hex, sign_event

PRIVATE_KEY_HEX = "2" * 64
RELAY_URL = "wss://relay.example"


class FakeWebSocket:
    """Minimal stand-in for a websockets connection.

    `send()` inspects outgoing AUTH events and auto-queues the matching
    OK response, like a scripted fake server, since the real event id
    depends on a wall-clock timestamp the test can't predict up front.
    """

    def __init__(
        self,
        auth_ok: bool = True,
        auth_message: str = "",
        hang: bool = False,
        respond_to_auth: bool = True,
    ):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []
        self.closed = False
        self._auth_ok = auth_ok
        self._auth_message = auth_message
        self._hang = hang
        self._respond_to_auth = respond_to_auth

    def push(self, message: dict | list) -> None:
        self.incoming.put_nowait(json.dumps(message))

    async def send(self, data: str) -> None:
        self.sent.append(data)
        parsed = json.loads(data)
        if parsed[0] == "AUTH" and self._respond_to_auth:
            event = parsed[1]
            self.push(["OK", event["id"], self._auth_ok, self._auth_message])

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._hang:
            await asyncio.Event().wait()
        try:
            return self.incoming.get_nowait()
        except asyncio.QueueEmpty:
            raise StopAsyncIteration from None


def _make_client(monkeypatch, ws: FakeWebSocket | None = None, **ws_kwargs):
    """Build an InboundClient wired to a FakeWebSocket, without connecting."""
    if ws is None:
        ws = FakeWebSocket(**ws_kwargs)

    async def fake_connect(url):
        return ws

    monkeypatch.setattr(inbound.websockets, "connect", fake_connect)
    return InboundClient(RELAY_URL, PRIVATE_KEY_HEX), ws


def _connected_client(monkeypatch, challenge="challenge-123"):
    """Build a client and drive it through a successful connect()."""
    ws = FakeWebSocket()
    ws.push(["AUTH", challenge])
    client, ws = _make_client(monkeypatch, ws=ws)
    asyncio.run(client.connect())
    return client, ws


def test_init_rejects_bad_private_key():
    with pytest.raises(InboundError, match="not valid hex or nsec"):
        InboundClient(RELAY_URL, "not-a-key")


def test_pubkey_is_derived_from_the_private_key():
    client = InboundClient(RELAY_URL, PRIVATE_KEY_HEX)
    assert client.pubkey == pubkey_hex(bytes.fromhex(PRIVATE_KEY_HEX))


def test_connect_authenticates(monkeypatch):
    ws = FakeWebSocket()
    # A stray NOTICE before the AUTH challenge, and another stray message
    # queued ahead of the eventual OK -- both must be skipped, not treated
    # as the messages connect() is actually waiting for.
    ws.push(["NOTICE", "hello"])
    ws.push(["AUTH", "challenge-123"])
    ws.push(["EVENT", "sub-1", {"id": "irrelevant"}])
    client, ws = _make_client(monkeypatch, ws=ws)
    asyncio.run(client.connect())

    auth_msg = json.loads(ws.sent[0])
    assert auth_msg[0] == "AUTH"
    assert auth_msg[1]["kind"] == 22242
    assert auth_msg[1]["tags"] == [
        ["relay", RELAY_URL],
        ["challenge", "challenge-123"],
    ]


def test_connect_raises_if_socket_closes_before_ack(monkeypatch):
    client, ws = _make_client(monkeypatch, respond_to_auth=False)
    ws.push(["AUTH", "challenge-123"])
    with pytest.raises(InboundError, match="closed before relay acknowledged AUTH"):
        asyncio.run(client.connect())


def test_connect_raises_on_rejected_auth(monkeypatch):
    client, ws = _make_client(
        monkeypatch, auth_ok=False, auth_message="restricted: not a relay member"
    )
    ws.push(["AUTH", "challenge-123"])
    with pytest.raises(InboundError, match="restricted: not a relay member"):
        asyncio.run(client.connect())


def test_connect_raises_if_socket_closes_before_challenge(monkeypatch):
    client, _ = _make_client(monkeypatch)
    with pytest.raises(
        InboundError, match="closed before relay sent an AUTH challenge"
    ):
        asyncio.run(client.connect())


def test_connect_times_out_waiting_for_challenge(monkeypatch):
    monkeypatch.setattr(inbound, "AUTH_TIMEOUT_SECONDS", 0.01)
    client, _ = _make_client(monkeypatch, hang=True)
    with pytest.raises(InboundError, match="timed out waiting for the relay"):
        asyncio.run(client.connect())


def test_subscribe_sends_req_with_filters(monkeypatch):
    client, ws = _connected_client(monkeypatch)
    asyncio.run(client.subscribe(["chan-1", "chan-2"], sub_id="sub-1", since=1000))

    req = json.loads(ws.sent[-1])
    assert req == [
        "REQ",
        "sub-1",
        {"kinds": [9], "#h": ["chan-1", "chan-2"], "since": 1000},
    ]


def test_subscribe_defaults_since_to_now_to_exclude_backlog(monkeypatch):
    """Without an explicit `since`, a relay's entire matching history would
    arrive indistinguishable from live events -- the daemon's owner-pubkey
    gate would then replay every old owner message as a fresh command on
    every restart. Defaulting to "now" is what prevents that.
    """
    client, ws = _connected_client(monkeypatch)
    before = int(time.time())
    asyncio.run(client.subscribe(["chan-1"]))
    after = int(time.time())

    req = json.loads(ws.sent[-1])
    assert before <= req[2]["since"] <= after


def test_subscribe_before_connect_raises():
    client = InboundClient(RELAY_URL, PRIVATE_KEY_HEX)
    with pytest.raises(InboundError, match="before connect"):
        asyncio.run(client.subscribe(["chan-1"]))


def test_events_before_connect_raises():
    client = InboundClient(RELAY_URL, PRIVATE_KEY_HEX)

    async def _drain():
        async for _ in client.events():
            pass

    with pytest.raises(InboundError, match="before connect"):
        asyncio.run(_drain())


def test_events_filters_non_events_and_duplicates(monkeypatch):
    client, ws = _make_client(monkeypatch)
    client._ws = ws  # skip auth; only exercising events() here

    good = sign_event(bytes.fromhex(PRIVATE_KEY_HEX), 1000, 9, [["h", "chan-1"]], "hi")

    ws.push(["EOSE", "sub-1"])
    ws.push(["EVENT", "sub-1", good])
    ws.push(["EVENT", "sub-1", good])  # duplicate, should be skipped

    async def _collect():
        return [event async for event in client.events()]

    events = asyncio.run(_collect())

    assert events == [good]


def test_events_drops_and_logs_an_invalid_signature(monkeypatch, capsys):
    client, ws = _make_client(monkeypatch)
    client._ws = ws  # skip auth; only exercising events() here

    good = sign_event(bytes.fromhex(PRIVATE_KEY_HEX), 1000, 9, [["h", "chan-1"]], "hi")
    # A distinct id (different content) so this isn't caught by the dedup
    # check first -- only the signature itself is wrong.
    forged = sign_event(
        bytes.fromhex(PRIVATE_KEY_HEX), 1001, 9, [["h", "chan-1"]], "bye"
    )
    forged["sig"] = "00" * 64

    ws.push(["EVENT", "sub-1", forged])
    ws.push(["EVENT", "sub-1", good])

    async def _collect():
        return [event async for event in client.events()]

    events = asyncio.run(_collect())

    assert events == [good]
    assert forged["id"] in capsys.readouterr().out


def test_events_surfaces_notice_and_closed_frames(monkeypatch, capsys):
    """A relay can reject part of a multi-channel subscription (e.g.
    "restricted: not a channel member" for one #h value) while still
    delivering events for the rest -- that must be visible, not silently
    indistinguishable from "no messages have arrived yet"."""
    client, ws = _make_client(monkeypatch)
    client._ws = ws  # skip auth; only exercising events() here

    good = sign_event(bytes.fromhex(PRIVATE_KEY_HEX), 1000, 9, [["h", "chan-1"]], "hi")

    ws.push(["NOTICE", "auth-required: authenticate before subscribing"])
    ws.push(["CLOSED", "swingbird", "restricted: not a channel member"])
    ws.push(["EVENT", "sub-1", good])

    async def _collect():
        return [event async for event in client.events()]

    events = asyncio.run(_collect())

    assert events == [good]
    out = capsys.readouterr().out
    assert "auth-required" in out
    assert "restricted: not a channel member" in out


def test_close_closes_open_socket(monkeypatch):
    client, ws = _connected_client(monkeypatch)
    asyncio.run(client.close())

    assert ws.closed is True


def test_close_without_connect_is_a_noop():
    client = InboundClient(RELAY_URL, PRIVATE_KEY_HEX)
    asyncio.run(client.close())
