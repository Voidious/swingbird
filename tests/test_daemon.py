# crispen: skip-file — this file is over max_file_lines because it covers every
# intent daemon.py handles (dispatch/confirm/cancel/recap/recap_action/recap_relay/
# recap_detail/reply-watch). A real fix means splitting it by concern, which is a
# bigger restructuring tracked in a follow-up ticket; this is a stopgap so pre-commit
# stops trying (and failing) to auto-split it in the meantime. Note this also exempts
# the file from crispen's other refactors (duplicate_extractor, function_splitter,
# tuple_dataclass, if_not_else), not just file_limiter -- crispen has no
# file-scoped-but-refactor-specific marker, only skip-file (all refactors) or
# skip=<name> (a single statement/function/entity).
import asyncio
import contextlib
import dataclasses
import json
import sys
import time

import pytest

from swingbird import daemon, outbound
from swingbird.audit import AuditLog
from swingbird.avatar import emoji_avatar_data_url
from swingbird.closed_items import ClosedItemStore
from swingbird.config import (
    AvatarConfig,
    ChannelConfig,
    Config,
    DispatchConfig,
    IdentityConfig,
    LLMConfig,
    OwnerConfig,
    RelayConfig,
    VoiceConfig,
    VoiceTTSConfig,
)
from swingbird.daemon import (
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_CLOSED_ITEMS_PATH,
    DEFAULT_CONFIG_PATH,
    Daemon,
    build_daemon,
)
from swingbird.inbound import InboundError
from swingbird.pending_actions import PendingActionStore
from swingbird.recap import RecapItem
from swingbird.recap_actions import RecapActionStore
from swingbird.recap_close import PendingClose, PendingCloseStore
from swingbird.recap_disambiguation import DisambiguationStore, PendingDisambiguation
from swingbird.router import Intent, IntentRouter

OWNER_PUBKEY = "owner-pubkey"
OTHER_PUBKEY = "someone-else"

CONFIG = Config(
    llm=LLMConfig(base_url="https://x", model="m", api_key_env="X_KEY"),
    relay=RelayConfig(url="wss://relay.example", private_key_env="RELAY_KEY"),
    channels=(
        ChannelConfig(id="chan-1", name="backend", write=True, agents=("Codex",)),
        ChannelConfig(id="chan-2", name="frontend", write=False, agents=("Goose",)),
    ),
    owner=OwnerConfig(pubkey=OWNER_PUBKEY, name="Voidious"),
)


class FakeLLM:
    """Duck-types `LLMClient`: a canned classification and/or recap reply.

    `json_response` backs `complete_json`, used by both the router's
    classification call and (since recap.py switched to structured output)
    the recap-building call -- a scenario spanning both (e.g. a recap
    request) needs a distinct response per call, so a list is consumed in
    call order the same way test_router.py's FakeCompletions does; a bare
    dict is broadcast to every call, which is all any dispatch/confirm/
    cancel-only test needs.
    """

    def __init__(self, json_response=None, text_response=""):
        self._json_response = json_response
        self._text_response = text_response
        self.calls: list[list[dict]] = []

    def complete_json(self, messages):
        self.calls.append(messages)
        responses = (
            self._json_response
            if isinstance(self._json_response, list)
            else [self._json_response]
        )
        index = min(len(self.calls) - 1, len(responses) - 1)
        return responses[index]

    def complete(self, messages):
        self.calls.append(messages)
        return self._text_response


class FakeInbound:
    """Duck-types `InboundClient`: replays a canned list of events."""

    def __init__(self, events, pubkey="bot-pubkey"):
        self._events = events
        self.connected = False
        self.subscribed = None
        self.since = None
        self.pubkey = pubkey

    async def connect(self):
        self.connected = True

    async def subscribe(self, channel_ids, since=None):
        self.subscribed = channel_ids
        self.since = since

    async def events(self):
        for event in self._events:
            yield event


def _event(pubkey=OWNER_PUBKEY, content="hi", tags=None, event_id="evt-1"):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "created_at": 1000,
        "kind": 9,
        "tags": [["p", "some-pubkey"], ["h", "dm-chan"]] if tags is None else tags,
        "content": content,
        "sig": "sig",
    }


def _daemon(
    tmp_path,
    llm,
    inbound=None,
    store=None,
    dm_id="dm-chan",
    recap_store=None,
    disambiguation=None,
    pending_close=None,
    closed_items=None,
    own_pubkey=None,
    config=None,
):
    config = config or CONFIG
    audit = AuditLog(tmp_path / "audit.jsonl")
    router = IntentRouter(llm, config, audit=audit)
    return Daemon(
        config,
        inbound or FakeInbound([]),
        router,
        store or PendingActionStore(),
        llm,
        audit,
        closed_items or ClosedItemStore(tmp_path / "closed_items.jsonl"),
        dm_id=dm_id,
        recap_store=recap_store,
        disambiguation=disambiguation,
        pending_close=pending_close,
        own_pubkey=own_pubkey,
    )


def _sent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        outbound, "send_message", lambda *a, **k: calls.append((a, k)) or "reply-evt"
    )
    return calls


def _handle_event_and_get_first_sent(
    tmp_path,
    llm,
    sent,
    event=None,
    store=None,
    recap_store=None,
    disambiguation=None,
    pending_close=None,
    closed_items=None,
    bot=None,
):
    bot = bot or _daemon(
        tmp_path,
        llm,
        store=store,
        recap_store=recap_store,
        disambiguation=disambiguation,
        pending_close=pending_close,
        closed_items=closed_items,
    )
    asyncio.run(bot._handle_event(event or _event()))
    return sent[0]


def _setup_ignored_event(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    bot = _daemon(tmp_path, llm)
    return sent, llm, bot


def test_ignores_events_not_from_owner(tmp_path, monkeypatch):
    sent, llm, bot = _setup_ignored_event(tmp_path, monkeypatch)

    asyncio.run(bot._handle_event(_event(pubkey=OTHER_PUBKEY)))

    assert sent == []
    assert llm.calls == []


def test_ignores_owner_messages_in_project_channels(tmp_path, monkeypatch):
    """The daemon only acts on commands sent via its DM with the owner --
    an @mention or plain message from the owner in a project channel (e.g.
    swingbird-dev) must never be misread as an instruction. Project channels
    stay subscribed for recap purposes only."""
    sent, llm, bot = _setup_ignored_event(tmp_path, monkeypatch)

    asyncio.run(bot._handle_event(_event(tags=[["h", "chan-1"]])))

    assert sent == []
    assert llm.calls == []


def test_ignores_events_before_the_dm_channel_is_resolved(tmp_path, monkeypatch):
    """Before `run()` resolves the DM id, `_dm_id` is None -- no channel can
    match, so everything is ignored rather than misrouted."""
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    bot = _daemon(tmp_path, llm, dm_id=None)

    asyncio.run(bot._handle_event(_event(tags=[["h", "dm-chan"]])))

    assert sent == []
    assert llm.calls == []


def test_handle_event_does_not_block_the_event_loop_during_a_slow_llm_call(
    tmp_path, monkeypatch
):
    """A slow `_process` call (e.g. a real recap's LLM/buzz-cli round trips)
    must run off the main thread -- otherwise it freezes the event loop for
    its whole duration, starving the inbound WebSocket's read/keepalive
    traffic long enough that the relay (or the `websockets` client's own
    ping timeout) drops the connection as idle. That drop ends `events()`'s
    `async for` silently (a clean close raises nothing), which looks
    exactly like the daemon exiting for no reason right after a recap --
    the bug this test guards against."""
    sent = _sent(monkeypatch)
    # "cancel" (rather than "chit_chat") so the router's retry-on-chit_chat
    # (see router.py) doesn't call complete_json a second time and throw
    # off the timing assertion below.
    llm = FakeLLM(json_response={"intent": "cancel"})
    real_complete_json = llm.complete_json

    def _slow_complete_json(messages):
        time.sleep(0.2)
        return real_complete_json(messages)

    llm.complete_json = _slow_complete_json
    bot = _daemon(tmp_path, llm)
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(15):
            await asyncio.sleep(0.01)
            ticks += 1

    async def scenario():
        start = time.monotonic()
        await asyncio.gather(bot._handle_event(_event()), ticker())
        return time.monotonic() - start

    elapsed = asyncio.run(scenario())

    # If _process ran on the main thread, the ticker couldn't advance until
    # after it finished, so the whole scenario would take at least
    # 0.2 + 15 * 0.01 seconds; running concurrently, it takes about
    # max(0.2, 0.15) seconds instead.
    assert elapsed < 0.3
    assert ticks == 15
    assert sent


def test_recap_reply_is_posted_back_to_the_source_channel(tmp_path, monkeypatch):
    from swingbird import recap_transcript

    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap"},
            {"text": "here's the recap", "items": []},
        ]
    )
    (args, kwargs) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert args == ("dm-chan", "here's the recap")
    assert kwargs == {"reply_to": "evt-1"}


def test_recap_stores_items_for_later_follow_up(tmp_path, monkeypatch):
    from swingbird import recap_transcript

    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    _sent(monkeypatch)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap"},
            {
                "text": "here's the recap",
                "items": [
                    {
                        "channel": "backend",
                        "label": "F4",
                        "summary": "unused-ignore propagation",
                        "instruction": "Fix the deterministic directive trip-check.",
                    }
                ],
            },
        ]
    )
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot._handle_event(_event()))

    stored = bot._recap_store.get("dm-chan")
    assert stored is not None
    assert stored[0].label == "F4"


def test_recap_logs_every_extracted_item(tmp_path, monkeypatch):
    from swingbird import recap_transcript

    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    _sent(monkeypatch)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap"},
            {
                "text": "here's the recap",
                "items": [
                    {
                        "channel": "backend",
                        "label": "F4",
                        "summary": "unused-ignore propagation",
                        "instruction": "Fix the deterministic directive trip-check.",
                    },
                    {
                        "channel": "backend",
                        "label": "F5",
                        "summary": "flaky retry test",
                        "instruction": "Stabilize the retry test.",
                        "is_primary": False,
                    },
                ],
            },
        ]
    )
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot._handle_event(_event()))

    records = [
        json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()
    ]
    (recap_built,) = [r for r in records if r["kind"] == "recap_built"]
    assert recap_built["thread_id"] == "dm-chan"
    assert [item["label"] for item in recap_built["items"]] == ["F4", "F5"]
    assert recap_built["items"][1]["is_primary"] is False


def test_process_tells_the_router_no_recap_is_open_when_none_is_stored(
    tmp_path, monkeypatch
):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "chit_chat"})

    _handle_event_and_get_first_sent(tmp_path, llm, sent)

    system_content = llm.calls[0][0]["content"]
    assert "no open recap" in system_content


def _setup_recap_detail_test(monkeypatch, item, channel="dm-chan"):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with(channel, item)
    llm = FakeLLM(json_response={"intent": "recap_detail", "message": "F4"})
    return sent, recap_store, llm


def test_process_tells_the_router_a_recap_is_open_for_this_thread(
    tmp_path, monkeypatch
):
    sent, recap_store, llm = _setup_recap_detail_test(monkeypatch, F4_ITEM)

    _handle_event_and_get_first_sent(tmp_path, llm, sent, recap_store=recap_store)

    system_content = llm.calls[0][0]["content"]
    assert "already has an open recap" in system_content


F4_ITEM = RecapItem(
    channel="backend",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
)
# A second channel's item, alongside F4_ITEM, for the recap follow-up tests
# that need a reference matching more than one item (see
# _setup_ambiguous_recap_action and the disambiguation tests below).
F5_ITEM = dataclasses.replace(F4_ITEM, label="F5", channel="frontend")


def _recap_store_with(thread_id, *items, channel=None):
    store = RecapActionStore()
    store.set(thread_id, tuple(items), channel=channel)
    return store


def test_recap_action_proposes_dispatch_for_the_matched_item(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response={"intent": "recap_action", "message": "F4"},
        text_response="Fix the deterministic directive trip-check.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    expected_reply = (
        "About to relay to backend (for Codex): 'Fix the deterministic "
        "directive trip-check.'. Confirm to send, or cancel."
    )
    assert args == ("dm-chan", expected_reply)


def test_recap_action_resolves_via_intent_channel_when_message_is_generic(
    tmp_path, monkeypatch
):
    """A generic reference like "the open items" never names a label or the
    channel in its own text -- only `intent.channel`, threaded through to
    `resolve_reference`, can resolve it to the channel's one item."""
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response={
            "intent": "recap_action",
            "channel": "backend",
            "message": "the open items",
        },
        text_response="Fix the deterministic directive trip-check.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    expected_reply = (
        "About to relay to backend (for Codex): 'Fix the deterministic "
        "directive trip-check.'. Confirm to send, or cancel."
    )
    assert args == ("dm-chan", expected_reply)


def test_recap_action_relays_the_dispatch_phrasing_rewrite(tmp_path, monkeypatch):
    # The relayed message is the LLM's rewrite, not the recap item's own
    # instruction verbatim -- that instruction may bundle several candidate
    # next steps the recap couldn't tell apart, so relaying it unchanged
    # would send the owner's whole undecided bundle to the coding agent
    # instead of just what the user picked.
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response={"intent": "recap_action", "message": "F4"},
        text_response="Go ahead and implement F4: add the trip-check.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert "Go ahead and implement F4: add the trip-check." in args[1]


def test_recap_action_sends_the_item_and_reference_to_dispatch_phrasing(
    tmp_path, monkeypatch
):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(json_response={"intent": "recap_action", "message": "F4"})

    _handle_event_and_get_first_sent(tmp_path, llm, sent, recap_store=recap_store)

    rephrase_call = llm.calls[-1]
    user_content = rephrase_call[1]["content"]
    assert F4_ITEM.summary in user_content
    assert F4_ITEM.instruction in user_content
    assert "F4" in user_content


def _confirm_event_pair(bot, event_id_1="evt-1", event_id_2="evt-2"):
    asyncio.run(bot._handle_event(_event(event_id=event_id_1)))
    bot._llm._json_response = {"intent": "confirm"}
    asyncio.run(bot._handle_event(_event(event_id=event_id_2)))


def test_recap_action_confirm_threads_to_the_items_source_event(tmp_path, monkeypatch):
    _sent(monkeypatch)
    relayed = []
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: relayed.append(a) or "posted-evt"
    )
    # Root resolution (see _reply_watch_id) is exercised by its own tests
    # below -- stubbed here so this test's own assertions (about what got
    # relayed) don't depend on it, and so it never shells out for real.
    monkeypatch.setattr(daemon, "fetch_thread_root", lambda *a: "source-evt")
    grounded_item = dataclasses.replace(F4_ITEM, source_event_id="source-evt")
    recap_store = _recap_store_with("dm-chan", grounded_item)
    llm = FakeLLM(
        json_response={"intent": "recap_action", "message": "F4"},
        text_response="Fix the deterministic directive trip-check.",
    )
    bot = _daemon(tmp_path, llm, recap_store=recap_store)

    _confirm_event_pair(bot)

    assert relayed == [
        (
            "chan-1",
            "Fix the deterministic directive trip-check.",
            "Voidious",
            OWNER_PUBKEY,
            "Codex",
            "source-evt",
        )
    ]


def _assert_last_sent(
    sent,
    channel="dm-chan",
    content="Fixed it.\n\nbuzz://message?channel=chan-1&id=reply-1",
    reply_to="evt-2",
):
    (args, kwargs) = sent[-1]
    assert args == (channel, content)
    assert kwargs == {"reply_to": reply_to}


def _setup_grounded_recap_bot(tmp_path, store):
    grounded_item = dataclasses.replace(F4_ITEM, source_event_id="source-evt")
    recap_store = _recap_store_with("dm-chan", grounded_item)
    llm = FakeLLM(
        json_response={"intent": "recap_action", "message": "F4"},
        text_response="Fixed it.",
    )
    bot = _daemon(tmp_path, llm, store=store, recap_store=recap_store)
    return bot, recap_store


def _setup_grounded_dispatch_test(monkeypatch, tmp_path):
    sent, store = _setup_dispatch_test(monkeypatch)
    monkeypatch.setattr(
        daemon, "fetch_thread_root", lambda channel_id, event_id: "thread-root-evt"
    )
    (bot, _) = _setup_grounded_recap_bot(tmp_path, store)
    return sent, store, bot


def test_grounded_dispatch_reply_watch_matches_the_threads_true_root(
    tmp_path, monkeypatch
):
    """A grounded (recap-action) dispatch threads to the item's source
    event, which can itself be nested in an existing thread. A coding
    agent's own status-update reply commonly threads to that thread's
    root instead of to the specific relayed message -- see the daemon
    module docstring -- so the wait-and-summarize must match a reply
    e-tagged only to the resolved root, not to `proposal.reply_to` or the
    relayed event's own id."""
    (sent, _, bot) = _setup_grounded_dispatch_test(monkeypatch, tmp_path)

    async def scenario():
        await bot._handle_event(_event(event_id="evt-1"))
        bot._llm._json_response = {"intent": "confirm"}
        await bot._handle_event(_event(event_id="evt-2"))
        # Neither "posted-evt" (the relayed message) nor "source-evt"
        # (what it was grounded in) -- only the resolved thread root.
        reply_event = _event(
            pubkey="codex-pubkey",
            content="Fixed it.",
            tags=[["h", "chan-1"], ["e", "thread-root-evt", "", "reply"]],
            event_id="reply-1",
        )
        await bot._handle_event(reply_event)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    _assert_last_sent(sent)
    assert bot._reply_watches == {}


def test_self_echo_of_the_relayed_dispatch_does_not_satisfy_its_own_reply_wait(
    tmp_path, monkeypatch
):
    """A grounded dispatch e-tags the thread root it was threaded into --
    exactly the id `_reply_watch_id` just resolved and registered a watch
    against (see the test above). Since the daemon is also subscribed to
    the project channel it just relayed into, its own subscription echoes
    that just-sent message straight back with matching tags, arriving
    before any real reply could exist. Without filtering out the daemon's
    own identity, that self-echo would satisfy the watch instantly and
    `summarize_reply` would "summarize" the relayed instruction itself --
    fabricating a plausible-looking reply from a message that was never a
    reply. Only the later, genuinely different-author reply should resolve
    the wait."""
    (sent, _, bot) = _setup_grounded_dispatch_test(monkeypatch, tmp_path)
    bot._own_pubkey = "bot-pubkey"

    async def scenario():
        await bot._handle_event(_event(event_id="evt-1"))
        bot._llm._json_response = {"intent": "confirm"}
        await bot._handle_event(_event(event_id="evt-2"))
        self_echo = _event(
            pubkey="bot-pubkey",
            content="@Codex Relaying instruction from Voidious: F4",
            tags=[["h", "chan-1"], ["e", "thread-root-evt", "", "root"]],
            event_id="posted-evt",
        )
        await bot._handle_event(self_echo)
        real_reply = _event(
            pubkey="codex-pubkey",
            content="Fixed it.",
            tags=[["h", "chan-1"], ["e", "thread-root-evt", "", "reply"]],
            event_id="reply-1",
        )
        await bot._handle_event(real_reply)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    # The summarizer must only ever have been asked to summarize the real
    # reply's content -- never the self-echoed instruction.
    summarize_call = bot._llm.calls[-1]
    assert summarize_call[-1]["content"] == "Fixed it."
    _assert_last_sent(sent)
    assert bot._reply_watches == {}


def test_reply_watch_id_falls_back_when_root_lookup_fails(
    tmp_path, monkeypatch, capsys
):
    """A thread-root lookup failure (e.g. the project channel isn't
    accessible for reads either) must not sink the confirm reply or the
    wait entirely -- it falls back to watching the relayed event's own id,
    same as before this behavior existed."""
    sent, store = _setup_dispatch_test(monkeypatch)

    def _fail(*a):
        raise outbound.RelayError("not found")

    monkeypatch.setattr(daemon, "fetch_thread_root", _fail)
    (bot, _) = _setup_grounded_recap_bot(tmp_path, store)

    async def scenario():
        await bot._handle_event(_event(event_id="evt-1"))
        bot._llm._json_response = {"intent": "confirm"}
        await bot._handle_event(_event(event_id="evt-2"))
        reply_event = _event(
            pubkey="codex-pubkey",
            content="Fixed it.",
            tags=[["h", "chan-1"], ["e", "posted-evt", "", "reply"]],
            event_id="reply-1",
        )
        await bot._handle_event(reply_event)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    _assert_last_sent(sent)
    assert "failed to resolve thread root for posted-evt: not found" in (
        capsys.readouterr().out
    )


def test_reply_target_ids_checks_every_e_tag_not_just_the_first(tmp_path, monkeypatch):
    """A reply nested two or more levels deep NIP-10-tags both a "root" and
    a "reply" id (root first, per the relay's own convention) -- the watch
    must still match even when the id it's keyed on isn't the first e-tag
    on the reply."""
    sent, store = _setup_dispatch_test(monkeypatch)
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        },
        text_response="Fixed it.",
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)

    async def scenario():
        await bot._handle_event(_event(event_id="evt-1"))
        bot._llm._json_response = {"intent": "confirm"}
        await bot._handle_event(_event(event_id="evt-2"))
        reply_event = _event(
            pubkey="codex-pubkey",
            content="Fixed it.",
            tags=[
                ["h", "chan-1"],
                ["e", "some-older-root", "", "root"],
                ["e", "posted-evt", "", "reply"],
            ],
            event_id="reply-1",
        )
        await bot._handle_event(reply_event)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    _assert_last_sent(sent)


def test_resolve_reply_watch_discards_an_already_done_future(tmp_path):
    """Defends against a narrow race between a reply-wait timing out and a
    reply for the same id arriving before the timeout's own cleanup pops
    the dict entry (see _await_and_summarize's TimeoutError branch): the
    stale, already-resolved future must be discarded, not fulfilled again,
    and the event must be reported as unhandled so it still falls through
    to normal routing."""
    bot = _daemon(tmp_path, FakeLLM())

    async def scenario():
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        bot._reply_watches["stale-evt"] = future
        event = _event(tags=[["h", "dm-chan"], ["e", "stale-evt", "", "reply"]])
        return bot._resolve_reply_watch(event)

    handled = asyncio.run(scenario())

    assert handled is False
    assert bot._reply_watches == {}


def _assert_recap_reference(
    recap_reference, reference="F4", channel="backend", label="F4"
):
    assert recap_reference["reference"] == reference
    assert recap_reference["channel"] == channel
    assert recap_reference["label"] == label


def _run_event_and_get_recap_reference(tmp_path, llm, recap_store, event_factory):
    bot = _daemon(tmp_path, llm, recap_store=recap_store)
    asyncio.run(bot._handle_event(event_factory()))

    records = [
        json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()
    ]
    (recap_reference,) = [r for r in records if r["kind"] == "recap_reference"]
    return recap_reference


def test_recap_action_logs_the_resolved_reference(tmp_path, monkeypatch):
    _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(json_response={"intent": "recap_action", "message": "F4"})

    recap_reference = _run_event_and_get_recap_reference(
        tmp_path, llm, recap_store, _event
    )
    assert recap_reference["recap_kind"] == "recap_action"
    _assert_recap_reference(recap_reference)


def test_recap_action_without_a_recent_recap_replies_helpfully(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "recap_action", "message": "F4"})

    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)

    assert args == (
        "dm-chan",
        (
            "Couldn't do that: I don't have a recent recap to reference here -- "
            "ask for a recap first."
        ),
    )


def _setup_ambiguous_recap_action(monkeypatch):
    """Shared Arrange phase for the ambiguous-reference tests below --
    same secondary item, recap store, and router response, differing only
    in what's asserted afterward."""
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM, F5_ITEM)
    llm = FakeLLM(json_response={"intent": "recap_action", "message": "all"})
    return sent, recap_store, llm, F5_ITEM


def test_recap_action_ambiguous_reference_lists_candidates(tmp_path, monkeypatch):
    sent, recap_store, llm, _ = _setup_ambiguous_recap_action(monkeypatch)

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args[1].startswith("Couldn't do that: That matches more than one item")
    assert "backend/F4" in args[1]
    assert "frontend/F5" in args[1]


def test_recap_action_ambiguous_reference_stores_a_pending_disambiguation(
    tmp_path, monkeypatch
):
    sent, recap_store, llm, other_item = _setup_ambiguous_recap_action(monkeypatch)
    disambiguation = DisambiguationStore()

    _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store, disambiguation=disambiguation
    )

    pending = disambiguation.get("dm-chan")
    assert pending.kind == "recap_action"
    assert pending.candidates == (F4_ITEM, other_item)
    assert pending.intent.message == "all"


def _pending_recap_action_disambiguation(candidates, message="all"):
    return PendingDisambiguation(
        "recap_action", candidates, Intent(kind="recap_action", message=message)
    )


def _pending_recap_relay_disambiguation(candidates, item_reference, message):
    return PendingDisambiguation(
        "recap_relay",
        candidates,
        Intent(kind="recap_relay", item_reference=item_reference, message=message),
    )


def _setup_recap_action_disambiguation(monkeypatch):
    """Shared setup for tests exercising an already-open recap_action
    disambiguation over F4_ITEM/F5_ITEM, differing only in the reply event
    and/or router response that follows."""
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM, F5_ITEM)
    pending = _pending_recap_action_disambiguation((F4_ITEM, F5_ITEM))
    disambiguation = DisambiguationStore()
    disambiguation.set("dm-chan", pending)
    return sent, recap_store, disambiguation, pending


def test_disambiguation_reply_by_number_resumes_recap_action(tmp_path, monkeypatch):
    sent, recap_store, disambiguation, _ = _setup_recap_action_disambiguation(
        monkeypatch
    )
    llm = FakeLLM(text_response="Fix the deterministic directive trip-check.")

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path,
        llm,
        sent,
        event=_event(content="1"),
        recap_store=recap_store,
        disambiguation=disambiguation,
    )

    assert args == (
        "dm-chan",
        (
            "About to relay to backend (for Codex): "
            "'Fix the deterministic directive trip-check.'. "
            "Confirm to send, or cancel."
        ),
    )
    # The router's own classification is never consulted -- resolving the
    # disambiguation answer short-circuits it (see daemon._process).
    assert len(llm.calls) == 1
    assert disambiguation.get("dm-chan") is None


def test_disambiguation_reply_by_label_resumes_recap_relay(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM, F5_ITEM)
    disambiguation = DisambiguationStore()
    disambiguation.set(
        "dm-chan",
        _pending_recap_relay_disambiguation(
            (F4_ITEM, F5_ITEM),
            "the open item",
            "couldn't we just pre-compile it?",
        ),
    )
    llm = FakeLLM(text_response="couldn't we just pre-compile it?")

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path,
        llm,
        sent,
        event=_event(content="backend/F4"),
        recap_store=recap_store,
        disambiguation=disambiguation,
    )

    assert args == (
        "dm-chan",
        (
            "About to relay to backend (for Codex): "
            '"couldn\'t we just pre-compile it?". Confirm to send, or cancel.'
        ),
    )
    assert disambiguation.get("dm-chan") is None


def test_disambiguation_unrecognized_reply_falls_through_to_routing(
    tmp_path, monkeypatch
):
    sent, recap_store, disambiguation, pending = _setup_recap_action_disambiguation(
        monkeypatch
    )
    llm = FakeLLM(json_response={"intent": "chit_chat"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path,
        llm,
        sent,
        event=_event(content="what's the weather like"),
        recap_store=recap_store,
        disambiguation=disambiguation,
    )

    assert args == ("dm-chan", daemon._CHIT_CHAT_REPLY)
    # An unrelated message doesn't answer the open question -- it stays
    # pending for a later reply, rather than being silently dropped.
    assert disambiguation.get("dm-chan") is pending


def test_cancel_clears_a_pending_disambiguation(tmp_path, monkeypatch):
    sent, _, disambiguation, _ = _setup_recap_action_disambiguation(monkeypatch)
    llm = FakeLLM(json_response={"intent": "cancel"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path,
        llm,
        sent,
        event=_event(content="cancel"),
        disambiguation=disambiguation,
    )

    assert args == ("dm-chan", "Cancelled -- nothing was sent.")
    assert disambiguation.get("dm-chan") is None


def test_fresh_recap_clears_a_stale_pending_disambiguation(tmp_path, monkeypatch):
    from swingbird import recap_transcript

    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    sent, recap_store, disambiguation, _ = _setup_recap_action_disambiguation(
        monkeypatch
    )
    llm = FakeLLM(
        json_response=[
            {"intent": "recap"},
            {"text": "here's the recap", "items": []},
        ]
    )

    _handle_event_and_get_first_sent(
        tmp_path,
        llm,
        sent,
        recap_store=recap_store,
        disambiguation=disambiguation,
    )

    assert disambiguation.get("dm-chan") is None


def test_recap_action_unknown_reference_becomes_a_reply(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(json_response={"intent": "recap_action", "message": "F9"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == ("dm-chan", "Couldn't do that: no recap item matches 'F9'")


def test_recap_close_proposes_closing_the_matched_item(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap_close", "message": "F4"},
            {"indices": [1]},
        ]
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == (
        "dm-chan",
        (
            "Close backend/F4 -- it won't be shown as open in future "
            "recaps? Confirm to close, or cancel."
        ),
    )


def test_recap_close_stores_a_pending_close(tmp_path, monkeypatch):
    _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap_close", "message": "F4"},
            {"indices": [1]},
        ]
    )
    bot = _daemon(tmp_path, llm, recap_store=recap_store)

    asyncio.run(bot._handle_event(_event()))

    pending = bot._pending_close.get("dm-chan")
    assert pending.items == (F4_ITEM,)


def test_recap_close_logs_the_resolved_reference(tmp_path, monkeypatch):
    _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap_close", "message": "F4"},
            {"indices": [1]},
        ]
    )

    recap_reference = _run_event_and_get_recap_reference(
        tmp_path, llm, recap_store, _event
    )
    assert recap_reference["recap_kind"] == "recap_close"
    _assert_recap_reference(recap_reference)


def test_recap_close_a_single_selection_ignores_a_sibling_sharing_its_source_event(
    tmp_path, monkeypatch
):
    """A recap message can cover more than one unrelated item for the same
    project (e.g. two independent PRIMARY items) -- selecting one of them
    by an exact label match must close only that item, never re-add the
    sibling just because they were grounded on the same message. This is
    the regression Voidious hit: no phrasing of "close dripbird F6 lint
    residue" could close just that item, since it kept getting batched with
    the unrelated "Merge 0.3.3 branch" item grounded on the same message."""
    sent = _sent(monkeypatch)
    sibling = dataclasses.replace(
        F4_ITEM, label="F5", source_event_id="src-evt", is_primary=False
    )
    grounded = dataclasses.replace(F4_ITEM, source_event_id="src-evt")
    recap_store = _recap_store_with("dm-chan", grounded, sibling)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap_close", "message": "F4"},
            {"indices": [1]},
        ]
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args[1] == (
        "Close backend/F4 -- it won't be shown as open in future recaps? "
        "Confirm to close, or cancel."
    )


def test_recap_close_without_a_source_event_closes_just_itself(tmp_path, monkeypatch):
    _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap_close", "message": "F4"},
            {"indices": [1]},
        ]
    )
    bot = _daemon(tmp_path, llm, recap_store=recap_store)

    asyncio.run(bot._handle_event(_event()))

    assert bot._pending_close.get("dm-chan").items == (F4_ITEM,)


def test_recap_close_without_a_recent_recap_replies_helpfully(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "recap_close", "message": "F4"})

    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)

    assert args == (
        "dm-chan",
        (
            "Couldn't do that: I don't have a recent recap to reference here -- "
            "ask for a recap first."
        ),
    )
    # Nothing to select against -- the selection LLM call is never made.
    assert len(llm.calls) == 1


def test_recap_close_all_items_for_one_channel_closes_every_match(
    tmp_path, monkeypatch
):
    """ "Close all swingbird items" (bullet 1/3 of Voidious's request) should
    close every matched item in one confirmation, not ask "which did you
    mean" the way a single-item recap_action/recap_relay reference would --
    see recap_close_selection.py."""
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM, F5_ITEM)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap_close", "message": "all"},
            {"indices": [1, 2]},
        ]
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args[1].startswith("That request covers 2 items:")
    assert "backend/F4" in args[1]
    assert "frontend/F5" in args[1]


def test_recap_close_no_match_becomes_a_reply(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap_close", "message": "F9"},
            {"indices": []},
        ]
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == ("dm-chan", "Couldn't do that: no recap item matches 'F9'")


def test_reset_closed_with_channel_clears_only_that_channel(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    closed_items.close(F4_ITEM)
    closed_items.close(F5_ITEM)
    llm = FakeLLM(json_response={"intent": "reset_closed", "channel": "backend"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, closed_items=closed_items
    )

    assert args == ("dm-chan", "Reset 1 closed item for backend.")
    assert [i.channel for i in closed_items.for_channels(None, since=0)] == ["frontend"]


def test_reset_closed_without_channel_clears_every_project(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    closed_items.close(F4_ITEM)
    closed_items.close(F5_ITEM)
    llm = FakeLLM(json_response={"intent": "reset_closed", "channel": None})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, closed_items=closed_items
    )

    assert args == ("dm-chan", "Reset 2 closed items for all projects.")
    assert closed_items.for_channels(None, since=0) == ()


def test_reset_closed_with_nothing_to_clear_replies_helpfully(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    llm = FakeLLM(json_response={"intent": "reset_closed", "channel": "backend"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, closed_items=closed_items
    )

    assert args == ("dm-chan", "Nothing to reset -- no closed items for backend.")


def test_reset_closed_unknown_channel_becomes_a_reply(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "reset_closed", "channel": "nope"})

    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)

    assert args == (
        "dm-chan",
        "Couldn't do that: unknown project channel: 'nope'",
    )


def test_reset_closed_logs_channel_and_count(tmp_path, monkeypatch):
    _sent(monkeypatch)
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    closed_items.close(F4_ITEM)
    llm = FakeLLM(json_response={"intent": "reset_closed", "channel": "backend"})
    bot = _daemon(tmp_path, llm, closed_items=closed_items)

    asyncio.run(bot._handle_event(_event()))

    records = [
        json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()
    ]
    (record,) = [r for r in records if r["kind"] == "closed_items_reset"]
    assert record["thread_id"] == "dm-chan"
    assert record["channel"] == "backend"
    assert record["count"] == 1


def _pending_close_bot(tmp_path, recap_store, closed_items=None):
    pending_close = PendingCloseStore()
    pending_close.set("dm-chan", PendingClose((F4_ITEM,)))
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    bot = _daemon(
        tmp_path,
        llm,
        recap_store=recap_store,
        pending_close=pending_close,
        closed_items=closed_items,
    )
    return bot, llm


def test_pending_close_confirm_persists_and_replies(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    bot, llm = _pending_close_bot(tmp_path, recap_store, closed_items=closed_items)

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, event=_event(content="yes"), bot=bot
    )

    assert args == ("dm-chan", "Closed item: backend/F4.")
    assert bot._pending_close.get("dm-chan") is None
    assert len(closed_items.for_channels(None, since=0)) == 1


def test_pending_close_confirm_logs_closed_items(tmp_path, monkeypatch):
    _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    bot, _ = _pending_close_bot(tmp_path, recap_store)

    asyncio.run(bot._handle_event(_event(content="yes")))

    records = [
        json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()
    ]
    (record,) = [r for r in records if r["kind"] == "closed_items"]
    assert record["thread_id"] == "dm-chan"
    assert record["items"] == [
        {"channel": "backend", "label": "F4", "source_event_id": None}
    ]


def test_pending_close_confirm_multiple_items_pluralizes(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM, F5_ITEM)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    pending_close = PendingCloseStore()
    pending_close.set("dm-chan", PendingClose((F4_ITEM, F5_ITEM)))
    bot = _daemon(tmp_path, llm, recap_store=recap_store, pending_close=pending_close)

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, event=_event(content="yes"), bot=bot
    )

    assert args == ("dm-chan", "Closed items: backend/F4, frontend/F5.")


def test_pending_close_cancel_discards_without_closing(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    closed_items = ClosedItemStore(tmp_path / "closed_items.jsonl")
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    bot, llm = _pending_close_bot(tmp_path, recap_store, closed_items=closed_items)

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, event=_event(content="cancel"), bot=bot
    )

    assert args == ("dm-chan", "Cancelled -- nothing was closed.")
    assert bot._pending_close.get("dm-chan") is None
    assert closed_items.for_channels(None, since=0) == ()


def test_pending_close_unrecognized_reply_falls_through_to_routing(
    tmp_path, monkeypatch
):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    bot, llm = _pending_close_bot(tmp_path, recap_store)

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path,
        llm,
        sent,
        event=_event(content="what's the weather like"),
        bot=bot,
    )

    assert args == ("dm-chan", daemon._CHIT_CHAT_REPLY)
    assert bot._pending_close.get("dm-chan") is not None
    # The router's retry-on-chit_chat (see router.py) fires twice here since
    # both calls return chit_chat -- what matters is that routing happened
    # at all, i.e. the pending close was left unresolved and fell through.
    assert len(llm.calls) == 2


def test_recap_relay_proposes_dispatch_with_the_relayed_message(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    relayed_text = "couldn't we just pre-compile it?"
    llm = FakeLLM(
        json_response={
            "intent": "recap_relay",
            "item_reference": "F4",
            "message": relayed_text,
        },
        text_response=relayed_text,
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    expected_reply = (
        f"About to relay to backend (for Codex): {relayed_text!r}. "
        "Confirm to send, or cancel."
    )
    assert args == ("dm-chan", expected_reply)


def test_recap_relay_sends_the_item_reference_and_message_to_relay_with_context(
    tmp_path, monkeypatch
):
    # The relayed call gets the item's own context (summary/instruction) so
    # it can resolve an ambiguous "it" -- and the user's message, which is
    # what actually gets forwarded (see recap_relay.py).
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response={
            "intent": "recap_relay",
            "item_reference": "F4",
            "message": "couldn't we just pre-compile it?",
        }
    )

    _handle_event_and_get_first_sent(tmp_path, llm, sent, recap_store=recap_store)

    relay_call = llm.calls[-1]
    user_content = relay_call[1]["content"]
    assert F4_ITEM.summary in user_content
    assert F4_ITEM.instruction in user_content
    assert "F4" in user_content
    assert "couldn't we just pre-compile it?" in user_content


def test_recap_relay_without_a_message_replies_helpfully(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(json_response={"intent": "recap_relay", "item_reference": "F4"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == (
        "dm-chan",
        "I didn't catch what to relay -- what should I tell the agent?",
    )


def test_recap_relay_logs_the_resolved_reference(tmp_path, monkeypatch):
    _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response={
            "intent": "recap_relay",
            "item_reference": "F4",
            "message": "couldn't we just pre-compile it?",
        }
    )

    recap_reference = _run_event_and_get_recap_reference(
        tmp_path, llm, recap_store, _event
    )
    assert recap_reference["recap_kind"] == "recap_relay"
    _assert_recap_reference(recap_reference)


def test_recap_relay_without_a_recent_recap_replies_helpfully(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(
        json_response={
            "intent": "recap_relay",
            "item_reference": "F4",
            "message": "couldn't we just pre-compile it?",
        }
    )

    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)

    assert args == (
        "dm-chan",
        (
            "Couldn't do that: I don't have a recent recap to reference here -- "
            "ask for a recap first."
        ),
    )


def test_recap_relay_confirm_threads_to_the_items_source_event(tmp_path, monkeypatch):
    _sent(monkeypatch)
    relayed = []
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: relayed.append(a) or "posted-evt"
    )
    monkeypatch.setattr(daemon, "fetch_thread_root", lambda *a: "source-evt")
    grounded_item = dataclasses.replace(F4_ITEM, source_event_id="source-evt")
    recap_store = _recap_store_with("dm-chan", grounded_item)
    relayed_text = "couldn't we just pre-compile it?"
    llm = FakeLLM(
        json_response={
            "intent": "recap_relay",
            "item_reference": "F4",
            "message": relayed_text,
        },
        text_response=relayed_text,
    )
    bot = _daemon(tmp_path, llm, recap_store=recap_store)

    _confirm_event_pair(bot)

    assert relayed == [
        (
            "chan-1",
            relayed_text,
            "Voidious",
            OWNER_PUBKEY,
            "Codex",
            "source-evt",
        )
    ]


def _setup_recap_test(monkeypatch, channel_id="dm-chan", item=F4_ITEM):
    monkeypatch.setattr(daemon, "fetch_recent_messages", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with(channel_id, item)
    return sent, recap_store


def test_recap_detail_elaborates_using_the_llm_not_a_flat_echo(tmp_path, monkeypatch):
    """recap_detail must not just replay the recap's own summary/instruction
    (see the recap follow-up plan's Goal 1) -- it goes through
    recap_detail.elaborate, an LLM call, and returns whatever that says."""
    sent, recap_store = _setup_recap_test(monkeypatch)
    llm = FakeLLM(
        json_response={"intent": "recap_detail", "message": "F4"},
        text_response="It's blocked on a design call about the trip-check scope.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == (
        "dm-chan",
        "It's blocked on a design call about the trip-check scope.",
    )


def test_recap_detail_resolves_via_intent_channel_when_message_is_generic(
    tmp_path, monkeypatch
):
    """Same gap as recap_action's equivalent test above, for recap_detail:
    "the open items" carries no label or channel text of its own."""
    sent, recap_store = _setup_recap_test(monkeypatch)
    llm = FakeLLM(
        json_response={
            "intent": "recap_detail",
            "channel": "backend",
            "message": "the open items",
        },
        text_response="More detail on the open items.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == ("dm-chan", "More detail on the open items.")


def test_recap_detail_elaborates_every_non_primary_item_for_additional_items(
    tmp_path, monkeypatch
):
    """ "tell me more about the additional items" should resolve to every
    non-primary item for the channel and elaborate on all of them in one
    call -- not just the leading item the recap text itself narrated."""
    monkeypatch.setattr(daemon, "fetch_recent_messages", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    other_item = RecapItem(
        channel="backend",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )
    third_item = RecapItem(
        channel="backend",
        label="F6",
        summary="lower priority follow-up",
        instruction="Revisit once F4/F5 land.",
        is_primary=False,
    )
    recap_store = _recap_store_with("dm-chan", F4_ITEM, other_item, third_item)
    llm = FakeLLM(
        json_response={
            "intent": "recap_detail",
            "channel": "backend",
            "message": "the additional items",
        },
        text_response=(
            "**backend -- F5:** More on F5.\n\n**backend -- F6:** More on F6."
        ),
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == (
        "dm-chan",
        "**backend -- F5:** More on F5.\n\n**backend -- F6:** More on F6.",
    )
    user_content = llm.calls[-1][1]["content"]
    assert "undefined-sentinel cloneDeep split" in user_content
    assert "lower priority follow-up" in user_content
    assert "unused-ignore propagation" not in user_content


def test_recap_detail_backfills_a_paragraph_the_llm_collapsed_away(
    tmp_path, monkeypatch
):
    """Live bug: resolving "the additional items" correctly found every
    non-primary item, but the elaboration call itself sometimes collapses
    several items into one combined answer instead of giving each its own
    paragraph, silently dropping the rest from what the user sees (see
    `recap_detail._backfill_missing_paragraphs`). The daemon must still
    surface every resolved item even when the LLM's own response doesn't
    name them."""
    monkeypatch.setattr(daemon, "fetch_recent_messages", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    other_item = RecapItem(
        channel="backend",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )
    third_item = RecapItem(
        channel="backend",
        label="F6",
        summary="lower priority follow-up",
        instruction="Revisit once F4/F5 land.",
        is_primary=False,
    )
    recap_store = _recap_store_with("dm-chan", F4_ITEM, other_item, third_item)
    llm = FakeLLM(
        json_response={
            "intent": "recap_detail",
            "channel": "backend",
            "message": "the additional items",
        },
        text_response="More on F5 and F6.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == (
        "dm-chan",
        (
            "More on F5 and F6."
            "\n\n**backend -- F5:** undefined-sentinel cloneDeep split "
            "Next: Design a fix for the cloneDeep split."
            "\n\n**backend -- F6:** lower priority follow-up "
            "Next: Revisit once F4/F5 land."
        ),
    )


def test_recap_detail_falls_back_to_the_recaps_own_scoped_channel(
    tmp_path, monkeypatch
):
    """Live bug (2026-09-12): after a project-scoped recap ("recap
    dripbird"), "tell me about the additional items" names no channel and
    the router's classifier -- seeing only this one message, with no
    memory of the earlier recap -- has no way to report one either. The
    store already knows which channel the recap it's holding was scoped
    to, so that should be used without needing `intent.channel` set at
    all."""
    monkeypatch.setattr(daemon, "fetch_recent_messages", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    other_item = RecapItem(
        channel="backend",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )
    recap_store = _recap_store_with("dm-chan", F4_ITEM, other_item, channel="backend")
    llm = FakeLLM(
        json_response={"intent": "recap_detail", "message": "the additional items"},
        text_response="More on F5.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == ("dm-chan", "More on F5.")


def test_recap_stores_the_scoped_channel_for_a_named_channel_recap(
    tmp_path, monkeypatch
):
    from swingbird import recap_transcript

    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    _sent(monkeypatch)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap", "channel": "backend"},
            {"text": "here's the recap", "items": []},
        ]
    )
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot._handle_event(_event()))

    assert bot._recap_store.channel_for("dm-chan") == "backend"


def test_recap_stores_no_scoped_channel_for_an_all_channels_recap(
    tmp_path, monkeypatch
):
    from swingbird import recap_transcript

    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    _sent(monkeypatch)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap"},
            {"text": "here's the recap", "items": []},
        ]
    )
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot._handle_event(_event()))

    assert bot._recap_store.channel_for("dm-chan") is None


def test_recap_detail_without_a_grounded_item_skips_the_thread_fetch(
    tmp_path, monkeypatch
):
    """F4_ITEM has no `source_event_id` -- nothing to fetch, so
    `fetch_thread_messages` must never be called for it."""
    thread_calls = []
    monkeypatch.setattr(
        daemon,
        "fetch_thread_messages",
        lambda *a, **k: thread_calls.append((a, k)) or [],
    )
    sent, recap_store = _setup_recap_test(monkeypatch)
    llm = FakeLLM(json_response={"intent": "recap_detail", "message": "F4"})

    _handle_event_and_get_first_sent(tmp_path, llm, sent, recap_store=recap_store)

    assert thread_calls == []


GROUNDED_F4_ITEM = RecapItem(
    channel="backend",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
    source_event_id="src-evt",
)


def test_recap_detail_fetches_the_grounded_thread_and_dm_history(tmp_path, monkeypatch):
    monkeypatch.setattr(
        daemon,
        "fetch_thread_messages",
        lambda channel_id, event_id: [
            {"created_at": 1, "content": f"original message ({channel_id}/{event_id})"}
        ],
    )
    monkeypatch.setattr(
        daemon,
        "fetch_recent_messages",
        lambda *a, **k: [{"created_at": 2, "content": "the recap dm history"}],
    )
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", GROUNDED_F4_ITEM)
    llm = FakeLLM(
        json_response={"intent": "recap_detail", "message": "F4"},
        text_response="more detail",
    )

    _handle_event_and_get_first_sent(tmp_path, llm, sent, recap_store=recap_store)

    user_content = llm.calls[-1][1]["content"]
    assert "original message (chan-1/src-evt)" in user_content
    assert "the recap dm history" in user_content


UNKNOWN_CHANNEL_ITEM = RecapItem(
    channel="ghost-channel",
    label="F4",
    summary="unused-ignore propagation",
    instruction="Fix the deterministic directive trip-check.",
    source_event_id="src-evt",
)


def test_recap_detail_unknown_channel_becomes_a_helpful_reply(tmp_path, monkeypatch):
    """`item.channel` is LLM-sourced (see recap.py) and never trusted
    blindly (§5) -- an unresolvable name is a reportable error, not a
    silent drop of the thread context or a crash."""
    sent, recap_store, llm = _setup_recap_detail_test(monkeypatch, UNKNOWN_CHANNEL_ITEM)

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args[1] == (
        "Couldn't do that: recap item names an unknown channel: 'ghost-channel'"
    )


def test_recap_detail_without_a_recent_recap_replies_helpfully(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "recap_detail", "message": "F4"})

    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)

    assert args[1].startswith("Couldn't do that: I don't have a recent recap")


def test_recap_detail_from_intent_selects_detailed_prompt(tmp_path, monkeypatch):
    from swingbird import recap, recap_transcript

    monkeypatch.setattr(recap_transcript, "fetch_messages_since", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap", "detail": "detailed"},
            {"text": "here's the detailed recap", "items": []},
        ]
    )

    _handle_event_and_get_first_sent(tmp_path, llm, sent)

    system_prompt = llm.calls[-1][0]["content"]
    assert system_prompt == recap._detailed_system_prompt(3)


def test_recap_list_renders_items_without_calling_the_llm(tmp_path, monkeypatch):
    """Unlike recap_detail, recap_list must never elaborate -- it lists the
    recap's own stored summary/instruction, so only the router's own
    classification call ever reaches the LLM."""
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(json_response={"intent": "recap_list", "message": "F4"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == (
        "dm-chan",
        (
            "**backend -- F4:** unused-ignore propagation "
            "Fix the deterministic directive trip-check."
        ),
    )
    assert len(llm.calls) == 1


def test_recap_list_resolves_via_intent_channel_when_message_is_generic(
    tmp_path, monkeypatch
):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM)
    llm = FakeLLM(
        json_response={
            "intent": "recap_list",
            "channel": "backend",
            "message": "the open items",
        }
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert "unused-ignore propagation" in args[1]


def test_recap_list_lists_every_non_primary_item_for_additional_items(
    tmp_path, monkeypatch
):
    sent = _sent(monkeypatch)
    other_item = RecapItem(
        channel="backend",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )
    third_item = RecapItem(
        channel="backend",
        label="F6",
        summary="lower priority follow-up",
        instruction="Revisit once F4/F5 land.",
        is_primary=False,
    )
    recap_store = _recap_store_with("dm-chan", F4_ITEM, other_item, third_item)
    llm = FakeLLM(
        json_response={
            "intent": "recap_list",
            "channel": "backend",
            "message": "the additional items",
        }
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert "undefined-sentinel cloneDeep split" in args[1]
    assert "lower priority follow-up" in args[1]
    assert "unused-ignore propagation" not in args[1]


def test_recap_list_falls_back_to_the_recaps_own_scoped_channel(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    other_item = RecapItem(
        channel="backend",
        label="F5",
        summary="undefined-sentinel cloneDeep split",
        instruction="Design a fix for the cloneDeep split.",
        is_primary=False,
    )
    recap_store = _recap_store_with("dm-chan", F4_ITEM, other_item, channel="backend")
    llm = FakeLLM(json_response={"intent": "recap_list", "message": "the other items"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert "undefined-sentinel cloneDeep split" in args[1]


def test_recap_list_no_other_items_says_so_plainly(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", F4_ITEM, channel="backend")
    llm = FakeLLM(json_response={"intent": "recap_list", "message": "the other items"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args[1].startswith("There are no other open items for that channel")
    assert "unused-ignore propagation" in args[1]


def test_recap_list_unmapped_channel_omits_the_source_link(tmp_path, monkeypatch):
    """Unlike recap_detail (which must fetch the item's source thread and so
    raises for a channel `config` doesn't recognize), recap_list never looks
    up the channel for anything but an optional link -- an unmapped channel
    just renders without one, same as recap_detail.append_source_links does
    for its own paragraphs."""
    sent = _sent(monkeypatch)
    recap_store = _recap_store_with("dm-chan", UNKNOWN_CHANNEL_ITEM)
    llm = FakeLLM(json_response={"intent": "recap_list", "message": "F4"})

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args[1] == (
        "**ghost-channel -- F4:** unused-ignore propagation "
        "Fix the deterministic directive trip-check."
    )


def test_recap_list_without_a_recent_recap_replies_helpfully(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "recap_list", "message": "F4"})

    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)

    assert args[1].startswith("Couldn't do that: I don't have a recent recap")


def test_dispatch_proposes_and_asks_for_confirmation(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    store = PendingActionStore()
    llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix the login timeout bug",
        }
    )
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent, store=store)
    expected_reply = (
        "About to relay to backend (for Codex): 'fix the login timeout bug'. "
        "Confirm to send, or cancel."
    )
    assert args == ("dm-chan", expected_reply)
    assert store.get("dm-chan") is not None


def test_dispatch_missing_target_asks_a_clarifying_question(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    store = PendingActionStore()
    llm = FakeLLM(
        json_response={"intent": "dispatch", "channel": None, "message": None}
    )
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent, store=store)
    assert "which project channel" in args[1]
    assert store.get("dm-chan") is None


def test_clarify_response_reuses_the_dispatch_path(tmp_path, monkeypatch):
    _sent(monkeypatch)
    store = PendingActionStore()
    llm = FakeLLM(
        json_response={
            "intent": "clarify_response",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix the bug",
        }
    )
    bot = _daemon(tmp_path, llm, store=store)

    asyncio.run(bot._handle_event(_event()))

    assert store.get("dm-chan") is not None


def test_confirm_posts_and_replies(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    relayed = []
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: relayed.append(a) or "posted-evt"
    )
    store = PendingActionStore()
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        }
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)
    # A bare "yes" typed as a new message, not a reply -- confirming still
    # resolves because it lands in the same channel the proposal was made in.
    _confirm_event_pair(bot)

    assert relayed == [("chan-1", "fix it", "Voidious", OWNER_PUBKEY, "Codex", None)]
    (args, _) = sent[-1]
    assert args == (
        "dm-chan",
        "Confirmed and relayed: buzz://message?channel=chan-1&id=posted-evt",
    )
    assert store.get("dm-chan") is None


def test_confirm_tells_the_router_a_dispatch_is_pending(tmp_path, monkeypatch):
    """A bare "confirm" carries no content of its own to classify from (see
    `router._PENDING_DISPATCH_NOTE`) -- the daemon must tell the router a
    proposal is actually waiting on this thread, or the classifier has no
    way to know "confirm" means confirm and not chit_chat (observed live:
    exactly this collapsed to chit_chat once, see the router module's own
    reasoning for `_PENDING_DISPATCH_NOTE`)."""
    _sent(monkeypatch)
    monkeypatch.setattr(outbound, "relay_dispatch", lambda *a: "posted-evt")
    store = PendingActionStore()
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        }
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)

    _confirm_event_pair(bot)

    first_call, second_call = dispatch_llm.calls
    assert "no dispatch proposal awaiting confirm" in first_call[0]["content"]
    assert "has a dispatch proposal awaiting confirm or" in second_call[0]["content"]


def test_confirm_against_an_inaccessible_channel_becomes_a_helpful_reply(
    tmp_path, monkeypatch
):
    """A private project channel the daemon isn't a member of shouldn't
    crash the daemon or vanish into a console-only log -- the owner should
    get told plainly that the dispatch didn't go through."""
    sent = _sent(monkeypatch)

    def _fail(*a):
        raise outbound.RelayError("restricted: not a channel member")

    monkeypatch.setattr(outbound, "relay_dispatch", _fail)
    store = PendingActionStore()
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        }
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)
    _confirm_event_pair(bot)

    (args, _) = sent[-1]
    assert args == (
        "dm-chan",
        "Couldn't do that: restricted: not a channel member",
    )
    assert store.get("dm-chan") is None


def _setup_dispatch_test(monkeypatch):
    sent = _sent(monkeypatch)
    monkeypatch.setattr(outbound, "relay_dispatch", lambda *a: "posted-evt")
    store = PendingActionStore()
    return sent, store


def test_confirm_summarizes_the_working_agents_reply(tmp_path, monkeypatch):
    sent, store = _setup_dispatch_test(monkeypatch)
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        },
        text_response="Fixed the bug and added a regression test.",
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)

    async def scenario():
        await bot._handle_event(_event(event_id="evt-1"))
        bot._llm._json_response = {"intent": "confirm"}
        await bot._handle_event(_event(event_id="evt-2"))
        reply_event = _event(
            pubkey="codex-pubkey",
            content="Fixed the bug and added a regression test.",
            tags=[["h", "chan-1"], ["e", "posted-evt", "", "reply"]],
            event_id="reply-1",
        )
        await bot._handle_event(reply_event)
        await asyncio.sleep(0.05)  # let the background summarize task finish

    asyncio.run(scenario())

    (args, kwargs) = sent[-1]
    assert args == (
        "dm-chan",
        (
            "Fixed the bug and added a regression test.\n\n"
            "buzz://message?channel=chan-1&id=reply-1"
        ),
    )
    assert kwargs == {"reply_to": "evt-2"}
    assert bot._reply_watches == {}


def test_reply_summary_failure_is_logged_not_raised(tmp_path, monkeypatch, capsys):
    _sent(monkeypatch)
    monkeypatch.setattr(outbound, "relay_dispatch", lambda *a: "posted-evt")

    def _fail_send(*a, **k):
        raise outbound.RelayError("boom")

    store = PendingActionStore()
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        },
        text_response="a summary",
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)

    async def scenario():
        await bot._handle_event(_event(event_id="evt-1"))
        bot._llm._json_response = {"intent": "confirm"}
        await bot._handle_event(_event(event_id="evt-2"))
        # send_message is only made to fail *after* confirm's own reply
        # goes out successfully, so only the background summarize call hits it.
        monkeypatch.setattr(outbound, "send_message", _fail_send)
        reply_event = _event(
            pubkey="codex-pubkey",
            content="fixed it",
            tags=[["h", "chan-1"], ["e", "posted-evt", "", "reply"]],
            event_id="reply-1",
        )
        await bot._handle_event(reply_event)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    assert "failed to summarize reply to posted-evt: boom" in capsys.readouterr().out


def test_reply_watch_does_not_swallow_unrelated_events(tmp_path, monkeypatch):
    """An event replying to some other, unwatched id is a normal event --
    it must still go through owner-gated routing rather than being silently
    consumed as if it resolved a reply wait."""
    (sent, _, bot) = _setup_ignored_event(tmp_path, monkeypatch)
    unrelated_reply = _event(
        tags=[["h", "dm-chan"], ["e", "some-other-event", "", "reply"]]
    )

    asyncio.run(bot._handle_event(unrelated_reply))

    assert "outside what I handle" in sent[0][0][1]


def test_confirm_reply_wait_times_out_without_a_reply(tmp_path, monkeypatch):
    sent, store = _setup_dispatch_test(monkeypatch)
    config = dataclasses.replace(
        CONFIG, dispatch=DispatchConfig(reply_wait_seconds=0.05)
    )
    audit = AuditLog(tmp_path / "audit.jsonl")
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        }
    )
    bot = Daemon(
        config,
        FakeInbound([]),
        IntentRouter(dispatch_llm, config, audit=audit),
        store,
        dispatch_llm,
        audit,
        ClosedItemStore(tmp_path / "closed_items.jsonl"),
        dm_id="dm-chan",
    )

    async def scenario():
        await bot._handle_event(_event(event_id="evt-1"))
        bot._llm._json_response = {"intent": "confirm"}
        await bot._handle_event(_event(event_id="evt-2"))
        await asyncio.sleep(0.15)  # let the wait time out

    asyncio.run(scenario())

    assert len(sent) == 2  # the proposal reply and the "confirmed" reply only
    assert bot._reply_watches == {}


def test_confirm_without_a_pending_action_is_safe(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: pytest.fail("should not dispatch")
    )
    llm = FakeLLM(json_response={"intent": "confirm"})
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert args == (
        "dm-chan",
        "Couldn't do that: no pending action for thread 'dm-chan'",
    )


def test_cancel_discards_without_posting(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    monkeypatch.setattr(
        outbound, "relay_dispatch", lambda *a: pytest.fail("should not dispatch")
    )
    store = PendingActionStore()
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": None,
            "message": "fix it",
        }
    )
    bot = _daemon(tmp_path, dispatch_llm, store=store)
    asyncio.run(bot._handle_event(_event(event_id="evt-1")))

    bot._llm._json_response = {"intent": "cancel"}
    asyncio.run(bot._handle_event(_event(event_id="evt-2")))

    (args, _) = sent[-1]
    assert args == ("dm-chan", "Cancelled -- nothing was sent.")
    assert store.get("dm-chan") is None


def test_chit_chat_reply(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert "outside what I handle" in args[1]


def test_router_error_becomes_a_reply_not_a_crash(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    llm = FakeLLM(json_response={"intent": "not-a-real-intent"})
    (args, _) = _handle_event_and_get_first_sent(tmp_path, llm, sent)
    assert args[1].startswith("Couldn't do that:")


def test_reply_send_failure_is_logged_not_raised(tmp_path, monkeypatch, capsys):
    def _fail(*a, **k):
        raise outbound.RelayError("buzz cli not found")

    monkeypatch.setattr(outbound, "send_message", _fail)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot._handle_event(_event()))

    assert "buzz cli not found" in capsys.readouterr().out


def _setup_outbound_mocks(monkeypatch):
    sent = _sent(monkeypatch)
    monkeypatch.setattr(outbound, "open_dm", lambda pubkey: "dm-chan")
    monkeypatch.setattr(
        outbound, "get_own_profile", lambda: {"display_name": "swingbird"}
    )
    monkeypatch.setattr(outbound, "join_channel", lambda channel_id: None)
    return sent


def test_run_connects_subscribes_and_survives_a_malformed_event(
    tmp_path, monkeypatch, capsys
):
    sent = _setup_outbound_mocks(monkeypatch)
    presence_calls = []
    monkeypatch.setattr(outbound, "set_presence", presence_calls.append)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    malformed = _event(event_id="bad-1", tags=[])
    good = _event(event_id="good-1", tags=[["h", "dm-chan"]])
    inbound = FakeInbound([malformed, good])
    bot = _daemon(tmp_path, llm, inbound=inbound)

    asyncio.run(bot.run())

    expected_reply = (
        "That's outside what I handle -- ask me for a recap, or to dispatch "
        "an instruction to a project channel."
    )
    assert inbound.connected is True
    assert inbound.subscribed == ["chan-1", "chan-2", "dm-chan"]
    assert sent == [(("dm-chan", expected_reply), {"reply_to": "good-1"})]
    assert "bad-1" in capsys.readouterr().out
    assert presence_calls == ["online", "offline"]


def _run_daemon_with_llm(tmp_path, json_response=None):
    llm = FakeLLM(json_response=json_response or {"intent": "chit_chat"})
    bot = _daemon(tmp_path, llm)

    asyncio.run(bot.run())
    return bot, llm


def test_run_joins_every_configured_project_channel(tmp_path, monkeypatch):
    sent = _setup_outbound_mocks(monkeypatch)
    monkeypatch.setattr(outbound, "set_presence", lambda status: None)
    joined = []
    monkeypatch.setattr(outbound, "join_channel", joined.append)
    _run_daemon_with_llm(tmp_path)

    assert joined == ["chan-1", "chan-2"]
    assert sent == []


def test_join_channel_failure_is_logged_and_does_not_block_startup(
    tmp_path, monkeypatch, capsys
):
    sent = _setup_outbound_mocks(monkeypatch)
    monkeypatch.setattr(outbound, "set_presence", lambda status: None)

    def _fail(channel_id):
        raise outbound.RelayError("restricted: channel is private")

    monkeypatch.setattr(outbound, "join_channel", _fail)
    _run_daemon_with_llm(tmp_path)

    out = capsys.readouterr().out
    assert "couldn't join channel 'backend'" in out
    assert "restricted: channel is private" in out
    assert sent == []  # startup still reached "connected and listening"


def _stub_outbound_network_calls(monkeypatch):
    monkeypatch.setattr(outbound, "set_presence", lambda status: None)
    monkeypatch.setattr(
        outbound, "get_own_profile", lambda: {"display_name": "swingbird"}
    )
    monkeypatch.setattr(outbound, "join_channel", lambda channel_id: None)


def test_run_subscribes_to_the_owners_dm_resolved_for_this_run(tmp_path, monkeypatch):
    sent = _sent(monkeypatch)
    calls = []
    monkeypatch.setattr(
        outbound, "open_dm", lambda pubkey: calls.append(pubkey) or "dm-chan"
    )
    _stub_outbound_network_calls(monkeypatch)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    dm_event = _event(tags=[["h", "dm-chan"]], event_id="dm-evt")
    inbound = FakeInbound([dm_event])
    bot = _daemon(tmp_path, llm, inbound=inbound)

    asyncio.run(bot.run())

    assert calls == [OWNER_PUBKEY]
    (args, kwargs) = sent[0]
    assert args[0] == "dm-chan"
    assert kwargs == {"reply_to": "dm-evt"}


def _run_daemon(tmp_path, llm=None, inbound=None, config=None):
    if llm is None:
        llm = FakeLLM(json_response={"intent": "chit_chat"})
    if inbound is None:
        inbound = FakeInbound([])
    bot = _daemon(tmp_path, llm, inbound=inbound, config=config)

    asyncio.run(bot.run())
    return bot, inbound


def test_run_captures_since_before_the_startup_network_round_trips(
    tmp_path, monkeypatch
):
    """A `since` taken only once `subscribe()` itself runs would exclude any
    message sent while `open_dm()`/`connect()` are still in flight -- both
    are real round-trips (a buzz-cli subprocess, then a WebSocket + NIP-42
    handshake), so the cutoff must be captured before either starts."""
    before = int(time.time())

    def _slow_open_dm(pubkey):
        time.sleep(1.1)
        return "dm-chan"

    monkeypatch.setattr(outbound, "open_dm", _slow_open_dm)
    _stub_outbound_network_calls(monkeypatch)
    (_, inbound) = _run_daemon(tmp_path)

    after = int(time.time())
    assert before <= inbound.since <= before + 1
    assert inbound.since < after


def test_presence_set_failure_is_logged_not_raised(tmp_path, monkeypatch, capsys):
    sent = _setup_outbound_mocks(monkeypatch)

    def _fail(status):
        raise outbound.RelayError("boom")

    monkeypatch.setattr(outbound, "set_presence", _fail)
    _run_daemon(tmp_path)

    assert sent == []
    out = capsys.readouterr().out
    assert "online" in out
    assert "offline" in out


def _stub_common_outbound(monkeypatch, outbound):
    monkeypatch.setattr(outbound, "open_dm", lambda pubkey: "dm-chan")
    monkeypatch.setattr(outbound, "set_presence", lambda status: None)
    monkeypatch.setattr(outbound, "join_channel", lambda channel_id: None)


def test_sync_identity_profile_updates_name_when_different(
    tmp_path, monkeypatch, capsys
):
    _stub_common_outbound(monkeypatch, outbound)
    monkeypatch.setattr(
        outbound, "get_own_profile", lambda: {"display_name": "old-name"}
    )
    update_calls = []
    monkeypatch.setattr(
        outbound, "update_profile", lambda **kwargs: update_calls.append(kwargs)
    )

    _run_daemon(tmp_path)

    assert update_calls == [{"name": "swingbird"}]
    assert "updated profile fields: ['name']" in capsys.readouterr().out


def test_sync_identity_profile_skips_when_already_matching(tmp_path, monkeypatch):
    _stub_common_outbound(monkeypatch, outbound)
    monkeypatch.setattr(
        outbound, "get_own_profile", lambda: {"display_name": "swingbird"}
    )
    monkeypatch.setattr(
        outbound,
        "update_profile",
        lambda **kwargs: pytest.fail("should not update"),
    )

    _run_daemon(tmp_path)


def test_sync_identity_profile_updates_description_when_configured(
    tmp_path, monkeypatch, capsys
):
    _stub_common_outbound(monkeypatch, outbound)
    monkeypatch.setattr(
        outbound, "get_own_profile", lambda: {"display_name": "swingbird"}
    )
    update_calls = []
    monkeypatch.setattr(
        outbound, "update_profile", lambda **kwargs: update_calls.append(kwargs)
    )
    config = dataclasses.replace(
        CONFIG, identity=IdentityConfig(name="swingbird", description="TPM bot")
    )

    _run_daemon(tmp_path, config=config)

    assert update_calls == [{"about": "TPM bot"}]


def test_sync_identity_profile_updates_avatar_when_configured(
    tmp_path, monkeypatch, capsys
):
    _stub_common_outbound(monkeypatch, outbound)
    monkeypatch.setattr(
        outbound, "get_own_profile", lambda: {"display_name": "swingbird"}
    )
    update_calls = []
    monkeypatch.setattr(
        outbound, "update_profile", lambda **kwargs: update_calls.append(kwargs)
    )
    avatar = AvatarConfig(style="emoji", emoji="🐦", color="#3399FF")
    config = dataclasses.replace(
        CONFIG, identity=IdentityConfig(name="swingbird", avatar=avatar)
    )

    _run_daemon(tmp_path, config=config)

    assert update_calls == [{"avatar": emoji_avatar_data_url("🐦", "#3399FF")}]


def test_sync_identity_profile_skips_avatar_when_already_matching(
    tmp_path, monkeypatch
):
    _stub_common_outbound(monkeypatch, outbound)
    wanted_avatar = emoji_avatar_data_url("🐦", "#3399FF")
    monkeypatch.setattr(
        outbound,
        "get_own_profile",
        lambda: {"display_name": "swingbird", "picture": wanted_avatar},
    )
    monkeypatch.setattr(
        outbound,
        "update_profile",
        lambda **kwargs: pytest.fail("should not update"),
    )
    avatar = AvatarConfig(style="emoji", emoji="🐦", color="#3399FF")
    config = dataclasses.replace(
        CONFIG, identity=IdentityConfig(name="swingbird", avatar=avatar)
    )

    _run_daemon(tmp_path, config=config)


def test_sync_identity_profile_failure_is_logged_not_raised(
    tmp_path, monkeypatch, capsys
):
    _stub_common_outbound(monkeypatch, outbound)

    def _fail():
        raise outbound.RelayError("boom")

    monkeypatch.setattr(outbound, "get_own_profile", _fail)

    _run_daemon(tmp_path)

    assert "failed to sync identity profile: boom" in capsys.readouterr().out


def test_build_daemon_wires_config_llm_and_inbound(tmp_path, monkeypatch):
    monkeypatch.setenv("X_API_KEY", "key")
    monkeypatch.setenv("RELAY_KEY", "1" * 64)
    config_path = tmp_path / "swingbird.toml"
    config_path.write_text(f"""
[llm]
base_url = "https://x"
model = "m"
api_key_env = "X_API_KEY"

[relay]
url = "wss://relay.example"
private_key_env = "RELAY_KEY"

[owner]
pubkey = "{OWNER_PUBKEY}"
name = "Voidious"

[[channels]]
id = "chan-1"
name = "backend"
write = true
agents = ["Codex"]
""")

    bot = build_daemon(
        str(config_path),
        str(tmp_path / "audit.jsonl"),
        str(tmp_path / "closed_items.jsonl"),
    )

    assert isinstance(bot, Daemon)
    assert bot._config.owner.pubkey == OWNER_PUBKEY


def test_build_daemon_raises_if_relay_key_env_is_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("X_API_KEY", "key")
    monkeypatch.delenv("RELAY_KEY", raising=False)
    config_path = tmp_path / "swingbird.toml"
    config_path.write_text(f"""
[llm]
base_url = "https://x"
model = "m"
api_key_env = "X_API_KEY"

[relay]
url = "wss://relay.example"
private_key_env = "RELAY_KEY"

[owner]
pubkey = "{OWNER_PUBKEY}"
name = "Voidious"

[[channels]]
id = "chan-1"
name = "backend"
write = true
agents = ["Codex"]
""")

    with pytest.raises(InboundError, match="RELAY_KEY"):
        build_daemon(
            str(config_path),
            str(tmp_path / "audit.jsonl"),
            str(tmp_path / "closed_items.jsonl"),
        )


class _FakeBuiltDaemon:
    def __init__(self):
        self.ran = False

    async def run(self):
        self.ran = True


def test_main_uses_default_paths_and_runs_the_daemon(monkeypatch):
    built = _FakeBuiltDaemon()
    calls = {}

    def fake_build_daemon(config_path, audit_log_path, closed_items_path):
        calls["config_path"] = config_path
        calls["audit_log_path"] = audit_log_path
        calls["closed_items_path"] = closed_items_path
        return built

    monkeypatch.setattr(daemon, "build_daemon", fake_build_daemon)
    monkeypatch.setattr(sys, "argv", ["swingbird"])

    daemon.main()

    assert calls["config_path"] == DEFAULT_CONFIG_PATH
    assert calls["audit_log_path"] == DEFAULT_AUDIT_LOG_PATH
    assert calls["closed_items_path"] == DEFAULT_CLOSED_ITEMS_PATH
    assert built.ran is True


def test_main_passes_through_custom_paths(monkeypatch):
    built = _FakeBuiltDaemon()
    calls = {}

    def fake_build_daemon(config_path, audit_log_path, closed_items_path):
        calls["config_path"] = config_path
        calls["audit_log_path"] = audit_log_path
        calls["closed_items_path"] = closed_items_path
        return built

    monkeypatch.setattr(daemon, "build_daemon", fake_build_daemon)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swingbird",
            "--config",
            "other.toml",
            "--audit-log",
            "other.jsonl",
            "--closed-items",
            "other-closed.jsonl",
        ],
    )

    daemon.main()

    assert calls["config_path"] == "other.toml"
    assert calls["audit_log_path"] == "other.jsonl"
    assert calls["closed_items_path"] == "other-closed.jsonl"


def _voice_config(config=None, **voice_overrides):
    voice_overrides.setdefault("enabled", True)
    voice_overrides.setdefault("tts", VoiceTTSConfig(voice="en_US-lessac-medium"))
    base = config or CONFIG
    return dataclasses.replace(base, voice=VoiceConfig(**voice_overrides))


def test_run_voice_turn_wakes_records_processes_replies_and_speaks(
    tmp_path, monkeypatch
):
    wake_calls = []
    monkeypatch.setattr(
        daemon.voice_wake,
        "listen_for_wake_word",
        lambda wake_word, mic: wake_calls.append((wake_word, mic)),
    )
    stt_calls = []
    transcripts = iter(["recap", None])

    def fake_record_and_transcribe(
        mic, stt, max_wait_seconds=daemon.voice_stt.MAX_UTTERANCE_SECONDS
    ):
        stt_calls.append((mic, stt, max_wait_seconds))
        return next(transcripts)

    monkeypatch.setattr(
        daemon.voice_stt, "record_and_transcribe", fake_record_and_transcribe
    )
    speak_calls = []
    monkeypatch.setattr(
        daemon.voice_tts,
        "speak",
        lambda text, tts, output: speak_calls.append((text, tts, output)),
    )
    sent = []
    event_ids = iter(["transcript-evt", "reply-evt"])
    monkeypatch.setattr(
        outbound,
        "send_message",
        lambda *a, **k: sent.append((a, k)) or next(event_ids),
    )
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    voice_config = _voice_config()
    bot = _daemon(tmp_path, llm, config=voice_config)

    asyncio.run(bot._run_voice_turn())

    assert wake_calls == [(voice_config.voice.wake_word, voice_config.voice.mic)]
    assert stt_calls == [
        (
            voice_config.voice.mic,
            voice_config.voice.stt,
            daemon.voice_stt.MAX_UTTERANCE_SECONDS,
        ),
        (
            voice_config.voice.mic,
            voice_config.voice.stt,
            voice_config.voice.follow_up_window_seconds,
        ),
    ]
    expected_reply = (
        "That's outside what I handle -- ask me for a recap, or to dispatch "
        "an instruction to a project channel."
    )
    assert sent == [
        (("dm-chan", "recap"), {}),
        (("dm-chan", expected_reply), {"reply_to": "transcript-evt"}),
    ]
    assert speak_calls == [
        (expected_reply, voice_config.voice.tts, voice_config.voice.output)
    ]


def test_run_voice_turn_registers_a_reply_wait_after_confirm(tmp_path, monkeypatch):
    monkeypatch.setattr(
        daemon.voice_wake, "listen_for_wake_word", lambda wake_word, mic: None
    )
    monkeypatch.setattr(daemon.voice_tts, "speak", lambda text, tts, output: None)
    monkeypatch.setattr(outbound, "relay_dispatch", lambda *a: "posted-evt")
    _sent(monkeypatch)
    # A `None` after each real transcript ends that turn's follow-up loop
    # immediately, so each `_run_voice_turn()` call below still does
    # exactly one wake word + one exchange, matching this test's intent.
    transcripts = iter(["dispatch it", None, "confirm", None])
    monkeypatch.setattr(
        daemon.voice_stt,
        "record_and_transcribe",
        lambda mic, stt, max_wait_seconds=daemon.voice_stt.MAX_UTTERANCE_SECONDS: next(
            transcripts
        ),
    )
    dispatch_llm = FakeLLM(
        json_response={
            "intent": "dispatch",
            "channel": "backend",
            "target_agent": "Codex",
            "message": "fix it",
        }
    )
    voice_config = _voice_config()
    bot = _daemon(tmp_path, dispatch_llm, config=voice_config)

    async def scenario():
        await bot._run_voice_turn()
        bot._llm._json_response = {"intent": "confirm"}
        await bot._run_voice_turn()

    asyncio.run(scenario())

    # The reply-wait registration consumes `_pending_watch` synchronously
    # inside `_run_voice_turn`, same as `_handle_event` -- nothing should be
    # left pending afterwards, and the watch itself should be live.
    assert bot._pending_watch is None
    assert len(bot._reply_watches) == 1


def test_run_voice_turn_keeps_listening_without_the_wake_word_until_follow_up_times_out(
    tmp_path, monkeypatch
):
    wake_calls = []
    monkeypatch.setattr(
        daemon.voice_wake,
        "listen_for_wake_word",
        lambda wake_word, mic: wake_calls.append((wake_word, mic)),
    )
    monkeypatch.setattr(daemon.voice_tts, "speak", lambda text, tts, output: None)
    _sent(monkeypatch)
    max_waits = []
    transcripts = iter(["recap", "recap", None])
    monkeypatch.setattr(
        daemon.voice_stt,
        "record_and_transcribe",
        lambda mic, stt, max_wait_seconds=daemon.voice_stt.MAX_UTTERANCE_SECONDS: (
            max_waits.append(max_wait_seconds) or next(transcripts)
        ),
    )
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    voice_config = _voice_config()
    bot = _daemon(tmp_path, llm, config=voice_config)

    asyncio.run(bot._run_voice_turn())

    # Only one wake-word wait for two exchanges -- the second one is a
    # follow-up (§V.11), not a fresh turn. The first record uses
    # record_and_transcribe's own default wait; only the follow-up
    # record(s) after it use the longer configured window, and the turn
    # ends once that window comes back empty.
    assert wake_calls == [(voice_config.voice.wake_word, voice_config.voice.mic)]
    assert max_waits == [
        daemon.voice_stt.MAX_UTTERANCE_SECONDS,
        voice_config.voice.follow_up_window_seconds,
        voice_config.voice.follow_up_window_seconds,
    ]


def test_safe_run_voice_turn_logs_and_swallows_a_failed_turn(
    tmp_path, monkeypatch, capsys
):
    def _fail(wake_word, mic):
        raise RuntimeError("mic gone")

    monkeypatch.setattr(daemon.voice_wake, "listen_for_wake_word", _fail)
    bot = _daemon(tmp_path, FakeLLM(), config=_voice_config())

    asyncio.run(bot._safe_run_voice_turn())

    out = capsys.readouterr().out
    assert "voice turn failed" in out
    assert "mic gone" in out


def test_run_voice_loop_runs_turns_back_to_back_until_cancelled(tmp_path):
    bot = _daemon(tmp_path, FakeLLM(), config=_voice_config())
    calls = []

    async def fake_safe_run_voice_turn():
        calls.append(1)
        await asyncio.sleep(0)

    bot._safe_run_voice_turn = fake_safe_run_voice_turn

    async def scenario():
        task = asyncio.create_task(bot._run_voice_loop())
        while len(calls) < 3:
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(calls) >= 3


class _YieldingInbound(FakeInbound):
    """Like `FakeInbound`, but cedes control to the event loop once before
    replaying its events -- needed so a task `run()` creates alongside the
    main `async for` (e.g. the voice loop) actually gets a turn to start
    before an empty event list lets `run()` fall straight through to
    `finally` and cancel it unstarted."""

    async def events(self):
        await asyncio.sleep(0)
        async for event in super().events():
            yield event


def test_run_starts_and_cancels_the_voice_loop_when_voice_is_enabled(
    tmp_path, monkeypatch
):
    sent = _setup_outbound_mocks(monkeypatch)
    started = []
    cancelled = []

    async def fake_run_voice_loop(self):
        started.append(1)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    monkeypatch.setattr(Daemon, "_run_voice_loop", fake_run_voice_loop)
    llm = FakeLLM(json_response={"intent": "chit_chat"})
    bot = _daemon(tmp_path, llm, inbound=_YieldingInbound([]), config=_voice_config())

    asyncio.run(bot.run())

    assert started == [1]
    assert cancelled == [1]
    assert sent == []
