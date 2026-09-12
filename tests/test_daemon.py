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
import dataclasses
import json
import sys
import time

import pytest

from swingbird import daemon, outbound
from swingbird.audit import AuditLog
from swingbird.config import (
    ChannelConfig,
    Config,
    DispatchConfig,
    LLMConfig,
    OwnerConfig,
    RelayConfig,
)
from swingbird.daemon import (
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_CONFIG_PATH,
    Daemon,
    build_daemon,
)
from swingbird.inbound import InboundError
from swingbird.pending_actions import PendingActionStore
from swingbird.recap import RecapItem
from swingbird.recap_actions import RecapActionStore
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
    own_pubkey=None,
):
    audit = AuditLog(tmp_path / "audit.jsonl")
    router = IntentRouter(llm, CONFIG, audit=audit)
    return Daemon(
        CONFIG,
        inbound or FakeInbound([]),
        router,
        store or PendingActionStore(),
        llm,
        audit,
        dm_id=dm_id,
        recap_store=recap_store,
        disambiguation=disambiguation,
        own_pubkey=own_pubkey,
    )


def _sent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        outbound, "send_message", lambda *a, **k: calls.append((a, k)) or "reply-evt"
    )
    return calls


def _handle_event_and_get_first_sent(
    tmp_path, llm, sent, event=None, store=None, recap_store=None, disambiguation=None
):
    bot = _daemon(
        tmp_path,
        llm,
        store=store,
        recap_store=recap_store,
        disambiguation=disambiguation,
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
    from swingbird import recap

    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
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
    from swingbird import recap

    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
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


def _recap_store_with(thread_id, *items):
    store = RecapActionStore()
    store.set(thread_id, tuple(items))
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


def _assert_last_sent(sent, channel="dm-chan", content="Fixed it.", reply_to="evt-2"):
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
    from swingbird import recap

    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
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
        text_response="More on F5 and F6.",
    )

    (args, _) = _handle_event_and_get_first_sent(
        tmp_path, llm, sent, recap_store=recap_store
    )

    assert args == ("dm-chan", "More on F5 and F6.")
    user_content = llm.calls[-1][1]["content"]
    assert "undefined-sentinel cloneDeep split" in user_content
    assert "lower priority follow-up" in user_content
    assert "unused-ignore propagation" not in user_content


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
    from swingbird import recap

    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    sent = _sent(monkeypatch)
    llm = FakeLLM(
        json_response=[
            {"intent": "recap", "detail": "detailed"},
            {"text": "here's the detailed recap", "items": []},
        ]
    )

    _handle_event_and_get_first_sent(tmp_path, llm, sent)

    system_prompt = llm.calls[-1][0]["content"]
    assert system_prompt == recap._DETAILED_SYSTEM_PROMPT


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
    assert args == ("dm-chan", "Confirmed and relayed (event posted-evt).")
    assert store.get("dm-chan") is None


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
    assert args == ("dm-chan", "Fixed the bug and added a regression test.")
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
    monkeypatch.setattr(outbound, "get_own_display_name", lambda: "swingbird")
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
    monkeypatch.setattr(outbound, "get_own_display_name", lambda: "swingbird")
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


def _run_daemon(tmp_path, llm=None, inbound=None):
    if llm is None:
        llm = FakeLLM(json_response={"intent": "chit_chat"})
    if inbound is None:
        inbound = FakeInbound([])
    bot = _daemon(tmp_path, llm, inbound=inbound)

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


def test_sync_display_name_updates_when_different(tmp_path, monkeypatch, capsys):
    _stub_common_outbound(monkeypatch, outbound)
    monkeypatch.setattr(outbound, "get_own_display_name", lambda: "old-name")
    set_calls = []
    monkeypatch.setattr(outbound, "set_display_name", set_calls.append)

    _run_daemon(tmp_path)

    assert set_calls == ["swingbird"]
    assert "updated display name 'old-name' -> 'swingbird'" in capsys.readouterr().out


def test_sync_display_name_skips_when_already_matching(tmp_path, monkeypatch):
    _stub_common_outbound(monkeypatch, outbound)
    monkeypatch.setattr(outbound, "get_own_display_name", lambda: "swingbird")
    monkeypatch.setattr(
        outbound, "set_display_name", lambda name: pytest.fail("should not update")
    )

    _run_daemon(tmp_path)


def test_sync_display_name_failure_is_logged_not_raised(tmp_path, monkeypatch, capsys):
    _stub_common_outbound(monkeypatch, outbound)

    def _fail():
        raise outbound.RelayError("boom")

    monkeypatch.setattr(outbound, "get_own_display_name", _fail)

    _run_daemon(tmp_path)

    assert "failed to sync display name: boom" in capsys.readouterr().out


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

    bot = build_daemon(str(config_path), str(tmp_path / "audit.jsonl"))

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
        build_daemon(str(config_path), str(tmp_path / "audit.jsonl"))


class _FakeBuiltDaemon:
    def __init__(self):
        self.ran = False

    async def run(self):
        self.ran = True


def test_main_uses_default_paths_and_runs_the_daemon(monkeypatch):
    built = _FakeBuiltDaemon()
    calls = {}

    def fake_build_daemon(config_path, audit_log_path):
        calls["config_path"] = config_path
        calls["audit_log_path"] = audit_log_path
        return built

    monkeypatch.setattr(daemon, "build_daemon", fake_build_daemon)
    monkeypatch.setattr(sys, "argv", ["swingbird"])

    daemon.main()

    assert calls["config_path"] == DEFAULT_CONFIG_PATH
    assert calls["audit_log_path"] == DEFAULT_AUDIT_LOG_PATH
    assert built.ran is True


def test_main_passes_through_custom_paths(monkeypatch):
    built = _FakeBuiltDaemon()
    calls = {}

    def fake_build_daemon(config_path, audit_log_path):
        calls["config_path"] = config_path
        calls["audit_log_path"] = audit_log_path
        return built

    monkeypatch.setattr(daemon, "build_daemon", fake_build_daemon)
    monkeypatch.setattr(
        sys,
        "argv",
        ["swingbird", "--config", "other.toml", "--audit-log", "other.jsonl"],
    )

    daemon.main()

    assert calls["config_path"] == "other.toml"
    assert calls["audit_log_path"] == "other.jsonl"
