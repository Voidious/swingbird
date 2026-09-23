"""Daemon: wires the inbound relay client to the router, pending-action
store, and audit log, and enforces the safety invariants from §5.

Only events from `config.owner.pubkey`, posted in the daemon's own DM with
the owner, are ever routed as a command. Project channels are subscribed to
for recap purposes only -- even a message from the owner there (e.g. an
@mention) is ignored as a command, so ordinary conversation in a dev channel
can never be misread as a dispatch/confirm/cancel. (Reacting to explicit
@mentions in project channels may be worth adding later; for now the DM is
the only interaction surface.) Everyone else's traffic in a subscribed
channel (including a coding agent's own replies) is ignored here too. A
single bad or unexpected event never kills the daemon: known, safety-relevant
failures (an ambiguous target, a bad LLM response, an unwritable channel,
...) become a reply to the sender instead of a crash, and anything truly
unexpected is logged and the loop continues.

Alongside the configured project channels, the daemon always subscribes to
its own 1:1 DM with the owner (resolved via `outbound.open_dm`) -- Buzz DMs
turn out to be ordinary #h-tagged channel events under the hood, so this
needed no new wire format, just one more channel id in the subscription.

On startup the daemon also joins every configured project channel it isn't
already a member of (`_join_project_channels`) -- good bot manners, since
membership isn't actually required to read or write an *open* channel (the
relay allows either membership or open visibility), so posting into
channels it never joined would otherwise look odd in the member list. A
join only fails for a private channel the identity isn't already in; that's
logged and never fatal. The same failure mode can also surface later, from
`relay_dispatch`/a recap fetch against a channel configured as private
without the daemon in it -- `_ACTIONABLE_ERRORS` including `RelayError`
turns that into a "Couldn't do that" reply to the owner (asking them to add
the bot) instead of a silently swallowed exception.

The daemon also publishes its own presence (online while the event loop is
running, offline on exit) so its availability dot in Buzz Desktop reflects
whether it's actually up -- a deployed daemon that isn't running should
never look identical to one that is.

After a confirmed dispatch, the daemon waits (up to `config.dispatch.
reply_wait_seconds`) for a reply to the relayed instruction, then
summarizes it back into the owner's DM -- see `_watch_for_reply` and
`_resolve_reply_watch`. This runs as a background asyncio task alongside
the main event loop rather than blocking it, since the reply (if any)
arrives as just another event on the same subscription; matching it to
the right wait is done by NIP-10 `e`-tag, not by agent identity, since
the daemon doesn't track individual coding agents' pubkeys.

A coding agent's own reply frequently doesn't `e`-tag the relayed message
directly: an async status update often threads to the whole conversation's
root instead of the specific message that carried the instruction, and a
reply nested two or more levels deep NIP-10-tags both a "root" and a
"reply" id, in either order. `_reply_watch_id` resolves the id actually
worth watching up front (the thread's true root for a grounded dispatch,
the relayed event's own id for a fresh top-level one), and
`_reply_target_ids` checks every `e`-tag id on an inbound event against
it, so either shape of reply is caught -- see both for why.

When `config.voice.enabled`, `run()` also starts `_run_voice_loop` as a
second background task alongside the main event subscription (Voice Mode
design doc §V.16 step 5) -- a voice turn begins whenever the wake word
fires, not as another event on the inbound subscription, so it can't be
driven from inside the same `async for`. `_run_voice_exchange` feeds each
transcript through `_process`, the exact same intent machinery a typed DM
uses, via the same `thread_id`/`event_id` shape `_process` already takes
for that reason (see `_process`'s own docstring); `_run_voice_turn` can run
several exchanges back-to-back within one wake word, per its own follow-up
window (§V.11).

Every voice reply plays through `voice_barge_in.speak_with_barge_in`
rather than `voice_tts.speak` directly (§V.12): it listens on the mic
concurrently with playback so the user can start talking over a reply
instead of having to wait it out, and returns the interrupting utterance's
own `Transcript` when that happens. `_run_voice_turn`'s loop treats that
exactly like a fresh capture from `voice_stt.record_and_transcribe` --
routing it through `_run_voice_exchange` (or speaking the low-confidence
reply) again -- rather than falling through to its own post-reply cue and
recording. Only when nothing interrupted a reply does the loop play the
cue and record the follow-up as before, passing `voice.debounce_seconds`
through as `record_and_transcribe`'s `mute_seconds` so the mic opens the
instant the cue finishes (not after an extra sleep) while still not
letting the reply's own trailing audio, or the cue itself, be misread as
the user talking during that same window.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from dataclasses import replace

from swingbird import outbound, voice_barge_in, voice_cues, voice_stt, voice_wake
from swingbird.audit import AuditLog
from swingbird.avatar import emoji_avatar_data_url
from swingbird.closed_items import ClosedItemStore
from swingbird.config import Config, load_config
from swingbird.dispatch_phrasing import rephrase_for_dispatch
from swingbird.history import (
    fetch_recent_messages,
    fetch_thread_messages,
    fetch_thread_root,
)
from swingbird.inbound import InboundClient, InboundError
from swingbird.llm import LLMClient, LLMError
from swingbird.pending_actions import (
    DispatchProposal,
    PendingActionError,
    PendingActionStore,
    cancel_dispatch,
    confirm_dispatch,
    propose_dispatch,
)
from swingbird.recap import RecapError, RecapItem, build_recap
from swingbird.recap_actions import (
    RecapActionError,
    RecapActionStore,
    ResolvedReference,
    resolve_reference,
)
from swingbird.recap_close import PendingClose, PendingCloseStore, resolve_close_reply
from swingbird.recap_close_selection import select_items_to_close
from swingbird.recap_detail import elaborate
from swingbird.recap_disambiguation import (
    AmbiguousRecapReference,
    DisambiguationStore,
    PendingDisambiguation,
    format_choices,
    resolve_choice,
)
from swingbird.recap_list import render_items
from swingbird.recap_relay import relay_with_context
from swingbird.reply_summary import summarize_reply
from swingbird.router import Intent, IntentRouter, RouterError
from swingbird.voice_render import render_for_speech

DEFAULT_CONFIG_PATH = "swingbird.toml"
DEFAULT_AUDIT_LOG_PATH = "audit.jsonl"
DEFAULT_CLOSED_ITEMS_PATH = "closed_items.jsonl"
# How much DM history to give recap_detail's elaboration as "the recap
# conversation so far" -- generous enough to cover the recap and this
# follow-up (plus a bit of prior back-and-forth) without pulling in an
# unbounded amount of unrelated DM history.
_RECAP_DETAIL_DM_HISTORY_LIMIT = 20

_CHIT_CHAT_REPLY = (
    "That's outside what I handle -- ask me for a recap, or to dispatch an "
    "instruction to a project channel."
)
_ACTIONABLE_ERRORS = (
    RouterError,
    PendingActionError,
    RecapError,
    RecapActionError,
    LLMError,
    outbound.RelayError,
)


class DaemonError(Exception):
    """Raised when an inbound event can't be handled (e.g. malformed tags)."""


class Daemon:
    """Routes verified inbound events from `config.owner.pubkey` to actions."""

    def __init__(
        self,
        config: Config,
        inbound: InboundClient,
        router: IntentRouter,
        store: PendingActionStore,
        llm: LLMClient,
        audit: AuditLog,
        closed_items: ClosedItemStore,
        dm_id: str | None = None,
        recap_store: RecapActionStore | None = None,
        disambiguation: DisambiguationStore | None = None,
        pending_close: PendingCloseStore | None = None,
        own_pubkey: str | None = None,
    ) -> None:
        self._config = config
        self._inbound = inbound
        self._router = router
        self._store = store
        self._llm = llm
        self._audit = audit
        # Persistent record of items the user has marked closed, so a
        # future recap stops re-surfacing them -- see closed_items.py.
        self._closed_items = closed_items
        # The last recap's structured items per thread, so a follow-up DM
        # ("go ahead with F4") can resolve "F4" back to a real channel and
        # instruction -- see recap_actions.py.
        self._recap_store = (
            recap_store if recap_store is not None else RecapActionStore()
        )
        # An open "which did you mean" question per thread, so the next DM
        # can answer it instead of being misrouted as a new command -- see
        # recap_disambiguation.py.
        self._disambiguation = (
            disambiguation if disambiguation is not None else DisambiguationStore()
        )
        # An open "close these items?" confirmation per thread, resolved by
        # a deterministic yes/no before the message ever reaches the router
        # -- see recap_close.py.
        self._pending_close = (
            pending_close if pending_close is not None else PendingCloseStore()
        )
        # Resolved fresh in run() via outbound.open_dm(); only events posted
        # in this channel are ever routed as a command (see module
        # docstring). None until run() sets it, which means no channel can
        # match and every event is ignored -- the safe default for a daemon
        # that hasn't finished starting up.
        self._dm_id = dm_id
        # The daemon's own identity, so a reply-wait never mistakes one of
        # its own outbound events (see `_resolve_reply_watch`) for a
        # working agent's reply. None until run() sets it via
        # `inbound.pubkey`, same lazy-init rationale as `_dm_id`.
        self._own_pubkey = own_pubkey
        # Relayed-instruction event id -> a Future resolved with whatever
        # event replies to it, so a confirm's background wait-and-summarize
        # task (see _watch_for_reply) can be woken from _handle_event.
        self._reply_watches: dict[str, asyncio.Future] = {}
        # Set by _confirm (which runs off-thread inside _process, see
        # _handle_event) to hand a reply-wait registration back to the main
        # thread rather than calling _watch_for_reply directly -- that
        # creates an asyncio Future and Task, neither of which is safe to
        # touch from a thread other than the one running their loop. Always
        # consumed (and cleared) by _handle_event in the same event's
        # handling before the next event can start one, so there's no
        # cross-event clobbering.
        self._pending_watch: tuple[str, str, str] | None = None

    async def run(self) -> None:
        # Captured before open_dm()/connect() so the backlog cutoff covers
        # the whole startup handshake -- both involve real network
        # round-trips (buzz-cli subprocess, then WebSocket + NIP-42 auth),
        # and a `since` taken only once subscribe() itself runs would
        # silently drop any message the owner sends while the daemon is
        # still coming up.
        since = int(time.time())
        self._own_pubkey = self._inbound.pubkey
        self._sync_identity_profile()
        self._join_project_channels()
        # Resolved here rather than in build_daemon() so construction stays
        # side-effect-free; this is the daemon's own DM with the owner,
        # opened (or resurfaced) fresh each run via the same buzz-cli path
        # outbound.py already uses for every other write.
        self._dm_id = outbound.open_dm(self._config.owner.pubkey)
        channel_ids = [channel.id for channel in self._config.channels] + [self._dm_id]
        print(
            f"swingbird: resolved DM channel {self._dm_id!r}; "
            f"subscribing to {channel_ids}"
        )
        await self._inbound.connect()
        await self._inbound.subscribe(channel_ids, since=since)
        self._set_presence("online")
        print("swingbird: connected and listening")
        # Runs alongside the main event loop, same rationale as the
        # reply-wait background task (see module docstring): a voice turn
        # arrives on its own schedule (whenever the wake word fires), not as
        # another event on this subscription, so it can't be driven from
        # inside `async for event in self._inbound.events()`.
        voice_task = (
            asyncio.create_task(self._run_voice_loop())
            if self._config.voice.enabled
            else None
        )
        try:
            async for event in self._inbound.events():
                await self._safe_handle(event)
        finally:
            self._set_presence("offline")
            if voice_task is not None:
                voice_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await voice_task

    def _set_presence(self, status: str) -> None:
        # Best-effort: a presence hiccup is cosmetic (it only drives the
        # availability dot in Buzz Desktop) and must never take the daemon
        # down or block it from processing real events.
        try:
            outbound.set_presence(status)
        except outbound.RelayError as exc:
            print(f"swingbird: failed to set presence to {status!r}: {exc}")

    def _join_project_channels(self) -> None:
        # Best-effort, like presence/display-name sync: joining a channel
        # the identity already belongs to (or an open one) always succeeds
        # on the relay, so a failure here means a real, reportable problem
        # (a private channel the identity isn't in) -- never a reason to
        # block startup. See module docstring.
        for channel in self._config.channels:
            try:
                outbound.join_channel(channel.id)
            except outbound.RelayError as exc:
                print(
                    f"swingbird: couldn't join channel {channel.name!r} "
                    f"({channel.id}) -- if it's private, add me to it: {exc}"
                )

    def _sync_identity_profile(self) -> None:
        # Keeps a fresh identity (or a renamed deployment) from showing up
        # under a stale/default name, bio, or avatar. Best-effort like
        # presence -- a lookup/update hiccup here is cosmetic and must never
        # block startup or take the daemon down.
        identity = self._config.identity
        wanted_avatar = (
            emoji_avatar_data_url(identity.avatar.emoji, identity.avatar.color)
            if identity.avatar is not None
            else None
        )
        try:
            current = outbound.get_own_profile()
            updates: dict[str, str] = {}
            if current.get("display_name") != identity.name:
                updates["name"] = identity.name
            if (
                identity.description is not None
                and current.get("about") != identity.description
            ):
                updates["about"] = identity.description
            if wanted_avatar is not None and current.get("picture") != wanted_avatar:
                updates["avatar"] = wanted_avatar
            if updates:
                outbound.update_profile(**updates)
                print(f"swingbird: updated profile fields: {sorted(updates)}")
        except outbound.RelayError as exc:
            print(f"swingbird: failed to sync identity profile: {exc}")

    async def _safe_handle(self, event: dict) -> None:
        try:
            await self._handle_event(event)
        except Exception as exc:  # noqa: BLE001 - last-resort net, see module docstring
            print(f"swingbird: failed to handle event {event.get('id')}: {exc}")

    async def _handle_event(self, event: dict) -> None:
        print(f"swingbird: event {event.get('id')} from {event.get('pubkey')}")
        if self._resolve_reply_watch(event):
            return
        if event["pubkey"] != self._config.owner.pubkey:
            print(
                f"swingbird: ignoring -- not from owner ({self._config.owner.pubkey})"
            )
            return
        channel_id = _channel_of(event)
        if channel_id != self._dm_id:
            print(
                f"swingbird: ignoring -- not the owner's DM channel ({self._dm_id!r})"
            )
            return
        # `_process` (an LLM call, and for a recap several buzz-cli
        # subprocess calls too) and `send_message` are both synchronous,
        # blocking calls -- run off-thread so they can't starve this
        # coroutine's event loop of cycles to service the inbound
        # WebSocket's read/keepalive traffic while they're in flight. A
        # recap in particular can block long enough that the relay (or the
        # `websockets` client's own ping timeout) drops the connection as
        # idle; since a clean server-initiated close ends `events()`'s
        # `async for` silently (no exception), that stalled event loop was
        # otherwise indistinguishable from the daemon just exiting.
        reply = await asyncio.to_thread(
            self._process, event["content"], channel_id, event["id"]
        )
        self._start_pending_watch()
        try:
            await asyncio.to_thread(
                outbound.send_message, channel_id, reply, reply_to=event["id"]
            )
        except outbound.RelayError as exc:
            print(f"swingbird: failed to send reply to {event['id']}: {exc}")

    async def _run_voice_loop(self) -> None:
        """Run voice turns back-to-back for as long as the daemon is up
        (§V.16 step 5) -- cancelled by `run()` on shutdown, same as any
        other background task started there.

        Deliberately `while True` with no exit condition of its own: a
        voice turn's only natural end is its follow-up window (§V.11)
        elapsing with nothing said, at which point the daemon should
        immediately go back to listening for the wake word, not stop.
        """
        while True:
            await self._safe_run_voice_turn()

    async def _safe_run_voice_turn(self) -> None:
        try:
            await self._run_voice_turn()
        except Exception as exc:  # noqa: BLE001 - last-resort net, see module docstring
            print(f"swingbird: voice turn failed: {exc}")

    async def _run_voice_turn(self) -> None:
        """Wait for the wake word, then run voice exchanges back-to-back
        without requiring it again until `follow_up_window_seconds` passes
        with nothing said (§V.11) -- at which point this turn ends and
        `_run_voice_loop` calls back in here, requiring the wake word once
        more. Each exchange's reply DM carries a transcript footer (§V.8) --
        see `_run_voice_exchange`'s own docstring.

        A captured utterance that `voice_stt.transcribe` flags as
        low-confidence (§V.10) never reaches `_run_voice_exchange` -- it's
        not posted as a DM, and never routed through `_process`, so a
        misheard word can't accidentally read as a recap/dispatch/confirm.
        The owner instead just hears `voice_stt.LOW_CONFIDENCE_REPLY`
        spoken back, and the turn keeps listening exactly like it would
        after a normal reply.

        `listen_for_wake_word` runs off-thread for the same reason
        `_run_voice_exchange`'s blocking calls do -- see its docstring.

        Every reply (confident-exchange or low-confidence apology) speaks
        through `voice_barge_in.speak_with_barge_in` (§V.12), which returns
        the interrupting `Transcript` if the user talked over it. That
        transcript becomes `transcript` for the *next* loop iteration
        directly -- skipping the cue/debounce/record below entirely, since
        the mic was already listening and already captured it -- exactly as
        if it had come from `record_and_transcribe`. Only a reply nobody
        interrupted falls through to the normal cue-then-listen path.

        A `voice_cues.play_listening_started` chime plays every time the
        mic is about to start a *fresh* deliberate listen (after the wake
        word, and again after an uninterrupted reply while the follow-up
        window stays open) -- not after a barge-in, which was already
        listening throughout. `play_listening_stopped` plays once, when the
        window finally elapses and this turn ends -- see `voice_cues.py`'s
        own docstring. The follow-up listen passes `voice.debounce_seconds`
        (§V.12) to `record_and_transcribe` as `mute_seconds`, so the mic
        opens the moment the cue finishes -- not after an extra wait -- but
        audio from that same window can't trigger "speech started," giving
        the reply's own trailing audio (and the cue itself) room to finish
        leaving the speaker without either eating the user's own first word
        or being misheard as it -- see `voice_stt.record_utterance`'s own
        docstring for why that's not needed on the barge-in path, which
        never stopped listening in the first place.
        """
        voice = self._config.voice
        await asyncio.to_thread(
            voice_wake.listen_for_wake_word, voice.wake_word, voice.mic
        )
        await asyncio.to_thread(voice_cues.play_listening_started, voice.output)
        # The first exchange after the wake word waits up to
        # wake_word_window_seconds for speech to start; later ones in this
        # same turn use the longer follow-up window instead.
        transcript = await asyncio.to_thread(
            voice_stt.record_and_transcribe,
            voice.mic,
            voice.stt,
            voice.wake_word_window_seconds,
        )
        while transcript is not None:
            if transcript.is_confident:
                barge_in = await self._run_voice_exchange(transcript.text)
            else:
                barge_in = await asyncio.to_thread(
                    voice_barge_in.speak_with_barge_in,
                    voice_stt.LOW_CONFIDENCE_REPLY,
                    voice.tts,
                    voice.output,
                    voice.mic,
                    voice.stt,
                )
            if barge_in is not None:
                transcript = barge_in
                continue
            await asyncio.to_thread(voice_cues.play_listening_started, voice.output)
            transcript = await asyncio.to_thread(
                voice_stt.record_and_transcribe,
                voice.mic,
                voice.stt,
                voice.follow_up_window_seconds,
                voice.debounce_seconds,
            )
        await asyncio.to_thread(voice_cues.play_listening_stopped, voice.output)

    async def _run_voice_exchange(self, transcript: str) -> voice_stt.Transcript | None:
        """Route one already-captured `transcript` through the exact same
        intent machinery a typed DM uses -- the first fully voice-driven
        exchange (§V.16 step 5), and every exchange after it within the
        same wake word's follow-up window (§V.11). Returns the barge-in
        `Transcript` if the user talked over the spoken reply, or `None` if
        nothing interrupted it -- see `_run_voice_turn`'s own docstring for
        how that return value is used.

        Mirrors `_handle_event`'s to_thread structure for the same reason:
        `_process` and `speak_with_barge_in` are both blocking calls (an
        LLM call, or subprocess audio I/O), so each runs off-thread to keep
        servicing the inbound WebSocket's read/keepalive traffic while it's
        in flight.
        """
        voice = self._config.voice
        # §V.3: voice shares the DM's thread id, every turn posts its own
        # DM first -- so the transcript is on the record, and `_process`
        # has an event id to anchor replies/reply-waits to, before routing.
        # It stays its own message rather than folding into the reply
        # below: `_confirm` (inside `_process`) needs a real, already-posted
        # event id to anchor a dispatch's later async reply-wait to, and
        # that has to exist before the reply text (or its own event) does.
        event_id = await asyncio.to_thread(
            outbound.send_message, self._dm_id, transcript
        )
        reply = await asyncio.to_thread(
            self._process, transcript, self._dm_id, event_id
        )
        self._start_pending_watch()
        # §V.8: the reply DM appends the transcript as a footer so it reads
        # on its own -- e.g. in a push notification, or scrolled past its
        # paired transcript message -- rather than relying on thread
        # position alone to show which command it's answering.
        await asyncio.to_thread(
            outbound.send_message,
            self._dm_id,
            f"{reply}\n\n-- {transcript}",
            reply_to=event_id,
        )
        return await asyncio.to_thread(
            voice_barge_in.speak_with_barge_in,
            render_for_speech(reply),
            voice.tts,
            voice.output,
            voice.mic,
            voice.stt,
        )

    def _start_pending_watch(self) -> None:
        """Consume `_pending_watch` (set by `_confirm`, running off-thread
        inside `_process`) and register the reply-wait it describes --
        shared by `_handle_event` and `_run_voice_exchange`, the two
        callers that call `_process` and then need to hand any resulting
        watch request back to the main thread (see `_pending_watch`'s own
        docstring for why it can't be created off-thread directly)."""
        if self._pending_watch is not None:
            watch, self._pending_watch = self._pending_watch, None
            self._watch_for_reply(*watch)

    def _resolve_reply_watch(self, event: dict) -> bool:
        """If `event` replies anywhere within a thread we're waiting on,
        fulfill that wait and report it as handled.

        Checked against every NIP-10 `e`-tag id on `event`
        (`_reply_target_ids`), not just one -- see `_reply_watch_id` for why
        a reply can legitimately carry a different id than the one that was
        actually watched while still belonging to the same watched thread.
        Matched by tag id alone, not by who posted it -- a reply is never
        itself treated as an owner command (§5), regardless of its author.

        The one author this must never match on is the daemon's own
        identity: a grounded dispatch (`relay_dispatch(reply_to=...)`)
        naturally e-tags the very thread root `_reply_watch_id` just
        resolved and registered a watch against, so the daemon's own
        subscription echoes that just-sent message straight back with
        matching tags. Without this check, that self-echo satisfies its own
        reply-wait instantly, and `summarize_reply` ends up "summarizing"
        the relayed instruction itself rather than any real reply --
        producing a fabricated-looking response before the actual working
        agent has said anything.
        """
        if event["pubkey"] == self._own_pubkey:
            return False
        for target_id in _reply_target_ids(event):
            future = self._reply_watches.pop(target_id, None)
            if future is None:
                continue
            if future.done():
                return False
            future.set_result(event)
            return True
        return False

    def _watch_for_reply(
        self, relayed_event_id: str, dm_channel_id: str, dm_reply_to: str
    ) -> None:
        future = asyncio.get_running_loop().create_future()
        self._reply_watches[relayed_event_id] = future
        asyncio.create_task(
            self._safe_await_and_summarize(
                relayed_event_id, dm_channel_id, dm_reply_to, future
            )
        )

    async def _safe_await_and_summarize(
        self,
        relayed_event_id: str,
        dm_channel_id: str,
        dm_reply_to: str,
        future: asyncio.Future,
    ) -> None:
        # This runs detached (via asyncio.create_task, never awaited by the
        # main loop), so it needs its own safety net -- same rationale as
        # _safe_handle wrapping _handle_event.
        try:
            await self._await_and_summarize(
                relayed_event_id, dm_channel_id, dm_reply_to, future
            )
        except Exception as exc:  # noqa: BLE001 - background task, must never propagate
            print(f"swingbird: failed to summarize reply to {relayed_event_id}: {exc}")

    async def _await_and_summarize(
        self,
        relayed_event_id: str,
        dm_channel_id: str,
        dm_reply_to: str,
        future: asyncio.Future,
    ) -> None:
        try:
            reply_event = await asyncio.wait_for(
                future, timeout=self._config.dispatch.reply_wait_seconds
            )
        except TimeoutError:
            self._reply_watches.pop(relayed_event_id, None)
            return
        # Same reasoning as _handle_event's to_thread calls -- this runs
        # concurrently with the main loop, so blocking here starves the
        # inbound WebSocket just as much as blocking on the main path would.
        summary = await asyncio.to_thread(
            summarize_reply, self._llm, reply_event["content"]
        )
        # Appended so the owner can jump straight to the working agent's own
        # reply in the project channel, not just read a paraphrase of it.
        link = outbound.message_link(_channel_of(reply_event), reply_event["id"])
        await asyncio.to_thread(
            outbound.send_message,
            dm_channel_id,
            f"{summary}\n\n{link}",
            reply_to=dm_reply_to,
        )

    def _process(self, content: str, thread_id: str, event_id: str) -> str:
        """Route one piece of owner-authored text through the daemon's
        existing intent machinery and return the reply text.

        Deliberately takes plain `content`/`event_id` rather than a Buzz
        event dict -- the pubkey/channel checks that decide whether
        something *is* an owner command already happened one level up, in
        `_handle_event`, and nothing below this point needs any other field
        off the event. This is what lets `_run_voice_exchange` (see the
        Voice Mode design doc §V.3/§V.4, §V.16 step 5) feed a transcript
        through the exact same recap/dispatch/confirm/safety machinery as a
        text DM, without a real inbound Buzz event to hang it off of -- it
        just needs a thread id (the shared DM channel, for voice) and an
        event id to anchor replies/reply-waits to (the DM
        `_run_voice_exchange` posts for that exchange, for voice).
        """
        try:
            pending = self._disambiguation.get(thread_id)
            if pending is not None:
                resumed = self._resume_disambiguation(pending, content, thread_id)
                if resumed is not None:
                    return resumed
            pending_close = self._pending_close.get(thread_id)
            if pending_close is not None:
                resumed = self._resume_pending_close(pending_close, content, thread_id)
                if resumed is not None:
                    return resumed
            open_recap_items = self._recap_store.get(thread_id)
            has_open_recap = open_recap_items is not None
            has_pending_dispatch = self._store.get(thread_id) is not None
            intent = self._router.route(
                content,
                thread_id=thread_id,
                has_open_recap=has_open_recap,
                has_pending_dispatch=has_pending_dispatch,
                open_recap_items=tuple(
                    (item.channel, item.label) for item in open_recap_items or ()
                ),
            )
            return self._act(intent, thread_id, event_id)
        except _ACTIONABLE_ERRORS as exc:
            return f"Couldn't do that: {exc}"

    def _resume_disambiguation(
        self, pending: PendingDisambiguation, text: str, thread_id: str
    ) -> str | None:
        """Answer an open "which did you mean" question (see
        `recap_disambiguation.py`) if `text` resolves to one of its
        candidates; `None` otherwise, so `_process` falls through to
        normal intent routing for anything that isn't answering it."""
        item = resolve_choice(pending.candidates, text)
        if item is None:
            return None
        self._disambiguation.clear(thread_id)
        if pending.kind == "recap_action":
            return self._apply_recap_action(item, pending.intent, thread_id)
        return self._apply_recap_relay(item, pending.intent, thread_id)

    def _resume_pending_close(
        self, pending: PendingClose, text: str, thread_id: str
    ) -> str | None:
        """Answer an open "close these items?" confirmation (see
        `recap_close.py`) if `text` resolves as a deterministic yes/no;
        `None` otherwise, so `_process` falls through to normal intent
        routing for anything that isn't answering it -- same idiom as
        `_resume_disambiguation`, kept as its own check rather than folded
        into that one since closing never touches the pending-dispatch
        confirm/cancel machinery at all (see `recap_close.py`'s module
        docstring)."""
        answer = resolve_close_reply(text)
        if answer is None:
            return None
        self._pending_close.clear(thread_id)
        if not answer:
            return "Cancelled -- nothing was closed."
        for item in pending.items:
            self._closed_items.close(item)
        self._audit.log_closed_items(thread_id, pending.items)
        labels = ", ".join(f"{item.channel}/{item.label}" for item in pending.items)
        noun = "item" if len(pending.items) == 1 else "items"
        return f"Closed {noun}: {labels}."

    def _act(self, intent: Intent, thread_id: str, event_id: str) -> str:
        if intent.kind == "recap":
            channel_names = [intent.channel] if intent.channel else None
            built_recap = build_recap(
                self._llm,
                self._config,
                channel_names=channel_names,
                detail=intent.detail,
                closed_items=self._closed_items,
            )
            self._audit.log_recap_built(thread_id, built_recap.items)
            self._recap_store.set(thread_id, built_recap.items, channel=intent.channel)
            # A fresh recap replaces this thread's items outright (see
            # RecapActionStore.set) -- any open disambiguation referred to
            # the old ones, so resuming it now would resolve against
            # stale candidates.
            self._disambiguation.clear(thread_id)
            return built_recap.text
        if intent.kind in ("dispatch", "clarify_response"):
            return self._dispatch_or_ask(intent, thread_id)
        if intent.kind == "confirm":
            return self._confirm(thread_id, event_id)
        if intent.kind == "cancel":
            if self._disambiguation.get(thread_id) is not None:
                self._disambiguation.clear(thread_id)
                return "Cancelled -- nothing was sent."
            cancel_dispatch(self._store, thread_id, audit=self._audit)
            return "Cancelled -- nothing was sent."
        if intent.kind == "recap_action":
            return self._recap_action(intent, thread_id)
        if intent.kind == "recap_detail":
            return self._recap_detail(intent, thread_id)
        if intent.kind == "recap_list":
            return self._recap_list(intent, thread_id)
        if intent.kind == "recap_relay":
            return self._recap_relay(intent, thread_id)
        if intent.kind == "recap_close":
            return self._recap_close(intent, thread_id)
        if intent.kind == "reset_closed":
            return self._reset_closed(intent, thread_id)
        return _CHIT_CHAT_REPLY

    def _recap_action(self, intent: Intent, thread_id: str) -> str:
        item = self._resolve_or_store_single_recap_item(
            thread_id, intent, "recap_action", intent.message
        )
        return self._apply_recap_action(item, intent, thread_id)

    def _apply_recap_action(
        self, item: RecapItem, intent: Intent, thread_id: str
    ) -> str:
        self._audit.log_recap_reference(thread_id, "recap_action", intent.message, item)
        instruction = rephrase_for_dispatch(self._llm, item, intent.message)
        dispatch_intent = Intent(
            kind="dispatch",
            channel=item.channel,
            message=instruction,
            reply_to=item.source_event_id,
        )
        return self._dispatch_or_ask(dispatch_intent, thread_id)

    def _recap_relay(self, intent: Intent, thread_id: str) -> str:
        """Forward the user's own question/comment about a recap item to
        its agent (see `recap_relay.py`).

        Unlike `_recap_action` (which relays the *recap's own* instruction,
        narrowed by `rephrase_for_dispatch`), `intent.message` here is new
        content from the user that didn't come from the recap at all --
        `relay_with_context` forwards it close to verbatim rather than
        rewriting it. Reuses the exact same item-resolution
        (`_resolve_or_store_single_recap_item`) and dispatch/confirm/
        reply-wait path `_recap_action` does, so this gets "threaded to the
        item's source message" and "wait-and-summarize back into the same
        DM thread" for free -- see `AGENTS.md`'s safety invariants.
        """
        if intent.message is None:
            return "I didn't catch what to relay -- what should I tell the agent?"
        item = self._resolve_or_store_single_recap_item(
            thread_id, intent, "recap_relay", intent.item_reference
        )
        return self._apply_recap_relay(item, intent, thread_id)

    def _apply_recap_relay(
        self, item: RecapItem, intent: Intent, thread_id: str
    ) -> str:
        self._audit.log_recap_reference(
            thread_id, "recap_relay", intent.item_reference, item
        )
        relayed = relay_with_context(
            self._llm, item, intent.item_reference, intent.message
        )
        dispatch_intent = Intent(
            kind="dispatch",
            channel=item.channel,
            message=relayed,
            reply_to=item.source_event_id,
        )
        return self._dispatch_or_ask(dispatch_intent, thread_id)

    def _require_recap_items(self, thread_id: str) -> tuple[RecapItem, ...]:
        """Return this thread's stored recap items, or raise the same
        "ask for a recap first" error both `_recap_close` and
        `_resolve_recap_items` need when nothing's been recapped yet."""
        items = self._recap_store.get(thread_id)
        if not items:
            raise RecapActionError(
                "I don't have a recent recap to reference here -- ask for a "
                "recap first."
            )
        return items

    def _recap_close(self, intent: Intent, thread_id: str) -> str:
        items = self._require_recap_items(thread_id)
        selected = select_items_to_close(self._llm, items, intent.message)
        for item in selected:
            self._audit.log_recap_reference(
                thread_id, "recap_close", intent.message, item
            )
        return self._propose_close(selected, intent, thread_id)

    def _propose_close(
        self, items: tuple[RecapItem, ...], intent: Intent, thread_id: str
    ) -> str:
        """Propose closing `items` exactly as `recap_close_selection.py`
        selected them -- one item, a project's worth, an explicit list, or
        any combination its scoping grammar supports.

        This used to also expand a single explicitly-selected item to every
        other item sharing its `source_event_id` ("all the work items
        grounded on this message"), from when close resolution reused
        `recap_actions.resolve_reference`'s single-item-only matching and
        that expansion was the only way to catch "did you mean everything
        this message covers." Now that `select_items_to_close` already
        resolves the full requested scope itself -- including an exact
        label match "regardless of project" per its own system prompt --
        that expansion only second-guesses an already-precise selection: a
        recap message routinely covers more than one unrelated item for the
        same project (e.g. two independent PRIMARY items), and grounding by
        message silently re-added the sibling no matter how exactly the
        user named just one of them. Trusting the selection as final is
        what makes "close X, not Y" actually close just X.

        This asks for a deterministic yes/no confirmation
        (`_resume_pending_close`) before persisting anything, mirroring the
        confirm-before-write invariant `pending_actions.py` enforces for a
        dispatch, without reusing that store -- closing never relays
        anything, so there's nothing to confirm through the dispatch
        confirm/cancel path. Listing every item in the proposal (not just a
        count) is what lets the user catch and cancel a wrongly-scoped
        selection before anything is persisted.
        """
        self._pending_close.set(thread_id, PendingClose(items))
        labels = ", ".join(f"{i.channel}/{i.label}" for i in items)
        if len(items) == 1:
            return (
                f"Close {labels} -- it won't be shown as open in future "
                "recaps? Confirm to close, or cancel."
            )
        return (
            f"That request covers {len(items)} items: {labels}. Close all "
            "of them -- none will be shown as open in future recaps? "
            "Confirm to close, or cancel."
        )

    def _reset_closed(self, intent: Intent, thread_id: str) -> str:
        """Clear previously closed items (see `ClosedItemStore.reset`), so
        future recaps surface them as open work again -- Voidious's
        requested fail-safe for a close made in error.

        No confirm-before-write step here, unlike `_recap_close`/
        `_propose_close`: closing an item risks silently hiding real,
        still-open work forever if the selection is wrong, which is exactly
        what that confirmation guards against (see `recap_close.py`'s
        module docstring). A reset's worst case is the opposite and far
        milder -- an already-finished item briefly reappears in one future
        recap -- and it's trivially self-correcting: just close it again.
        That asymmetry is why this writes immediately instead of going
        through `PendingCloseStore` or a store of its own.
        """
        if (
            intent.channel is not None
            and self._config.channel_by_name(intent.channel) is None
        ):
            raise RecapActionError(f"unknown project channel: {intent.channel!r}")
        count = self._closed_items.reset(intent.channel)
        self._audit.log_closed_items_reset(thread_id, intent.channel, count)
        scope = intent.channel or "all projects"
        if count == 0:
            return f"Nothing to reset -- no closed items for {scope}."
        noun = "item" if count == 1 else "items"
        return f"Reset {count} closed {noun} for {scope}."

    def _recap_detail(self, intent: Intent, thread_id: str) -> str:
        resolved = self._resolve_recap_items(thread_id, intent.message, intent.channel)
        for item in resolved.items:
            self._audit.log_recap_reference(
                thread_id, "recap_detail", intent.message, item
            )
        threads = [self._fetch_item_thread(item) for item in resolved.items]
        dm_messages = fetch_recent_messages(
            thread_id, limit=_RECAP_DETAIL_DM_HISTORY_LIMIT
        )
        return elaborate(
            self._llm,
            resolved.items,
            threads,
            dm_messages,
            intent.message,
            self._config,
            no_other_items=resolved.degraded,
        )

    def _recap_list(self, intent: Intent, thread_id: str) -> str:
        """Enumerate the recap's additional/open items at the recap's own
        level of detail (see `recap_list.render_items`) -- unlike
        `_recap_detail`, this never calls the LLM or fetches an item's
        source thread: there's nothing to ground beyond each item's own
        already-grounded `summary`/`instruction`, so doing either would
        just be wasted work.
        """
        resolved = self._resolve_recap_items(thread_id, intent.message, intent.channel)
        for item in resolved.items:
            self._audit.log_recap_reference(
                thread_id, "recap_list", intent.message, item
            )
        return render_items(resolved.items, resolved.degraded, self._config)

    def _fetch_item_thread(self, item: RecapItem) -> list[dict]:
        """Return every message in the thread `item.source_event_id`
        belongs to, so `recap_detail`'s elaboration can draw on more than
        the recap's own condensed summary/instruction.

        `[]` when the item wasn't grounded in one specific transcript
        message (see `RecapItem.source_event_id`) -- nothing to fetch,
        never guessed. `item.channel` is always one of the config's known
        channel names in practice (see `recap.py`'s extraction prompt), but
        an LLM-sourced field is never trusted blindly (§5) -- an unresolved
        name raises the same as an unknown dispatch target does in
        `pending_actions.propose_dispatch`, rather than guessing or
        silently dropping the thread context.
        """
        if item.source_event_id is None:
            return []
        channel = self._config.channel_by_name(item.channel)
        if channel is None:
            raise RecapActionError(
                f"recap item names an unknown channel: {item.channel!r}"
            )
        return fetch_thread_messages(channel.id, item.source_event_id)

    def _resolve_recap_items(
        self,
        thread_id: str,
        reference: str | None,
        channel: str | None = None,
    ) -> ResolvedReference:
        """Resolve `reference` against the last recap's items for `thread_id`.

        `channel` is `intent.channel` from the router's own classification --
        passed through as `resolve_reference`'s authoritative channel signal
        rather than leaving it to re-derive a channel by searching `reference`
        for a known name, which misses whenever the router already stripped
        the channel out of the leftover reference text. When `channel` is
        `None`, this falls back to the channel the last recap in this thread
        was itself scoped to (`RecapActionStore.channel_for`) -- the router
        only ever classifies the current message in isolation, so it has no
        way to report a channel for e.g. "tell me more" after "recap
        dripbird" even though there's nothing left to disambiguate; the
        thread already answered "which project" the moment it got a
        project-scoped recap, and every follow-up in it should keep that
        answer rather than needing the user to repeat it.

        `.items` can hold more than one item -- e.g. "the additional items"
        resolves to every non-primary item for a channel (see `recap_actions.
        resolve_reference`). `_recap_detail` elaborates on however many come
        back, and passes `.degraded` through to `elaborate` so the user is
        told when that resolved to the same single item rather than a real
        additional one; `_resolve_single_recap_item` (for `recap_action`,
        which posts a dispatch and so can't act on a batch or care about
        `.degraded` -- see recap_actions.py's module docstring) narrows
        `.items` further to exactly one.
        """
        items = self._require_recap_items(thread_id)
        if channel is None:
            channel = self._recap_store.channel_for(thread_id)
        return resolve_reference(items, reference, channel)

    def _resolve_single_recap_item(
        self,
        thread_id: str,
        reference: str | None,
        channel: str | None = None,
    ) -> RecapItem:
        """Like `_resolve_recap_items`, but raises `AmbiguousRecapReference`
        (a `RecapActionError`, caught centrally, see `_ACTIONABLE_ERRORS`)
        for anything that isn't exactly one match -- v1 scope for
        `recap_action`/`recap_relay`: a reference matching more than one
        item (including an explicit "all") is treated the same as an
        ambiguous single-item reference, since it's about to become a
        dispatch and can't act on a batch. Carrying the candidates on the
        exception (rather than just the formatted message) is what lets
        `_resolve_or_store_single_recap_item` remember them for a
        disambiguation follow-up.
        """
        matched = tuple(self._resolve_recap_items(thread_id, reference, channel).items)
        if len(matched) > 1:
            raise AmbiguousRecapReference(
                "That matches more than one item, and I can only act on "
                "one at a time for now -- which did you mean: "
                f"{format_choices(matched)}? Reply with the number.",
                matched,
            )
        return matched[0]

    def _resolve_or_store_single_recap_item(
        self,
        thread_id: str,
        intent: Intent,
        kind: str,
        reference: str | None,
    ) -> RecapItem:
        """Like `_resolve_single_recap_item`, but on an ambiguous reference
        also remembers `intent` and the candidates as a
        `PendingDisambiguation` for this thread (see
        `recap_disambiguation.py`), so the user's next reply can answer
        the "which did you mean" question instead of being misrouted as a
        new command -- then re-raises, so the question still reaches the
        user exactly like it did before this existed.
        """
        try:
            return self._resolve_single_recap_item(thread_id, reference, intent.channel)
        except AmbiguousRecapReference as exc:
            self._disambiguation.set(
                thread_id, PendingDisambiguation(kind, exc.candidates, intent)
            )
            raise

    def _dispatch_or_ask(self, intent: Intent, thread_id: str) -> str:
        if intent.channel is None or intent.message is None:
            return (
                "I didn't catch which channel or what to relay -- which "
                "project channel should I dispatch to, and what should the "
                "message say?"
            )
        if intent.kind != "dispatch":
            intent = replace(intent, kind="dispatch")
        proposal = propose_dispatch(
            self._store, self._config, thread_id, intent, audit=self._audit
        )
        target = f" (for {proposal.target_agent})" if proposal.target_agent else ""
        return (
            f"About to relay to {intent.channel}{target}: "
            f"{proposal.instruction!r}. Confirm to send, or cancel."
        )

    def _confirm(self, thread_id: str, dm_reply_to: str) -> str:
        # Runs off-thread (see _handle_event) -- stash the watch request for
        # _handle_event to register via _watch_for_reply once it's back on
        # the main thread, rather than creating the asyncio Future/Task here.
        event_id, proposal = confirm_dispatch(
            self._store, thread_id, self._config.owner, audit=self._audit
        )
        self._pending_watch = (
            self._reply_watch_id(event_id, proposal),
            thread_id,
            dm_reply_to,
        )
        link = outbound.message_link(proposal.channel_id, event_id)
        return f"Confirmed and relayed: {link}"

    def _reply_watch_id(self, event_id: str, proposal: DispatchProposal) -> str:
        """Return the id to register a reply-wait against for this dispatch.

        A fresh (ungrounded) dispatch posts top-level, so it's already its
        own thread root and `event_id` is exactly right. A recap-action
        dispatch instead threads to `proposal.reply_to` (see
        `pending_actions.py`) -- a message inside an existing thread whose
        real root can be several messages further up, and a coding agent's
        own status-update reply often threads to *that* root rather than to
        the specific relayed message (see module docstring). Resolving the
        true root here, once, up front, is what lets a single dict lookup in
        `_resolve_reply_watch` match either shape of reply. Best-effort: a
        lookup failure falls back to `event_id` (matching a direct reply
        only, the pre-existing behavior) rather than losing the
        wait-and-summarize outright.
        """
        if proposal.reply_to is None:
            return event_id
        try:
            return fetch_thread_root(proposal.channel_id, proposal.reply_to)
        except outbound.RelayError as exc:
            print(f"swingbird: failed to resolve thread root for {event_id}: {exc}")
            return event_id


def _channel_of(event: dict) -> str:
    """Return the channel (#h tag) an event was posted in.

    Also doubles as the pending-action store's "thread id": the channel a
    message arrives in *is* the conversation, so a bare "yes" typed as a new
    message in that channel resolves the proposal made there without needing
    to reply-thread it, and confirm/cancel can never cross channels.
    """
    for tag in event.get("tags", []):
        if tag and tag[0] == "h":
            return tag[1]
    raise DaemonError(f"event {event.get('id')} has no channel (#h) tag")


def _reply_target_ids(event: dict) -> list[str]:
    """Return every event id referenced by an `e` tag on `event` (NIP-10).

    A reply can carry a "root" tag, a "reply" tag, or both depending on
    thread depth, in either order -- see `Daemon._reply_watch_id` -- so
    matching a reply-wait needs to check all of them, not just the first.
    """
    return [tag[1] for tag in event.get("tags", []) if len(tag) >= 2 and tag[0] == "e"]


def build_daemon(
    config_path: str, audit_log_path: str, closed_items_path: str
) -> Daemon:
    """Load config and wire up a `Daemon`, reading the relay key from env."""
    config = load_config(config_path)
    private_key = _read_private_key(config.relay.private_key_env)
    audit = AuditLog(audit_log_path)
    llm = LLMClient(config.llm)
    router = IntentRouter(llm, config, audit=audit)
    inbound = InboundClient(config.relay.url, private_key)
    closed_items = ClosedItemStore(closed_items_path)
    return Daemon(
        config, inbound, router, PendingActionStore(), llm, audit, closed_items
    )


def _read_private_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise InboundError(f"environment variable {env_var} is not set")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the swingbird TPM agent daemon.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG_PATH)
    parser.add_argument("--closed-items", default=DEFAULT_CLOSED_ITEMS_PATH)
    args = parser.parse_args()

    daemon = build_daemon(args.config, args.audit_log, args.closed_items)
    asyncio.run(daemon.run())
