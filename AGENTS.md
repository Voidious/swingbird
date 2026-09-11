# AGENTS.md

Instructions for coding agents working in this repo. `README.md` is the source of truth for
what swingbird does, how to configure it, and how to run it -- read that first if you haven't.
This file is about how to work *in* the codebase.

## Setup

```bash
uv sync
```

Requires Python 3.12+ (`.python-version` pins 3.13). `uv sync` also pulls the dev group
(`crispen`, `pytest-cov`, `ruff`, `pre-commit`).

## Build / test / lint

```bash
uv run pytest        # 100% branch coverage enforced (--cov-fail-under=100, see pyproject.toml)
uv run ruff check .
uv run ruff format .
```

A change isn't done until `pytest` passes at 100% branch coverage and `ruff check` is clean.
Coverage is checked per-run over the whole `swingbird` package, not per-file -- a new module or
branch needs its own tests in the same change, not a follow-up.

Pre-commit (`.pre-commit-config.yaml`) runs `crispen` on the staged diff, `ruff-check --fix`,
`ruff-format`, and the full `pytest` run with coverage on every commit. Don't bypass it with
`--no-verify`; if a hook fails, fix the underlying issue.

## Code conventions

- `from __future__ import annotations` at the top of every module; use modern type hints
  (`list[dict]`, `str | None`) directly, no `typing.List`/`typing.Optional`.
- Line length 88 (`ruff` config in `pyproject.toml`), enforced via `E501`.
- Module and function docstrings explain *why* a design choice was made (and reference the
  design doc's `§n.n` sections where relevant), not what the code obviously does. Match that
  style in new modules -- see any existing file under `swingbird/` for the pattern.
- Config precedence: `swingbird.toml` holds shareable defaults; the gitignored
  `.swingbird.toml` next to it overrides on top, merged key-by-key per section (a list like
  `channels` is replaced outright, not merged element-by-element). Never put a real secret or
  real channel ID in `swingbird.toml` -- those belong in `.swingbird.toml` or an env var named
  by an `*_env` config key.
- Writes to Buzz go through the `buzz` CLI subprocess (`outbound.py`'s `run_buzz_cli`, also
  reused by `history.py`). Reads of live channel activity use the direct WebSocket client in
  `inbound.py`. Don't blur this line -- it's what lets the daemon hold a persistent
  subscription while still using the CLI's own identity handling for writes.

## Safety invariants (don't relax these without asking Voidious)

- Only messages from `config.owner.pubkey`, in swingbird's own DM with the owner, are ever
  routed as a command. Anything in a project channel -- even from the owner, even an
  `@mention` of swingbird -- is read-only context for recaps, never a command.
  See `daemon.py`.
- A `dispatch` intent never posts directly; it becomes a `DispatchProposal` in
  `pending_actions.py` and only reaches `outbound.relay_dispatch` via `confirm_dispatch`.
  A `recap_action` ("go ahead with F4") and a `recap_relay` ("for dripbird F4, couldn't we
  just pre-compile it?") both resolve to a `dispatch` `Intent` and go through this exact same
  path -- there is no separate posting route for a recap follow-up. When the matched
  `RecapItem` has a `source_event_id`, the relayed message threads to it
  (`Intent.reply_to` -> `DispatchProposal.reply_to` -> `relay_dispatch`); a fresh dispatch has
  no such message and always posts top-level.
- On any ambiguity (unresolvable channel/agent, multiple pending proposals for a bare
  "confirm"), the router/store asks the user rather than guessing. Don't add
  best-guess fallbacks here -- guessing wrong means dispatching to the wrong channel. The same
  applies to `RecapItem.source_event_id`: `recap.py` only sets it when the LLM cites one
  specific transcript message, never a "closest guess" like the last message in the channel
  (rejected explicitly -- an item can be an older, still-unfinished thing).
- A single bad or unexpected inbound event must never crash the daemon loop; known failure
  modes become a reply to the sender, unexpected ones are logged and the loop continues.

## Architecture map

| Module | Responsibility |
| --- | --- |
| `daemon.py` | Wires inbound events to the router, pending-action store, and audit log; owns the safety invariants above. |
| `inbound.py` | Persistent WebSocket client to the relay (NIP-42 auth + subscription). |
| `outbound.py` | All writes, via the `buzz` CLI subprocess. |
| `nostr_crypto.py` | NIP-01 key parsing / signing / verification, used only by the direct WebSocket path. |
| `llm.py` | Thin OpenAI-compatible client wrapper (`LLMClient`); config-driven base URL/key/model so swapping providers is a config change. |
| `router.py` | LLM call that classifies an inbound DM into an intent (recap / dispatch / confirm / cancel / recap_action / recap_detail / recap_relay / chit-chat). |
| `pending_actions.py` | Confirm/cancel state machine for proposed dispatches. |
| `history.py` | One-shot fetch of recent channel messages (via `outbound.run_buzz_cli`), for recaps. |
| `recap.py` | LLM-summarizes recent channel activity into a short recap, plus structured per-channel `RecapItem`s. |
| `recap_actions.py` | Stores the latest recap's items per thread and resolves a "go ahead with X" / "tell me more about X" / "for X, ..." reference against them. |
| `dispatch_phrasing.py` | Narrows/rewrites a recap item's own instruction into a directive, for `recap_action`. |
| `recap_relay.py` | Forwards the user's own question/comment about a recap item to its agent near-verbatim, resolving ambiguous references (e.g. "it") against the item's context, for `recap_relay`. |
| `reply_summary.py` | LLM-summarizes a coding agent's reply to a relayed dispatch, for the owner's DM. |
| `audit.py` | Local append-only JSON-lines log of inbound events and proposal outcomes, independent of Buzz's own event log. |
| `config.py` | Loads and merges `swingbird.toml` + `.swingbird.toml`. |

## Tests

One `tests/test_<module>.py` per `swingbird/<module>.py`. Match that layout for new modules.
Tests must not require live network/relay/LLM access -- mock `LLMClient`, `run_buzz_cli`, and
the WebSocket client rather than hitting real services.
