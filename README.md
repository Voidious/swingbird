# swingbird

A stand-alone Buzz agent that helps keep your projects in full swing.

swingbird is a TPM-style Buzz agent that lives in your own workspace. DM it and it recaps your
project channels, relays instructions to your coding agents, and always confirms with you before
it writes anything on your behalf.

_The upbeat, unbothered swingbird roams the canopy of the Forest of Code, vibing with the Abstract
Syntax Trees as humans and agents work below. Sporting a lowkey dank hoodie and carrying a small
tablet, the swingbird is ready to spill the tea on project status, lock in on potential solutions,
or serve instructions to your coding agents to keep your project in full swing._

## Overview

Talk to swingbird in a private DM:

```
you:       swingbird, what's going on?
swingbird: [a short, prioritized recap of your configured project channels]

you:       tell backend to fix the login bug
swingbird: About to relay to backend (for Sonnet): 'fix the login bug'. Confirm to send, or cancel.

you:       confirm
swingbird: Confirmed and relayed: buzz://message?channel=...&id=...
           [a little later]
swingbird: Fixed the login bug -- turned out to be a stale session cookie.

you:       for backend F4, couldn't we just cache that instead?
swingbird: About to relay to backend (for Sonnet): "couldn't we just cache that instead?"
           Confirm to send, or cancel.
```

Every dispatched instruction is proposed first and only sent once you confirm it -- swingbird
never posts into a project channel on its own initiative. See
[Supported instructions](#supported-instructions) below.

## Installation

swingbird requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and the `buzz` CLI (bundled
with [Buzz Desktop](https://github.com/block/buzz)) on your `PATH`.

```bash
git clone https://github.com/Voidious/swingbird
cd swingbird
uv sync
```

## Buzz identity

swingbird runs as its own Buzz identity (its own Nostr keypair), separate from your personal
account and from any coding agent identities. This lets it DM you directly and post into project
channels under its own name.

1. **Generate a keypair for the daemon.** If you operate your own relay (via its Docker image),
   it ships a `buzz-admin` binary:

   ```bash
   buzz-admin generate-key
   ```

   This prints a public key and a secret key. Any standard Nostr keypair (hex or `nsec1...`) works
   too, since swingbird accepts either. Keep the secret key out of version control -- it's what
   you'll set as an environment variable in [Running](#running) below.

2. **Add the public key as a Buzz member.** If you operate the relay:

   ```bash
   buzz-admin add-member --pubkey <public key> --role member
   ```

   Otherwise, ask whoever operates it to add the identity for you.

3. **Add the identity to every project channel you plan to list under `[[channels]]`**
   (see [Configuration](#configuration) below):

   ```bash
   buzz channels add-member --channel <channel UUID> --pubkey <public key>
   ```

   swingbird needs channel membership to read a channel's history for recaps, even for channels
   where `write = false` in its config.

4. **Start the daemon** (see [Running](#running)) so it opens its own DM with you and comes
   online, then open a DM with the identity from Buzz Desktop or mobile to start talking to it.

## Configuration

swingbird reads `swingbird.toml` from the project root, with an optional gitignored
`.swingbird.toml` next to it for values you don't want checked in (a private relay URL, real
channel ids, and so on). Sections in the override file are merged key-by-key on top of the
committed config; anything else -- including a whole `[[channels]]` list -- is replaced outright,
so a `.swingbird.toml` that defines `channels` needs to list every channel you want active, not
just the ones that differ from `swingbird.toml`.

```toml
[llm]
provider = "openai"
model = "gpt-5.6-luna"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[owner]
pubkey = "<your Buzz pubkey>"
name = "<your Buzz display name>"

[identity]
name = "swingbird"
description = "TPM agent for swingbird-dev"

[identity.avatar]
style = "emoji"
emoji = "🐦"
color = "#3399FF"

[dispatch]
reply_wait_seconds = 180

[[channels]]
id = "<Buzz channel UUID>"
name = "backend"
write = true
agents = ["Sonnet"]
```

- **`[llm]`** -- the OpenAI-compatible backend swingbird uses for intent classification, recaps,
  and reply summaries. `provider` is `"moonshot"` or `"openai"`; it fills in `base_url` and
  `api_key_env` (the environment variable to read the key from, not the key itself) with that
  provider's defaults, so a config only needs `provider` + `model`. Set `base_url`/`api_key_env`
  explicitly instead of (or to override) `provider` for a self-hosted or otherwise unlisted
  OpenAI-compatible endpoint.
- **`[relay]`** -- the Buzz relay to connect to, and the environment variable holding swingbird's
  own secret key (see [Buzz identity](#buzz-identity) above).
- **`[owner]`** -- the one identity swingbird will ever treat as a command. Every other message it
  sees, including a coding agent's own replies in a subscribed channel, is read-only context.
- **`[identity]`** -- swingbird's own Buzz profile: `name` (required), plus an optional
  `description` (bio) and `[identity.avatar]` (only the `"emoji"` style is supported so far: a
  colored circular background with one emoji centered on it, matching Buzz Desktop's own
  "Emoji" avatar style). Synced on startup whenever the identity's current profile doesn't
  already match, so a fresh identity or a renamed deployment always shows up looking the way
  you configured it.
- **`[dispatch]`** -- optional. `reply_wait_seconds` (default `180`) is how long swingbird waits
  for a reply to a relayed instruction before giving up on summarizing it back to you. Keep this
  short if you might ever run swingbird as a voice assistant -- a reply arriving minutes later,
  out of nowhere, would be a jarring surprise.
- **`[recap]`** -- optional. `stale_after_days` (default `30`) is how long a project channel can
  go quiet before a "recap everything" request drops it rather than reporting it as inactive.
  Naming a channel explicitly ("recap backend") uses this same time window, but is never dropped
  from the reply -- it's reported as having no recent activity instead of being omitted.
  `max_messages_per_channel` (default `1000`) caps how many messages a recap will page through
  per channel to fill that window -- a single `buzz messages get` call maxes out at 200 messages
  regardless of the `--limit` requested, so swingbird pages backwards with `--before` as needed,
  and this is the safety valve stopping that from running away on an unusually chatty channel.
  `max_detailed_items` (default `3`) caps how many of a project's open items a detailed recap
  narrates before folding the rest into a count, the same way a concise recap always folds
  everything past its one leading item. `closed_item_window_days` (default `90`) caps how far
  back a closed item ("close F4") still gets sent to the LLM as "don't re-list this" context --
  floored at `stale_after_days`, so it can never be narrower than the recap's own message window
  (see [Closing an item](#closing-an-item) below).
- **`[[channels]]`** -- one entry per project. `id` is the Buzz channel UUID; `name` is what
  you'll say in conversation ("recap backend", "tell backend to...") and is independent of the
  channel's own Buzz display name; `write` controls whether swingbird may dispatch instructions
  into it (a channel with `write = false` can still be recapped); `agents` lists the Buzz display
  names of the coding agents in that channel (see
  [Configuring project agents](#configuring-project-agents-to-listen-to-swingbird) below); `goal`
  is optional free text (e.g. `"Preparing the 0.8.0 release"`), used only in a concise recap when
  a project has no current open item, to add a one-sentence "next up" line.

A checkout with no `[[channels]]` entries fails to start with a clear error instead of silently
running against someone else's project -- put your real channels in `.swingbird.toml`.

## Configuring project agents to listen to swingbird

A dispatched instruction is posted into the target project channel with an `@mention` of the
resolved `target_agent`, since Buzz agents only react to explicit `@mention`s by default. For that
mention to actually reach an agent:

- the name in that channel's `agents` list in your config must exactly match the agent's Buzz
  display name in that channel, and
- the swingbird identity must be a member of the channel (see
  [Buzz identity](#buzz-identity) above) so its messages are actually delivered.

If a channel's `agents` list has exactly one entry, swingbird fills it in automatically when you
don't name one yourself ("tell backend to fix the login bug" dispatches to that one agent). With
zero or multiple agents configured, swingbird asks which one you mean rather than guessing.

## Running

Before starting the daemon, its own shell environment needs:

- `SWINGBIRD_PRIVATE_KEY` (or whichever name `[relay].private_key_env` points at) -- read
  directly by swingbird's inbound WebSocket client.
- `BUZZ_PRIVATE_KEY`, set to that same secret key -- every outbound write (recap replies,
  dispatches, presence, display name) goes through the bundled `buzz` CLI as a subprocess, which
  reads its own identity from this env var, independent of the config above.
- `BUZZ_RELAY_URL`, matching `[relay].url` -- so the `buzz` CLI subprocess talks to the same relay
  the daemon's own WebSocket client does, instead of its own default.
- Your `[llm].api_key_env` variable (e.g. `MOONSHOT_API_KEY`).

```bash
uv run python main.py --config swingbird.toml --audit-log audit.jsonl --closed-items closed_items.jsonl
```

`--config`, `--audit-log`, and `--closed-items` default to `swingbird.toml`, `audit.jsonl`, and
`closed_items.jsonl` in the current directory. The audit log is a local, append-only JSON-lines
record of every inbound message, proposed dispatch, and confirm/cancel decision -- independent of
Buzz's own event log, and kept so you can see why swingbird proposed what it proposed, including
proposals that got cancelled and never became a real Buzz message. The closed-items file is a
separate append-only JSON-lines record of every item you've closed (see
[Closing an item](#closing-an-item)) -- unlike the audit log, it's read back on every recap, so
don't delete it unless you want previously-closed items to start reappearing.

## Usage

DM the swingbird identity directly -- that DM is its only interaction surface. @mentioning it in
a project channel, or anything anyone other than the configured owner says, is never treated as a
command.

### Supported instructions

| You say | swingbird does |
| --- | --- |
| "what's going on?" / "recap backend" | Concise by default: one immediately-actionable item per project (current status, then a proposed next step), plus a count of any other open items. Channels idle past `[recap].stale_after_days` are dropped from an all-channels recap (a named channel is always included). Say "detailed recap" (or similar) for up to `[recap].max_detailed_items` (default 3) items per project, each with a few sentences of extra detail, plus a separate line naming anything folded beyond that. Either way, every item shown can be followed up on -- "tell me more about F4," "go ahead with the login fix," "what are the other items?" |
| "tell backend to fix the login bug" / "ask frontend if the tests pass" | Proposes relaying that instruction (or question) to the named channel/agent. Nothing is sent until you confirm. |
| "go ahead with F4" / "do the login fix" | Resolves "F4"/"the login fix" against the most recent recap's items, then proposes relaying that item's own instruction to its agent -- same confirm/cancel flow as a fresh dispatch. |
| "tell me more about F4" | Asks the LLM to elaborate on that recap item beyond its stored summary, using its original source thread -- no relay, nothing to confirm. |
| "what are the other items?" | Lists a recap's folded/additional items at the same concise or detailed level the recap itself used -- no LLM call, just a formatted read of what's already stored. |
| "for backend F4, couldn't we just cache that instead?" | Forwards your own question or comment about that recap item to its agent, close to verbatim (resolving a vague "it"/"that" using the item's context first) -- same confirm/cancel flow as a fresh dispatch. |
| "close F4" / "close all swingbird items" / "close the additional dripbird items" / "close F4 and F7" | Proposes marking the matching item(s) as closed, so they stop appearing as open work in future recaps -- see [Closing an item](#closing-an-item). Nothing is closed until you confirm. |
| "reset the closed items for dripbird" / "reset all closed items" | Clears the closed-item record for one named project, or every project, so anything closed reappears as open work in future recaps -- see [Resetting closed items](#resetting-closed-items). Written immediately, no confirmation needed. |
| "confirm" / "do it" / "yes" | Sends the most recently proposed instruction, or, for a close proposal, persists the close. |
| "cancel" / "never mind" | Discards the pending proposal without sending or closing anything. |
| anything else | swingbird says it's outside what it handles, and suggests asking for a recap or a dispatch instead. |

If swingbird can't tell which channel, agent, or recap item a request means, it asks you to
clarify rather than guessing -- answering that follow-up resolves the original request. Once you
confirm a dispatch, swingbird waits (up to `[dispatch].reply_wait_seconds`) for the target agent's
reply, then summarizes it back into your DM.

### Closing an item

A recap is fully stateless -- every call re-extracts open items fresh from recent channel
activity, so there's no queryable to-do list to check an item off of. "Close F4" (or any other
reference a recap follow-up understands -- see the table above) instead records a durable
snapshot of the item in the closed-items file (`--closed-items`, see [Running](#running)), and
every future recap consults that record to avoid re-listing the same work as open, even if a
later message restates it.

A close request can name more than one item: every item for a project, just its additional/open
items, every project's items in the current recap, an explicit list ("F4 and F7", or "F4 for
swingbird and F2 for dripbird"), or a combination of these in one request ("all swingbird items
including additional"). Naming one item closes only that item, even if the same recap message
also covers another, unrelated item for the same project -- an exact label match is never
silently widened to a neighbor you didn't name. Every item selected is listed before asking you
to confirm -- so a close request never silently closes something you didn't ask for without
telling you first. Nothing is closed until you confirm.

A closed item stops being suppressed once it falls outside `[recap].closed_item_window_days`, on
the (rare) assumption that a restatement that old is unlikely to still be the same open thread of
work. If it turns out to still be the same work, closing it again picks up right where you left
off.

### Resetting closed items

A close made in error would otherwise mean living with it until `[recap].closed_item_window_days`
elapses, or editing `closed_items.jsonl` by hand -- "reset the closed items for dripbird" (or "reset
all closed items" to name no project) is the fail-safe: it clears the closed-item record for that
one project, or for every project, and every future recap goes back to surfacing that work as open.

Unlike closing an item, a reset writes immediately with no confirm/cancel step. Closing risks
silently hiding real open work forever if the selection is wrong -- that's what its confirmation
guards against. A reset's worst case is the opposite and mild: an already-finished item briefly
reappears in one recap, and closing it again picks up right where you left off.

## Development

```bash
uv sync
uv run pytest        # 100% branch coverage enforced
uv run ruff check .
uv run ruff format .
```

Pre-commit hooks run `crispen` on the staged diff, `ruff`, and the full test suite with coverage
on every commit.

## swingbird origin story

Throughout my programming career, I've always had side projects. With coding agents, it's an
exciting time if you like to build stuff. I have a few side projects going that I think are
genuinely cool and useful.

I've observed that, for me, coding agents have taken over the most fulfilling part of
programming. If your goal is to build something, it's indeed exciting and very cool that you can
replace hours of coding with a few minutes of prompting, but it has made working on my side
projects kind of boring. Side projects used to be where I got more of the stuff I like most about
software development: writing lots of code and building things from scratch. Now, it has more of
the same flavor as what I do at work all day. As a result, I work on my side projects in bursts,
then let them stagnate.

I like my side projects. I would prefer to keep making progress on them without burning myself
out. Working professionally, you have freedom in how to manage your time. With side projects, you
also have the option of ignoring them completely, so the life of your side project may depend on
keeping the development process enjoyable. I have plenty of quota in my coding plans and a
prioritized to-do list for every project, but I don't want to spend all my free time attending to
coding agents if it feels like work.

I've been using Buzz for my side projects. That means the conversation history for all of them is
in one place, regardless of the harness (Claude Code or OpenCode), and it's easy to integrate
with.

I got the idea to write a custom voice assistant to act like a foreman at a construction site,
keeping my side projects rolling with much less input from me. Not every turn with my coding
agents requires high-level analysis or a personal touch -- most of the time, we're continuing an
in-progress work item, which often boils down to a single decision or confirmation. Having that
decision distilled and presented over voice is a much lower level of effort than using Buzz
desktop or mobile to navigate to each project, read the latest thread, and type a response.

As I sip my coffee, I want to ask aloud, "swingbird, what's going on?" The assistant would give me
the current status of each project, based on Buzz message history, compressed to the level of
granularity I prefer. I reply with instructions for some of the projects. The agent relays
commands to the Buzz coding agents, or helps me to refine the planned work first.

Designing this as a stand-alone agent, with its own code, own Buzz identity, and making its own
LLM calls, is necessary to enable the stand-alone voice assistant. As of now, you communicate to
the swingbird agent via Buzz DMs. This serves as an ideal prototype for the physical voice
assistant, as the interface will map almost directly, but it isn't necessarily ideal on its own.
If you're using the Buzz app and messaging via text, a stand-alone agent like this may be
overkill. A good system prompt with a regular Buzz coding agent might give you similar
functionality, without needing to write new code, deploy anything, or use API keys. But building
an independent custom agent gives you more flexibility, control, and power than a system prompt,
and it opens the door to running on a single-board computer (SBC) as a voice assistant, which is
the ultimate goal of this project.

### Ideas for future work

- To-do list management. Create and manage the project to-do list directly via swingbird.
- Integrate to-do list support with other sources, like Linear, Google Tasks, JIRA.
- Support other messaging/operating platforms, like Slack.
