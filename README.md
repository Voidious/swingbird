# swingbird

A stand-alone Buzz agent that helps keep your projects in full swing.

swingbird is a TPM-style Buzz agent that lives in your own workspace. DM it and it recaps your
project channels, relays instructions to your coding agents, and always confirms with you before
it writes anything on your behalf.

_The upbeat, unbothered swingbird roams the canopy of the Forest of Code, vibing with the Abstract
Syntax Trees as humans and agents work below. Sporting feathers of the dankest colors, headphones
that slap, and a small tablet, the swingbird is ready to spill the tea on project status, lock in
on potential solutions, or yeet commands to your coding agents to keep your projects in full
swing._

## Overview

Talk to swingbird in a private DM:

```
you:       swingbird, what's going on?
swingbird: [a short, prioritized recap of your configured project channels]

you:       tell backend to fix the login bug
swingbird: About to relay to backend (for Sonnet): 'fix the login bug'. Confirm to send, or cancel.

you:       confirm
swingbird: Confirmed and relayed (event ...).
           [a little later]
swingbird: Fixed the login bug -- turned out to be a stale session cookie.
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
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[owner]
pubkey = "<your Buzz pubkey>"
name = "<your Buzz display name>"

[identity]
name = "swingbird"

[dispatch]
reply_wait_seconds = 90

[[channels]]
id = "<Buzz channel UUID>"
name = "backend"
write = true
agents = ["Sonnet"]
```

- **`[llm]`** -- the OpenAI-compatible backend swingbird uses for intent classification, recaps,
  and reply summaries. `api_key_env` names the environment variable to read the key from, not the
  key itself.
- **`[relay]`** -- the Buzz relay to connect to, and the environment variable holding swingbird's
  own secret key (see [Buzz identity](#buzz-identity) above).
- **`[owner]`** -- the one identity swingbird will ever treat as a command. Every other message it
  sees, including a coding agent's own replies in a subscribed channel, is read-only context.
- **`[identity]`** -- swingbird's own Buzz display name. Synced on startup whenever the identity's
  current profile name doesn't match, so a fresh identity or a renamed deployment always shows up
  under a name you'll recognize.
- **`[dispatch]`** -- optional. `reply_wait_seconds` (default `90`) is how long swingbird waits
  for a reply to a relayed instruction before giving up on summarizing it back to you. Keep this
  short if you might ever run swingbird as a voice assistant -- a reply arriving minutes later,
  out of nowhere, would be a jarring surprise.
- **`[[channels]]`** -- one entry per project. `id` is the Buzz channel UUID; `name` is what
  you'll say in conversation ("recap backend", "tell backend to...") and is independent of the
  channel's own Buzz display name; `write` controls whether swingbird may dispatch instructions
  into it (a channel with `write = false` can still be recapped); `agents` lists the Buzz display
  names of the coding agents in that channel (see
  [Configuring project agents](#configuring-project-agents-to-listen-to-swingbird) below).

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
uv run python main.py --config swingbird.toml --audit-log audit.jsonl
```

`--config` and `--audit-log` both default to `swingbird.toml` and `audit.jsonl` in the current
directory. The audit log is a local, append-only JSON-lines record of every inbound message,
proposed dispatch, and confirm/cancel decision -- independent of Buzz's own event log, and kept so
you can see why swingbird proposed what it proposed, including proposals that got cancelled and
never became a real Buzz message.

## Usage

DM the swingbird identity directly -- that DM is its only interaction surface. @mentioning it in
a project channel, or anything anyone other than the configured owner says, is never treated as a
command.

### Supported instructions

| You say | swingbird does |
| --- | --- |
| "what's going on?" / "recap backend" | Summarizes recent activity across all configured channels, or just the one you named -- blockers and decisions first, then in-flight work, then what recently finished. |
| "tell backend to fix the login bug" / "ask frontend if the tests pass" | Proposes relaying that instruction (or question) to the named channel/agent. Nothing is sent until you confirm. |
| "confirm" / "do it" / "yes" | Sends the most recently proposed instruction. |
| "cancel" / "never mind" | Discards the pending proposal without sending anything. |
| anything else | swingbird says it's outside what it handles, and suggests asking for a recap or a dispatch instead. |

If swingbird can't tell which channel or agent a dispatch request means, it asks you to clarify
rather than guessing -- answering that follow-up resolves the original request. Once you confirm a
dispatch, swingbird waits (up to `[dispatch].reply_wait_seconds`) for the target agent's reply,
then summarizes it back into your DM.

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
