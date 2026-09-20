"""Render a DM-formatted reply string as speech-safe text (Voice Mode design
doc §V.7).

Every reply string `Daemon._process` returns today was written assuming a DM
reader, not a listener -- `outbound.message_link` deep links, Python `!r`
repr quoting, and `@name` mention syntax are all fine on a screen and odd or
actively wrong read aloud ("at Sonnet", literal quote marks, escaped
characters). This is deliberately its own explicit rendering pass over the
already-built DM text, not a second "dual-purpose" return value threaded
through `_act` -- the DM text and the spoken text are genuinely different
artifacts with different constraints, even though they usually start from
the same content.

Scope is narrow and rule-based, matching each rule to a specific known
call site rather than guessing at general text-cleanup rules:

- `outbound.message_link` output, as produced by `Daemon._confirm`
  ("Confirmed and relayed: {link}") and `Daemon._await_and_summarize`
  ("{summary}\\n\\n{link}") -- stripped entirely, per §V.7. `recap.py`'s own
  link placement (`outbound.append_paragraph_link`, mid-paragraph) is
  deliberately not handled yet -- §V.7 asks to review real recap output
  before guessing at that shape, not before this module exists.
- `Daemon._dispatch_or_ask`'s `!r`-repr'd proposal instruction -- unquoted
  back to plain text via `ast.literal_eval` (the exact inverse of Python's
  own `repr()`, so it's correct for whichever quote style/escaping `repr()`
  picked, not just the common case).
- Buzz's `@name` mention syntax -- the `@` is dropped, leaving just the name.
"""

from __future__ import annotations

import ast
import re

_MESSAGE_LINK = r"buzz://message\?channel=\S+&id=\S+"
# "Confirmed and relayed: {link}" -> "Confirmed and relayed."
_LINK_AFTER_LABEL_RE = re.compile(r":\s*" + _MESSAGE_LINK)
# "{summary}\n\n{link}" -> "{summary}" (a link occupying its own trailing
# paragraph, as `_await_and_summarize` posts it).
_LINK_OWN_PARAGRAPH_RE = re.compile(r"\n{2,}" + _MESSAGE_LINK + r"\s*$")
# Fallback for any other placement -- just remove the link itself.
_MESSAGE_LINK_RE = re.compile(_MESSAGE_LINK)

# Matches `_dispatch_or_ask`'s fixed
# `f": {proposal.instruction!r}. Confirm to send, or cancel."` tail. The
# quoted group is whatever `repr()` produced -- either quote style, with
# `repr()`'s own escaping -- so it's parsed with `ast.literal_eval` (repr's
# exact inverse) rather than a hand-rolled unescape.
_PROPOSAL_INSTRUCTION_RE = re.compile(
    r": (['\"].*)\. Confirm to send, or cancel\.$", re.DOTALL
)

# Buzz's own `@name` mention syntax -- spoken as just the name.
_MENTION_RE = re.compile(r"@(\w[\w.-]*)")


def render_for_speech(text: str) -> str:
    """Return `text` (a DM-formatted reply) rendered for TTS."""
    text = _strip_message_links(text)
    text = _unquote_proposal_instruction(text)
    return _MENTION_RE.sub(r"\1", text)


def _strip_message_links(text: str) -> str:
    text = _LINK_AFTER_LABEL_RE.sub(".", text)
    text = _LINK_OWN_PARAGRAPH_RE.sub("", text)
    text = _MESSAGE_LINK_RE.sub("", text)
    return text.rstrip()


def _unquote_proposal_instruction(text: str) -> str:
    match = _PROPOSAL_INSTRUCTION_RE.search(text)
    if match is None:
        return text
    try:
        instruction = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        # Not actually a repr'd string (shouldn't happen against this
        # codebase's own output) -- leave the text as-is rather than risk
        # mangling something unexpected.
        return text
    return text[: match.start()] + f": {instruction}. Confirm to send, or cancel."
