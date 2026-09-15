"""LLM-backed selection of which recap items a "close ..." request refers to
(swingbird-dev, 2026-09-15).

`recap_close`'s original resolution (see `recap_close.py`'s module
docstring/history) reused `recap_actions.resolve_reference` -- built for
`recap_action`/`recap_relay`, which can only ever act on exactly one item --
so a close request that matched more than one item raised
`AmbiguousRecapReference` and asked "which did you mean" instead of just
closing all of them. That's the wrong default for closing specifically:
Voidious wants "close all swingbird items", "close the additional items for
dripbird", "close everything", an explicit list of items (one project or
several), and combinations of these in one request ("all swingbird items
including additional", "swingbird items and dripbird items") to all just
work, since closing already confirms the exact batch before persisting
anything.

Encoding that scoping grammar deterministically (extending `resolve_
reference`'s bidirectional/word-boundary matching with "primary vs
additional" and "per-project vs cross-project" logic on top) would mean
re-deriving, in Python, exactly the kind of compositional free-text parsing
an LLM already does well -- so this instead hands the *whole* recap store's
items for the thread (every channel, not narrowed to one) and the user's
raw request to one JSON-forced LLM call, the same "trust the model, verify
downstream" pattern `router.py`/`recap.py` already use elsewhere in this
codebase.

This is safe to do here specifically because closing has its own
confirm-before-persist step (`recap_close.py`'s `PendingCloseStore`) that
lists every selected item by name before anything is written -- a
misjudged selection is caught and cancelled by the user, not silently
persisted. That's exactly the property that lets AGENTS.md's "no guessing
on ambiguity" invariant (written for dispatch routing, where a wrong guess
relays to the wrong channel) not apply the same way to item selection here.
"""

from __future__ import annotations

from swingbird.llm import LLMClient
from swingbird.recap import RecapItem
from swingbird.recap_actions import RecapActionError

_SYSTEM_PROMPT = """You are matching a user's "close ..." request against a \
numbered list of open work items from a status recap, to decide which ones \
they want marked closed/done.

Each item is tagged PRIMARY (the recap's own lead item for that project) or \
ADDITIONAL (an open item the recap only counted, not detailed) and belongs \
to one named project. Interpret the request's scope the way its own words \
describe it:

- "all <project> items" / "every item for <project>" / "close out \
<project>", with no mention of additional/other/open items, means every \
PRIMARY item for that project only -- not its ADDITIONAL items.
- "the additional/other/open items for <project>" means every ADDITIONAL \
item for that project only, not its PRIMARY item(s).
- "<project> items including additional" (or similar wording that \
explicitly asks for both) means every item -- PRIMARY and ADDITIONAL -- \
for that project.
- "all items" / "everything", naming no project, means every PRIMARY item \
across every project -- never pull in an ADDITIONAL item this way unless \
the request also asked for additional items.
- A specific label, short code (e.g. "F4"), or description matches exactly \
the item(s) it identifies, regardless of project or PRIMARY/ADDITIONAL -- \
the user may name several, for one project or across several (e.g. "F4 \
and F7", "F4 for swingbird and F2 for dripbird").
- Several of the above can combine in one request (e.g. "all swingbird \
items including additional, and the dripbird Kimi item") -- select the \
union of everything asked for.

Only select an item you're confident the request actually refers to -- \
never invent a match for a label/code/description that isn't in the list, \
and never pull in a project's items unless the request names or clearly \
describes that project. If nothing in the list matches, return an empty \
list rather than guessing.

Respond with JSON only, matching this shape:
{"indices": [<1-based item numbers to close, possibly empty>]}"""


def select_items_to_close(
    llm: LLMClient, items: tuple[RecapItem, ...], request: str | None
) -> tuple[RecapItem, ...]:
    """Return the subset of `items` that `request` refers to.

    Raises `RecapActionError` (same error type and message shape
    `resolve_reference` raises for a plain no-match) when the request
    doesn't select anything -- `daemon.py`'s `_ACTIONABLE_ERRORS` already
    catches this and replies "Couldn't do that: ...", so no separate
    handling is needed at the call site.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Open items:\n{_format_items(items)}\n\n"
                f"Request: {request or '(unspecified)'}"
            ),
        },
    ]
    response = llm.complete_json(messages)
    selected = _resolve_indices(items, response.get("indices"))
    if not selected:
        raise RecapActionError(f"no recap item matches {request!r}")
    return selected


def _format_items(items: tuple[RecapItem, ...]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        tag = "PRIMARY" if item.is_primary else "ADDITIONAL"
        keywords = f" (keywords: {', '.join(item.keywords)})" if item.keywords else ""
        lines.append(f"{i}. [{item.channel}] {tag} {item.label}{keywords}")
    return "\n".join(lines)


def _resolve_indices(
    items: tuple[RecapItem, ...], indices: object
) -> tuple[RecapItem, ...]:
    if not isinstance(indices, list):
        return ()
    selected = []
    seen = set()
    for index in indices:
        in_range = isinstance(index, int) and 1 <= index <= len(items)
        if not in_range or index in seen:
            continue
        seen.add(index)
        selected.append(items[index - 1])
    return tuple(selected)
