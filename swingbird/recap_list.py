"""Enumerate a recap's additional/open items without elaborating on them
(see `daemon._recap_list`).

`recap_detail.elaborate` exists to answer "tell me more about X" -- a fresh
LLM call that draws on an item's full source thread to say something
genuinely beyond its own stored summary/instruction. A bare "what are the
other items" isn't asking for that: it's asking to see items the recap
itself never rendered, at the *same* concise/detailed level the recap gave
its primary item(s) -- not a deeper one. Routing both through `elaborate`
made "the other items" always come back more detailed than the primary
item(s) the recap actually narrated, even for a concise recap (observed
live -- see the recap follow-up plan). `render_items` never calls the LLM
at all: it formats each item's own already-grounded `summary`/`instruction`
directly, so an enumeration can never read as more detailed than the recap
that surfaced these items in the first place.
"""

from __future__ import annotations

from swingbird.config import Config
from swingbird.recap import RecapItem
from swingbird.recap_detail import append_source_links

_NO_OTHER_ITEMS_NOTE = (
    "There are no other open items for that channel -- here's the one "
    "already in the recap:"
)


def render_items(items: list[RecapItem], no_other_items: bool, config: Config) -> str:
    """Return a plain, recap-level listing of `items` -- one paragraph per
    item, each starting with its channel and label in bold Markdown, the
    same "**channel -- label:**" shape `recap_detail.elaborate` uses (see
    `append_source_links`, shared with that module so both attach links the
    same way) -- but built directly from each item's stored `summary`/
    `instruction` rather than a fresh LLM elaboration.

    `no_other_items` is `ResolvedReference.degraded` from `recap_actions.
    resolve_reference` -- True when the user asked for "the other/
    additional items" but the channel only ever had the one item already in
    `items` (see `ResolvedReference`'s docstring). Without this, the
    listing would silently show that one item as if it were a genuine
    additional one; with it, a plain note is prepended instead, mirroring
    `elaborate`'s own handling of the same case.
    """
    parts = [_NO_OTHER_ITEMS_NOTE] if no_other_items else []
    for item in items:
        body = f"{item.summary} {item.instruction}".strip()
        parts.append(f"**{item.channel} -- {item.label}:** {body}")
    text = "\n\n".join(parts)
    return append_source_links(text, items, config)
