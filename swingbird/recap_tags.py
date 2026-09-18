from __future__ import annotations


def _resolve_tag(
    mapping: dict[str, dict[str, str]], channel: str, tag: object
) -> str | None:
    """Resolve an LLM-cited `tag` (e.g. "m3") against `mapping` for `channel`.

    Never trusts the LLM's tag blindly -- a missing tag, an empty string, an
    unknown channel, or a tag that doesn't match any message actually shown
    for that channel all resolve to `None` (a plain dict miss, in the latter
    three cases) rather than a guess. Shared by both `RecapItem.
    source_event_id` and `source_content`, resolved from the same tag against
    two parallel maps built in `_build_transcript`.

    `channel` is trusted exactly as the LLM wrote it, with no independent
    check that it's actually where `tag` came from -- what makes that safe
    is `_build_transcript` numbering every tag globally rather than
    restarting at "m1" per channel (see its own docstring): a tag exists in
    exactly one channel's sub-`mapping`, so a mislabeled `channel` finds no
    entry there and still falls through to `None`, rather than coincidentally
    matching a same-numbered tag that belongs to some other, unrelated
    channel."""
    if not isinstance(tag, str):
        return None
    return mapping.get(channel, {}).get(tag)
