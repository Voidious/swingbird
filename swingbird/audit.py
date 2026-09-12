"""Append-only audit log (JSON lines), per §4.2/§5.

Independent of Buzz's own signed event log -- which is the canonical record
of what was actually *sent* -- this local log exists so you can debug why
the agent proposed what it proposed, including proposals that got
cancelled and never became a Buzz event at all. Hooked in at the router
(every inbound message, step 5) and the pending-action store (proposals
and their confirm/cancel resolution, step 6).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """Appends one JSON record per line to `path`, creating it if needed."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_transcript_in(self, thread_id: str | None, text: str) -> None:
        self._write("transcript_in", thread_id, text=text)

    def log_proposed_action(self, thread_id: str, proposal: Any) -> None:
        self._write(
            "proposed_action",
            thread_id,
            channel_id=proposal.channel_id,
            instruction=proposal.instruction,
            target_agent=proposal.target_agent,
            reply_to=proposal.reply_to,
        )

    def log_recap_reference(
        self, thread_id: str, recap_kind: str, reference: str | None, item: Any
    ) -> None:
        self._write(
            "recap_reference",
            thread_id,
            recap_kind=recap_kind,
            reference=reference,
            channel=item.channel,
            label=item.label,
            source_event_id=item.source_event_id,
        )

    def log_recap_built(self, thread_id: str, items: tuple[Any, ...]) -> None:
        """Records every `RecapItem` a recap extracted, not just the ones a
        later reference happens to resolve to -- `RecapActionStore` only
        keeps them in memory, gone once the next recap replaces them, so
        this is the only record of what a given `resolve_reference` call
        (see `log_recap_reference`) was actually matching against."""
        self._write(
            "recap_built",
            thread_id,
            items=[
                {
                    "channel": item.channel,
                    "label": item.label,
                    "keywords": list(item.keywords),
                    "is_primary": item.is_primary,
                    "source_event_id": item.source_event_id,
                }
                for item in items
            ],
        )

    def log_decision(
        self,
        thread_id: str | None,
        decision: str,
        proposal: Any,
        event_id: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "channel_id": proposal.channel_id,
            "instruction": proposal.instruction,
            "target_agent": proposal.target_agent,
        }
        if event_id is not None:
            fields["event_id"] = event_id
        self._write("decision", thread_id, decision=decision, **fields)

    def _write(self, kind: str, thread_id: str | None, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "kind": kind,
            "thread_id": thread_id,
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
