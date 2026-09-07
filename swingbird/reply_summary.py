"""Summarize a working agent's reply to a relayed dispatch (§4.2, step 4
of the follow-on task list).

A single LLM call, matching the style of `recap.py`'s naive summarizer:
correct and simple, no incremental state to maintain.
"""

from __future__ import annotations

from swingbird.llm import LLMClient

_SYSTEM_PROMPT = """You are a TPM agent summarizing a coding agent's reply \
to an instruction relayed on the user's behalf. Write a concise 1-3 \
sentence summary of what was done or found, and anything that needs the \
user's attention. Skip preamble and filler -- just the substance."""


def summarize_reply(llm: LLMClient, reply_text: str) -> str:
    """Return a short summary of `reply_text` for the requesting owner."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": reply_text},
    ]
    return llm.complete(messages)
