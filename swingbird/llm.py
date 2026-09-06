"""Thin OpenAI-compatible LLM client wrapper.

Config-driven per the design doc (§6/§7): base_url + api_key + model
come from `LLMConfig`, so swapping providers (or eventually a local
backend) is a config change, not a rewrite.
"""

from __future__ import annotations

import json
import os

import openai
from openai import OpenAI

from swingbird.config import LLMConfig


class LLMError(Exception):
    """Raised when the LLM backend fails or returns something unusable."""


class LLMClient:
    def __init__(self, config: LLMConfig, client: OpenAI | None = None) -> None:
        self._model = config.model
        if client is not None:
            self._client = client
            return
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise LLMError(f"environment variable {config.api_key_env} is not set")
        self._client = OpenAI(base_url=config.base_url, api_key=api_key)

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant's plain-text reply to `messages`."""
        response = self._chat(messages)
        return response.choices[0].message.content or ""

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        """Return the assistant's reply to `messages`, parsed as JSON.

        Forces JSON-mode output on the backend so callers (the intent
        router, dispatch-proposal builder, etc.) get structured data
        back instead of having to parse free text themselves. Some
        models (e.g. Moonshot Kimi) still wrap the JSON in a Markdown
        code fence despite JSON-mode being forced, so that's stripped
        before parsing rather than treated as invalid output.
        """
        response = self._chat(messages, response_format={"type": "json_object"})
        content = response.choices[0].message.content or ""
        try:
            return json.loads(_strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM did not return valid JSON: {content!r}") from exc

    def _chat(self, messages: list[dict[str, str]], **kwargs):
        try:
            return self._client.chat.completions.create(
                model=self._model, messages=messages, **kwargs
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc


def _strip_code_fence(text: str) -> str:
    """Strip a wrapping ```json ... ``` (or ``` ... ```) code fence, if present."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3]
        first_line, _, rest = text.partition("\n")
        if first_line.strip().isalpha():
            text = rest
    return text.strip()
