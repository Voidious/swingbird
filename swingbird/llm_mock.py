"""Record-and-replay LLM mock for testing (`[debug].mock_llm`, see config.py).

Every `LLMClient` call site (router.py's intent classification, recap.py's
structured items, reply_summary.py, dispatch_phrasing.py, ...) expects its own
response shape, so a single canned reply -- or a fixed pool of them -- would
break `complete_json` callers that parse a specific JSON structure back out.
Keying the cache on the exact prompt instead sidesteps that: the first call
for a given prompt always goes to the real backend and gets a real,
correctly-shaped response, which is then replayed verbatim for every later
call with that same exact prompt.

This is built for voice-pipeline tuning (mic gain, echo cancellation,
barge-in), where the same test phrase gets spoken into the mic over and over
while a physical setting is adjusted -- identical STT transcripts produce
identical prompts, so only the first repeat of each phrase costs a real call.
A genuinely new phrase always costs one real call; nothing here fakes intent
classification or recap content, it just stops paying for exact repeats.

The cache key normalizes "user"-role message content (lowercased, non-
alphanumeric characters stripped) before hashing, so STT noise the mic-tuning
loop doesn't care about -- "Recap." vs "recap" vs "recap!" -- still hits the
same entry; the real backend still gets the untouched text on a cache miss,
so this only affects what counts as a repeat, never what's actually sent.
"system"-role content (static prompt prose, not user speech) is hashed as-is.
Deliberately *not* a second cache layer keyed on the classified intent
instead of the exact prompt: router.py's own call is the only one that
embeds the raw phrase, so once it's normalized, a downstream call like
recap.py's `build_recap` -- whose prompt is built from channel history, not
the phrase -- already replays across "recap" vs "give me a recap" for free,
with no extra layer. Calls that instead relay or rewrite the user's own
wording (recap_relay.py, dispatch_phrasing.py, recap_action's reference
resolution) genuinely depend on that wording per §4.2/§5 -- keying those on
intent alone would let two different somethings collapse onto one cached
reply, which is the wrong failure mode for a dispatch-adjacent path even in
a test-only mock.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from swingbird.llm import LLMClient

_NON_ALNUM = re.compile(r"[^a-z0-9]")


class MockLLMClient:
    def __init__(self, real: LLMClient, cache_path: str | Path) -> None:
        self._real = real
        self._cache_path = Path(cache_path)
        self._cache = _load_cache(self._cache_path)

    def complete(self, messages: list[dict[str, str]]) -> str:
        key = _cache_key("complete", messages)
        if key in self._cache:
            return self._cache[key]
        response = self._real.complete(messages)
        self._store(key, response)
        return response

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        key = _cache_key("complete_json", messages)
        if key in self._cache:
            return json.loads(self._cache[key])
        response = self._real.complete_json(messages)
        self._store(key, json.dumps(response))
        return response

    def _store(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")


def _cache_key(method: str, messages: list[dict[str, str]]) -> str:
    normalized = [_normalize_message(message) for message in messages]
    payload = json.dumps({"method": method, "messages": normalized}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_message(message: dict[str, str]) -> dict[str, str]:
    if message.get("role") != "user":
        return message
    return {**message, "content": _NON_ALNUM.sub("", message["content"].lower())}


def _load_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
