"""Test doubles: a scripted chat model and a bag-of-words embedder.

The scripted model tells the Wisdom Core's prompts apart by comparing the
whole system prompt with the constants in wisdom_core, so a reworded prompt
can never be mistaken for another one.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import wisdom_core as wc

# Every database a test opens is kept here and closed when its test module finishes (each module's
# tearDownModule is close_stores), so Python 3.13 and later find no unclosed connection to warn about.
# Tests that close their own cores are unaffected: closing twice is harmless.
_OPEN_STORES: list[Any] = []
_store_init = wc._Store.__init__


def _tracked_store_init(self: Any, path: str) -> None:
    _store_init(self, path)
    _OPEN_STORES.append(self)


wc._Store.__init__ = _tracked_store_init  # type: ignore[method-assign]


def close_stores() -> None:
    """Close every database the tests have opened so far (a store is only kept once it opened)."""
    while _OPEN_STORES:
        _OPEN_STORES.pop().close()


STOP = {
    "the", "and", "for", "with", "that", "this", "you", "your", "are", "was", "were", "has", "have",
    "not", "but", "all", "any", "can", "its", "into", "from", "when", "what", "which", "who", "why",
    "how", "use", "used", "one", "each", "they", "them", "then", "than", "there", "their", "been",
}


def bow_embed(texts: list[str], dims: int = 1024) -> list[list[float]]:
    """Bag of words hashed into ``dims`` buckets: texts sharing words point the same way."""
    out = []
    for text in texts:
        vec = [0.0] * dims
        for word in re.findall(r"[a-z0-9]+", str(text).lower()):
            if len(word) < 3 or word in STOP:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % dims
            vec[bucket] += 1.0
        out.append(vec)
    return out


def combined(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The default answer to "combine these principles": the first of them, as one principle that keeps its
    fields (a test that cares queues its own reply)."""
    listing = str(messages[-1]["content"]).split("PRINCIPLES TO COMBINE INTO ONE\n", 1)[-1]
    first = listing.splitlines()[0].split(". ", 1)[-1]
    name = re.split(r"\. (?:Why|Scope|Use when):", first)[0].rstrip(".")
    return {"principle": {"name": name, "importance": "", "extent": "", "use_case": "", "related": []}}


class ScriptedModel:
    """Plays the model. Each kind of prompt has its own queue of replies.

    A reply may be a string, a dict/list (sent as JSON), an exception (raised),
    or a function of the messages.
    """

    KINDS = ("situation", "lesson", "reflect", "execute", "combine", "calibrate", "answer")

    def __init__(self, dims: int = 1024, embed_name: str = "bow") -> None:
        self.dims = dims
        self.name = "scripted"
        self.embed_name = embed_name
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        self.embed_calls: list[list[str]] = []
        self.replies: dict[str, list[Any]] = {k: [] for k in self.KINDS}
        self.defaults: dict[str, Any] = {
            "situation": "A general request.",
            "lesson": {"principle": "", "corrects": False},   # names no lesson: the whole synthesis runs
            "reflect": "LESSON: nothing new to learn from this task.",
            "execute": {"actions": []},
            "combine": combined,
            "calibrate": {"pairs": []},          # no pairs: a calibration on the agent's principles is skipped
            "answer": "OK.",
        }

    @staticmethod
    def kind_of(messages: list[dict[str, Any]]) -> str:
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        if system == wc.SITUATION_SYSTEM:
            return "situation"
        if system == getattr(wc, "LESSON_SYSTEM", None):
            return "lesson"
        if system == wc.REFLECT_SYSTEM:
            return "reflect"
        if system == wc.EXECUTE_SYSTEM:
            return "execute"
        if system == getattr(wc, "COMBINE_SYSTEM", None):
            return "combine"
        if system == getattr(wc, "DOMAIN_SYSTEM", None):
            return "calibrate"
        return "answer"

    def queue(self, kind: str, *replies: Any) -> ScriptedModel:
        assert kind in self.KINDS, kind
        self.replies[kind].extend(replies)
        return self

    def chat(self, messages, *, json_mode: bool = False, temperature: float | None = None) -> str:
        kind = self.kind_of(messages)
        self.calls.append((kind, [dict(m) for m in messages], {"json_mode": json_mode, "temperature": temperature}))
        queue = self.replies[kind]
        reply = queue.pop(0) if queue else self.defaults[kind]
        if callable(reply) and not isinstance(reply, type):
            reply = reply(messages)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, (dict, list)):
            reply = json.dumps(reply)
        return reply

    def embed(self, texts):
        texts = list(texts)
        self.embed_calls.append(texts)
        return bow_embed(texts, self.dims)

    # helpers for assertions
    def prompts(self, kind: str) -> list[list[dict[str, Any]]]:
        return [m for k, m, _ in self.calls if k == kind]

    def kinds(self) -> list[str]:
        return [k for k, _, _ in self.calls]

    def last_user(self, kind: str) -> str:
        return self.prompts(kind)[-1][-1]["content"]


def principle(name: str, importance: str = "", extent: str = "", use_case: str = "",
              related: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "importance": importance, "extent": extent, "use_case": use_case,
            "related": related or []}


def seed(core: wc.WisdomCore, *items: Any) -> list[wc.Principle]:
    """Put principles straight into a core, as if it had learned them."""
    added = []
    for item in items:
        fields = item if isinstance(item, dict) else principle(item)
        report = wc.SynthesisReport(task="seed")
        core._apply([{"type": "add", "principle": fields}], {}, report, "seed")
        assert report.added, f"seed was refused: {report.lines()}"
        added.extend(report.added)
    return added


def many(n: int, topic: str = "budget money rule") -> list[dict[str, Any]]:
    """``n`` principles that share ``topic`` but are otherwise clearly different."""
    return [principle(f"{topic} {w}{i}a {w}{i}b {w}{i}c") for i, w in
            zip(range(n), (f"word{j}" for j in range(n)))]
