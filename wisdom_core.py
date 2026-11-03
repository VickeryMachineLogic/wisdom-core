# Wisdom Core ------------------------------------------------------------
# MIT License
#
# Copyright (c) 2026 Vickery Machine Logic, LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -------------------------------------------------------------------------
"""Wisdom Core: an AI agent framework driven by a Core Axiom.

You write one sentence, the Core Axiom. The agent works out the rest.

* **Principles** are lessons the agent writes for itself. They live in a small
  database that is searched by meaning. Each is embedded twice: what it is
  about (to be recalled when its situation comes up) and what it says (to
  tell whether two principles teach the same thing), each written the way the
  embedding model expects (see ``EMBED_PROMPTS``).
* **Injection**: every time the agent has to decide something, the Core Axiom
  and the principles most relevant to that decision (12 by default) go into
  its prompt exactly as written.
* **Wisdom Synthesis**: after every task the agent reads what the user
  said (how they reacted to its previous answer, what they prefer, what they
  said about themselves), its own reasoning and its answer, and writes down
  the piece of wisdom the exchange taught - what the thinking established
  about the subject, or what the user showed - as one principle in service
  of the Core Axiom. It takes a quick first look before anything else: if
  that lesson measures as one of its principles already, nothing more runs.
  If not, the exchange is new ground and it must write a principle.
* **Wisdom Distillation**: during synthesis it merges principles that say
  nearly the same thing into one that fits the Core Axiom, and removes any
  principle that conflicts with the axiom or turned out to be wrong.
* **Principles earn their place**: the agent remembers which principles
  shaped each answer and how the user received it. A principle behind
  answers the user keeps pushing back on comes into doubt and is reviewed
  (removed, revised, or found to stand); one that has served well many
  times reads as proven. Nothing is removed on a count alone.

Quick start, with Ollama running locally::

    from wisdom_core import Agent

    agent = Agent(core_axiom="Give accurate answers and say plainly when you're unsure.")
    print(agent.ask("What should I check before signing a lease?"))
    print(agent.last_report.summary())      # what the Wisdom Core just did

From a terminal::

    python -m wisdom_core --axiom "Give accurate answers and say plainly when you're unsure."

Inside an agent you already have, use the Wisdom Core on its own::

    from wisdom_core import WisdomCore

    core = WisdomCore(core_axiom="...", model=my_model)
    system_prompt = core.inject(task)           # Core Axiom + relevant principles
    result = run_my_agent(system_prompt, task)
    core.synthesize(task, result)               # Wisdom Synthesis + Distillation

The module needs nothing beyond the Python standard library. If numpy is
installed, searching large principle sets is faster.
"""

from __future__ import annotations

import argparse
import ast
import http.client
import inspect
import json
import logging
import math
import operator
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from array import array
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:  # optional speed-up for large principle sets; everything works without it
    import numpy as _np
except Exception:  # pragma: no cover - depends on the environment
    _np = None

__version__ = "0.8.1"

__all__ = [
    "Agent",
    "WisdomCore",
    "Levels",
    "Principle",
    "SynthesisReport",
    "Event",
    "Ollama",
    "OpenAICompatible",
    "Anthropic",
    "Combined",
    "FunctionModel",
    "WisdomCoreError",
    "ModelError",
    "resolve_model",
    "fit_history",
    "context_window_of",
    "check_embeddings",
    "embed_prompts_for",
    "EMBED_PROMPTS",
    "main",
    "__version__",
]

log = logging.getLogger("wisdom_core")
log.addHandler(logging.NullHandler())   # silent unless the application configures logging

# ---------------------------------------------------------------------------
# The semantic levels: the defaults of ``Levels``, which decides how close
# counts as close and how much gets written. Everything else below is fixed.
# ---------------------------------------------------------------------------
FORCE_PRINCIPLE = True        # every task drafts its lesson as a principle; measurement decides the rest
DUPLICATE_SIMILARITY = 0.95   # a drafted principle this close to a held one counts as covered, not added
COMBINE_SIMILARITY = 0.85     # principles this alike are combined into one by the Wisdom Core itself (0: off)
AUTO_LEVELS = ("new_ground", "covered", "combine")   # depend on the embedding model's scale: calibrated unless set
MERGE_SIMILARITY = 0.0        # how alike principles must be for a merge to go through (0: the model decides)
RECALL_FLOOR = 0.0            # a principle must be at least this relevant to be injected (0: no floor)
DEFAULT_CONTEXT = 12          # principles injected at each decision
MAX_ADDS_PER_TASK = 1         # new principles one task may add: its single most important lesson
NEW_GROUND_SIMILARITY = 0.85  # a first-look lesson below this against every held principle is new ground (0: always)
# What an automatic level falls back to while it is not measured, or when the model cannot be calibrated.
FALLBACK_LEVELS = {"new_ground": NEW_GROUND_SIMILARITY, "covered": DUPLICATE_SIMILARITY, "combine": COMBINE_SIMILARITY}

# ---------------------------------------------------------------------------
# Fixed numbers. None of these need tuning for normal use.
# ---------------------------------------------------------------------------
CANDIDATES_PER_QUERY = 8      # principles pulled per search when synthesizing
MAX_CANDIDATES = 20           # most principles the Wisdom Core looks at in one pass
MAX_COMBINES_PER_TASK = 2     # near-duplicates combined automatically after one task (each costs a model call)
STATE_VERSION = 1             # the shape of export_state() / import_state()
ANTHROPIC_VERSION = "2023-06-01"   # the Messages API version the Anthropic backend speaks
ANTHROPIC_MAX_OUTPUT = 8192   # longest reply asked of a Claude model whose own limit is not known
TASK_CHARS = 4000             # how much of a task the Wisdom Core reads
RESULT_CHARS = 6000           # how much of a result the Wisdom Core reads
HISTORY_CHARS = 4000          # longest single message kept in the agent's prompt
EARLIER_CHARS = 800           # longest earlier message shown to the Wisdom Core
PREVIOUS_CHARS = 1500         # how much of the answer the user is responding to the Wisdom Core reads
ANSWER_CHARS = 3000           # how much of the agent's own answer the Wisdom Core reads when it learns
REASONING_CHARS = 3000        # how much of the agent's reasoning before it answered the Wisdom Core reads
NAME_CHARS = 240              # longest principle statement kept
FIELD_CHARS = 400             # longest other principle field kept
RELATED_MAX = 8               # most related concepts kept on a principle
EMBED_BATCH = 32              # texts per embedding request when re-embedding
DEFAULT_NUM_CTX = 16384       # Ollama context window, when the model holds that much
FALLBACK_NUM_CTX = 8192       # Ollama context window when the model's own limit cannot be read
REPLY_ROOM = 2048             # tokens kept free for the reply when the conversation is fitted
CHARS_PER_TOKEN = 3.5         # cautious estimate used to fit text into a context window
SCHEMA_VERSION = 1            # the identity column added in 0.5 is read and written by older versions unharmed
EMBED_LAYOUT = "2"            # what is embedded for a principle and how; a change re-embeds every principle

# How an embedding model wants a search, a stored text and a plain comparison written. Many retrieval models
# are trained with these markers and score far lower without them. The family is looked for anywhere in the
# embedding model's name (so "hf.co/.../embeddinggemma-300m-GGUF" matches); the first match wins. "{}" is the
# text. A backend can set its own with an ``embed_prompts`` attribute: {"query": "...{}...", "document": "...{}...", "similarity": "...{}..."}.
# Checked against each model's own documentation, October 2026.
EMBED_PROMPTS: tuple[tuple[str, dict[str, str]], ...] = (
    ("embeddinggemma", {"query": "task: search result | query: {}", "document": "title: none | text: {}",
                        "similarity": "task: sentence similarity | query: {}"}),
    ("nomic-embed", {"query": "search_query: {}", "document": "search_document: {}", "similarity": "clustering: {}"}),
    ("mxbai-embed", {"query": "Represent this sentence for searching relevant passages: {}"}),
    ("snowflake-arctic-embed", {"query": "Represent this sentence for searching relevant passages: {}"}),
    ("bge-m3", {}),
    ("bge-", {"query": "Represent this sentence for searching relevant passages: {}"}),
    ("multilingual-e5", {"query": "query: {}", "document": "passage: {}", "similarity": "query: {}"}),
    ("e5-", {"query": "query: {}", "document": "passage: {}", "similarity": "query: {}"}),
    ("qwen3-embedding", {"query": "Instruct: Given a question, retrieve passages that answer it\nQuery: {}",
                         "similarity": "Instruct: Retrieve text that says the same thing\nQuery: {}"}),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class WisdomCoreError(Exception):
    """Base class for errors raised by wisdom_core."""


class ModelError(WisdomCoreError):
    """A model backend failed: it could not be reached, refused, or replied oddly."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class _ParseError(WisdomCoreError):
    """The Wisdom Core could not read the model's list of changes."""


class _AlreadyHeld(Exception):
    """The first look found the lesson already held: Wisdom Synthesis stops there (never raised to callers)."""


# ---------------------------------------------------------------------------
# Prompts. The first line of each system prompt is distinct, so a scripted
# model in tests can tell them apart by comparing the whole constant.
# ---------------------------------------------------------------------------
AGENT_PREAMBLE = (
    "You are an AI agent. Everything you do serves your Core Axiom, and you are guided by "
    "principles you have learned from your own past work."
)

PRINCIPLES_PREAMBLE = (
    "These are lessons you learned from your own past work, chosen because they are the most "
    "relevant to this task (most relevant first). Apply the ones that fit. If a principle would "
    "work against the Core Axiom, the axiom wins."
)

NO_PRINCIPLES = "You have not learned any principles yet. Act directly from the Core Axiom."

SITUATION_SYSTEM = """You help an AI agent recall its own principles before it acts.
Describe the situation below in one sentence, the way a principle's "use when" line describes the situation that calls for it: what the person is dealing with and what they need, naming the subject in plain words (for example: "someone asks whether to stay loyal to a friend who lied to them").
Do not answer or solve the task. Write only the description."""

SITUATION_USER = """CORE AXIOM
{axiom}

{earlier}TASK
{task}"""

REFLECT_SYSTEM = """You are the Wisdom Core of an AI agent.
You keep the agent's principles: short, reusable pieces of wisdom - truths about the world, about people and about how to act well - each one in service of the agent's Core Axiom. The agent reasons from them the next time a situation calls for them. The Core Axiom is the deciding factor in everything you do here.

The agent has just answered the user. Learn from the user and from the substance of the thinking, never from the form of the answer. Work through these steps:
1. WHAT THE USER SHOWED: Read what the user said. If it responds to the agent's previous answer, say how: a correction, a complaint, approval, a follow-up that shows what was missing, or a change of direction. Note any preference or instruction about how they want to be helped, and any fact, value or goal they stated about themselves or their situation. Quote the user's own words where you can. If the message is only a question, say what it shows about what the user is after.
2. WHAT THE THINKING ESTABLISHED: Read the agent's reasoning and its answer for what they worked out about the subject itself - the key ideas, distinctions, causes, trade-offs and conclusions - not for how the answer was organised. Which of them would still be true and useful in a different situation?
3. LESSON: Name the one piece of wisdom this exchange taught, taken from steps 1 and 2: what the user showed, or the most valuable insight the thinking reached about the subject. Work it out from this exchange before you compare it with the listed principles. A lesson about how answers should be written (their structure, frameworks, length, layout or tone) is not wisdom: it counts only if the user asked for it or reacted to it; never write one just because the answer did it. If your lesson only restates a listed principle, it is not what this exchange added: look again for what is new. It must be reusable, not a fact about this particular conversation.
4. PRINCIPLE: Draft that lesson as one principle that fits the Core Axiom, and say which of the listed principles come closest to it. HOW MUCH TO WRITE, at the end, says when to draft one and when a listed principle is enough.
5. DUPLICATES: If two or more listed principles say nearly the same thing, draft one principle that combines them and fits the Core Axiom.
6. CONFLICTS: If a listed principle conflicts with the Core Axiom, contradicts another principle, or turned out to be wrong - above all if the user's own words or this exchange's reasoning show it was wrong - it must be removed. Say which one and why. If two principles conflict and each holds part of the truth, combine them into one consistent principle instead.
7. REVIEW: A listed principle marked IN DOUBT shaped answers the user pushed back on far more often than this agent's answers in general. Judge it by this exchange, its record and the Core Axiom: if it is wrong, remove it; if it is half right, revise it; if it is sound and the pushbacks were about something else, say that it stands. A principle marked PROVEN has served well many times: do not remove it over one disagreement.

Write a short memo that works through the steps in order. Refer to listed principles by their labels (P1, P2, ...) and only to labels that are listed. Do not write JSON."""

# How much to write: the memo's policy, set by Levels.force.
FORCE_MEMO = """Every exchange teaches something, however ordinary it was, so draft the principle for this exchange's lesson every time - even when one of the listed principles looks close. Take it from what the user showed or from what the thinking established about the subject (steps 1 and 2), never from how the answer was laid out. When the user only asked a question, the lesson is the most valuable insight the thinking reached in answering it. Whether the principle is new or already held is measured afterwards by the Wisdom Core, not decided by you. Write it as wisdom the agent can reason from next time, not as a description of this exchange."""

CAREFUL_MEMO = """Be conservative. Change only what this task actually shows. If one of the listed principles already covers the lesson closely enough, say which one and draft nothing. Most tasks add at most one principle, and many add none."""

FORCE_ACTIONS = """The memo drafted a principle for this task. Include it as an "add" action, written out in full. Do not use "covered": the Wisdom Core measures that itself."""

FORCE_RETRY = """A principle is required from every task. Reply again with the same JSON object, this time including an "add" action holding the memo's lesson written as a principle."""

REFLECT_USER = """CORE AXIOM
{axiom}

{earlier}{previous}WHAT THE USER SAID
{task}

{reasoning}THE AGENT'S ANSWER
{result}

{outcome}PRINCIPLES THAT MAY RELATE
{principles}

HOW MUCH TO WRITE
{policy}"""

EXECUTE_SYSTEM = """You turn the Wisdom Core's memo into changes to the agent's principles.
Reply with one JSON object and nothing else, in this form:
{"actions": [
  {"type": "covered", "id": "P2"},
  {"type": "add", "principle": {"name": "...", "importance": "...", "extent": "...", "use_case": "...", "related": ["...", "..."]}},
  {"type": "merge", "ids": ["P1", "P3"], "principle": {"name": "...", "importance": "...", "extent": "...", "use_case": "...", "related": ["..."]}},
  {"type": "remove", "id": "P4", "reason": "..."},
  {"type": "stands", "id": "P5", "reason": "..."}
]}

Action types:
- covered: the lesson is already held by this listed principle. Nothing changes.
- add: a new principle.
- merge: two or more listed principles are replaced by one principle that combines them. One listed principle with a new principle revises it: the new wording replaces the old.
- remove: a listed principle is deleted because it conflicts with the Core Axiom, contradicts another principle, or proved wrong. Always give the reason.
- stands: a listed principle marked "in doubt" was reviewed in the memo and found sound. Give the reason.

Use only the labels listed below. Include only the actions the memo decided on. If nothing changes, reply {"actions": []}.

Principle fields. The agent finds its principles by meaning, by comparing them with what a person says, so write them in plain words that name the subject:
- name: the principle itself, as one short sentence stating the truth or the way of acting it teaches, naming its subject plainly (at most 25 words).
- importance: why it matters for the Core Axiom, in one sentence.
- extent: where it applies and where it stops, in one sentence.
- use_case: the situation that calls for it, described the way a person would bring it up and naming the subject, in one sentence (for example: "someone asks whether to stay loyal to a friend who lied to them").
- related: three to six words or short phrases a person might use when this comes up, the subject itself first."""

EXECUTE_USER = """CORE AXIOM
{axiom}

LISTED PRINCIPLES
{labels}

MEMO
{memo}{policy}"""

ADD_LIMIT = """Add at most {count} new principle{s} from this task. If the memo found more lessons than that, keep the one that matters most for the Core Axiom."""

COMBINE_SYSTEM = """You are the Wisdom Core of an AI agent. The principles below say nearly the same thing, so the agent should hold them as one.
Write the single principle that replaces them: keep everything any of them teaches, drop the repetition, and make it fit the Core Axiom.
Reply with one JSON object and nothing else, in this form:
{"principle": {"name": "...", "importance": "...", "extent": "...", "use_case": "...", "related": ["...", "..."]}}

Principle fields. The agent finds its principles by meaning, by comparing them with what a person says, so write them in plain words that name the subject:
- name: the principle itself, as one short sentence stating the truth or the way of acting it teaches, naming its subject plainly (at most 25 words).
- importance: why it matters for the Core Axiom, in one sentence.
- extent: where it applies and where it stops, in one sentence.
- use_case: the situation that calls for it, described the way a person would bring it up and naming the subject, in one sentence (for example: "someone asks whether to stay loyal to a friend who lied to them").
- related: three to six words or short phrases a person might use when this comes up, the subject itself first."""

COMBINE_USER = """CORE AXIOM
{axiom}

PRINCIPLES TO COMBINE INTO ONE
{principles}"""

# The first look: one quick call before Wisdom Synthesis. It does not see the principles, so it cannot lean on
# them; the Wisdom Core measures its lesson against them to decide whether the exchange is new ground.
LESSON_SYSTEM = """You are the Wisdom Core of an AI agent, taking a first look at an exchange that just finished.
Name the one piece of wisdom this exchange taught, in service of the agent's Core Axiom: what the user showed (a correction, a complaint, approval, a preference, or a fact, value or goal they stated) or the most valuable insight the agent's thinking reached about the subject itself. Never take it from how the answer was written (its structure, frameworks, length, layout or tone) unless the user asked for that or reacted to it. It must be reusable wisdom, not a fact about this conversation.
Also say whether the user's message corrects, disagrees with or complains about the agent's previous answer.
Reply with one JSON object and nothing else, in this form:
{"principle": {"name": "...", "importance": "...", "extent": "...", "use_case": "...", "related": ["...", "..."]}, "corrects": false}

Principle fields. The agent finds its principles by meaning, by comparing them with what a person says, so write them in plain words that name the subject:
- name: the principle itself, as one short sentence stating the truth or the way of acting it teaches, naming its subject plainly (at most 25 words).
- importance: why it matters for the Core Axiom, in one sentence.
- extent: where it applies and where it stops, in one sentence.
- use_case: the situation that calls for it, described the way a person would bring it up and naming the subject, in one sentence (for example: "someone asks whether to stay loyal to a friend who lied to them").
- related: three to six words or short phrases a person might use when this comes up, the subject itself first.
corrects: true only if the user's message pushes back on the agent's previous answer."""

LESSON_USER = """CORE AXIOM
{axiom}

{earlier}{previous}WHAT THE USER SAID
{task}

{reasoning}THE AGENT'S ANSWER
{result}"""

DOMAIN_SYSTEM = """You help calibrate how an AI agent compares its principles by meaning.
For each principle below, write two sentences:
- reworded: the same lesson in different words. Keep exactly its meaning; change the wording and the sentence structure.
- related: a different lesson on the same subject, one a person could hold alongside it. It must not say the same thing, and it must not contradict it.
Reply with one JSON object and nothing else, in this form:
{"pairs": [{"id": "P1", "reworded": "...", "related": "..."}]}"""

DOMAIN_USER = """PRINCIPLES
{principles}"""

JSON_RETRY = "That was not valid JSON. Reply again with only the JSON object, starting with { and ending with }."


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    """UTC timestamp with microseconds, so two events in one second still sort."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _ms(started: float) -> int:
    """Milliseconds since ``started`` (a time.monotonic() reading)."""
    return int((time.monotonic() - started) * 1000)


def _clip(text: Any, limit: int) -> str:
    """Trim text to about ``limit`` characters, keeping its start and its end."""
    text = "" if text is None else str(text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = max(limit - head, 0)
    cut = len(text) - head - tail
    end = text[-tail:].lstrip() if tail else ""
    return f"{text[:head].rstrip()}\n[... {cut} characters cut ...]\n{end}".rstrip()


def _strip_thinking(text: Any) -> str:
    """Drop a reasoning model's <think>...</think> block and keep the answer."""
    text = "" if text is None else str(text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif text.lstrip().lower().startswith("<think>"):
        text = ""  # the reply was cut off while still thinking
    return text.strip()


def _thinking_part(text: Any) -> str:
    """The reasoning a model wrote inline as <think>...</think> before its answer (what _strip_thinking drops)."""
    text = "" if text is None else str(text)
    if "</think>" in text:
        head = text.rsplit("</think>", 1)[0]
    elif text.lstrip().lower().startswith("<think>"):
        head = text
    else:
        return ""
    head = head.strip()
    return (head[7:] if head.lower().startswith("<think>") else head).strip()


def _sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _unit(vec: Iterable[Any]) -> array:
    """float32 copy of ``vec`` scaled to length 1, so cosine similarity is a dot product."""
    out = array("f")
    for x in vec:
        out.append(float(x))
    norm = math.sqrt(math.fsum(x * x for x in out))
    if norm > 0 and math.isfinite(norm):
        out = array("f", (x / norm for x in out))
    return out


def _dot(a: array, b: array) -> float:
    return math.fsum(map(operator.mul, a, b))


def _binomial_tail(k: int, n: int, p: float) -> float:
    """The chance of ``k`` or more successes in ``n`` tries at probability ``p``."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    total = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return min(max(total, 0.0), 1.0)


def _level(value: Any, name: str) -> float:
    """A similarity level: a number from 0 to 1."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number from 0 to 1") from None
    if not (0.0 <= out <= 1.0) or not math.isfinite(out):
        raise ValueError(f"{name} must be a number from 0 to 1")
    return out


def _whole(value: Any, name: str, low: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be at least {low}") from None
    if out < low:
        raise ValueError(f"{name} must be at least {low}")
    return out


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class Levels:
    """The semantic levels: how close counts as close, and how much gets written.

    ``new_ground`` decides whether Wisdom Synthesis runs at all. A quick first
    look writes the exchange's lesson without seeing the principles, and the
    Wisdom Core measures it against every principle held: at or above
    ``new_ground`` the agent already holds that lesson and nothing more runs;
    below it the exchange is new ground and the full synthesis runs. The first
    look is skipped (the synthesis always runs) while no principles are held,
    when an outcome is given, when the user pushed back on the previous answer,
    and when ``new_ground`` is 0.

    On new ground with ``force`` on (the default), the task must draft its
    lesson as a principle, and ``covered`` decides by measurement whether the
    agent already holds it: a draft that similar to a held principle is counted as covered,
    anything less alike is added. With ``force`` off, the model decides for
    itself whether a task taught anything, and ``covered`` only stops a draft
    that is nearly identical to one already held.

    Every level that compares principles (``new_ground``, ``covered``,
    ``combine``, ``merge``) measures what they say: the statement alone,
    embedded as a plain comparison. ``recall_floor`` measures what a principle
    is about (its statement, the situation that calls for it and the words a
    person would use) against what the person said.

    **Calibrated automatically.** Embedding models do not share a scale, so
    ``new_ground``, ``covered`` and ``combine`` are None by default, meaning
    "measure them": the first time a Wisdom Core uses an embedding model it
    runs ``check_embeddings`` once, keeps the result in its database, and uses
    the levels measured there (see ``WisdomCore.effective_levels``). A model
    that cannot tell the comparisons apart keeps the fallbacks (0.85, 0.95 and
    0.85). A number you set yourself is always used as given.

    ``combine`` is enforced by the Wisdom Core itself: a drafted principle at
    least that close to a held one (but below ``covered``) is merged with it
    instead of being added, and principles a task brings up that are that
    close to each other are combined into one. 0 turns it off.

    ``merge`` is how alike two principles must be for a merge the model
    proposes to go through, and ``recall_floor`` how relevant a principle must
    be to be injected at all; 0 for either leaves that decision where it is
    today. ``context`` is how many principles are injected, and ``max_adds``
    how many new ones one task may add.
    """

    new_ground: float | None = None          # None: measured for the embedding model (fallback 0.85)
    force: bool = FORCE_PRINCIPLE
    covered: float | None = None             # None: measured (fallback 0.95)
    combine: float | None = None             # None: measured (fallback 0.85)
    merge: float = MERGE_SIMILARITY
    recall_floor: float = RECALL_FLOOR
    context: int = DEFAULT_CONTEXT
    max_adds: int = MAX_ADDS_PER_TASK

    def __post_init__(self) -> None:
        self.force = bool(self.force)
        self.covered = None if self.covered is None else _level(self.covered, "covered")
        self.combine = None if self.combine is None else _level(self.combine, "combine")
        self.merge = _level(self.merge, "merge")
        self.recall_floor = _level(self.recall_floor, "recall_floor")
        self.context = _whole(self.context, "context", 1)
        self.max_adds = _whole(self.max_adds, "max_adds", 1)
        self.new_ground = None if self.new_ground is None else _level(self.new_ground, "new_ground")

    def to_dict(self) -> dict[str, Any]:
        return {"new_ground": self.new_ground, "force": self.force, "covered": self.covered, "combine": self.combine, "merge": self.merge,
                "recall_floor": self.recall_floor, "context": self.context, "max_adds": self.max_adds}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Levels:
        """Levels from a saved dict; anything missing keeps its default, anything unknown is ignored."""
        data = dict(data or {})
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data and data[f] is not None}
        return cls(**known)

    def replace(self, **changes: Any) -> Levels:
        """A copy with ``changes``; None for a measured level makes it automatic again."""
        return Levels.from_dict({**self.to_dict(), **changes})

    def automatic(self) -> list[str]:
        """The levels left to calibration (None)."""
        return [k for k in AUTO_LEVELS if getattr(self, k) is None]

    def resolved(self, measured: Mapping[str, Any] | None = None) -> Levels:
        """A copy with every automatic level filled in: from ``measured`` (a calibration's levels) when it has
        one, else the fallback. Levels set by hand are kept as they are."""
        fill = {k: (measured or {}).get(k, FALLBACK_LEVELS[k]) for k in self.automatic()}
        return Levels.from_dict({**self.to_dict(), **fill})
@dataclass
class Principle:
    """One principle the agent holds (or once held).

    ``name`` is the principle itself, one sentence telling the agent how to act.
    ``importance``, ``extent`` and ``use_case`` say why it matters, where it
    stops, and when it applies. ``related`` lists related concepts, which help
    the right principle come up in a search.
    """

    id: str
    name: str
    importance: str = ""
    extent: str = ""
    use_case: str = ""
    related: list[str] = field(default_factory=list)
    status: str = "active"          # active | merged | removed
    created_at: str = ""
    learned_from: str = ""          # the task it was written after
    times_injected: int = 0
    shaped: int = 0                 # answers it shaped (among the most relevant injected) whose reception was seen
    questioned: int = 0             # of those, answers the user pushed back on (or that an application marked failed)
    review: dict[str, Any] | None = None   # its last review, when it was in doubt and found to stand
    ended_at: str | None = None
    end_reason: str | None = None
    merged_into: str | None = None
    score: float | None = None      # similarity to the current task; not stored

    def since_review(self) -> tuple[int, int]:
        """Answers shaped and questioned since the principle was last reviewed and found to stand."""
        r = self.review or {}
        return max(self.shaped - int(r.get("shaped") or 0), 0), max(self.questioned - int(r.get("questioned") or 0), 0)

    def render(self) -> str:
        """The principle as the model sees it."""
        parts = [_sentence(self.name)]
        if self.importance:
            parts.append(f"Why: {_sentence(self.importance)}")
        if self.extent:
            parts.append(f"Scope: {_sentence(self.extent)}")
        if self.use_case:
            use = re.sub(r"^\s*(?:use\s+)?when\s+", "", self.use_case, flags=re.I) or self.use_case
            parts.append(f"Use when: {_sentence(use)}")
        return " ".join(parts)

    def embed_text(self) -> str:
        """The whole principle as one text, labels included (what earlier versions embedded)."""
        parts = [self.render()]
        if self.related:
            parts.append("Related: " + ", ".join(self.related) + ".")
        return " ".join(parts)

    def recall_text(self) -> str:
        """What the principle is about, embedded so it is found when its situation comes up: the statement, the
        situation that calls for it and the words a person would use, with no labels shared by every principle."""
        parts = [_sentence(self.name)]
        if self.use_case:
            parts.append(_sentence(self.use_case[0].upper() + self.use_case[1:]))
        if self.related:
            parts.append(", ".join(self.related) + ".")
        return " ".join(parts)

    def identity_text(self) -> str:
        """What the principle says, embedded to tell whether two principles teach the same thing: the statement
        alone, so the shared shape of the other fields cannot make different lessons look alike."""
        return _sentence(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "importance": self.importance,
            "extent": self.extent,
            "use_case": self.use_case,
            "related": list(self.related),
            "status": self.status,
            "created_at": self.created_at,
            "learned_from": self.learned_from,
            "times_injected": self.times_injected,
            "shaped": self.shaped,
            "questioned": self.questioned,
            "review": dict(self.review) if self.review else None,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "merged_into": self.merged_into,
        }


@dataclass
class Event:
    """One entry in the Wisdom Core's log of what it did."""

    at: str
    kind: str                       # axiom | added | merged | removed | covered | nothing | skipped | failed
    principle_id: str | None
    sources: list[str]
    detail: str
    task: str


@dataclass
class SynthesisReport:
    """What one round of Wisdom Synthesis did."""

    task: str
    added: list[Principle] = field(default_factory=list)
    merged: list[tuple[list[Principle], Principle]] = field(default_factory=list)
    removed: list[tuple[Principle, str]] = field(default_factory=list)
    covered: list[Principle] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    memo: str = ""
    error: str | None = None
    first_look: dict[str, Any] = field(default_factory=dict)   # the quick first look and what it measured
    reaction: dict[str, Any] = field(default_factory=dict)     # how the answer before was received, and by what
    questioned: list[Principle] = field(default_factory=list)  # principles behind an answer the user pushed back on
    supported: list[Principle] = field(default_factory=list)   # principles behind an answer the user carried on from
    doubt: list[Principle] = field(default_factory=list)       # principles that came into doubt this round
    stands: list[tuple[Principle, str]] = field(default_factory=list)   # reviewed while in doubt and found sound

    @property
    def changed(self) -> bool:
        """True if any principle was added, merged or removed."""
        return bool(self.added or self.merged or self.removed)

    def lines(self) -> list[str]:
        out: list[str] = []
        if self.error:
            out.append(f"! learning failed: {self.error}")
        for p in self.added:
            out.append(f"+ added: {p.name}" + (f" (nearest held: {p.score:.2f})" if p.score is not None else ""))
        for sources, p in self.merged:
            if len(sources) == 1:
                out.append(f"= revised: {sources[0].name} -> {p.name}")
            else:
                out.append(f"= merged {len(sources)} into: {p.name}" + (f" ({p.score:.2f} alike)" if p.score is not None else ""))
        for p, reason in self.removed:
            out.append(f"- removed: {p.name} ({reason})")
        for p in self.covered:
            out.append(f"~ already covered by: {p.name}" + (f" ({p.score:.2f})" if p.score is not None else ""))
        for p, reason in self.stands:
            out.append(f"= stands: {p.name} ({reason})")
        for note in self.skipped:
            out.append(f"? ignored: {note}")
        look = self.first_look
        if look.get("held"):
            out.append(f"~ already held, so nothing more ran (lesson: {look.get('lesson', '')}; measured "
                       f"{look.get('score', 0):.2f}, new ground below {look.get('level', 0):.2f})")
        if not out:
            out.append("~ nothing new to learn")
        kind = self.reaction.get("kind")             # how the answer was received comes after what was learned
        if kind and self.questioned:
            out.append(("v the user pushed back" if kind == "pushback" else "v the outcome was poor")
                       + f": questioned {len(self.questioned)} principle(s) behind the answer: "
                       + "; ".join(p.name for p in self.questioned))
        elif kind and self.supported:
            out.append(("^ the user carried on" if kind == "accepted" else "^ the outcome was good")
                       + f": supported {len(self.supported)} principle(s) behind the answer")
        for p in self.doubt:
            shaped, questioned = p.since_review()
            out.append(f"! in doubt: {p.name} (the user pushed back on {questioned} of the {shaped} answers it "
                       f"shaped; it is reviewed next time it comes up)")
        return out

    def summary(self) -> str:
        return "\n".join(self.lines())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "added": [{**p.to_dict(), "score": p.score} for p in self.added],
            "merged": [{"sources": [s.to_dict() for s in sources], "result": p.to_dict(), "score": p.score}
                       for sources, p in self.merged],
            "removed": [{"principle": p.to_dict(), "reason": reason} for p, reason in self.removed],
            "covered": [{**p.to_dict(), "score": p.score} for p in self.covered],
            "skipped": list(self.skipped),
            "memo": self.memo,
            "first_look": dict(self.first_look),
            "reaction": dict(self.reaction),
            "questioned": [p.to_dict() for p in self.questioned],
            "supported": [p.to_dict() for p in self.supported],
            "doubt": [p.to_dict() for p in self.doubt],
            "stands": [{"principle": p.to_dict(), "reason": reason} for p, reason in self.stands],
            "error": self.error,
            "changed": self.changed,
            "lines": self.lines(),
        }


# ---------------------------------------------------------------------------
# Model backends. A model is any object with:
#     chat(messages, *, json_mode=False, temperature=None) -> str
#     embed(texts) -> list[list[float]]
#     name, embed_name       (embed_name identifies the embedding space)
# chat() may also accept on_token (stream the reply), stats (filled with token
# counts) and think (switch a reasoning model's thinking); they are passed only
# to a chat() that declares them.
# ---------------------------------------------------------------------------
def _error_text(body: str) -> str:
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return _clip(body, 300) or "(no details)"
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            err = err.get("message") or err
        return _clip(err if isinstance(err, str) else json.dumps(err), 300)
    return _clip(body, 300)


def _error_body(exc: urllib.error.HTTPError) -> str:
    """The body of a server's error reply, read and then closed (Python 3.14 warns about one left open). A body
    that cannot be read is empty, so the caller still gets a ModelError with the status."""
    try:
        return exc.read().decode("utf-8", "replace")
    except (OSError, http.client.HTTPException):
        return ""
    finally:
        exc.close()


def _http_json(
    url: str,
    payload: Any = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    hint: str = "",
) -> Any:
    """POST (or GET when there is no payload) JSON and return the decoded reply.

    The default timeout is None: a slow local model is waited for, never cut off.
    """
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = _error_body(exc)
        raise ModelError(
            f"{url} answered HTTP {exc.code}: {_error_text(body)}", status=exc.code, body=body
        ) from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        msg = f"Could not reach {url}: {reason}"
        raise ModelError(f"{msg}. {hint}" if hint else msg) from None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ModelError(f"{url} did not return JSON: {raw[:200]!r}") from None


def _http_stream(
    url: str,
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    hint: str = "",
    sse: bool = False,
) -> Generator[Any, None, None]:
    """POST JSON and yield each JSON object the server streams back.

    Ollama streams one JSON object per line; OpenAI-style servers send
    server-sent events ("data: {...}"). Closing the generator closes the
    connection, which tells the server to stop generating.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Accept", "text/event-stream" if sse else "application/x-ndjson")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = _error_body(exc)
        raise ModelError(
            f"{url} answered HTTP {exc.code}: {_error_text(body)}", status=exc.code, body=body
        ) from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        msg = f"Could not reach {url}: {reason}"
        raise ModelError(f"{msg}. {hint}" if hint else msg) from None
    with resp:
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if sse:
                    if not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                    if line == "[DONE]":
                        return
                try:
                    yield json.loads(line)
                except ValueError:
                    log.debug("skipped a streamed line that is not JSON: %r", line[:200])
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise ModelError(f"The connection to {url} dropped while the reply was streaming: {exc}") from None


def _ollama_host(host: str | None) -> str:
    """Normalise an Ollama address; accepts the same forms as $OLLAMA_HOST."""
    host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").strip()
    if "://" not in host:
        host = "http://" + host
    parts = urllib.parse.urlsplit(host)
    hostname = parts.hostname or "localhost"
    if hostname in ("0.0.0.0", "::"):   # a bind address, not somewhere to connect to
        hostname = "localhost"
    port = parts.port or (443 if parts.scheme == "https" else 11434)
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parts.scheme}://{netloc}:{port}{parts.path.rstrip('/')}"


def _tag(name: str) -> str:
    """Ollama treats "x" and "x:latest" as the same model."""
    return name if ":" in name else f"{name}:latest"


_EMBED_PREFIXES = (
    "all-minilm", "bge-", "bge:", "mxbai", "snowflake-arctic", "granite-embedding",
    "paraphrase-multilingual", "jina", "nomic-embed", "e5-", "gte-", "multilingual-e5",
)


class Ollama:
    """Chat and embeddings through a local (or LAN) Ollama server.

    With no arguments it uses the first chat model and the first embedding
    model the server has. The address comes from ``host``, then $OLLAMA_HOST,
    then http://localhost:11434. The context window sent with every request
    is read from the model itself and capped at 16,384 tokens, unless you pass
    ``num_ctx``.
    """

    def __init__(
        self,
        model: str | None = None,
        embed_model: str | None = None,
        *,
        host: str | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
        keep_alive: str | int | None = None,
    ) -> None:
        self.host = _ollama_host(host)
        self._model = model
        self._embed_model = embed_model
        self.num_ctx = int(num_ctx) if num_ctx else None
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._legacy_embed = False
        self._windows: dict[str, int] = {}     # model -> context window worked out from /api/show
        self._no_think: set[str] = set()       # models whose server refused the think switch
        self._starved: set[str] = set()        # models that thought without answering

    def __repr__(self) -> str:
        return f"Ollama(model={self._model!r}, embed_model={self._embed_model!r}, host={self.host!r})"

    _HINT = "Is Ollama running? Start it with: ollama serve"

    def _call(self, path: str, payload: Any = None) -> Any:
        return _http_json(f"{self.host}{path}", payload, timeout=self.timeout, hint=self._HINT)

    @property
    def model(self) -> str:
        if not self._model:
            self._detect()
        return self._model  # type: ignore[return-value]

    @model.setter
    def model(self, name: str | None) -> None:
        self._model = name or None

    @property
    def embed_model(self) -> str:
        if not self._embed_model:
            self._detect()
        return self._embed_model  # type: ignore[return-value]

    @embed_model.setter
    def embed_model(self, name: str | None) -> None:
        self._embed_model = name or None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    @property
    def embed_name(self) -> str:
        return f"ollama:{_tag(self.embed_model)}"

    @property
    def context_window(self) -> int:
        """The num_ctx sent with every chat request: the model's own limit, capped at 16,384."""
        if self.num_ctx:
            return self.num_ctx
        name = self.model
        if name not in self._windows:
            limit = self.model_limit(name)
            self._windows[name] = min(limit, DEFAULT_NUM_CTX) if limit else FALLBACK_NUM_CTX
        return self._windows[name]

    def model_limit(self, name: str | None = None) -> int | None:
        """The longest context the model can hold, from its own metadata; None if it does not say."""
        try:
            info = self._call("/api/show", {"model": name or self.model})
        except ModelError:
            return None
        details = info.get("model_info") if isinstance(info, dict) else None
        if isinstance(details, dict):
            for key, value in details.items():
                if str(key).endswith(".context_length") and isinstance(value, (int, float)) and value > 0:
                    return int(value)
        return None

    def list_models(self) -> list[dict[str, Any]]:
        data = self._call("/api/tags")
        models = data.get("models") if isinstance(data, dict) else None
        return [m for m in (models or []) if isinstance(m, dict)]

    def is_embedding_model(self, entry: dict[str, Any] | str) -> bool:
        """True for a model that only makes embeddings (asked of the server, then guessed from the name)."""
        if isinstance(entry, str):
            entry = {"name": entry}
        name = str(entry.get("name") or entry.get("model") or "")
        try:
            info = self._call("/api/show", {"model": name})
        except ModelError:
            info = {}
        caps = info.get("capabilities") if isinstance(info, dict) else None
        if isinstance(caps, list) and caps:
            return "embedding" in caps and "completion" not in caps
        details = entry.get("details") or {}
        families = " ".join(str(f) for f in (details.get("families") or []))
        families += " " + str(details.get("family") or "")
        low = name.lower()
        return "embed" in low or "bert" in families.lower() or low.startswith(_EMBED_PREFIXES)

    _is_embedding_model = is_embedding_model

    def _detect(self) -> None:
        chat: str | None = self._model
        embed: str | None = self._embed_model
        for entry in self.list_models():
            if chat and embed:
                break
            name = str(entry.get("name") or entry.get("model") or "")
            if not name:
                continue
            if self.is_embedding_model(entry):
                embed = embed or name
            else:
                chat = chat or name
        if not chat:
            raise ModelError(
                f"Ollama at {self.host} has no chat model. Pull one first, for example: ollama pull llama3.2"
            )
        if not embed:
            raise ModelError(
                f"Ollama at {self.host} has no embedding model, which Wisdom Core needs to search "
                "principles by meaning. Pull one first, for example: ollama pull nomic-embed-text"
            )
        self._model, self._embed_model = chat, embed
        log.info("Ollama at %s: chat model %s, embedding model %s", self.host, chat, embed)

    @staticmethod
    def _missing_model(exc: ModelError, name: str) -> ModelError:
        body = exc.body.lower()
        if exc.status == 404 and "not found" in body and "model" in body:
            return ModelError(f"Ollama does not have the model {name!r}. Pull it first: ollama pull {name}",
                              status=404, body=exc.body)
        return exc

    def chat(self, messages: Sequence[dict[str, Any]], *, json_mode: bool = False,
             temperature: float | None = None, on_token: Callable[[str, str], None] | None = None,
             stats: dict[str, Any] | None = None, think: bool | None = None) -> str:
        """One reply.

        ``on_token(text, kind)`` receives the reply as it is written, kind being
        "content" or "thinking". ``stats`` is filled with token counts, the
        window used and why generation stopped. ``think=False`` switches a
        reasoning model's thinking off; None leaves the model's default. A model
        that thinks without writing an answer is asked once more with thinking
        off, and remembered.
        """
        name = self.model
        options: dict[str, Any] = {"num_ctx": self.context_window}
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict[str, Any] = {
            "model": name,
            "messages": [dict(m) for m in messages],
            "stream": on_token is not None,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if think is None and name in self._starved:
            think = False
        if think is not None and name not in self._no_think:
            payload["think"] = bool(think)
        content, thinking, info = self._exchange(payload, on_token)
        if (not content.strip() and thinking.strip() and payload.get("think") is not False
                and name not in self._no_think):
            self._starved.add(name)
            payload["think"] = False
            content, more, info = self._exchange(payload, on_token)
            thinking += more
            info["note"] = "The model thought without answering, so it was asked again with thinking off."
        if stats is not None:
            stats.update(info)
            stats["thinking"] = thinking
        return content

    @staticmethod
    def _info(data: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": payload["model"],
            "num_ctx": payload["options"]["num_ctx"],
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "done_reason": data.get("done_reason"),
        }

    def _exchange(self, payload: dict[str, Any],
                  on_token: Callable[[str, str], None] | None) -> tuple[str, str, dict[str, Any]]:
        try:
            if payload["stream"]:
                assert on_token is not None
                return self._stream(payload, on_token)
            data = self._call("/api/chat", payload)
        except ModelError as exc:
            if exc.status == 400 and "think" in payload and "think" in exc.body.lower():
                self._no_think.add(payload["model"])      # an older server, or a model with no switch
                return self._exchange({k: v for k, v in payload.items() if k != "think"}, on_token)
            raise self._missing_model(exc, payload["model"]) from None
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelError(f"Unexpected reply from Ollama /api/chat: {_clip(json.dumps(data), 200)}")
        thinking = message.get("thinking") if isinstance(message, dict) else None
        return content, thinking if isinstance(thinking, str) else "", self._info(data, payload)

    def _stream(self, payload: dict[str, Any],
                on_token: Callable[[str, str], None]) -> tuple[str, str, dict[str, Any]]:
        content: list[str] = []
        thinking: list[str] = []
        info = self._info({}, payload)
        chunks = _http_stream(f"{self.host}/api/chat", payload, timeout=self.timeout, hint=self._HINT)
        try:
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise ModelError(f"Ollama stopped with an error: {chunk['error']}")
                message = chunk.get("message") or {}
                piece = message.get("thinking")
                if isinstance(piece, str) and piece:
                    thinking.append(piece)
                    on_token(piece, "thinking")
                piece = message.get("content")
                if isinstance(piece, str) and piece:
                    content.append(piece)
                    on_token(piece, "content")
                if chunk.get("done"):
                    info = self._info(chunk, payload)
        finally:
            chunks.close()
        return "".join(content), "".join(thinking), info

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = [str(t) for t in texts]
        if not texts:
            return []
        name = self.embed_model
        if not self._legacy_embed:
            try:
                data = self._call("/api/embed", {"model": name, "input": texts, "truncate": True})
            except ModelError as exc:
                missing = self._missing_model(exc, name)
                if missing is not exc or exc.status != 404:
                    raise missing from None
                self._legacy_embed = True      # Ollama older than 0.3.4: one text per call
            else:
                vecs = data.get("embeddings") if isinstance(data, dict) else None
                if not isinstance(vecs, list) or len(vecs) != len(texts):
                    raise ModelError(f"Unexpected reply from Ollama /api/embed: {_clip(json.dumps(data), 200)}")
                return vecs
        out: list[list[float]] = []
        for text in texts:
            try:
                data = self._call("/api/embeddings", {"model": name, "prompt": text})
            except ModelError as exc:
                raise self._missing_model(exc, name) from None
            vec = data.get("embedding") if isinstance(data, dict) else None
            if not isinstance(vec, list) or not vec:
                raise ModelError(f"Unexpected reply from Ollama /api/embeddings: {_clip(json.dumps(data), 200)}")
            out.append(vec)
        return out


class OpenAICompatible:
    """Chat and embeddings through any server that speaks the OpenAI API.

    That covers OpenAI itself and local servers such as LM Studio, vLLM,
    llama.cpp's llama-server and Ollama's /v1 endpoint. ``base_url`` defaults
    to $OPENAI_BASE_URL, then https://api.openai.com/v1; ``api_key`` defaults
    to $OPENAI_API_KEY. ``model`` may be left out to call ``list_models()`` only.
    """

    def __init__(
        self,
        model: str | None = None,
        embed_model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if model is not None and not str(model).strip():
            raise ValueError("OpenAICompatible needs a chat model name")
        self.model = model
        self.embed_model = embed_model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self._json_ok = True     # some servers reject response_format
        self._temp_ok = True     # some reasoning models reject temperature
        self._usage_ok = True    # some servers reject stream_options (token counts while streaming)

    def __repr__(self) -> str:
        return f"OpenAICompatible(model={self.model!r}, embed_model={self.embed_model!r}, base_url={self.base_url!r})"

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    @property
    def embed_name(self) -> str:
        return f"openai:{self.base_url}:{self.embed_model}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _post(self, path: str, payload: Any) -> Any:
        return _http_json(f"{self.base_url}{path}", payload, headers=self._headers(), timeout=self.timeout)

    def list_models(self) -> list[str]:
        """The model names the server offers (its /models list)."""
        data = _http_json(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ModelError(f"Unexpected reply from {self.base_url}/models: {_clip(json.dumps(data), 200)}")
        names: list[str] = []
        for item in items:
            name = item.get("id") if isinstance(item, dict) else item
            if isinstance(name, str) and name.strip() and name not in names:
                names.append(name)
        return names

    def chat(self, messages: Sequence[dict[str, Any]], *, json_mode: bool = False,
             temperature: float | None = None, on_token: Callable[[str, str], None] | None = None,
             stats: dict[str, Any] | None = None, think: bool | None = None) -> str:
        """One reply; see ``Ollama.chat``. ``think`` is accepted and not sent: there is no standard switch."""
        if not self.model:
            raise ModelError("OpenAICompatible needs a chat model name, for example OpenAICompatible('gpt-4.1-mini')")
        last: ModelError | None = None
        for _ in range(4):
            payload: dict[str, Any] = {"model": self.model, "messages": [dict(m) for m in messages]}
            if json_mode and self._json_ok:
                payload["response_format"] = {"type": "json_object"}
            if temperature is not None and self._temp_ok:
                payload["temperature"] = temperature
            if on_token is not None:
                payload["stream"] = True
                if self._usage_ok:
                    payload["stream_options"] = {"include_usage": True}
            try:
                if on_token is not None:
                    content, thinking, info = self._stream(payload, on_token)
                else:
                    content, thinking, info = self._read(self._post("/chat/completions", payload))
                break
            except ModelError as exc:
                last = exc
                if exc.status not in (400, 422):
                    raise
                body = exc.body.lower()
                if "stream_options" in payload and ("stream_options" in body or "include_usage" in body):
                    self._usage_ok = False
                elif "temperature" in payload and "temperature" in body:
                    self._temp_ok = False
                elif "response_format" in payload:
                    self._json_ok = False
                elif "temperature" in payload:
                    self._temp_ok = False
                else:
                    raise
        else:
            assert last is not None
            raise last
        if stats is not None:
            stats.update(info)
            stats["thinking"] = thinking
        return content

    def _read(self, data: Any) -> tuple[str, str, dict[str, Any]]:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelError(f"Unexpected reply from {self.base_url}/chat/completions: "
                             f"{_clip(json.dumps(data), 200)}") from None
        thinking = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        info = {"model": self.model, "prompt_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"), "done_reason": choice.get("finish_reason")}
        return (content if isinstance(content, str) else ""), (thinking if isinstance(thinking, str) else ""), info

    def _stream(self, payload: dict[str, Any],
                on_token: Callable[[str, str], None]) -> tuple[str, str, dict[str, Any]]:
        content: list[str] = []
        thinking: list[str] = []
        info: dict[str, Any] = {"model": self.model, "prompt_tokens": None, "output_tokens": None,
                                "done_reason": None}
        chunks = _http_stream(f"{self.base_url}/chat/completions", payload, headers=self._headers(),
                              timeout=self.timeout, sse=True)
        try:
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise ModelError(f"{self.base_url} stopped with an error: {_error_text(json.dumps(chunk))}")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(piece, str) and piece:
                        thinking.append(piece)
                        on_token(piece, "thinking")
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        content.append(piece)
                        on_token(piece, "content")
                    if choice.get("finish_reason"):
                        info["done_reason"] = choice["finish_reason"]
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    info["prompt_tokens"] = usage.get("prompt_tokens")
                    info["output_tokens"] = usage.get("completion_tokens")
        finally:
            chunks.close()
        return "".join(content), "".join(thinking), info

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = [str(t) for t in texts]
        if not texts:
            return []
        if not self.embed_model:
            raise ModelError(
                "OpenAICompatible needs embed_model=... so principles can be searched by meaning "
                "(for OpenAI, for example text-embedding-3-small)."
            )
        data = self._post("/embeddings", {"model": self.embed_model, "input": texts})
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise ModelError(f"Unexpected reply from {self.base_url}/embeddings: {_clip(json.dumps(data), 200)}")
        try:
            items = sorted(items, key=lambda it: int(it.get("index", 0)))
            return [list(it["embedding"]) for it in items]
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ModelError(f"Unexpected reply from {self.base_url}/embeddings: "
                             f"{_clip(json.dumps(data), 200)}") from None


_ANTHROPIC_STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class Anthropic:
    """Chat through Anthropic's Messages API (the Claude models).

    ``api_key`` defaults to $ANTHROPIC_API_KEY and ``base_url`` to
    https://api.anthropic.com. Anthropic has no embedding model, so pair it
    with a backend that has one::

        Combined(Anthropic("claude-sonnet-4-5"), Ollama(embed_model="embeddinggemma"))

    ``model`` may be left out to call ``list_models()`` only. Replies are
    capped at the model's own output limit (known once ``list_models()`` has
    run), or ANTHROPIC_MAX_OUTPUT tokens, or ``max_tokens`` if you pass it.
    """

    def __init__(self, model: str | None = None, *, api_key: str | None = None, base_url: str | None = None,
                 timeout: float | None = None, max_tokens: int | None = None) -> None:
        if model is not None and not str(model).strip():
            raise ValueError("Anthropic needs a model name, for example Anthropic('claude-sonnet-4-5')")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        base = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.base_url = base[:-3] if base.endswith("/v1") else base      # the paths below carry /v1
        self.timeout = timeout
        self.max_tokens = int(max_tokens) if max_tokens else None
        self._temp_ok = True                  # models after Claude Opus 4.6 refuse a temperature
        self._limits: dict[str, int] = {}     # model -> most output tokens it allows, from list_models()

    def __repr__(self) -> str:
        return f"Anthropic(model={self.model!r}, base_url={self.base_url!r})"

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    @property
    def embed_name(self) -> str:
        return "anthropic:no-embeddings"

    def _headers(self) -> dict[str, str]:
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def list_models(self) -> list[dict[str, Any]]:
        """The models this key can use, as Anthropic lists them (newest first)."""
        found: list[dict[str, Any]] = []
        after = ""
        for _ in range(50):
            query = "limit=1000" + (f"&after_id={urllib.parse.quote(after)}" if after else "")
            data = _http_json(f"{self.base_url}/v1/models?{query}", headers=self._headers(), timeout=self.timeout)
            items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ModelError(f"Unexpected reply from {self.base_url}/v1/models: {_clip(json.dumps(data), 200)}")
            found += [m for m in items if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]]
            if not data.get("has_more") or not data.get("last_id"):
                break
            after = str(data["last_id"])
        for m in found:
            limit = m.get("max_tokens")
            if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
                self._limits[m["id"]] = limit
        return found

    def _output_limit(self) -> int:
        if self.max_tokens:
            return self.max_tokens
        known = self._limits.get(str(self.model))
        return min(known, ANTHROPIC_MAX_OUTPUT) if known else ANTHROPIC_MAX_OUTPUT

    def _payload(self, messages: Sequence[dict[str, Any]], temperature: float | None,
                 stream: bool) -> dict[str, Any]:
        system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system").strip()
        turns: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            text = str(m.get("content") or "").strip() or "(empty)"      # the API refuses empty turns
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + text
            else:
                turns.append({"role": role, "content": text})
        while turns and turns[0]["role"] != "user":                     # a conversation opens with the person
            turns.pop(0)
        if not turns:
            raise ModelError("There is no message for the model to answer.")
        payload: dict[str, Any] = {"model": self.model, "max_tokens": self._output_limit(), "messages": turns}
        if system:
            payload["system"] = system
        if temperature is not None and self._temp_ok:
            payload["temperature"] = min(max(float(temperature), 0.0), 1.0)
        if stream:
            payload["stream"] = True
        return payload

    def _info(self, prompt: Any, output: Any, stop: Any) -> dict[str, Any]:
        return {"model": self.model, "prompt_tokens": prompt, "output_tokens": output,
                "done_reason": _ANTHROPIC_STOP.get(stop, stop) if stop else None}

    def chat(self, messages: Sequence[dict[str, Any]], *, json_mode: bool = False,
             temperature: float | None = None, on_token: Callable[[str, str], None] | None = None,
             stats: dict[str, Any] | None = None, think: bool | None = None) -> str:
        """One reply; see ``Ollama.chat``. ``json_mode`` and ``think`` are accepted: the prompts already ask
        for JSON where it is needed, and extended thinking is left off."""
        if not self.model:
            raise ModelError("Anthropic needs a model name, for example Anthropic('claude-sonnet-4-5')")
        last: ModelError | None = None
        for _ in range(3):
            payload = self._payload(messages, temperature, on_token is not None)
            try:
                if on_token is not None:
                    content, thinking, info = self._stream(payload, on_token)
                else:
                    content, thinking, info = self._read(_http_json(
                        f"{self.base_url}/v1/messages", payload, headers=self._headers(), timeout=self.timeout))
                break
            except ModelError as exc:
                last = exc
                body = exc.body.lower()
                if exc.status == 400 and "temperature" in payload and "temperature" in body:
                    self._temp_ok = False
                elif exc.status == 400 and "max_tokens" in body and payload["max_tokens"] > 4096:
                    self._limits[str(self.model)] = 4096
                else:
                    raise
        else:
            assert last is not None
            raise last
        if stats is not None:
            stats.update(info)
            stats["thinking"] = thinking
        return content

    def _read(self, data: Any) -> tuple[str, str, dict[str, Any]]:
        blocks = data.get("content") if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            raise ModelError(f"Unexpected reply from {self.base_url}/v1/messages: {_clip(json.dumps(data), 200)}")
        text = "".join(str(b.get("text") or "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        thinking = "".join(str(b.get("thinking") or "") for b in blocks
                           if isinstance(b, dict) and b.get("type") == "thinking")
        usage = _as_dict(data.get("usage"))
        return text, thinking, self._info(usage.get("input_tokens"), usage.get("output_tokens"), data.get("stop_reason"))

    def _stream(self, payload: dict[str, Any],
                on_token: Callable[[str, str], None]) -> tuple[str, str, dict[str, Any]]:
        content: list[str] = []
        thinking: list[str] = []
        info = self._info(None, None, None)
        events = _http_stream(f"{self.base_url}/v1/messages", payload, headers=self._headers(),
                              timeout=self.timeout, sse=True)
        try:
            for event in events:
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind == "error":
                    err = _as_dict(event.get("error"))
                    raise ModelError(f"Anthropic stopped with an error: {err.get('message') or err.get('type') or event}")
                if kind == "message_start":
                    message = _as_dict(event.get("message"))
                    usage = _as_dict(message.get("usage"))
                    info["prompt_tokens"] = usage.get("input_tokens")
                elif kind == "content_block_delta":
                    delta = _as_dict(event.get("delta"))
                    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str) and delta["text"]:
                        content.append(delta["text"])
                        on_token(delta["text"], "content")
                    elif (delta.get("type") == "thinking_delta" and isinstance(delta.get("thinking"), str)
                          and delta["thinking"]):
                        thinking.append(delta["thinking"])
                        on_token(delta["thinking"], "thinking")
                elif kind == "message_delta":
                    delta = _as_dict(event.get("delta"))
                    if delta.get("stop_reason"):
                        info["done_reason"] = _ANTHROPIC_STOP.get(delta["stop_reason"], delta["stop_reason"])
                    usage = _as_dict(event.get("usage"))
                    if usage.get("output_tokens") is not None:
                        info["output_tokens"] = usage["output_tokens"]
                elif kind == "message_stop":
                    break
        finally:
            events.close()
        return "".join(content), "".join(thinking), info

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ModelError("Anthropic has no embedding model. Search principles with another backend: "
                         "Combined(Anthropic(...), Ollama(embed_model=...)).")


def _accepted_kwargs(fn: Callable[..., Any]) -> set[str] | None:
    """Names ``fn`` accepts as keywords; None means it takes **kwargs."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return set()
    names = set()
    for p in params:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            names.add(p.name)
    return names


class FunctionModel:
    """Use your own functions as the model.

    ``chat(messages) -> str`` receives OpenAI-style message dicts. If it also
    accepts ``json_mode``, ``temperature``, ``on_token``, ``stats`` or ``think``
    keywords (or **kwargs), those are passed along. ``embed(texts) -> list of
    vectors``. Change ``embed_name`` whenever you switch embedding models, so
    stored principles are re-embedded in the new space.
    """

    def __init__(
        self,
        chat: Callable[..., str],
        embed: Callable[[list[str]], Sequence[Sequence[float]]],
        *,
        name: str = "custom",
        embed_name: str | None = None,
    ) -> None:
        self._chat = chat
        self._embed = embed
        self.name = name
        self.embed_name = embed_name or f"custom:{name}"
        self._kwargs = _accepted_kwargs(chat)

    def __repr__(self) -> str:
        return f"FunctionModel(name={self.name!r}, embed_name={self.embed_name!r})"

    def chat(self, messages: Sequence[dict[str, Any]], *, json_mode: bool = False,
             temperature: float | None = None, on_token: Callable[[str, str], None] | None = None,
             stats: dict[str, Any] | None = None, think: bool | None = None) -> str:
        hints: dict[str, Any] = {"json_mode": json_mode, "temperature": temperature}
        hints.update({k: v for k, v in (("on_token", on_token), ("stats", stats), ("think", think))
                      if v is not None})
        if self._kwargs is not None:
            hints = {k: v for k, v in hints.items() if k in self._kwargs}
        out = self._chat([dict(m) for m in messages], **hints)
        text = "" if out is None else str(out)
        if on_token is not None and "on_token" not in hints and text:
            on_token(text, "content")      # a function that cannot stream still shows its reply
        return text

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(v) for v in self._embed([str(t) for t in texts])]


class Combined:
    """Chat through one backend and search principles through another.

    For a chat model that has no embeddings (Anthropic), or to keep the
    searching on a small local model::

        Combined(Anthropic("claude-sonnet-4-5"), Ollama(embed_model="embeddinggemma"))

    ``chat_backend`` and ``embed_backend`` can be swapped at any time; the
    Wisdom Core re-embeds its principles when the embedding model changes.
    """

    def __init__(self, chat: Any, embed: Any) -> None:
        if not callable(getattr(chat, "chat", None)):
            raise TypeError("Combined(chat, embed): chat must be a backend with a chat() method")
        if not callable(getattr(embed, "embed", None)):
            raise TypeError("Combined(chat, embed): embed must be a backend with an embed() method")
        self.chat_backend = chat
        self.embed_backend = embed

    def __repr__(self) -> str:
        return f"Combined(chat={self.chat_backend!r}, embed={self.embed_backend!r})"

    @property
    def name(self) -> str:
        return str(getattr(self.chat_backend, "name", type(self.chat_backend).__name__))

    @property
    def embed_name(self) -> str:
        return _embed_name(self.embed_backend)

    @property
    def context_window(self) -> int | None:
        return context_window_of(self.chat_backend)

    def chat(self, messages: Sequence[dict[str, Any]], *, json_mode: bool = False,
             temperature: float | None = None, on_token: Callable[[str, str], None] | None = None,
             stats: dict[str, Any] | None = None, think: bool | None = None) -> str:
        accepted = _accepted_kwargs(self.chat_backend.chat)
        hints: dict[str, Any] = {"json_mode": json_mode, "temperature": temperature}
        hints.update({k: v for k, v in (("on_token", on_token), ("stats", stats), ("think", think))
                      if v is not None})
        if accepted is not None:
            hints = {k: v for k, v in hints.items() if k in accepted}
        out = self.chat_backend.chat([dict(m) for m in messages], **hints)
        text = "" if out is None else str(out)
        if on_token is not None and "on_token" not in hints and text:
            on_token(text, "content")      # a backend that cannot stream still shows its reply
        return text

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(v) for v in self.embed_backend.embed([str(t) for t in texts])]


def resolve_model(model: Any = None) -> Any:
    """Turn the ``model`` argument into a backend.

    None means Ollama with automatic model choice; a string is an Ollama chat
    model name; anything with ``chat`` and ``embed`` methods is used as is.
    """
    if model is None:
        return Ollama()
    if isinstance(model, str):
        return Ollama(model=model)
    if callable(getattr(model, "chat", None)) and callable(getattr(model, "embed", None)):
        return model
    raise TypeError(
        "model must be None, an Ollama model name, or an object with chat() and embed() methods "
        "(for example Ollama(...), OpenAICompatible(...), Combined(...) or FunctionModel(...))"
    )


def _embed_name(model: Any) -> str:
    name = getattr(model, "embed_name", None)
    return str(name) if name else type(model).__name__


# ---------------------------------------------------------------------------
# Search by meaning: an in-memory index of unit vectors.
# ---------------------------------------------------------------------------
class _Index:
    """Two vectors per principle. ``recall``: what it is about, searched with what a person says. ``ident``:
    what it says, compared with other principles to tell whether two teach the same thing."""

    KINDS = ("recall", "ident")

    def __init__(self) -> None:
        self.ids: list[str] = []
        self._vecs: dict[str, list[array]] = {k: [] for k in self.KINDS}
        self._pos: dict[str, int] = {}
        self._matrix: dict[str, Any] = {k: None for k in self.KINDS}
        self.dims: int | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, pid: object) -> bool:
        return pid in self._pos

    def put(self, pid: str, vec: array, ident: array | None = None) -> None:
        """Hold a principle's vectors; without ``ident`` the recall vector stands in for both."""
        ident = vec if ident is None else ident
        for v in (vec, ident):
            if self.dims is None:
                self.dims = len(v)
            elif len(v) != self.dims:
                raise WisdomCoreError(f"vector size {len(v)} does not match the index ({self.dims})")
        if pid in self._pos:
            i = self._pos[pid]
            self._vecs["recall"][i], self._vecs["ident"][i] = vec, ident
        else:
            self._pos[pid] = len(self.ids)
            self.ids.append(pid)
            self._vecs["recall"].append(vec)
            self._vecs["ident"].append(ident)
        self._matrix = {k: None for k in self.KINDS}

    def drop(self, pid: str) -> None:
        i = self._pos.pop(pid, None)
        if i is None:
            return
        last = len(self.ids) - 1
        if i != last:
            self.ids[i] = self.ids[last]
            for vecs in self._vecs.values():
                vecs[i] = vecs[last]
            self._pos[self.ids[i]] = i
        self.ids.pop()
        for vecs in self._vecs.values():
            vecs.pop()
        self._matrix = {k: None for k in self.KINDS}
        if not self.ids:
            self.dims = None

    def vector(self, pid: str, kind: str = "ident") -> array | None:
        i = self._pos.get(pid)
        return None if i is None else self._vecs[kind][i]

    def scores(self, query: array, kind: str = "recall") -> list[float]:
        if not self.ids:
            return []
        if len(query) != self.dims:
            raise WisdomCoreError(f"query vector size {len(query)} does not match the index ({self.dims})")
        vecs = self._vecs[kind]
        if _np is not None:
            if self._matrix[kind] is None:
                blob = b"".join(v.tobytes() for v in vecs)
                self._matrix[kind] = _np.frombuffer(blob, dtype=_np.float32).reshape(len(self.ids), len(query))
            q = _np.frombuffer(query.tobytes(), dtype=_np.float32)
            return [float(x) for x in self._matrix[kind] @ q]
        return [_dot(v, query) for v in vecs]

    def top(self, queries: Sequence[array], k: int, exclude: Iterable[str] = (),
            kind: str = "recall") -> list[tuple[str, float]]:
        """Best ``k`` ids; each id scores the best of its similarities to the queries."""
        if k <= 0 or not self.ids:
            return []
        skip = set(exclude)
        best: dict[str, float] = {}
        for q in queries:
            for pid, sc in zip(self.ids, self.scores(q, kind), strict=True):
                if pid not in skip and sc > best.get(pid, -2.0):
                    best[pid] = sc
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


# ---------------------------------------------------------------------------
# Storage: one SQLite file per agent.
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS axioms (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    text   TEXT NOT NULL,
    set_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS principles (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    importance     TEXT NOT NULL DEFAULT '',
    extent         TEXT NOT NULL DEFAULT '',
    use_case       TEXT NOT NULL DEFAULT '',
    related        TEXT NOT NULL DEFAULT '[]',
    status         TEXT NOT NULL DEFAULT 'active',
    created_at     TEXT NOT NULL,
    axiom_id       INTEGER,
    learned_from   TEXT NOT NULL DEFAULT '',
    times_injected INTEGER NOT NULL DEFAULT 0,
    last_injected  TEXT,
    ended_at       TEXT,
    end_reason     TEXT,
    merged_into    TEXT,
    embedding      BLOB,
    embed_name     TEXT,
    dims           INTEGER,
    identity       BLOB,
    shaped         INTEGER NOT NULL DEFAULT 0,
    questioned     INTEGER NOT NULL DEFAULT 0,
    review         TEXT
);
CREATE INDEX IF NOT EXISTS principles_status ON principles(status);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    at           TEXT NOT NULL,
    kind         TEXT NOT NULL,
    principle_id TEXT,
    sources      TEXT NOT NULL DEFAULT '[]',
    detail       TEXT NOT NULL DEFAULT '',
    task         TEXT NOT NULL DEFAULT ''
);
"""


def _vec_blob(vec: array) -> bytes:
    return vec.tobytes()


def _json_list(value: Any) -> list[str]:
    """The list held in a JSON text column (or a list as it already stands)."""
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    try:
        out = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [str(x) for x in out] if isinstance(out, list) else []


def _blob_vec(blob: bytes | None) -> array | None:
    if not blob:
        return None
    out = array("f")
    out.frombytes(blob)
    return out


def _review_from(raw: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _principle_from_row(row: sqlite3.Row) -> Principle:
    try:
        related = json.loads(row["related"] or "[]")
    except ValueError:
        related = []
    return Principle(
        id=row["id"],
        name=row["name"],
        importance=row["importance"],
        extent=row["extent"],
        use_case=row["use_case"],
        related=[str(r) for r in related] if isinstance(related, list) else [],
        status=row["status"],
        created_at=row["created_at"],
        learned_from=row["learned_from"],
        times_injected=row["times_injected"],
        shaped=int(row["shaped"] or 0),
        questioned=int(row["questioned"] or 0),
        review=_review_from(row["review"]),
        ended_at=row["ended_at"],
        end_reason=row["end_reason"],
        merged_into=row["merged_into"],
    )


class _Store:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:" and not path.startswith("file:"):
            folder = os.path.dirname(os.path.abspath(path))
            os.makedirs(folder, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None,
                                    uri=path.startswith("file:"))
        self.conn.row_factory = sqlite3.Row
        try:
            with self.lock:
                tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                if tables and "meta" not in tables:
                    raise WisdomCoreError(f"{path} is a database that Wisdom Core did not create; "
                                          "give each agent its own file.")
                self.conn.executescript(_SCHEMA)
                columns = {r[1] for r in self.conn.execute("PRAGMA table_info(principles)")}
                if "identity" not in columns:      # a database from before 0.5: its principles are re-embedded
                    self.conn.execute("ALTER TABLE principles ADD COLUMN identity BLOB")
                for column, kind in (("shaped", "INTEGER NOT NULL DEFAULT 0"),   # before 0.8: no standing yet
                                     ("questioned", "INTEGER NOT NULL DEFAULT 0"), ("review", "TEXT")):
                    if column not in columns:
                        self.conn.execute(f"ALTER TABLE principles ADD COLUMN {column} {kind}")
                version = self.meta("schema_version")
                if version is None:
                    self.set_meta("schema_version", str(SCHEMA_VERSION))
                elif int(version) > SCHEMA_VERSION:
                    raise WisdomCoreError(f"{path} was written by a newer version of wisdom_core "
                                          f"(schema {version}); upgrade wisdom_core.")
        except sqlite3.DatabaseError as exc:
            self.conn.close()
            raise WisdomCoreError(f"{path} is not a Wisdom Core database ({exc}).") from None
        except BaseException:
            self.conn.close()
            raise

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def data_version(self) -> int:
        with self.lock:
            return int(self.conn.execute("PRAGMA data_version").fetchone()[0])

    # meta ------------------------------------------------------------------
    def meta(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                              "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def bump(self, key: str) -> None:
        with self.tx() as c:
            row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            value = int(row[0]) + 1 if row else 1
            c.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))

    # axiom -----------------------------------------------------------------
    def current_axiom(self) -> tuple[int, str] | None:
        with self.lock:
            row = self.conn.execute("SELECT id, text FROM axioms ORDER BY id DESC LIMIT 1").fetchone()
        return None if row is None else (int(row[0]), row[1])

    def set_axiom(self, text: str, previous: str | None) -> int:
        with self.tx() as c:
            at = _now()
            cur = c.execute("INSERT INTO axioms(text, set_at) VALUES(?, ?)", (text, at))
            detail = "set" if previous is None else f"changed from: {previous}"
            self._event(c, "axiom", None, [], detail, "", at)
            return int(cur.lastrowid or 0)

    def axioms(self) -> list[tuple[str, str]]:
        with self.lock:
            rows = self.conn.execute("SELECT text, set_at FROM axioms ORDER BY id").fetchall()
        return [(r[0], r[1]) for r in rows]

    # principles ------------------------------------------------------------
    def insert(self, c: sqlite3.Connection, p: Principle, vecs: tuple[array, array], embed_name: str,
               axiom_id: int | None) -> None:
        vec, ident = vecs
        c.execute(
            "INSERT INTO principles(id, name, importance, extent, use_case, related, status, created_at, "
            "axiom_id, learned_from, embedding, embed_name, dims, identity) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.id, p.name, p.importance, p.extent, p.use_case, json.dumps(p.related), "active",
             p.created_at, axiom_id, p.learned_from, _vec_blob(vec), embed_name, len(vec), _vec_blob(ident)),
        )

    def end(self, c: sqlite3.Connection, pid: str, status: str, reason: str, merged_into: str | None = None) -> None:
        c.execute("UPDATE principles SET status = ?, ended_at = ?, end_reason = ?, merged_into = ? "
                  "WHERE id = ? AND status = 'active'", (status, _now(), reason, merged_into, pid))

    def get(self, pid: str) -> Principle | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM principles WHERE id = ?", (pid,)).fetchone()
        return None if row is None else _principle_from_row(row)

    def get_many(self, ids: Sequence[str]) -> dict[str, Principle]:
        if not ids:
            return {}
        with self.lock:
            marks = ",".join("?" * len(ids))
            rows = self.conn.execute(f"SELECT * FROM principles WHERE id IN ({marks})", list(ids)).fetchall()
        return {r["id"]: _principle_from_row(r) for r in rows}

    def list(self, include_ended: bool) -> list[Principle]:
        sql = "SELECT * FROM principles"
        if not include_ended:
            sql += " WHERE status = 'active'"
        with self.lock:
            rows = self.conn.execute(sql + " ORDER BY created_at, id").fetchall()
        return [_principle_from_row(r) for r in rows]

    def active_vectors(self) -> list[tuple[str, bytes | None, bytes | None, str | None]]:
        """(id, recall vector, identity vector, embedding space) for every active principle."""
        with self.lock:
            rows = self.conn.execute("SELECT id, embedding, identity, embed_name FROM principles "
                                     "WHERE status = 'active' ORDER BY created_at, id").fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def count_active(self) -> int:
        with self.lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM principles WHERE status = 'active'").fetchone()[0])

    def set_embedding(self, pid: str, vecs: tuple[array, array], embed_name: str) -> None:
        vec, ident = vecs
        with self.lock:
            self.conn.execute("UPDATE principles SET embedding = ?, identity = ?, embed_name = ?, dims = ? "
                              "WHERE id = ?", (_vec_blob(vec), _vec_blob(ident), embed_name, len(vec), pid))

    def mark_shaped(self, ids: Sequence[str], questioned: bool) -> None:
        """Count an answer these principles shaped, and whether it was pushed back on."""
        if not ids:
            return
        with self.tx() as c:
            c.executemany("UPDATE principles SET shaped = shaped + 1, questioned = questioned + ? "
                          "WHERE id = ? AND status = 'active'", [(1 if questioned else 0, pid) for pid in ids])

    def set_review(self, pid: str, review: Mapping[str, Any]) -> None:
        with self.lock:
            self.conn.execute("UPDATE principles SET review = ? WHERE id = ?", (json.dumps(dict(review)), pid))

    def mark_injected(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        with self.tx() as c:
            at = _now()
            c.executemany("UPDATE principles SET times_injected = times_injected + 1, last_injected = ? "
                          "WHERE id = ?", [(at, pid) for pid in ids])

    # events ----------------------------------------------------------------
    @staticmethod
    def _event(c: sqlite3.Connection, kind: str, pid: str | None, sources: Sequence[str],
               detail: str, task: str, at: str | None = None) -> None:
        c.execute("INSERT INTO events(at, kind, principle_id, sources, detail, task) VALUES(?,?,?,?,?,?)",
                  (at or _now(), kind, pid, json.dumps(list(sources)), detail, task))

    def event(self, kind: str, pid: str | None = None, sources: Sequence[str] = (), detail: str = "",
              task: str = "") -> None:
        with self.tx() as c:
            self._event(c, kind, pid, sources, detail, task)

    # backup and restore ------------------------------------------------------
    _PRINCIPLE_COLUMNS = ("id", "name", "importance", "extent", "use_case", "related", "status", "created_at",
                          "axiom_id", "learned_from", "times_injected", "last_injected", "ended_at",
                          "end_reason", "merged_into", "shaped", "questioned", "review")
    _PRINCIPLE_DEFAULTS: Mapping[str, int] = {"times_injected": 0, "shaped": 0, "questioned": 0}
    _EVENT_COLUMNS = ("at", "kind", "principle_id", "sources", "detail", "task")

    def is_empty(self) -> bool:
        with self.lock:
            return not self.conn.execute("SELECT 1 FROM principles LIMIT 1").fetchone()

    def rows_for_backup(self) -> dict[str, Any]:
        """Everything in the database except the vectors, as plain JSON-safe rows."""
        with self.lock:
            meta = {r[0]: r[1] for r in self.conn.execute("SELECT key, value FROM meta")}
            axioms = [{"id": r[0], "text": r[1], "set_at": r[2]}
                      for r in self.conn.execute("SELECT id, text, set_at FROM axioms ORDER BY id")]
            principles = [dict(zip(self._PRINCIPLE_COLUMNS, r, strict=True)) for r in self.conn.execute(
                f"SELECT {', '.join(self._PRINCIPLE_COLUMNS)} FROM principles ORDER BY created_at, id")]
            events = [dict(zip(self._EVENT_COLUMNS, r, strict=True)) for r in self.conn.execute(
                f"SELECT {', '.join(self._EVENT_COLUMNS)} FROM events ORDER BY id")]
        for row in principles:
            row["related"] = _json_list(row["related"])
        for row in events:
            row["sources"] = _json_list(row["sources"])
        return {"meta": meta, "axioms": axioms, "principles": principles, "events": events}

    def restore(self, state: Mapping[str, Any]) -> dict[str, int]:
        """Write backed-up rows into this database, replacing whatever is in it. No vectors: the
        principles are embedded again the first time they are searched."""
        axioms = [r for r in state.get("axioms") or [] if str(r.get("text") or "").strip()]
        principles = list(state.get("principles") or [])
        events = list(state.get("events") or [])
        meta = dict(state.get("meta") or {})
        with self.tx() as c:
            for table in ("events", "principles", "axioms", "meta"):
                c.execute(f"DELETE FROM {table}")
            for row in axioms:
                c.execute("INSERT INTO axioms(id, text, set_at) VALUES(?, ?, ?)",
                          (row.get("id"), str(row["text"]), str(row.get("set_at") or _now())))
            for row in principles:
                values = dict(row)
                values["related"] = json.dumps(_json_list(values.get("related")))
                if isinstance(values.get("review"), dict):
                    values["review"] = json.dumps(values["review"])
                elif not isinstance(values.get("review"), str):
                    values["review"] = None
                c.execute(
                    f"INSERT INTO principles({', '.join(self._PRINCIPLE_COLUMNS)}) "
                    f"VALUES({', '.join('?' * len(self._PRINCIPLE_COLUMNS))})",
                    [values.get(col) if values.get(col) is not None else self._PRINCIPLE_DEFAULTS.get(col)
                     for col in self._PRINCIPLE_COLUMNS])
            for row in events:
                c.execute(f"INSERT INTO events({', '.join(self._EVENT_COLUMNS)}) "
                          f"VALUES({', '.join('?' * len(self._EVENT_COLUMNS))})",
                          [json.dumps(_json_list(row.get("sources"))) if col == "sources"
                           else row.get(col) for col in self._EVENT_COLUMNS])
            for key, value in meta.items():
                c.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (str(key), str(value)))
        return {"axioms": len(axioms), "principles": len(principles), "events": len(events)}

    def events(self, limit: int | None) -> list[Event]:
        sql = "SELECT at, kind, principle_id, sources, detail, task FROM events ORDER BY id DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        with self.lock:
            rows = self.conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            try:
                sources = [str(s) for s in json.loads(r[3] or "[]")]
            except ValueError:
                sources = []
            out.append(Event(at=r[0], kind=r[1], principle_id=r[2], sources=sources, detail=r[4], task=r[5]))
        return out

    def counts(self) -> dict[str, int]:
        with self.lock:
            status = dict(self.conn.execute("SELECT status, COUNT(*) FROM principles GROUP BY status").fetchall())
            kinds = dict(self.conn.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall())
        return {
            "active": int(status.get("active", 0)),
            "merged": int(status.get("merged", 0)),
            "removed": int(status.get("removed", 0)),
            "added": int(kinds.get("added", 0)),
            "merges": int(kinds.get("merged", 0)),
            "removals": int(kinds.get("removed", 0)),
            "covered": int(kinds.get("covered", 0)),
            "failed": int(kinds.get("failed", 0)),
        }


# ---------------------------------------------------------------------------
# Reading the model's list of changes. Small local models get the format
# slightly wrong in predictable ways, so this is forgiving.
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _loads_lenient(text: str) -> Any:
    s = _strip_thinking(text)
    fence = _FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()
    tries = [s]
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(s.rfind("}"), s.rfind("]"))
        if end > start:
            tries.append(s[start:end + 1])
    for candidate in list(tries):
        repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate.translate(_SMART_QUOTES))
        if repaired != candidate:
            tries.append(repaired)
    for candidate in tries:
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    for candidate in tries:   # Python-style literals: single quotes, True/None
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
    raise _ParseError(f"could not read JSON from: {_clip(text, 300)}")


def _parse_actions(text: str) -> list[dict[str, Any]]:
    data = _loads_lenient(text)
    if isinstance(data, dict):
        keyed = {str(k).strip().lower(): v for k, v in data.items()}
        if "actions" in keyed:
            data = keyed["actions"]
        elif "type" in keyed or "action" in keyed:
            data = [data]
        elif not data:
            data = []
        else:
            raise _ParseError(f"no actions in: {_clip(json.dumps(data), 300)}")
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise _ParseError(f"actions must be a list, got: {_clip(str(data), 200)}")
    return [a for a in data if isinstance(a, dict)]


_TYPE_ALIASES = {
    "add": "add", "new": "add", "create": "add", "addprinciple": "add", "write": "add",
    "covered": "covered", "cover": "covered", "keep": "covered", "exists": "covered",
    "alreadycovered": "covered", "coveredby": "covered", "held": "covered",
    "merge": "merge", "combine": "merge", "distill": "merge", "consolidate": "merge",
    "remove": "remove", "delete": "remove", "retire": "remove", "drop": "remove", "conflict": "remove",
    "stands": "stands", "stand": "stands", "sound": "stands", "affirm": "stands", "affirmed": "stands",
    "upheld": "stands", "uphold": "stands", "reviewed": "stands", "standsreviewed": "stands",
    "none": "none", "nothing": "none", "noop": "none", "noaction": "none", "nochange": "none",
    "skip": "none",
}


def _action_type(raw: dict[str, Any]) -> str:
    value = raw.get("type", raw.get("action", raw.get("op", "")))
    key = re.sub(r"[^a-z]", "", str(value).lower())
    return _TYPE_ALIASES.get(key, key or "(missing)")


def _norm_keys(obj: dict[str, Any]) -> dict[str, Any]:
    return {re.sub(r"[^a-z]", "", str(k).lower()): v for k, v in obj.items()}


def _pick(norm: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = norm.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _truthy(value: Any) -> bool:
    """A yes/no the model wrote: true, "true", "yes", 1."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1")
    return bool(value) if isinstance(value, (bool, int, float)) else False


def _shorten(text: str, limit: int) -> str:
    """One line of at most ``limit`` characters, cut at a word and ending in an ellipsis: a principle field is read
    as a sentence, so it is never cut in the middle the way ``_clip`` cuts long text."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit - 1]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def _as_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v).strip() for v in value if str(v).strip())
    elif isinstance(value, dict):
        value = "; ".join(f"{k}: {v}" for k, v in value.items())
    return _shorten(str(value), limit)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else re.split(r"[,;\n]", str(value))
    out: list[str] = []
    for item in items:
        text = _shorten(str(item), 60)
        if text and text.lower() not in (o.lower() for o in out):
            out.append(text)
    return out[:RELATED_MAX]


def _principle_fields(obj: Any) -> dict[str, Any] | None:
    """Pull principle fields out of whatever shape the model used."""
    if isinstance(obj, str):
        name = _as_text(obj, NAME_CHARS)
        return {"name": name, "importance": "", "extent": "", "use_case": "", "related": []} if name else None
    if not isinstance(obj, dict):
        return None
    norm = _norm_keys(obj)
    name = _pick(norm, "name", "principle", "statement", "rule", "title", "text", "lesson")
    if isinstance(name, dict):         # {"principle": {...}} nested one level deeper
        return _principle_fields(name)
    name = _as_text(name, NAME_CHARS)
    if not name:
        return None
    return {
        "name": name,
        "importance": _as_text(_pick(norm, "importance", "why", "whyitmatters", "whyimportant", "rationale"),
                               FIELD_CHARS),
        "extent": _as_text(_pick(norm, "extent", "scope", "limits", "limit", "boundary", "boundaries",
                                 "appliesto"), FIELD_CHARS),
        "use_case": _as_text(_pick(norm, "usecase", "usewhen", "when", "trigger", "applieswhen", "situation"),
                             FIELD_CHARS),
        "related": _as_list(_pick(norm, "related", "edges", "relatedconcepts", "concepts", "tags", "keywords")),
    }


_LABEL_RE = re.compile(r"^\s*[\[(]?\s*(?:p|principle)?\s*#?\s*(\d+)\s*[\])]?\s*[.:]?\s*$", re.I)


def _resolve_label(value: Any, labels: dict[str, Principle]) -> Principle | None:
    """Map what the model wrote ("P2", 2, "[P2]", a name) to a listed principle."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return labels.get(f"P{int(value)}")
    text = str(value).strip()
    m = _LABEL_RE.match(text)
    if m:
        return labels.get(f"P{int(m.group(1))}")
    low = text.lower().rstrip(".")
    for p in labels.values():
        if low in (p.id.lower(), p.name.lower().rstrip(".")):
            return p
    return None


_PLACEHOLDERS = {"", "n/a", "na", "none", "null", "name", "string", "text", "reason", "tbd", "todo",
                 "principle", "why", "example"}


def _is_placeholder(text: str) -> bool:
    """True for "...", "<reason>", "N/A" and the like: a model copying the example, not saying anything."""
    core = re.sub(r"[\s.\u2026\"'`<>\[\](){}_-]", "", text or "").lower()
    return core in _PLACEHOLDERS or sum(ch.isalpha() for ch in core) < 3


def _action_targets(raw: dict[str, Any], *keys: str) -> list[Any]:
    norm = _norm_keys(raw)
    for key in keys:
        value = norm.get(key)
        if value in (None, "", []):
            continue
        return list(value) if isinstance(value, (list, tuple)) else [value]
    return []


def _action_principle(raw: dict[str, Any]) -> dict[str, Any] | None:
    """The principle carried by an add or merge action, wherever the model put it."""
    norm = _norm_keys(raw)
    for key in ("principle", "newprinciple", "merged", "mergedprinciple", "combined", "result", "new"):
        value = norm.get(key)
        if isinstance(value, (dict, str)) and value:
            fields = _principle_fields(value)
            if fields:
                return fields
    loose = {k: v for k, v in raw.items()
             if re.sub(r"[^a-z]", "", str(k).lower()) not in
             ("type", "action", "op", "id", "ids", "targetids", "targets", "sources", "reason")}
    return _principle_fields(loose)


def _earlier_block(conversation: Sequence[dict[str, Any]] | None, messages: int = 4) -> str:
    """The last few messages before this task, for the Wisdom Core to read."""
    if not conversation:
        return ""
    lines = []
    for m in list(conversation)[-messages:]:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        lines.append(f"{'User' if role == 'user' else 'Agent'}: {_clip(content, EARLIER_CHARS)}")
    return "EARLIER IN THE CONVERSATION\n" + "\n".join(lines) + "\n\n" if lines else ""


def _previous_answer(conversation: Sequence[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], str]:
    """The conversation before the agent's last answer, and that answer: what the user's new message responds to."""
    messages = [m for m in (conversation or []) if m.get("role") in ("user", "assistant")]
    if messages and messages[-1].get("role") == "assistant" and str(messages[-1].get("content") or "").strip():
        return messages[:-1], str(messages[-1]["content"])
    return messages, ""


def _describe(values: Sequence[Any]) -> str:
    return ", ".join(repr(v) if not isinstance(v, str) else v for v in values) or "(none given)"


# ---------------------------------------------------------------------------
# The Wisdom Core
# ---------------------------------------------------------------------------
def embed_prompts_for(model: Any) -> dict[str, str]:
    """How an embedding model wants a search ("query"), a stored text ("document") and a plain comparison
    ("similarity") written, "{}" standing for the text: the backend's own ``embed_prompts`` if it has them,
    else the first entry in ``EMBED_PROMPTS`` found in its embedding model's name. Empty: no markers needed."""
    own = getattr(model, "embed_prompts", None)
    if isinstance(own, Mapping):
        return {str(k): str(v) for k, v in own.items() if "{}" in str(v)}
    name = _embed_name(model).lower()
    for family, prompts in EMBED_PROMPTS:
        if family in name:
            return dict(prompts)
    return {}


def _embed_as(model: Any, texts: Sequence[str], roles: Sequence[str]) -> list[array]:
    """Unit vectors for ``texts``, each written the way ``model`` wants for its role, in one request."""
    prompts = embed_prompts_for(model)
    sent = [prompts.get(role, "{}").replace("{}", str(t) if str(t).strip() else "(empty)")   # some APIs reject ""
            for t, role in zip(texts, roles, strict=True)]
    raw = model.embed(sent)
    if not isinstance(raw, (list, tuple)) or len(raw) != len(sent):
        got = len(raw) if isinstance(raw, (list, tuple)) else "no"
        raise ModelError(f"The embedding model returned {got} vectors for {len(sent)} texts")
    out = []
    for vec in raw:
        try:
            unit = _unit(vec)
        except (TypeError, ValueError):
            raise ModelError("The embedding model returned something that is not a list of numbers") from None
        if not len(unit):
            raise ModelError("The embedding model returned an empty vector")
        out.append(unit)
    if len({len(v) for v in out}) > 1:
        raise ModelError("The embedding model returned vectors of different sizes")
    return out


# The fixed set check_embeddings() measures. Each entry is a principle (with the situation that calls for it and
# the words a person would use), the same lesson in other words, a different lesson on the same subject, and a
# question the principle answers. Unrelated pairs are each principle beside one from another subject.
def _cp(name: str, use_case: str, related: list[str]) -> Principle:
    return Principle(id="check", name=name, use_case=use_case, related=related)


_CHECK_SET: tuple[dict[str, Any], ...] = (
    {"principle": _cp("Ask when the money will be needed before suggesting where to keep savings",
                      "someone asks where to keep money they are saving for something",
                      ["savings", "time horizon", "where to keep money"]),
     "reworded": "Before recommending where to put savings, find out when the person will need the money.",
     "related": "Keep an emergency fund in an account that can be reached the same day.",
     "question": "Where should I put the $5,000 I'm saving for a house?"},
    {"principle": _cp("Pay off the debt with the highest interest rate first",
                      "someone has several debts and asks which to pay first", ["debt", "interest rate", "repayment"]),
     "reworded": "Put extra money toward whichever debt charges the most interest before the others.",
     "related": "Check whether a loan charges a penalty for paying it off early.",
     "question": "I have a credit card at 24% and a car loan at 6%. Which should I pay down first?"},
    {"principle": _cp("Loyalty stops being a virtue when it requires betraying your conscience",
                      "someone asks whether to stay loyal to a friend or group that did something wrong",
                      ["loyalty", "conscience", "friendship"]),
     "reworded": "Being loyal is no longer virtuous once it demands that you act against your conscience.",
     "related": "Loyalty is earned through steady support over time, not demanded.",
     "question": "Is it wrong to stay loyal to a friend who lied to me?"},
    {"principle": _cp("Honest feedback is kinder in the long run than comfortable silence",
                      "someone wonders whether to tell a friend or colleague an uncomfortable truth",
                      ["honesty", "feedback", "hard conversations"]),
     "reworded": "Telling someone a hard truth does them more good over time than keeping quiet to spare them.",
     "related": "Praise people in public and criticise them in private.",
     "question": "My friend's business plan has a big flaw. Should I tell her even though she is excited?"},
    {"principle": _cp("Get the exact error message before suggesting a fix",
                      "a customer reports that an app or device keeps failing",
                      ["error message", "troubleshooting", "crash"]),
     "reworded": "Before proposing a solution, find out precisely what the error message says.",
     "related": "Tell the customer how long a fix will take before starting it.",
     "question": "The app keeps crashing when I log in. What should I do?"},
    {"principle": _cp("Restart the router before escalating a connection fault",
                      "someone's internet connection has stopped working", ["internet", "router", "connection"]),
     "reworded": "Power-cycle the router first, and only escalate the connection problem if that does not fix it.",
     "related": "Check whether other devices on the same network are also offline.",
     "question": "My Wi-Fi stopped working this morning. What should I try?"},
    {"principle": _cp("Season food a little at a time and taste as you go",
                      "someone is cooking and unsure how much salt or seasoning to add",
                      ["seasoning", "salt", "cooking"]),
     "reworded": "Add salt gradually and keep tasting rather than seasoning everything at once.",
     "related": "Let meat rest for a few minutes after cooking before cutting it.",
     "question": "How much salt should I put in this soup?"},
    {"principle": _cp("Increase training load gradually to avoid injury",
                      "someone wants to train harder or start a new exercise routine",
                      ["exercise", "training", "injury"]),
     "reworded": "Build up exercise intensity slowly so the body can adapt without getting hurt.",
     "related": "Sleep matters as much as training for building strength.",
     "question": "Can I go from running twice a week to running every day?"},
    {"principle": _cp("Give feedback on the work, not on the person",
                      "a manager needs to tell someone their work fell short", ["feedback", "management", "criticism"]),
     "reworded": "When giving criticism, talk about what was done rather than about who the person is.",
     "related": "Agree on clear goals with each team member at the start of a project.",
     "question": "How do I tell an employee his report was sloppy without upsetting him?"},
    {"principle": _cp("Put the main point in the first sentence",
                      "someone is writing an email, report or article and wants it to be read",
                      ["writing", "clarity", "structure"]),
     "reworded": "Lead with the most important idea instead of building up to it.",
     "related": "Cut every word that does not add meaning.",
     "question": "How should I start this email to my boss about the delay?"},
    {"principle": _cp("Practise recalling material instead of rereading it",
                      "someone is studying for an exam and wants to remember more", ["studying", "memory", "exam"]),
     "reworded": "Testing yourself on what you studied works better than reading it again.",
     "related": "Spread study sessions over several days rather than cramming.",
     "question": "What is the best way to study for my history exam next week?"},
    {"principle": _cp("Set limits for children calmly and keep them consistent",
                      "a parent is dealing with a child who keeps testing the rules",
                      ["parenting", "children", "discipline"]),
     "reworded": "Hold the same boundaries for kids every time, without losing your temper.",
     "related": "Praise a child's effort rather than their talent.",
     "question": "My six-year-old keeps refusing to go to bed. What should I do?"},
    {"principle": _cp("Write a failing test before fixing a bug",
                      "a developer has found a bug and is about to fix it", ["bug", "testing", "debugging"]),
     "reworded": "Reproduce the bug with a test that fails before you change any code.",
     "related": "Keep each commit to one logical change.",
     "question": "I found a bug where totals are off by one. How should I go about fixing it?"},
    {"principle": _cp("Keep copies of your passport and documents separate from the originals",
                      "someone is preparing for a trip abroad", ["travel", "passport", "documents"]),
     "reworded": "Store a copy of your passport somewhere other than where you keep the passport itself.",
     "related": "Learn a few words of the local language before you travel.",
     "question": "What should I sort out before my first trip to Japan?"},
    {"principle": _cp("Water plants deeply and less often rather than a little every day",
                      "someone is caring for a garden or houseplants and unsure how to water them",
                      ["watering", "plants", "garden"]),
     "reworded": "Give plants a thorough soaking now and then instead of a light sprinkle daily.",
     "related": "Plant according to how much sun each part of the garden gets.",
     "question": "How often should I water my tomato plants?"},
)
_CHECK_UNRELATED_OFFSET = 7    # principle i is set beside principle i + 7: always another subject
CALIBRATION_BOUNDS = (0.05, 0.99)   # a measured level is kept inside these

# Keeping the calibration true over time. The fingerprint is one fixed sentence, embedded and kept: if the same
# model name later embeds it differently, the model changed underneath (new weights pulled under the same tag, a
# new runtime), so every principle is embedded again and the model measured again. The agent's own principles
# are used to calibrate once it holds enough of them, and again as it learns: its subject is what matters.
FINGERPRINT_TEXT = "Checking the facts before acting prevents most costly mistakes."
FINGERPRINT_EVERY = 1         # lessons between fingerprint checks: every one (one short request; a model that
                              # changed and went unnoticed for 10 lessons made wrong decisions in simulation)
FINGERPRINT_MATCH = 0.9995    # a fingerprint at least this alike is the same model (allows for float noise)
DOMAIN_MIN = 8                # principles held before the agent's own are used to calibrate
DOMAIN_EVERY = 25             # new principles between calibrations on the agent's own principles
DOMAIN_SAMPLE = 6             # principles reworded each time
DOMAIN_PAIRS_MIN = 4          # usable pairs needed for a calibration on the agent's own principles

# A principle's standing: the record of how the answers it shaped were received. A principle shapes an answer
# when it is among the few most relevant principles injected for it. The user's next message says how that
# answer was received (the first look reads it: a pushback, or carrying on), and an application can say so
# outright (``success``). A principle is IN DOUBT when its record would be surprising for a sound principle:
# the chance of that many pushbacks, if its answers were pushed back on as often as this user's other answers,
# is below DOUBT_ALPHA. It is then reviewed in the next Wisdom Synthesis that lists it, where the model removes
# it, revises it, or finds that it stands. Nothing is removed on a count alone.
SHAPING_TOP = 4               # the most relevant principles injected for an answer are the ones that shaped it
DOUBT_MIN_QUESTIONED = 3      # pushbacks (since its last review) before a principle can be in doubt
DOUBT_ALPHA = 0.02            # how unlikely its record must be, for a sound principle, to put it in doubt
PROVEN_SHAPED = 8             # answers shaped, with at most a tenth questioned, before a principle reads as proven


def _percentile(values: Sequence[float], pct: float) -> float:
    """The ``pct`` percentile (0-100) of ``values``, interpolated."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    at = (len(ordered) - 1) * pct / 100
    low = math.floor(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _levels_from(reworded: Sequence[float], same: Sequence[float]) -> tuple[str, dict[str, float] | None]:
    """The separation of rewordings from related lessons, and the levels that fit it (None when they overlap).
    Clear: 90% of rewordings above 90% of related lessons; partial: three quarters of each apart."""
    low, high = _percentile(reworded, 10), _percentile(same, 90)
    q1_reworded, q3_same = _percentile(reworded, 25), _percentile(same, 75)
    lo_bound, hi_bound = CALIBRATION_BOUNDS
    if high < low:
        separation, level = "clear", (high + low) / 2
    elif q3_same < q1_reworded:
        separation, level = "partial", (q3_same + q1_reworded) / 2
    else:
        return "none", None
    level = round(min(max(level, lo_bound), hi_bound), 2)
    covered = round(min(max(_percentile(reworded, 90), level + 0.03), hi_bound), 2)
    return separation, {"new_ground": level, "combine": level, "covered": covered}


def check_embeddings(model: Any, levels: Levels | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Measure an embedding model and work out the levels that fit it: its calibration.

    One embedding request over a fixed set of 15 principles from different subjects: each beside the same
    lesson in other words, beside a different lesson on the same subject, and beside a principle from another
    subject, plus 15 questions each looked up among the 15 principles. Embedding models do not share a scale
    (one scores a rewording 0.95, another 0.75), so the levels that compare principles are set from this.

    - ``separation`` is "clear" when 90% of rewordings score above 90% of the related lessons, "partial" when
      three quarters of each are apart, and "none" otherwise.
    - ``calibrated`` holds the measured ``new_ground`` and ``combine`` (the midpoint between related lessons
      and rewordings) and ``covered`` (where the closest rewordings sit), or None when there is no separation.
    - ``recall`` says how many questions found their own principle first.
    - ``verdict`` and ``recall_verdict`` say it in words.

    ``levels`` (or the defaults) is only reported beside the result. ``model`` is a backend or an Ollama model
    name. Nothing is stored.
    """
    model = resolve_model(model)
    levels = levels if isinstance(levels, Levels) else Levels.from_dict(levels)
    n = len(_CHECK_SET)
    texts = ([e["principle"].identity_text() for e in _CHECK_SET] + [e["reworded"] for e in _CHECK_SET]
             + [e["related"] for e in _CHECK_SET] + [e["principle"].recall_text() for e in _CHECK_SET]
             + [e["question"] for e in _CHECK_SET])
    roles = ["similarity"] * (3 * n) + ["document"] * n + ["query"] * n
    vecs = _embed_as(model, texts, roles)
    base, reword, related, docs, questions = (vecs[i * n:(i + 1) * n] for i in range(5))
    reworded = [_dot(base[i], reword[i]) for i in range(n)]
    same = [_dot(base[i], related[i]) for i in range(n)]
    unrelated = [_dot(base[i], base[(i + _CHECK_UNRELATED_OFFSET) % n]) for i in range(n)]
    right, next_best, found = [], [], 0
    for i, q in enumerate(questions):
        scores = [_dot(q, d) for d in docs]
        right.append(scores[i])
        next_best.append(max(s for j, s in enumerate(scores) if j != i))
        found += scores[i] > next_best[-1]

    low, high = _percentile(reworded, 10), _percentile(same, 90)
    mid_reworded, mid_same = _percentile(reworded, 50), _percentile(same, 50)
    q1_reworded, q3_same = _percentile(reworded, 25), _percentile(same, 75)
    separation, calibrated = _levels_from(reworded, same)
    level = None if calibrated is None else calibrated["combine"]
    if separation == "clear":
        verdict = (f"Rewordings of one lesson measured {low:.2f} or more and different lessons on the same subject "
                   f"{high:.2f} or less, so this model tells them apart. New ground and Combine fit it at "
                   f"{level:.2f}, and Already covered at {(calibrated or {}).get('covered', 0):.2f}.")
    elif separation == "partial":
        verdict = (f"Most rewordings measured {q1_reworded:.2f} or more and most related lessons {q3_same:.2f} or "
                   f"less, but some of each overlap. New ground and Combine are set to {level:.2f}, between the two; "
                   "a few near-duplicates may slip through and a few related lessons may be combined.")
    else:
        verdict = (f"On this model, different lessons on the same subject and rewordings of one lesson overlap too "
                   f"much (a quarter of the related lessons scored {q3_same:.2f} or more, and a quarter of the "
                   f"rewordings {q1_reworded:.2f} or less), so no level tells them apart. The fallback levels stay "
                   "in use; a different embedding model would do better.")
    recall_verdict = (f"The right principle came first for {found} of {n} questions (about "
                      f"{sum(right) / n:.2f} against {sum(next_best) / n:.2f} for the next best).")
    return {"model": _embed_name(model), "prompts": bool(embed_prompts_for(model)), "count": n,
            "reworded": reworded, "same_subject": same, "unrelated": unrelated,
            "lowest_reworded": low, "highest_same_subject": high,
            "median_reworded": mid_reworded, "median_same_subject": mid_same,
            "separation": separation, "separates": separation == "clear", "calibrated": calibrated,
            "suggested": None if calibrated is None else calibrated["combine"],
            "levels": {k: getattr(levels, k) for k in AUTO_LEVELS},
            "recall": {"right": right, "next_best": next_best, "found": found, "of": n},
            "verdict": verdict, "recall_verdict": recall_verdict}


class WisdomCore:
    """The framework on its own: Core Axiom, principles, injection and synthesis.

    Use it inside an agent you already have::

        core = WisdomCore(core_axiom="...", model=Ollama("llama3.1"))
        prompt = core.inject(task)          # at every decision
        ...
        core.synthesize(task, result)       # after every task

    ``Agent`` wraps it in a ready-made chat loop.

    Args:
        core_axiom: The one sentence the agent serves. Required the first time a
            database is used; after that it is read from the database. Passing
            a different sentence later changes the axiom (the old one is kept in
            the history, and principles that conflict with the new one are
            removed as they come up).
        model: None (Ollama, choosing models automatically), an Ollama model
            name, or a backend object such as Ollama, OpenAICompatible or
            FunctionModel.
        path: The SQLite file that holds this agent's principles. Use one file
            per agent. ":memory:" keeps everything in memory.
        context: How many principles are injected at each decision (default 12).
            A shortcut for ``levels.context``.
        levels: The semantic levels (see ``Levels``): how close counts as
            already covered, whether every task must write a principle, and how
            many are injected. The object is kept as given, so changing it later
            changes what this Wisdom Core does from the next task on.
        on_event: Optional function called with a dict for every step the
            Wisdom Core takes: recall, each model call (and every streamed
            token), each synthesis stage and its result. Use it to watch or log
            the framework working. An ordinary exception it raises is logged and
            ignored; KeyboardInterrupt-style exceptions (BaseException) pass
            through, which is how a caller can stop a run.
    """

    def __init__(self, core_axiom: str | None = None, model: Any = None, *, path: str = "wisdom.db",
                 context: int | None = None, levels: Levels | Mapping[str, Any] | None = None,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.levels = levels if isinstance(levels, Levels) else Levels.from_dict(levels)
        if context is not None:
            self.levels.context = _whole(context, "context", 1)
        self.model = resolve_model(model)
        self.path = path
        self.on_event = on_event
        self._lock = threading.RLock()
        self._index: _Index | None = None
        self._index_version: int | None = None
        self._index_space: str | None = None
        self._kw_cache: tuple[int, set[str] | None] | None = None
        self.last_stats: dict[str, Any] = {}
        self._calib: dict[str, Any] | None = None
        self._fp_lessons: int | None = None      # lessons since the fingerprint was checked (None: not yet)
        self._store = _Store(path)
        try:
            current = self._store.current_axiom()
            axiom = (core_axiom or "").strip()
            if axiom:
                if current is None or current[1] != axiom:
                    if current is not None:
                        log.warning("The Core Axiom for %s changed. Principles learned under the old one are "
                                    "kept; any that conflict with the new one are removed as they come up.", path)
                    self._store.set_axiom(axiom, None if current is None else current[1])
            elif current is None:
                raise WisdomCoreError('A Core Axiom is required the first time a database is used: '
                                      'WisdomCore(core_axiom="...")')
        except BaseException:
            self._store.close()
            raise

    def __repr__(self) -> str:
        return f"WisdomCore(path={self.path!r}, model={self.model!r})"

    @property
    def context(self) -> int:
        """How many principles are injected at each decision (``levels.context``)."""
        return self.levels.context

    @context.setter
    def context(self, value: int) -> None:
        self.levels.context = _whole(value, "context", 1)

    def __enter__(self) -> WisdomCore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the database."""
        self._store.close()

    # -- watching it work ----------------------------------------------------
    def _emit(self, event: str, **fields: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event({"type": event, **fields})
        except Exception:  # a broken watcher must never break the agent
            log.warning("on_event raised while handling %s", event, exc_info=True)

    def _model_kwargs(self) -> set[str] | None:
        """Which optional keywords the model's chat() accepts (None: any)."""
        key = id(self.model)
        if self._kw_cache is None or self._kw_cache[0] != key:
            self._kw_cache = (key, _accepted_kwargs(self.model.chat))
        return self._kw_cache[1]

    def _chat(self, purpose: str, messages: Sequence[dict[str, Any]], *, json_mode: bool = False,
              temperature: float | None = None, think: bool | None = None) -> str:
        """Every model call goes through here, so each one can be watched and streamed."""
        call = uuid.uuid4().hex[:12]
        accepted = self._model_kwargs()
        stats: dict[str, Any] = {}

        def token(text: str, kind: str = "content") -> None:
            self._emit("model.token", call=call, purpose=purpose, text=text, kind=kind)

        kwargs: dict[str, Any] = {"json_mode": json_mode, "temperature": temperature}
        for key, value in (("on_token", token if self.on_event else None), ("stats", stats), ("think", think)):
            if value is not None and (accepted is None or key in accepted):
                kwargs[key] = value
        self._emit("model.start", call=call, purpose=purpose, model=str(getattr(self.model, "name", "")),
                   messages=[dict(m) for m in messages], json_mode=json_mode, temperature=temperature,
                   think=think)
        started = time.monotonic()
        try:
            reply = self.model.chat(messages, **kwargs)
        except BaseException as exc:
            self._emit("model.done", call=call, purpose=purpose, reply=None,
                       error=f"{type(exc).__name__}: {exc}", ms=_ms(started), stats=stats)
            raise
        self._emit("model.done", call=call, purpose=purpose, reply=reply, error=None, ms=_ms(started),
                   stats=stats)
        self.last_stats = stats         # read by Agent.respond for the reasoning behind the answer
        return reply

    # -- the Core Axiom -----------------------------------------------------
    @property
    def core_axiom(self) -> str:
        current = self._store.current_axiom()
        return current[1] if current else ""

    def set_core_axiom(self, text: str) -> None:
        """Change the Core Axiom. The old one stays in ``axiom_history()``."""
        text = (text or "").strip()
        if not text:
            raise ValueError("The Core Axiom cannot be empty")
        with self._lock:
            current = self._store.current_axiom()
            if current is None or current[1] != text:
                self._store.set_axiom(text, None if current is None else current[1])

    def axiom_history(self) -> list[tuple[str, str]]:
        """Every Core Axiom this database has had, oldest first, as (text, set_at)."""
        return self._store.axioms()

    # -- reading the principles ---------------------------------------------
    def principles(self, *, include_ended: bool = False) -> list[Principle]:
        """Principles in the order they were learned. ``include_ended`` adds merged and removed ones."""
        return self._store.list(include_ended)

    def get(self, principle_id: str) -> Principle | None:
        return self._store.get(principle_id)

    def history(self, limit: int | None = 50) -> list[Event]:
        """What the Wisdom Core has done, newest first."""
        return self._store.events(limit)

    def stats(self) -> dict[str, int]:
        out = self._store.counts()
        out["syntheses"] = int(self._store.meta("syntheses") or 0)
        out["held_at_first_look"] = int(self._store.meta("held_at_first_look") or 0)
        out["answers_reacted"] = int(self._store.meta("reactions_seen") or 0)
        out["answers_questioned"] = int(self._store.meta("reactions_questioned") or 0)
        out["in_doubt"] = sum(1 for p in self._store.list(False) if self.standing(p)["in_doubt"])
        return out

    # -- backup and restore ---------------------------------------------------
    def export_state(self) -> dict[str, Any]:
        """Everything this database holds, as plain data that ``import_state`` can rebuild elsewhere:
        the Core Axiom and every axiom before it, every principle as written (including merged and
        removed ones), and the log of what the Wisdom Core did. The vectors are left out; they are made
        again from the text by whichever embedding model the new copy uses."""
        with self._lock:
            state = self._store.rows_for_backup()
            return {"state_version": STATE_VERSION, "wisdom_core": __version__, "exported_at": _now(),
                    "axiom": self.core_axiom, "levels": self.levels.to_dict(),
                    "counts": self.stats(), **state}

    def import_state(self, state: Mapping[str, Any], *, replace: bool = False) -> dict[str, int]:
        """Rebuild an exported state in this database. The principles keep their text, their history and
        their ids; their vectors are made again the first time they are searched. Refuses a database that
        already holds principles unless ``replace`` is set, which clears it first."""
        if not isinstance(state, Mapping):
            raise WisdomCoreError("that is not an exported Wisdom Core state")
        try:
            version = int(state.get("state_version"))    # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise WisdomCoreError("that is not an exported Wisdom Core state") from None
        if version > STATE_VERSION:
            raise WisdomCoreError(f"that backup was written by a newer Wisdom Core (state version {version}; "
                                  f"this one reads {STATE_VERSION})")
        with self._lock:
            if not replace and not self._store.is_empty():
                raise WisdomCoreError("this database already holds principles; import into a new one")
            written = self._store.restore(state)
            self._index = None
            self._index_version = None
            if self._store.current_axiom() is None:
                axiom = str(state.get("axiom") or "").strip()
                if not axiom:
                    raise WisdomCoreError("that backup has no Core Axiom")
                self._store.set_axiom(axiom, None)
                written["axioms"] = 1
            self._emit("principles.changed")
            return written

    def remove(self, principle_id: str, reason: str = "removed by hand") -> Principle:
        """Remove a principle yourself. It stays in the log with the reason."""
        with self._lock:
            p = self._store.get(principle_id)
            if p is None:
                raise KeyError(principle_id)
            if p.status != "active":
                raise WisdomCoreError(f"principle {principle_id} is already {p.status}")
            reason = reason.strip() or "removed by hand"
            with self._store.tx() as c:
                self._store.end(c, p.id, "removed", reason)
                self._store._event(c, "removed", p.id, [], f"{p.name} ({reason})", "")
            if self._index is not None:
                self._index.drop(p.id)
            p.status, p.end_reason = "removed", reason
            return p

    # -- embeddings and the search index -------------------------------------
    def embed_prompts(self) -> dict[str, str]:
        """How the embedding model in use wants each kind of text written (see ``embed_prompts_for``)."""
        return embed_prompts_for(self.model)

    def _space(self) -> str:
        """The embedding space a stored vector belongs to: the model, plus how principles are embedded."""
        return f"{_embed_name(self.model)} (layout {EMBED_LAYOUT})"

    def _embed(self, texts: Sequence[str], purpose: str = "", role: str | None = None,
               roles: Sequence[str] | None = None) -> list[array]:
        """Embed texts, each written the way the model wants for its ``role`` (one for all, or ``roles``)."""
        kinds = list(roles) if roles is not None else [role or "document"] * len(texts)
        if not texts:
            return []
        started = time.monotonic()
        out = _embed_as(self.model, texts, kinds)
        self._emit("embed", purpose=purpose, count=len(texts), ms=_ms(started),
                   model=str(getattr(self.model, "embed_name", "")))
        return out

    def _embed_principles(self, principles: Sequence[Principle], purpose: str) -> list[tuple[array, array]]:
        """Both vectors of each principle, in one request: what it is about (recall) and what it says (identity)."""
        if not principles:
            return []
        texts = [p.recall_text() for p in principles] + [p.identity_text() for p in principles]
        roles = ["document"] * len(principles) + ["similarity"] * len(principles)
        vecs = self._embed(texts, purpose, roles=roles)
        return list(zip(vecs[:len(principles)], vecs[len(principles):], strict=True))

    def _reembed(self, ids: Sequence[str], name: str) -> None:
        found = self._store.get_many(list(ids))
        todo = [found[i] for i in ids if i in found]
        if todo:
            log.info("Embedding %d principle(s) with %s", len(todo), name)
            self._emit("reindex", count=len(todo), space=_embed_name(self.model))
        for start in range(0, len(todo), EMBED_BATCH):
            batch = todo[start:start + EMBED_BATCH]
            for p, vecs in zip(batch, self._embed_principles(batch, "reindex"), strict=True):
                self._store.set_embedding(p.id, vecs, name)

    def _ensure_index(self) -> _Index:
        """The search index, rebuilt if another process changed the database or the embedding model changed."""
        with self._lock:
            version = self._store.data_version()
            space = self._space()
            if self._index is not None and version == self._index_version and space == self._index_space:
                return self._index
            index = _Index()
            rows = self._store.active_vectors()
            if rows:
                stale = [pid for pid, blob, ident, row_space in rows if not blob or not ident or row_space != space]
                if stale:
                    self._reembed(stale, space)
                    rows = self._store.active_vectors()
                if len({len(b) for _, blob, ident, _ in rows for b in (blob, ident) if b}) > 1:
                    self._reembed([r[0] for r in rows], space)
                    rows = self._store.active_vectors()
                for pid, blob, ident, _ in rows:
                    vec, ivec = _blob_vec(blob), _blob_vec(ident)
                    if vec is not None and ivec is not None:
                        index.put(pid, vec, ivec)
            self._index = index
            self._index_version = self._store.data_version()
            self._index_space = space
            return index

    def _sized(self, dims: int) -> _Index:
        """The index, re-embedded first if the embedding model now returns vectors of another size."""
        index = self._ensure_index()
        if len(index) and dims != index.dims:
            log.warning("The embedding size changed (%s -> %s); re-embedding every principle.", index.dims, dims)
            self._reembed(list(index.ids), self._space())
            self._index = None
            index = self._ensure_index()
            if len(index) and dims != index.dims:
                raise ModelError("The embedding model returns vectors of inconsistent sizes")
        return index

    def _query_vectors(self, texts: Sequence[str], purpose: str = "", role: str = "query") -> list[array]:
        """Embed texts to search with (``role`` "query"), or to compare with what principles say ("similarity"),
        re-embedding stored principles if the vector size changed."""
        self._ensure_index()
        vecs = self._embed(texts, purpose, role)
        if vecs:
            self._sized(len(vecs[0]))
        return vecs

    def _principle_vectors(self, p: Principle, purpose: str = "principle") -> tuple[array, array]:
        """Both vectors of a principle about to be stored or measured."""
        self._ensure_index()
        (vecs,) = self._embed_principles([p], purpose)
        self._sized(len(vecs[0]))
        return vecs

    def check_embeddings(self) -> dict[str, Any]:
        """Measure the embedding model in use against this core's levels (see ``check_embeddings``)."""
        with self._lock:
            return check_embeddings(self.model, self.levels)

    # -- calibration: the levels that fit the embedding model -------------------
    def calibration(self, *, measure: bool = True) -> dict[str, Any] | None:
        """The calibration of the embedding model in use, as kept in this database: ``space``, ``at``,
        ``levels`` (the measured new_ground, covered and combine in use, or None when the model could not be
        calibrated), ``source`` ("general" or "agent": from the agent's own principles), ``general`` and
        ``domain`` (each calibration's own result), ``separation``, ``verdict`` and the rest of
        ``check_embeddings``' result. When there is none for this model yet and ``measure`` is set, it is
        measured now (one embedding request). None if it could not be measured (the model could not be
        reached); that is tried again on the next call. With ``measure`` off it only reads, and never waits for
        a lesson in progress."""
        saved = self._saved_calibration()
        if saved is not None or not measure:
            return saved
        with self._lock:
            saved = self._saved_calibration()
            return saved if saved is not None else self.calibrate()

    def _saved_calibration(self) -> dict[str, Any] | None:
        try:
            space = self._space()
        except ModelError:            # which model is in use cannot be known yet (the service is down)
            return None
        cached = self._calib
        if cached is not None and cached.get("space") == space:
            return cached
        try:
            saved = json.loads(self._store.meta("calibration") or "null")
        except ValueError:
            saved = None
        if isinstance(saved, dict) and saved.get("space") == space:
            self._calib = saved
            return saved
        return None

    def calibrate(self) -> dict[str, Any] | None:
        """Measure the embedding model in use now with the general check (``check_embeddings``) and keep the
        result as this database's calibration, replacing any before it (a calibration on the agent's own
        principles starts again from here). Returns it, or None if the model could not be reached."""
        with self._lock:
            space = self._space()
            started = time.monotonic()
            try:
                result = check_embeddings(self.model, self.levels)
            except ModelError as exc:
                log.warning("The embedding model could not be calibrated: %s", exc)
                self._emit("calibrated", ok=False, error=str(exc), space=_embed_name(self.model))
                return None
            general = {"levels": result["calibrated"], "separation": result["separation"],
                       "verdict": result["verdict"], "at": _now()}
            calib = {**result, "space": space, "at": general["at"], "levels": result["calibrated"],
                     "source": "general", "general": general, "domain": None, "domain_added": None}
            measured = calib["levels"]
            detail = (f"{_embed_name(self.model)}: " + (
                f"new ground and combine {measured['new_ground']:.2f}, already covered {measured['covered']:.2f}"
                if measured else "could not be calibrated; the fallback levels are used") + f". {calib['verdict']}")
            self._keep_calibration(calib, detail)
            self._emit("calibrated", ok=True, calibration=calib, source="general", space=_embed_name(self.model),
                       ms=_ms(started))
            if self._fp_lessons is None:          # keep the fingerprint of the model just measured
                try:
                    self.check_fingerprint()
                except ModelError as exc:
                    log.warning("The embedding model's fingerprint could not be taken: %s", exc)
            return calib

    def _keep_calibration(self, calib: dict[str, Any], detail: str) -> None:
        self._store.set_meta("calibration", json.dumps(calib))
        self._store.event("calibrated", detail=_clip(detail, 1000))
        self._calib = calib

    def calibrate_on_principles(self) -> dict[str, Any] | None:
        """Calibrate on this agent's own principles: the chat model rewords a sample of them and writes a
        different lesson on the same subject for each, and the levels are set from how the embedding model
        scores those pairs. What an agent learns about sits closer together than the general check's mixed
        subjects, so this is what the levels should fit once it holds ``DOMAIN_MIN`` principles. When its
        pairs overlap, the general calibration stays in use. Returns the calibration, or None when there is
        none to build on or too few principles. Model problems are raised (``ModelError``)."""
        with self._lock:
            calib = self.calibration()
            held = sorted(self.principles(), key=lambda p: (p.created_at, p.id))
            if calib is None or len(held) < DOMAIN_MIN:
                return None
            started = time.monotonic()
            step = len(held) / min(DOMAIN_SAMPLE, len(held))
            sample = [held[int(i * step)] for i in range(min(DOMAIN_SAMPLE, len(held)))]   # spread over its life
            labels = {f"P{i}": p for i, p in enumerate(sample, 1)}
            messages = [{"role": "system", "content": DOMAIN_SYSTEM},
                        {"role": "user", "content": DOMAIN_USER.format(
                            principles="\n".join(f"{label}. {p.name}" for label, p in labels.items()))}]
            data: Any = None
            reply = ""
            for attempt in range(2):
                reply = self._chat("calibrate", messages, json_mode=True, temperature=0.3, think=False)
                try:
                    data = _loads_lenient(reply)
                    break
                except _ParseError:
                    if attempt:
                        raise
                    messages = messages + [{"role": "assistant", "content": _clip(reply, 4000) or "(empty)"},
                                           {"role": "user", "content": JSON_RETRY}]
            items = data.get("pairs") if isinstance(data, dict) else data
            pairs: list[tuple[Principle, str, str]] = []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                norm = _norm_keys(item)
                target = _resolve_label(_pick(norm, "id", "label", "principle"), labels)
                reworded = _as_text(_pick(norm, "reworded", "rewording", "same"), NAME_CHARS)
                related = _as_text(_pick(norm, "related", "different", "other"), NAME_CHARS)
                if (target is None or any(t.id == target.id for t, _, _ in pairs)
                        or _is_placeholder(reworded) or _is_placeholder(related)):
                    continue
                said = {target.name.lower().strip(" ."), reworded.lower().strip(" ."), related.lower().strip(" .")}
                if len(said) == 3:                 # a copy is no test
                    pairs.append((target, reworded, related))
            added = self._store.counts()["added"]
            if len(pairs) < DOMAIN_PAIRS_MIN:
                calib["domain_added"] = added - DOMAIN_EVERY + 5     # try again after a few more principles
                self._keep_calibration(calib, f"own principles: only {len(pairs)} usable pairs came back, so the "
                                              f"{calib['source']} calibration stays in use")
                self._emit("calibrated", ok=False, error=f"only {len(pairs)} usable pairs", source="agent",
                           space=_embed_name(self.model))
                return calib
            texts = [t for p, rw, rel in pairs for t in (p.identity_text(), rw, rel)]
            vecs = self._query_vectors(texts, "calibrate", role="similarity")
            rw_scores = [_dot(vecs[3 * i], vecs[3 * i + 1]) for i in range(len(pairs))]
            rel_scores = [_dot(vecs[3 * i], vecs[3 * i + 2]) for i in range(len(pairs))]
            separation, levels = _levels_from(rw_scores, rel_scores)
            n = len(pairs)
            if levels is not None:
                verdict = (f"On {n} of the agent's own principles, rewordings measured {min(rw_scores):.2f} to "
                           f"{max(rw_scores):.2f} and related lessons {min(rel_scores):.2f} to {max(rel_scores):.2f} "
                           f"({separation} separation). New ground and Combine are set to {levels['combine']:.2f} "
                           f"and Already covered to {levels['covered']:.2f} for what this agent knows.")
            else:
                verdict = (f"On {n} of the agent's own principles, rewordings ({min(rw_scores):.2f} to "
                           f"{max(rw_scores):.2f}) and related lessons ({min(rel_scores):.2f} to "
                           f"{max(rel_scores):.2f}) overlap, so the general calibration stays in use.")
            calib["domain"] = {"at": _now(), "held": len(held), "separation": separation, "levels": levels,
                               "verdict": verdict, "reworded": rw_scores, "same_subject": rel_scores,
                               "pairs": [{"principle": p.name, "reworded": rw, "related": rel,
                                          "reworded_score": a, "related_score": b}
                                         for (p, rw, rel), a, b in zip(pairs, rw_scores, rel_scores, strict=True)]}
            calib["domain_added"] = added
            calib["levels"] = levels if levels is not None else calib["general"]["levels"]
            calib["source"] = "agent" if levels is not None else "general"
            self._keep_calibration(calib, f"own principles: {verdict}")
            self._emit("calibrated", ok=True, calibration=calib, source="agent", space=_embed_name(self.model),
                       ms=_ms(started))
            return calib

    def _maybe_calibrate_on_principles(self) -> None:
        """After a lesson: calibrate on the agent's own principles when it first holds enough of them, and again
        every ``DOMAIN_EVERY`` new ones. Only for automatic levels; a failure waits for the next time."""
        if not self.levels.automatic():
            return
        calib = self.calibration(measure=False)
        if calib is None or self._store.count_active() < DOMAIN_MIN:
            return
        last = calib.get("domain_added")
        if last is not None and self._store.counts()["added"] - int(last) < DOMAIN_EVERY:
            return
        try:
            self.calibrate_on_principles()
        except (ModelError, _ParseError) as exc:
            log.warning("Calibrating on the agent's own principles failed: %s", exc)
            calib["domain_added"] = self._store.counts()["added"] - DOMAIN_EVERY + 5
            self._store.set_meta("calibration", json.dumps(calib))
            self._emit("calibrated", ok=False, error=str(exc), source="agent", space=_embed_name(self.model))

    # -- the fingerprint: is the embedding model still the one that was measured? --
    def check_fingerprint(self) -> bool:
        """Embed ``FINGERPRINT_TEXT`` and compare it with the vector kept from before. If the same model name now
        embeds it differently, the model changed underneath (new weights under the same tag, a new runtime):
        every principle is embedded again and the model is measured again. Returns True when it had changed.
        The first check for a model only keeps the fingerprint."""
        with self._lock:
            space = self._space()
            vec = self._embed([FINGERPRINT_TEXT], "fingerprint", "similarity")[0]
            self._fp_lessons = 0
            try:
                saved = json.loads(self._store.meta("fingerprint") or "null")
            except ValueError:
                saved = None
            keep = json.dumps({"space": space, "vector": [round(float(x), 7) for x in vec], "at": _now()})
            if not isinstance(saved, dict) or saved.get("space") != space or not saved.get("vector"):
                self._store.set_meta("fingerprint", keep)
                return False
            old = _unit(saved["vector"])
            score = _dot(old, vec) if len(old) == len(vec) else 0.0
            if score >= FINGERPRINT_MATCH:
                return False
            log.warning("The embedding model %s changed underneath (fingerprint %.4f); re-embedding and measuring "
                        "it again.", _embed_name(self.model), score)
            self._store.set_meta("fingerprint", keep)
            active = [pid for pid, _, _, _ in self._store.active_vectors()]
            self._reembed(active, space)
            self._index = None
            self._store.event("calibrated", detail=_clip(
                f"{_embed_name(self.model)} changed underneath: the fixed sentence measured {score:.4f} against how "
                f"it was embedded before, so {len(active)} principle(s) were embedded again and the model is "
                "measured again", 1000))
            self._emit("model_changed", space=_embed_name(self.model), score=score, reembedded=len(active))
            self._calib = None
            self._store.set_meta("calibration", "")
            if self.levels.automatic():
                self.calibrate()
            return True

    def _maybe_check_fingerprint(self, lesson: bool) -> None:
        """Check the fingerprint the first time this core searches or learns, and every ``FINGERPRINT_EVERY``
        lessons after that. A model that cannot be reached is checked again next time."""
        if self._fp_lessons is not None:
            if lesson:
                self._fp_lessons += 1
            if self._fp_lessons < FINGERPRINT_EVERY:
                return
        elif not self._store.count_active() and self.calibration(measure=False) is None:
            return                              # nothing embedded or measured yet, so nothing to compare with
        try:
            self.check_fingerprint()
        except ModelError as exc:
            log.warning("The embedding model's fingerprint could not be checked: %s", exc)

    def effective_levels(self) -> Levels:
        """The levels in force: ``levels`` with each automatic one (None) filled in from the calibration of the
        embedding model in use, measured now if there is none yet. Levels set by hand are used as given."""
        with self._lock:
            if not self.levels.automatic():
                return self.levels.resolved()
            calib = self.calibration()
            return self.levels.resolved((calib or {}).get("levels"))

    # -- Injection ------------------------------------------------------------
    def recall(self, task: str, *, conversation: Sequence[dict[str, Any]] | None = None,
               k: int | None = None, expand: bool = True) -> list[Principle]:
        """The principles most relevant to ``task``, most relevant first.

        When there are more principles than ``k``, the model first writes a
        one-sentence description of the situation, and principles are ranked
        by the better of their similarity to the task and to that description.
        Principles describe how to act while tasks describe a subject, so the
        description finds them more reliably. ``expand=False`` skips that call.
        Principles less relevant than ``levels.recall_floor`` are left out.
        """
        k = self.levels.context if k is None else int(k)
        with self._lock:
            started = time.monotonic()
            if self._fp_lessons is None:
                self._maybe_check_fingerprint(lesson=False)
            index = self._ensure_index()
            self._emit("recall.start", task=_clip(task, 500), held=len(index), k=k)
            if k <= 0 or not len(index):
                self._emit("recall.done", principles=[], held=len(index), k=k, situation="", ms=_ms(started))
                return []
            texts = [_clip(task, TASK_CHARS)]
            situation = ""
            if expand and len(index) > k:
                situation = self._situation(task, conversation)
                if situation:
                    texts.append(situation)
            queries = self._query_vectors(texts, "recall")
            ranked = self._ensure_index().top(queries, k)
            floor, below = self.levels.recall_floor, 0
            if floor > 0:
                kept = [(pid, score) for pid, score in ranked if score >= floor]
                below, ranked = len(ranked) - len(kept), kept
            found = self._store.get_many([pid for pid, _ in ranked])
            out: list[Principle] = []
            for pid, score in ranked:
                p = found.get(pid)
                if p is not None and p.status == "active":
                    p.score = score
                    out.append(p)
            self._store.mark_injected([p.id for p in out])
            for p in out:
                p.times_injected += 1
            self._emit("recall.done", principles=[{**p.to_dict(), "score": p.score} for p in out],
                       held=len(self._ensure_index()), k=k, situation=situation, floor=floor, below=below,
                       ms=_ms(started))
            return out

    def _situation(self, task: str, conversation: Sequence[dict[str, Any]] | None) -> str:
        user = SITUATION_USER.format(axiom=self.core_axiom, earlier=_earlier_block(conversation),
                                     task=_clip(task, TASK_CHARS))
        reply = self._chat("situation", [{"role": "system", "content": SITUATION_SYSTEM},
                                         {"role": "user", "content": user}], temperature=0.2, think=False)
        return _clip(_strip_thinking(reply), 1200)

    def format_context(self, principles: Sequence[Principle]) -> str:
        """The text injected into the agent's prompt: the Core Axiom and the principles, as written."""
        lines = ["CORE AXIOM", self.core_axiom, "", "PRINCIPLES"]
        if principles:
            lines.append(PRINCIPLES_PREAMBLE)
            lines += [f"{i}. {p.render()}" for i, p in enumerate(principles, 1)]
        else:
            lines.append(NO_PRINCIPLES)
        return "\n".join(lines)

    def inject(self, task: str, *, conversation: Sequence[dict[str, Any]] | None = None,
               k: int | None = None, expand: bool = True) -> str:
        """Recall the principles for ``task`` and return the text to put in the agent's prompt."""
        return self.format_context(self.recall(task, conversation=conversation, k=k, expand=expand))

    # -- Wisdom Synthesis and Distillation -------------------------------------
    def synthesize(self, task: str, result: str, *, outcome: str | None = None,
                   conversation: Sequence[dict[str, Any]] | None = None,
                   used: Iterable[Principle | str] = (), reasoning: str | None = None,
                   previous: Iterable[Principle | str] = (), success: bool | None = None) -> SynthesisReport:
        """Learn from a finished task.

        The model reads the Core Axiom, what the user said (``task``), the
        agent's previous answer that it responds to (the last answer in
        ``conversation``), the agent's ``reasoning`` before it answered (a
        reasoning model's thinking), the agent's answer (``result``) and
        ``outcome`` (anything known about how it went), plus the principles most
        related to it, including those in ``used`` (the ones injected for the
        task). It learns what the user showed - a reaction to the previous
        answer, a preference, a fact, value or goal they stated - and what the
        thinking established about the subject itself, and writes that as a
        piece of wisdom. A lesson about how the answer was written counts only
        if the user asked for it or reacted to it. It then keeps the principles as they are, adds one,
        revises or merges principles, or removes ones the user's words or the
        Core Axiom show to be wrong.

        It starts with a quick first look (see ``Levels.new_ground``): one
        short call writes the lesson without seeing the principles, and if that
        lesson measures as a principle already held, the synthesis stops there.

        Principles earn their place. ``previous`` is the principles that were
        injected for the agent's previous answer, most relevant first (the one
        the user is responding to): the first look reads whether the user pushed
        back on it, and the most relevant few of them are counted as questioned
        or supported accordingly. ``success`` says outright how this task's
        answer went, when the application knows; the most relevant few of
        ``used`` are counted by it. A principle questioned far more often than
        this agent's answers in general is in doubt and is listed for review in
        the memo (see ``standing``).

        Problems with the model (unreachable, or a reply that cannot be read)
        are reported in ``SynthesisReport.error`` rather than raised, so a
        failed lesson never costs the agent its answer.
        """
        report = SynthesisReport(task="" if task is None else str(task))
        note = _clip(task, 300)
        used, previous = list(used or ()), list(previous or ())
        with self._lock:
            started = time.monotonic()
            self._maybe_check_fingerprint(lesson=True)
            self._emit("synthesis.start", task=note)
            try:
                lesson = self._first_look(task, result, outcome, conversation, reasoning, report, note)
                reviewing = self._react(report, note, used, previous, success)
                if report.first_look.get("held"):
                    raise _AlreadyHeld
                candidates = self._candidates(task, result, [*used, *reviewing])
                labels = {f"P{i}": p for i, p in enumerate(candidates, 1)}
                self._emit("synthesis.candidates",
                           labels={label: {"id": p.id, "name": p.name} for label, p in labels.items()})
                report.memo = self._reflect(task, result, outcome, conversation, labels, reasoning)
                if not report.memo:
                    raise _ParseError("the model wrote an empty memo")
                self._emit("synthesis.memo", memo=report.memo)
                actions = self._execute(report.memo, labels)
                self._emit("synthesis.actions", actions=actions)
                drafted = self._apply(actions, labels, report, note)
                if self.levels.force and not drafted and not report.merged:
                    if lesson:          # it wrote none: the first look already named one, so that is the principle
                        forced = [{"type": "add", "principle": lesson}]
                    else:               # ask once more for one
                        forced = self._forced_add(report.memo, labels)
                    self._emit("synthesis.actions", actions=forced, forced=True)
                    if not self._apply(forced, labels, report, note, adds_only=True):
                        report.skipped.append("forced: the model drafted no principle for this task")
                self._distill(labels, report, note)
            except _AlreadyHeld:
                self._store.bump("held_at_first_look")
            except (ModelError, _ParseError) as exc:
                report.error = str(exc)
                log.warning("Wisdom Synthesis failed: %s", exc)
                self._store.event("failed", detail=_clip(str(exc), 500), task=note)
            finally:
                if report.changed:
                    self._emit("principles.changed")
            if not report.error and not (report.changed or report.covered or report.skipped):
                self._store.event("nothing", detail=_clip(report.memo, 500), task=note)
            if not report.error and report.added:
                self._maybe_calibrate_on_principles()
            self._store.bump("syntheses")
            self._emit("synthesis.done", report=report.to_dict(), ms=_ms(started))
        return report

    def _first_look(self, task: str, result: str, outcome: str | None,
                    conversation: Sequence[dict[str, Any]] | None, reasoning: str | None,
                    report: SynthesisReport, note: str) -> dict[str, Any] | None:
        """The quick first look before Wisdom Synthesis: name the exchange's lesson without seeing the principles,
        then measure it against every principle held. Fills ``report.first_look``; ``held`` there means the agent
        already holds the lesson and nothing more should run. Returns the lesson's principle fields (None if there
        was no first look or it could not be read), which a forced principle uses instead of asking again."""
        index = self._ensure_index()
        given = outcome is not None and str(outcome).strip()
        level = self.effective_levels().new_ground or 0.0 if len(index) and not given else 0.0
        why = ("skipped: no principles are held yet" if not len(index) else
               "skipped: an outcome was given, so the whole synthesis runs" if given else
               "off: new ground is set to 0" if not level else "")
        if why:
            report.first_look = {"ran": False, "why": why, "level": level}
            self._emit("synthesis.first_look", **report.first_look)
            return None
        earlier, previous = _previous_answer(conversation)
        user = LESSON_USER.format(
            axiom=self.core_axiom,
            earlier=_earlier_block(earlier),
            previous=(f"THE AGENT'S PREVIOUS ANSWER, WHICH THE USER IS RESPONDING TO\n{_clip(previous, PREVIOUS_CHARS)}\n\n"
                      if previous else ""),
            task=_clip(task, TASK_CHARS),
            reasoning=(f"THE AGENT'S REASONING BEFORE IT ANSWERED\n{_clip(reasoning, REASONING_CHARS)}\n\n"
                       if reasoning and reasoning.strip() else ""),
            result=_clip(result, ANSWER_CHARS) or "(no answer)")
        messages = [{"role": "system", "content": LESSON_SYSTEM}, {"role": "user", "content": user}]
        reply = ""
        data: Any = None
        try:
            for attempt in range(2):
                reply = self._chat("lesson", messages, json_mode=True, temperature=0.0, think=False)
                try:
                    data = _loads_lenient(reply)
                    break
                except _ParseError:
                    if attempt:
                        raise
                    messages = messages + [{"role": "assistant", "content": _clip(reply, 4000) or "(empty)"},
                                           {"role": "user", "content": JSON_RETRY}]
        except (ModelError, _ParseError) as exc:      # the full synthesis still runs, and reports its own errors
            log.warning("the first look could not be read: %s", exc)
            report.first_look = {"ran": True, "why": "its reply could not be read, so the whole synthesis runs",
                                 "level": level}
            self._emit("synthesis.first_look", **report.first_look)
            return None
        fields = _action_principle(data) if isinstance(data, dict) else None
        corrects = _truthy(_pick(_norm_keys(data), "corrects", "correction", "pushback", "disagrees")) \
            if isinstance(data, dict) else False
        if not fields or _LABEL_RE.match(fields["name"]) or _is_placeholder(fields["name"]):
            report.first_look = {"ran": True, "why": "it named no lesson, so the whole synthesis runs", "level": level,
                                 "corrects": corrects}
            self._emit("synthesis.first_look", **report.first_look)
            return None
        draft = Principle(id="", name=fields["name"], importance=fields["importance"], extent=fields["extent"],
                          use_case=fields["use_case"], related=list(fields["related"]))
        lesson_vec = self._query_vectors([draft.identity_text()], "lesson", role="similarity")[0]
        best = self._ensure_index().top([lesson_vec], 1, kind="ident")
        score = best[0][1] if best else 0.0
        nearest = self._store.get(best[0][0]) if best else None
        look: dict[str, Any] = {"ran": True, "lesson": fields["name"], "score": score, "level": level,
                                "nearest": nearest.name if nearest else "", "corrects": corrects, "held": False}
        if corrects:
            look["why"] = "the user pushed back on the previous answer, so the whole synthesis runs"
        elif nearest is not None and nearest.status == "active" and score >= level:
            look["held"] = True
            look["why"] = "already held"
            nearest.score = score
            report.covered.append(nearest)
            self._store.event("covered", nearest.id, task=note,
                              detail=f"{nearest.name} (first look: the lesson measured {score:.2f}, at or above "
                                     f"the new-ground level of {level:.2f}, so nothing more ran: {fields['name']})")
        else:
            look["why"] = "new ground"
        report.first_look = look
        self._emit("synthesis.first_look", **look)
        return fields

    # -- standing: how the answers a principle shaped were received ---------------
    def _shaping(self, items: Iterable[Principle | str]) -> list[Principle]:
        """The active principles among the most relevant few given (most relevant first), fresh from the store."""
        ids: list[str] = []
        for item in items or ():
            pid = item.id if isinstance(item, Principle) else str(item)
            if pid not in ids:
                ids.append(pid)
            if len(ids) == SHAPING_TOP:
                break
        found = self._store.get_many(ids)
        return [found[pid] for pid in ids if pid in found and found[pid].status == "active"]

    def _react(self, report: SynthesisReport, note: str, used: Iterable[Principle | str],
               previous: Iterable[Principle | str], success: bool | None) -> list[Principle]:
        """Count how the answer before was received against the principles that shaped it (``previous``, read
        from the first look), or how this answer went when ``success`` says (``used``). Returns the principles
        that should be listed for review in this round's memo: the ones questioned, and any in doubt."""
        marks: list[tuple[list[Principle], bool, str]] = []
        if success is not None:
            shaped = self._shaping(used)
            if shaped:
                marks.append((shaped, not success, "outcome"))
        look = report.first_look
        if look.get("ran") and "corrects" in look:
            shaped = self._shaping(previous)
            if shaped:
                marks.append((shaped, bool(look.get("corrects")), "pushback" if look.get("corrects") else "accepted"))
        reviewing: list[Principle] = []
        for principles, questioned, kind in marks:
            before = {p.id: self.standing(p)["in_doubt"] for p in principles}
            self._store.mark_shaped([p.id for p in principles], questioned)
            self._store.bump("reactions_seen")
            if questioned:
                self._store.bump("reactions_questioned")
            fresh = self._store.get_many([p.id for p in principles])
            principles = [fresh[p.id] for p in principles if p.id in fresh]
            report.reaction = {"kind": kind, "questioned": questioned, "principles": [p.id for p in principles]}
            if questioned:
                report.questioned.extend(principles)
                self._store.event("questioned", sources=[p.id for p in principles], task=note,
                                  detail=f"{len(principles)} principle(s) behind the answer the user pushed on: "
                                         + "; ".join(p.name for p in principles))
                reviewing.extend(principles)
                for p in principles:
                    standing = self.standing(p)
                    if standing["in_doubt"] and not before.get(p.id):
                        report.doubt.append(p)
                        self._store.event("doubt", p.id, detail=f"{p.name} (the user pushed back on "
                                          f"{standing['questioned']} of the {standing['shaped']} answers it shaped, "
                                          f"against about {standing['usual']:.0%} of this agent's other answers)",
                                          task=note)
            else:
                report.supported.extend(principles)
        unique: list[Principle] = []
        for p in reviewing:
            if all(r.id != p.id for r in unique):
                unique.append(p)
        return unique

    def standing(self, principle: Principle) -> dict[str, Any]:
        """A principle's standing: how the answers it shaped were received, since its last review.

        ``shaped`` and ``questioned`` count answers it shaped and the ones the user pushed back on; ``usual``
        is the share of this user's other answers (not shaped by this principle) that are pushed back on;
        ``surprise`` is how likely that many pushbacks would be for a sound principle, pushed back on as often
        as the usual; ``in_doubt`` when that is below ``DOUBT_ALPHA`` after at least ``DOUBT_MIN_QUESTIONED``
        pushbacks; ``proven`` when it has served many times with almost no pushback; ``reviews`` how many
        times it was reviewed and found to stand.
        """
        shaped, questioned = principle.since_review()
        usual = self.usual_pushback(excluding=principle)
        surprise = _binomial_tail(questioned, shaped, usual) if shaped else 1.0
        rate = questioned / shaped if shaped else 0.0
        in_doubt = questioned >= DOUBT_MIN_QUESTIONED and surprise < DOUBT_ALPHA
        proven = principle.shaped >= PROVEN_SHAPED and principle.questioned * 10 <= principle.shaped
        return {"shaped": shaped, "questioned": questioned, "rate": rate, "usual": usual, "surprise": surprise,
                "in_doubt": in_doubt, "proven": proven, "reviews": int((principle.review or {}).get("count") or 0),
                "total_shaped": principle.shaped, "total_questioned": principle.questioned}

    def usual_pushback(self, excluding: Principle | None = None) -> float:
        """The share of this agent's answers the user pushes back on, with a mild prior for a new agent; with
        ``excluding``, the share among the answers that principle did not shape."""
        seen = int(self._store.meta("reactions_seen") or 0)
        questioned = int(self._store.meta("reactions_questioned") or 0)
        if excluding is not None:
            seen, questioned = max(seen - excluding.shaped, 0), max(questioned - excluding.questioned, 0)
        return (questioned + 1) / (seen + 4)

    def _record_note(self, p: Principle) -> str:
        """What the memo is told about a principle's standing: only when it is in doubt or proven."""
        standing = self.standing(p)
        if standing["in_doubt"]:
            reviewed = (f"; reviewed and found to stand {standing['reviews']} time(s) before"
                        if standing["reviews"] else "")
            return (f" [IN DOUBT: the user pushed back on {standing['questioned']} of the {standing['shaped']} "
                    f"answers it shaped, against about {standing['usual']:.0%} of this agent's other answers"
                    f"{reviewed}. Review it.]")
        if standing["proven"]:
            return (f" [PROVEN: shaped {p.shaped} answers, "
                    f"{'none' if not p.questioned else p.questioned} questioned.]")
        return ""

    def _candidates(self, task: str, result: str, used: Iterable[Principle | str]) -> list[Principle]:
        index = self._ensure_index()
        if not len(index):
            return []
        order: list[str] = []
        for item in used or ():
            pid = item.id if isinstance(item, Principle) else str(item)
            if pid in index and pid not in order:
                order.append(pid)
        queries = self._query_vectors([_clip(task, TASK_CHARS), _clip(f"{task}\n\n{result}", TASK_CHARS)],
                                      "candidates")
        index = self._ensure_index()
        for q in queries:
            for pid, _ in index.top([q], CANDIDATES_PER_QUERY):
                if pid not in order:
                    order.append(pid)
        order = order[:MAX_CANDIDATES]
        found = self._store.get_many(order)
        return [found[pid] for pid in order if pid in found and found[pid].status == "active"]

    def _reflect(self, task: str, result: str, outcome: str | None,
                 conversation: Sequence[dict[str, Any]] | None, labels: dict[str, Principle],
                 reasoning: str | None = None) -> str:
        listing = "\n".join(f"{label}. {p.render()}{self._record_note(p)}" for label, p in labels.items())
        outcome_text = _clip(outcome, RESULT_CHARS) if outcome is not None else ""
        earlier, previous = _previous_answer(conversation)
        user = REFLECT_USER.format(
            axiom=self.core_axiom,
            earlier=_earlier_block(earlier),
            previous=(f"THE AGENT'S PREVIOUS ANSWER, WHICH THE USER IS RESPONDING TO\n{_clip(previous, PREVIOUS_CHARS)}\n\n"
                      if previous else ""),
            task=_clip(task, TASK_CHARS),
            reasoning=(f"THE AGENT'S REASONING BEFORE IT ANSWERED\n{_clip(reasoning, REASONING_CHARS)}\n\n"
                       if reasoning and reasoning.strip() else ""),
            result=_clip(result, ANSWER_CHARS) or "(no answer)",
            outcome=f"WHAT HAPPENED AFTERWARDS\n{outcome_text}\n\n" if outcome_text else "",
            principles=listing or "(none: the agent holds no principles yet, so there are no labels to refer to)",
            policy=FORCE_MEMO if self.levels.force else CAREFUL_MEMO,
        )
        reply = self._chat("reflect", [{"role": "system", "content": REFLECT_SYSTEM},
                                       {"role": "user", "content": user}], temperature=0.2, think=False)
        return _strip_thinking(reply)

    def _execute_messages(self, memo: str, labels: dict[str, Principle]) -> list[dict[str, Any]]:
        listing = "\n".join(f"{label}. {p.name}" + (" (in doubt)" if self.standing(p)["in_doubt"] else "")
                            for label, p in labels.items()) or "(none)"
        return [{"role": "system", "content": EXECUTE_SYSTEM},
                {"role": "user", "content": EXECUTE_USER.format(
                    axiom=self.core_axiom, labels=listing, memo=_clip(memo, RESULT_CHARS),
                    policy=self._action_policy())}]

    def _action_policy(self) -> str:
        """What the executor is told about how much to write: the add limit, and the forced principle."""
        count = self.levels.max_adds
        policy = "\n\n" + ADD_LIMIT.format(count=count, s="" if count == 1 else "s")
        return policy + (f"\n{FORCE_ACTIONS}" if self.levels.force else "")

    def _execute(self, memo: str, labels: dict[str, Principle]) -> list[dict[str, Any]]:
        messages = self._execute_messages(memo, labels)
        reply = self._chat("execute", messages, json_mode=True, temperature=0.0, think=False)
        try:
            return _parse_actions(reply)
        except _ParseError:
            retry = messages + [{"role": "assistant", "content": _clip(reply, 4000) or "(empty)"},
                                {"role": "user", "content": JSON_RETRY}]
            return _parse_actions(self._chat("execute", retry, json_mode=True, temperature=0.0, think=False))

    def _forced_add(self, memo: str, labels: dict[str, Principle]) -> list[dict[str, Any]]:
        """Ask once more for the memo's lesson as a principle, when a task wrote none and force is on."""
        messages = self._execute_messages(memo, labels) + [
            {"role": "assistant", "content": '{"actions": []}'},
            {"role": "user", "content": FORCE_RETRY}]
        try:
            actions = _parse_actions(self._chat("execute", messages, json_mode=True, temperature=0.0, think=False))
        except (_ParseError, ModelError) as exc:
            log.warning("the forced principle could not be read: %s", exc)
            return []
        return [a for a in actions if _action_type(a) == "add"]

    # -- Wisdom Distillation, enforced: principles nearly the same are combined into one -----
    def _distill(self, labels: dict[str, Principle], report: SynthesisReport, note: str) -> None:
        """Combine the principles this task brought up (and any it just added) that measure at or above the
        combine level against each other, most alike first, at most MAX_COMBINES_PER_TASK times."""
        level = self.effective_levels().combine
        if not level:
            return
        index = self._ensure_index()
        pool: list[Principle] = []
        for p in [*labels.values(), *report.added]:
            if p.status == "active" and p.id in index and all(q.id != p.id for q in pool):
                pool.append(p)
        pairs = []
        for i, a in enumerate(pool):
            va = index.vector(a.id)
            for b in pool[i + 1:]:
                vb = index.vector(b.id)
                if va is not None and vb is not None and len(va) == len(vb):
                    score = _dot(va, vb)
                    if score >= level:
                        pairs.append((score, a, b))
        pairs.sort(key=lambda t: -t[0])
        used: set[str] = set()
        done = 0
        for score, a, b in pairs:
            if done >= MAX_COMBINES_PER_TASK:
                break
            if a.id in used or b.id in used or a.status != "active" or b.status != "active":
                continue
            if self._combine_into([a, b], score, report, note, used, persisted=[a, b]):
                done += 1
                report.added[:] = [p for p in report.added if p.id not in used]   # an addition folded into another

    def _combine_into(self, principles: Sequence[Principle], score: float, report: SynthesisReport, note: str,
                      consumed: set[str], *, persisted: Sequence[Principle]) -> bool:
        """Ask the model for the one principle that replaces ``principles`` and store it; ``persisted`` are the
        ones held in the database (a draft that was never stored is not). Returns False if it could not."""
        fields = self._combined_fields(principles)
        if fields is None:
            report.skipped.append("combine: the model wrote no combined principle for "
                                  + " | ".join(p.name for p in principles))
            return False
        axiom = self._store.current_axiom()
        merged = Principle(id=uuid.uuid4().hex, created_at=_now(), learned_from=note, **fields)
        vecs = self._principle_vectors(merged)
        reason = f"combined into: {merged.name} (measured {score:.2f} alike)"
        with self._store.tx() as c:
            self._store.insert(c, merged, vecs, self._space(), axiom[0] if axiom else None)
            for p in persisted:
                self._store.end(c, p.id, "merged", reason, merged_into=merged.id)
            self._store._event(c, "merged", merged.id, [p.id for p in persisted],
                               f"{merged.name} (combined automatically: measured {score:.2f} alike, at or above "
                               f"the combine level of {self.effective_levels().combine or 0:.2f})", note)
        index = self._ensure_index()
        for p in persisted:
            index.drop(p.id)
            p.status, p.merged_into, p.end_reason = "merged", merged.id, reason
            consumed.add(p.id)
        index.put(merged.id, *vecs)
        merged.score = score
        report.merged.append((list(principles), merged))
        return True

    def _combined_fields(self, principles: Sequence[Principle]) -> dict[str, Any] | None:
        listing = "\n".join(f"{i}. {p.render()}" for i, p in enumerate(principles, 1))
        messages = [{"role": "system", "content": COMBINE_SYSTEM},
                    {"role": "user", "content": COMBINE_USER.format(axiom=self.core_axiom, principles=listing)}]
        reply = ""
        for attempt in range(2):
            try:
                reply = self._chat("combine", messages, json_mode=True, temperature=0.0, think=False)
                data = _loads_lenient(reply)
            except _ParseError:
                if attempt:
                    return None
                messages = messages + [{"role": "assistant", "content": _clip(reply, 4000) or "(empty)"},
                                       {"role": "user", "content": JSON_RETRY}]
                continue
            except ModelError as exc:
                log.warning("principles could not be combined: %s", exc)
                return None
            fields = _action_principle(data) if isinstance(data, dict) else None
            if fields and not _LABEL_RE.match(fields["name"]) and not _is_placeholder(fields["name"]):
                return fields
            return None
        return None

    def _least_alike(self, principles: Sequence[Principle]) -> tuple[float, str, str] | None:
        """The least alike pair among principles about to be merged, as (similarity, name, name)."""
        index = self._ensure_index()
        vectors = [(p, index.vector(p.id)) for p in principles]
        worst: tuple[float, str, str] | None = None
        for i, (a, va) in enumerate(vectors):
            for b, vb in vectors[i + 1:]:
                if va is None or vb is None or len(va) != len(vb):
                    continue
                score = _dot(va, vb)
                if worst is None or score < worst[0]:
                    worst = (score, a.name, b.name)
        return worst

    def _apply(self, actions: list[dict[str, Any]], labels: dict[str, Principle],
               report: SynthesisReport, note: str, adds_only: bool = False) -> int:
        """Carry out the model's changes. Returns how many principles this pass drafted, whether they were
        added or measured as already held."""
        levels = self.effective_levels()
        covered_at = levels.covered if levels.covered is not None else FALLBACK_LEVELS["covered"]
        axiom = self._store.current_axiom()
        axiom_id = axiom[0] if axiom else None
        space = self._space()
        consumed: set[str] = set()       # merged or removed during this pass
        adds = len(report.added)
        drafted = 0

        def cover(p: Principle, detail: str, score: float | None = None) -> None:
            p.score = score
            if all(c.id != p.id for c in report.covered):
                report.covered.append(p)
            self._store.event("covered", p.id, detail=detail, task=note)

        for raw in actions:
            kind = _action_type(raw)
            if kind == "none" or (adds_only and kind != "add"):
                continue

            if kind == "covered":
                targets = _action_targets(raw, "id", "ids", "target", "targetid", "label", "coveredby",
                                          "principle")
                target = _resolve_label(targets[0], labels) if targets else None
                if target is None:
                    report.skipped.append(f"covered: {_describe(targets)} is not a listed principle")
                elif target.id in consumed:
                    report.skipped.append(f"covered: {target.name} was merged or removed in this pass")
                else:
                    cover(target, target.name)

            elif kind == "add":
                fields = _action_principle(raw)
                if not fields or _LABEL_RE.match(fields["name"]) or _is_placeholder(fields["name"]):
                    report.skipped.append("add: no principle was given")
                    continue
                if adds >= levels.max_adds:
                    report.skipped.append(f"add: only {levels.max_adds} new principle"
                                          f"{'' if levels.max_adds == 1 else 's'} per task ({fields['name']})")
                    continue
                draft = Principle(id=uuid.uuid4().hex, created_at=_now(), learned_from=note, **fields)
                vecs = self._principle_vectors(draft)
                index = self._ensure_index()
                best = index.top([vecs[1]], 1, exclude=consumed, kind="ident")
                nearest = best[0][1] if best else None
                if best and nearest is not None and nearest >= covered_at:
                    held = self._store.get(best[0][0])
                    if held is not None:
                        cover(held, f"{held.name} (measured {nearest:.2f} against the principle this task "
                                    f"would have written, at or above the covered level of "
                                    f"{covered_at:.2f}: {draft.name})", nearest)
                        drafted += 1
                        continue
                if best and nearest is not None and levels.combine and nearest >= levels.combine:
                    held = self._store.get(best[0][0])     # nearly the same: combined with it, never held twice
                    if held is not None and held.status == "active":
                        if self._combine_into([held, draft], nearest, report, note, consumed, persisted=[held]):
                            drafted += 1
                            continue
                        cover(held, f"{held.name} (measured {nearest:.2f}, at or above the combine level of "
                                    f"{levels.combine:.2f}; the two could not be combined, so the one held was "
                                    f"kept: {draft.name})", nearest)
                        drafted += 1
                        continue
                draft.score = nearest       # how close the nearest principle it already held was
                with self._store.tx() as c:
                    self._store.insert(c, draft, vecs, space, axiom_id)
                    self._store._event(c, "added", draft.id, [], draft.name, note)
                index.put(draft.id, *vecs)
                report.added.append(draft)
                adds += 1
                drafted += 1

            elif kind == "merge":
                targets = _action_targets(raw, "ids", "targetids", "targets", "sources", "merge", "from", "id")
                resolved = [_resolve_label(t, labels) for t in targets]
                unknown = [t for t, r in zip(targets, resolved, strict=True) if r is None]
                sources: list[Principle] = []
                for r in resolved:
                    if r is not None and all(s.id != r.id for s in sources):
                        sources.append(r)
                fields = _action_principle(raw)
                if unknown:
                    report.skipped.append(f"merge: {_describe(unknown)} not among the listed principles; "
                                          "nothing was merged")
                elif not sources:
                    report.skipped.append("merge: no listed principle was named")
                elif any(s.id in consumed for s in sources):
                    report.skipped.append("merge: one of those principles was already merged or removed in this pass")
                elif not fields or _LABEL_RE.match(fields["name"]) or _is_placeholder(fields["name"]):
                    report.skipped.append("merge: no combined principle was given")
                elif len(sources) > 1 and (apart := self._least_alike(sources)) is not None and apart[0] < levels.merge:
                    report.skipped.append(f"merge: {apart[1]} and {apart[2]} measured {apart[0]:.2f}, below the "
                                          f"merge level of {levels.merge:.2f}; nothing was merged")
                else:                       # two or more are merged; one alone is revised: its new wording replaces it
                    merged = Principle(id=uuid.uuid4().hex, created_at=_now(), learned_from=note, **fields)
                    vecs = self._principle_vectors(merged)
                    how = "merged into" if len(sources) > 1 else "revised as"
                    with self._store.tx() as c:
                        self._store.insert(c, merged, vecs, space, axiom_id)
                        for s in sources:
                            self._store.end(c, s.id, "merged", f"{how}: {merged.name}", merged_into=merged.id)
                        self._store._event(c, "merged", merged.id, [s.id for s in sources],
                                           merged.name if len(sources) > 1 else f"revised: {merged.name}", note)
                    index = self._ensure_index()
                    for s in sources:
                        index.drop(s.id)
                        s.status, s.merged_into, s.end_reason = "merged", merged.id, f"{how}: {merged.name}"
                        consumed.add(s.id)
                    index.put(merged.id, *vecs)
                    report.merged.append((sources, merged))

            elif kind == "remove":
                targets = _action_targets(raw, "id", "ids", "target", "targetid", "targets", "label", "principle")
                reason = _as_text(_pick(_norm_keys(raw), "reason", "why", "because", "explanation"), FIELD_CHARS)
                if not targets:
                    report.skipped.append("remove: no principle was named")
                for t in targets:
                    target = _resolve_label(t, labels)
                    if target is None:
                        report.skipped.append(f"remove: {_describe([t])} is not a listed principle")
                        continue
                    if _is_placeholder(reason):
                        report.skipped.append(f"remove: no reason was given, so {target.name} was kept")
                        continue
                    if target.id in consumed:
                        report.skipped.append(f"remove: {target.name} was already merged or removed in this pass")
                        continue
                    with self._store.tx() as c:
                        self._store.end(c, target.id, "removed", reason)
                        self._store._event(c, "removed", target.id, [], f"{target.name} ({reason})", note)
                    self._ensure_index().drop(target.id)
                    target.status, target.end_reason = "removed", reason
                    consumed.add(target.id)
                    report.removed.append((target, reason))

            elif kind == "stands":
                targets = _action_targets(raw, "id", "ids", "target", "targetid", "targets", "label", "principle")
                reason = _as_text(_pick(_norm_keys(raw), "reason", "why", "because", "explanation"), FIELD_CHARS)
                if not targets:
                    report.skipped.append("stands: no principle was named")
                for t in targets:
                    target = _resolve_label(t, labels)
                    if target is None:
                        report.skipped.append(f"stands: {_describe([t])} is not a listed principle")
                    elif target.id in consumed:
                        report.skipped.append(f"stands: {target.name} was merged or removed in this pass")
                    elif not self.standing(target)["in_doubt"]:
                        report.skipped.append(f"stands: {target.name} was not in doubt, so there was nothing to review")
                    else:
                        reason = reason if not _is_placeholder(reason) else "reviewed and found sound"
                        review = {"at": _now(), "shaped": target.shaped, "questioned": target.questioned,
                                  "count": int((target.review or {}).get("count") or 0) + 1, "reason": reason}
                        self._store.set_review(target.id, review)
                        self._store.event("stands", target.id, detail=f"{target.name} ({reason})", task=note)
                        target.review = review
                        report.stands.append((target, reason))

            else:
                report.skipped.append(f"unknown action {kind!r}")
        return drafted


def context_window_of(model: Any) -> int | None:
    """The context window a backend sends with each request, when it says (Ollama does); else None."""
    try:
        value = getattr(model, "context_window", None)
    except WisdomCoreError:
        return None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def fit_history(system: str, history: Sequence[dict[str, Any]], message: str,
                window: int | None) -> tuple[list[dict[str, Any]], int]:
    """Drop the oldest messages until the prompt leaves room for a reply inside ``window`` tokens.

    Only the conversation is trimmed. The system prompt (the Core Axiom and the
    principles) and the new message always go in whole, so a long conversation
    can never push the axiom out of the model's window. Returns the messages
    kept and how many were dropped.
    """
    kept = [dict(m) for m in history]
    if not window:
        return kept, 0
    room = max(window - min(REPLY_ROOM, window // 4), 0) * CHARS_PER_TOKEN
    size = len(system) + len(message) + sum(len(str(m.get("content", ""))) + 16 for m in kept) + 32
    dropped = 0
    while kept and size > room:
        gone = kept.pop(0)
        size -= len(str(gone.get("content", ""))) + 16
        dropped += 1
    if dropped and kept and kept[0].get("role") == "assistant":   # start on the person's turn
        kept.pop(0)
        dropped += 1
    return kept, dropped


# ---------------------------------------------------------------------------
# A ready-made agent
# ---------------------------------------------------------------------------
class Agent:
    """A chat agent built on the Wisdom Core.

    Every call to ``ask`` is one task: the Core Axiom and the most relevant
    principles are injected, the model answers, and then Wisdom Synthesis
    runs on that task and its answer. ``last_report`` says what it learned.

    Args:
        core_axiom: The one sentence the agent serves (see ``WisdomCore``).
        model: None, an Ollama model name, or a backend object.
        path: The SQLite file for this agent's principles.
        context: How many principles are injected at each decision (default 12).
        levels: The semantic levels (see ``Levels``).
        instructions: Optional standing instructions (tone, format, language)
            placed after the Core Axiom and principles.
        history: How many earlier exchanges of the conversation the model sees.
            Older ones are also dropped when the conversation would not leave
            room for a reply in the model's context window.
        learn: Set False to answer without running Wisdom Synthesis.
        on_event: Watch every step (see ``WisdomCore``). The answer call is
            reported as a model call with purpose "answer".
    """

    def __init__(self, core_axiom: str | None = None, model: Any = None, *, path: str = "wisdom.db",
                 context: int | None = None, levels: Levels | Mapping[str, Any] | None = None,
                 instructions: str | None = None, history: int = 6, learn: bool = True,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.core = WisdomCore(core_axiom, model, path=path, context=context, levels=levels, on_event=on_event)
        self.instructions = (instructions or "").strip()
        self.history = max(0, int(history))
        self.learn = bool(learn)
        self.messages: list[dict[str, str]] = []
        self.last_principles: list[Principle] = []
        self.last_report: SynthesisReport | None = None
        self._pending: tuple[str, str, list[dict[str, str]], list[Principle], str, list[Any]] | None = None
        self.last_reasoning = ""        # the thinking behind the last answer, handed to Wisdom Synthesis
        self.previous_principles: list[Principle | str] = []   # behind the answer the next message responds to

    def __repr__(self) -> str:
        return f"Agent(path={self.core.path!r}, model={self.core.model!r})"

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.core.close()

    @property
    def core_axiom(self) -> str:
        return self.core.core_axiom

    @property
    def on_event(self) -> Callable[[dict[str, Any]], None] | None:
        return self.core.on_event

    @on_event.setter
    def on_event(self, handler: Callable[[dict[str, Any]], None] | None) -> None:
        self.core.on_event = handler

    def principles(self) -> list[Principle]:
        return self.core.principles()

    def reset(self) -> None:
        """Forget the conversation. Principles are kept."""
        self.messages.clear()
        self.last_principles = []
        self.previous_principles = []
        self.last_report = None
        self._pending = None

    def system_prompt(self, principles: Sequence[Principle]) -> str:
        parts = [AGENT_PREAMBLE, "", self.core.format_context(principles)]
        if self.instructions:
            parts += ["", "INSTRUCTIONS", self.instructions]
        return "\n".join(parts)

    def _window(self) -> list[dict[str, str]]:
        if self.history <= 0:
            return []
        return [{"role": m["role"], "content": _clip(m["content"], HISTORY_CHARS)}
                for m in self.messages[-2 * self.history:]]

    def ask(self, message: str) -> str:
        """Answer one message, then learn from it (``respond`` followed by ``synthesize``)."""
        reply = self.respond(message)
        if self.learn:
            self.synthesize()
        return reply

    def respond(self, message: str) -> str:
        """Answer one message without learning from it yet.

        Use this when the answer should reach the person before the agent
        learns, then call ``synthesize()``, optionally with an outcome.
        """
        text = "" if message is None else str(message).strip()
        if not text:
            raise ValueError("message is empty")
        conversation = self._window()
        principles = self.core.recall(text, conversation=conversation)
        system = self.system_prompt(principles)
        fitted, dropped = fit_history(system, conversation, text, context_window_of(self.core.model))
        if dropped:
            self.core._emit("history.trimmed", dropped=dropped, kept=len(fitted))
        messages = [{"role": "system", "content": system}, *fitted, {"role": "user", "content": text}]
        raw = self.core._chat("answer", messages)
        reply = _strip_thinking(raw)
        # the thinking behind the answer: a reasoning model's separate thinking, or <think>...</think> written inline
        thoughts = [str(self.core.last_stats.get("thinking") or "").strip(), _thinking_part(raw)]
        self.last_reasoning = "\n\n".join(t for t in thoughts if t)
        self.messages.append({"role": "user", "content": text})
        self.messages.append({"role": "assistant", "content": reply})
        previous = list(self.previous_principles)       # what shaped the answer this message responds to
        self.previous_principles = list(principles)
        self.last_principles = principles
        self.last_report = None
        self._pending = (text, reply, conversation, principles, self.last_reasoning, previous)
        return reply

    def synthesize(self, outcome: str | None = None, success: bool | None = None) -> SynthesisReport:
        """Run Wisdom Synthesis on the last answer. ``outcome`` is anything known about how it went, in words;
        ``success`` says outright whether it went well, which counts for or against the principles behind it."""
        if self._pending is None:
            raise WisdomCoreError("There is no answered task to learn from yet.")
        text, reply, conversation, principles, reasoning, previous = self._pending
        self._pending = None
        try:
            report = self.core.synthesize(text, reply, outcome=outcome, conversation=conversation, used=principles,
                                          reasoning=reasoning, previous=previous, success=success)
        except Exception as exc:  # a lesson that fails must never cost the answer
            log.warning("Wisdom Synthesis raised %s", type(exc).__name__, exc_info=True)
            report = SynthesisReport(task=text, error=f"{type(exc).__name__}: {exc}")
        self.last_report = report
        return report


# ---------------------------------------------------------------------------
# Command line: python -m wisdom_core
# ---------------------------------------------------------------------------
_HELP = """Commands:
  /principles   list the principles the agent holds
  /history      what the Wisdom Core has done lately
  /stats        counts
  /axiom        show the Core Axiom
  /reset        forget the conversation (principles are kept)
  /quit         leave"""


def _color(text: str, code: str) -> str:
    if sys.stdout.isatty() and os.name != "nt" and "NO_COLOR" not in os.environ:
        return f"\033[{code}m{text}\033[0m"
    return text


def _print_principles(principles: Sequence[Principle]) -> None:
    if not principles:
        print("No principles yet.")
        return
    print(f"{len(principles)} principle{'s' if len(principles) != 1 else ''}:")
    for i, p in enumerate(principles, 1):
        print(f"{i:>3}. {p.name}")
        details = [f"why: {p.importance}" if p.importance else "",
                   f"scope: {p.extent}" if p.extent else "",
                   f"use when: {p.use_case}" if p.use_case else ""]
        for d in details:
            if d:
                print(f"     {d}")
        record = (f"; shaped {p.shaped} answer{'s' if p.shaped != 1 else ''}, {p.questioned} questioned"
                  if p.shaped else "")
        print(_color(f"     recalled {p.times_injected} time{'s' if p.times_injected != 1 else ''}{record}"
                     f"; learned from: {_clip(p.learned_from, 80) or '-'}", "2"))


def _print_history(events: Sequence[Event]) -> None:
    if not events:
        print("Nothing has happened yet.")
        return
    for e in events:
        try:
            stamp = datetime.fromisoformat(e.at).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            stamp = e.at[:19].replace("T", " ")
        task = f"  [task: {_clip(e.task, 60)}]" if e.task else ""
        print(f"{stamp}  {e.kind:<8} {_clip(e.detail, 160)}{task}")


def _build_model(args: argparse.Namespace) -> Any:
    if args.openai_base_url:
        if not args.model:
            raise WisdomCoreError("--model is required with --openai-base-url")
        return OpenAICompatible(args.model, args.embed_model, base_url=args.openai_base_url, api_key=args.api_key)
    return Ollama(args.model, args.embed_model, host=args.host)


def _chat_loop(agent: Agent) -> int:
    try:
        import readline  # noqa: F401  (arrow keys and history where available)
    except ImportError:  # pragma: no cover - Windows
        pass
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.split()[0].lower()
            if cmd in ("/quit", "/exit", "/q"):
                return 0
            if cmd == "/principles":
                _print_principles(agent.core.principles())
            elif cmd == "/history":
                _print_history(agent.core.history(30))
            elif cmd == "/stats":
                print(", ".join(f"{k}: {v}" for k, v in agent.core.stats().items()))
            elif cmd == "/axiom":
                print(agent.core.core_axiom)
            elif cmd == "/reset":
                agent.reset()
                print("Conversation cleared. Principles are kept.")
            elif cmd == "/help":
                print(_HELP)
            else:
                print("Unknown command. Type /help for the list.")
            continue
        try:
            reply = agent.respond(line)
        except ModelError as exc:
            print(f"\n[model error] {exc}")
            continue
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        print(f"\nagent> {reply}", flush=True)
        if agent.learn:
            try:
                report = agent.synthesize()
            except KeyboardInterrupt:
                print("\n(learning interrupted)")
                continue
            print(_color("\n".join(f"  wisdom core {line}" for line in report.lines()), "2"))


def main(argv: Sequence[str] | None = None) -> int:
    """Chat with an agent that learns its own principles under a Core Axiom."""
    parser = argparse.ArgumentParser(
        prog="wisdom-core",
        description="Chat with an agent that learns its own principles under a Core Axiom.",
    )
    parser.add_argument("--axiom", help="the Core Axiom (needed the first time a database is used)")
    parser.add_argument("--db", default="wisdom.db", help="the agent's principle database (default: wisdom.db)")
    parser.add_argument("--model", help="chat model (default: the first chat model Ollama has)")
    parser.add_argument("--embed-model", help="embedding model (default: the first embedding model Ollama has)")
    parser.add_argument("--host", help="Ollama address (default: $OLLAMA_HOST, then http://localhost:11434)")
    parser.add_argument("--openai-base-url", metavar="URL",
                        help="use an OpenAI-compatible server at URL instead of Ollama")
    parser.add_argument("--api-key", help="API key for --openai-base-url (default: $OPENAI_API_KEY)")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                        help=f"principles injected at each decision (default: {DEFAULT_CONTEXT})")
    parser.add_argument("--no-learn", action="store_true", help="answer without running Wisdom Synthesis")
    parser.add_argument("--principles", action="store_true", help="print the principles and exit")
    parser.add_argument("--history", action="store_true", help="print what the Wisdom Core has done and exit")
    parser.add_argument("--version", action="version", version=f"wisdom-core {__version__}")
    args = parser.parse_args(argv)

    if args.principles or args.history:
        if not os.path.exists(args.db):
            print(f"No principle database at {args.db}.", file=sys.stderr)
            return 1
        try:
            store = _Store(args.db)
        except WisdomCoreError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            axiom = store.current_axiom()
            print(f"Core Axiom: {axiom[1] if axiom else '(not set)'}\n")
            if args.principles:
                _print_principles(store.list(include_ended=False))
            if args.history:
                if args.principles:
                    print()
                _print_history(store.events(50))
        finally:
            store.close()
        return 0

    axiom = (args.axiom or "").strip()
    existing = None
    if os.path.exists(args.db):
        try:
            store = _Store(args.db)
        except WisdomCoreError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        existing = store.current_axiom()
        store.close()
    if axiom and existing is not None and existing[1] != axiom:
        print(f"Changing this agent's Core Axiom. It was: {existing[1]}")
        print("Its principles are kept; any that conflict with the new axiom are removed as they come up.\n")
    if not axiom and existing is None:
        try:
            axiom = input("Core Axiom (the one sentence this agent serves): ").strip()
        except (EOFError, KeyboardInterrupt):
            axiom = ""
        if not axiom:
            print("A Core Axiom is needed to start.", file=sys.stderr)
            return 1
    try:
        agent = Agent(axiom or None, _build_model(args), path=args.db, context=args.context,
                      learn=not args.no_learn)
    except (WisdomCoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with agent:
        try:
            chat_name, embed_name = agent.core.model.name, _embed_name(agent.core.model)
        except ModelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        count = len(agent.core.principles())
        print(f"Wisdom Core {__version__}")
        print(f"Core Axiom: {agent.core.core_axiom}")
        print(f"Principles: {count} in {args.db}")
        print(f"Models:     {chat_name} (chat), {embed_name} (embeddings)")
        print("Type /help for commands.")
        return _chat_loop(agent)


if __name__ == "__main__":
    raise SystemExit(main())
