"""A tiny offline stand-in for a real model, so the examples run with no Ollama and no network.

It is NOT an AI model. It answers every prompt with a fixed, rule-based reply:

* chat: it recognises which step of the Wisdom Core is asking (first look, memo, actions, ...)
  by comparing the system prompt with the prompt constants in wisdom_core, and returns a
  canned reply in the format that step expects. The "lesson" it writes is picked from a small
  table by keywords in the user's message.
* embed: a bag-of-words vector, so texts that share words count as similar.

It is wired in through wisdom_core.FunctionModel, the same public hook you would use to plug
in your own model functions. Swap it for Ollama(), OpenAICompatible(...) or Anthropic(...)
to get real answers and real lessons.
"""

from __future__ import annotations

import hashlib
import json
import re

import wisdom_core as wc  # the installed package, or the file in a source checkout

# keyword -> the principle the toy model "learns" when a message mentions it
LESSONS = [
    (("promo", "promotional"), {
        "name": "Check for promotional or 0% rates before ranking debts by their interest rate.",
        "importance": "A temporary 0% rate changes which debt costs the most right now.",
        "extent": "Applies while a promotional rate lasts; after it ends, compare the regular rates.",
        "use_case": "someone says a card or loan has a promotional or 0% rate",
        "related": ["promo rate", "0% APR", "balance transfer", "debt order"],
    }),
    (("credit", "card", "loan", "debt", "rate", "interest"), {
        "name": "Pay off the highest-interest debt first once minimum payments are covered.",
        "importance": "It saves the most money, which serves sound money decisions.",
        "extent": "Applies to ordinary consumer debt, not to tax or legal questions.",
        "use_case": "someone asks which of their debts to pay off first",
        "related": ["debt", "interest rate", "credit card", "loan payoff"],
    }),
    (("stock", "invest", "savings", "saved"), {
        "name": "Do not put all of your savings into a single stock.",
        "importance": "Spreading money out limits how much one bad bet can cost.",
        "extent": "General guidance, not personal investment advice.",
        "use_case": "someone asks whether to put their savings into one investment",
        "related": ["investing", "diversification", "single stock", "savings"],
    }),
    (("budget", "bills", "rent", "spend"), {
        "name": "Write every monthly bill down before deciding what is left to spend.",
        "importance": "A budget built on real numbers prevents surprises.",
        "extent": "Covers recurring household bills.",
        "use_case": "someone wants help adding up their monthly bills or budget",
        "related": ["budget", "monthly bills", "rent", "spending"],
    }),
]
FALLBACK = {
    "name": "Ask what the person is trying to achieve before giving advice.",
    "importance": "Advice only helps when it fits the goal.",
    "extent": "Any open-ended request.",
    "use_case": "someone asks a broad question without saying their goal",
    "related": ["goals", "clarifying questions"],
}
STOP = {"the", "and", "for", "with", "that", "this", "you", "your", "are", "was", "not", "but", "all",
        "any", "can", "its", "into", "from", "when", "what", "which", "who", "how", "one", "should"}


def _lesson_for(text: str) -> dict:
    words = set(re.findall(r"[a-z]+", text.lower()))
    for keys, principle in LESSONS:
        if words & set(keys):
            return dict(principle)
    return dict(FALLBACK)


def _said(user_prompt: str) -> str:
    """The user's message, as the Wisdom Core lays it out in its prompts."""
    part = user_prompt.split("WHAT THE USER SAID\n", 1)[-1]
    return part.split("\n\n", 1)[0]


def chat(messages: list[dict]) -> str:
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user = messages[-1]["content"]
    if system == wc.SITUATION_SYSTEM:                       # recall: describe the situation
        return "Someone asks for help with: " + user.rsplit("TASK\n", 1)[-1][:120]
    if system == wc.LESSON_SYSTEM:                          # synthesis: the quick first look
        said = _said(user)
        corrects = bool(re.search(r"\b(wrong|no,|disagree)\b", said.lower()))
        return json.dumps({"principle": _lesson_for(said), "corrects": corrects})
    if system == wc.REFLECT_SYSTEM:                         # synthesis: the memo
        doubted = re.search(r"^(P\d+)\. .*\[IN DOUBT", user, flags=re.M)
        if doubted:                                         # a real model would judge it; the toy always removes it
            return f"REVIEW: {doubted.group(1)} is in doubt and the user keeps pushing back, so remove it."
        return "LESSON: " + _lesson_for(_said(user))["name"]
    if system == wc.EXECUTE_SYSTEM:                         # synthesis: the memo turned into actions
        memo = user.split("MEMO\n", 1)[-1]
        review = re.search(r"REVIEW: (P\d+)", memo)
        if review:
            return json.dumps({"actions": [{"type": "remove", "id": review.group(1),
                                            "reason": "the user kept pushing back on answers it shaped"}]})
        return json.dumps({"actions": [{"type": "add", "principle": _lesson_for(memo)}]})
    if system == wc.COMBINE_SYSTEM:                         # distillation: keep the first one listed
        first = user.split("PRINCIPLES TO COMBINE INTO ONE\n", 1)[-1].splitlines()[0].split(". ", 1)[-1]
        return json.dumps({"principle": {"name": re.split(r"\. (?:Why|Scope|Use when):", first)[0]}})
    if system == wc.DOMAIN_SYSTEM:                          # calibration on the agent's own principles
        return json.dumps({"pairs": []})
    # Anything else is the agent's answer. Show which principles reached the prompt.
    held = re.findall(r"^\d+\. (.+?)(?: Why:|$)", system, flags=re.M)
    if held:
        return f"(toy answer) Following {len(held)} principle(s), starting with: {held[0]}"
    return "(toy answer) No principles yet, so I act from the Core Axiom alone."


def embed(texts: list[str], dims: int = 512) -> list[list[float]]:
    out = []
    for text in texts:
        vec = [0.0] * dims
        for word in re.findall(r"[a-z0-9]+", str(text).lower()):
            if len(word) > 2 and word not in STOP:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
        out.append(vec or [0.0] * dims)
    return out


def offline_model() -> wc.FunctionModel:
    return wc.FunctionModel(chat, embed, name="offline-toy", embed_name="offline-toy-bow")
