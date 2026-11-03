# Concepts

Wisdom Core is built from a few ideas. This page explains each one in plain words, then how they fit together in one task.

## The Core Axiom

One sentence that says what the agent is for, such as *"Help people make sound money decisions."* It is close to the only thing you have to write.

It does two jobs. It goes into the agent's prompt at every decision, like a system prompt. And it decides what the agent learns: every principle is written, combined or removed to fit it. Give two agents the same experiences under different axioms and they end up with different principles.

The axiom is stored in the agent's database. Changing it (`core.set_core_axiom(text)`, or passing a different `core_axiom=`) keeps the old one in `core.axiom_history()`. Existing principles are kept, and any that conflict with the new axiom are removed as they come up.

## Principles

A principle is a short piece of wisdom the agent writes for itself, in service of the axiom. Each one has:

| Field | What it holds |
|---|---|
| `name` | The principle itself: one sentence on how to act. |
| `importance` | Why it matters for the Core Axiom. |
| `extent` | Where it applies and where it stops. |
| `use_case` | The situation that calls for it, the way a person would bring it up. |
| `related` | A few words a person might use when it comes up. |

The agent sees each one rendered as a line like: *"Pay off the highest-interest debt first once minimum payments are covered. Why: ... Scope: ... Use when: ..."*.

Each principle also has a **status**:

- `active`: held now, and can be injected.
- `merged`: combined into another principle, or revised (replaced by new wording). `merged_into` names the principle it became.
- `removed`: taken out by the agent during synthesis, or by you with `core.remove(principle_id, reason)`.

Ending a principle retires it; it never deletes it. A retired principle keeps its text, its `end_reason` and `ended_at`, and shows in `core.principles(include_ended=True)` and in `core.history()`.

## Injection

Before every decision, the agent recalls the principles most relevant to what was just said, and the Core Axiom plus those principles go into the prompt exactly as written, most relevant first. Twelve is the default (`context=`).

Relevance is measured by meaning, with the embedding model. Once the agent holds more principles than it injects, the model first writes one sentence describing the situation, and each principle is ranked by the better of its match to the message and to that description. In the ready-made `Agent`, the model sees the last few exchanges (`history=6` by default), and when the backend reports its context window (Ollama does) older messages are dropped if the conversation would not leave room for a reply. The axiom and the principles are never trimmed.

`core.inject(task)` returns that text, ready for your prompt. `core.recall(task)` returns the principles themselves.

## Wisdom Synthesis

After every task, the agent works out the one piece of wisdom the exchange taught. That is either what the user showed (a correction, a preference, a fact about themselves) or what its own thinking established about the subject. It then makes sure it holds a principle for that lesson.

- **First look.** One short call names the lesson without seeing the principles, and says whether the user pushed back on the previous answer. If the lesson measures as something the agent already holds, synthesis stops there. The first look is skipped while no principles are held and when you pass an `outcome`; then the whole synthesis runs. When the user pushed back, the whole synthesis always runs.
- **New ground.** Otherwise the model writes a short memo, then turns it into actions: `add`, `merge` (which also covers revising one principle), `remove` with a reason, `covered`, or `stands`.
- **Checks.** Each action is checked before it is applied. Only principles shown to the model can be touched, placeholder text is refused, and one task adds one new principle at most (`max_adds`).
- **Measured, not judged.** Whether a new principle is really new is measured by similarity, not left to the model.

A failed synthesis never costs the answer. It is reported in `SynthesisReport.error`.

## Wisdom Distillation

During synthesis, principles that say nearly the same thing are combined into one that fits the Core Axiom (at most twice per task), and a principle that conflicts with the axiom, contradicts another, or turned out to be wrong is removed. The agent has full control of its principles, and every change is logged.

## Principles earn their place

The agent remembers which principles shaped each answer (the four most relevant ones injected) and how the answer was received:

- The first look reads whether the user's next message pushed back.
- Your application can also say outright how an answer went: `success=True` or `success=False` in `synthesize()`.

A principle comes into **doubt** when it has been pushed back on at least three times since its last review, and that record would be unlikely (under 2%) for a sound principle, given how often this user pushes back on the agent's other answers. The next synthesis that lists it reviews it: the model removes it, revises it, or says it stands. A principle that has shaped eight answers or more, with at most a tenth pushed back on, reads as **proven** and isn't removed over one disagreement. Nothing is removed on a count alone. `core.standing(principle)` returns the record.

## Semantic levels

`Levels` decides how close counts as close and how much gets written. Three of them (`new_ground`, `covered`, `combine`) are measured automatically for whichever embedding model you use, because embedding models score similarity on different scales. Any level you set yourself is used as given. See the README's *Semantic levels* section for the full table.

```python
from wisdom_core import Agent, Levels
agent = Agent(core_axiom="Help people make sound money decisions.", levels=Levels(context=8, combine=0.8))
```

## One task, end to end

```
 person says something
        │
        ▼
 RECALL      principles ranked by meaning ──► INJECT  Core Axiom + top principles into the prompt
        │
        ▼
 ANSWER      the model answers (its reasoning is kept, if it shows any)
        │
        ▼
 FIRST LOOK  name the lesson; did the user push back on the last answer?
        │          └── already held? stop here
        ▼
 SYNTHESIS   memo ──► actions: add / merge / remove / covered / stands ──► checked, applied, logged
        │
        ▼
 DISTILL     combine near-duplicates the task brought up
```

## What one task costs

With learning on, one task is one model call for the answer, one for the first look once principles exist, and two more (memo and actions) on new ground. Now and then there is also a situation description (once the agent holds more principles than it injects), a combination, or a calibration, plus embedding requests. `learn=False` skips all of the learning calls.

## Where it all lives

One SQLite file per agent (`wisdom.db` unless you pass `path=`; `":memory:"` keeps it in memory). It holds every axiom, every principle (active or retired), the log of every change, and the calibration. `core.export_state()` returns all of it as plain data, without the vectors, and `core.import_state(state)` rebuilds it elsewhere. Timestamps are UTC.
