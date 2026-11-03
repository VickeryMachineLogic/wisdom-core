"""Principles earn their place: pushback and outcomes count against the principles behind an answer.

    python examples/04_feedback_and_events.py

respond() answers without learning yet, so the answer can go out first; synthesize() learns
afterwards and can take success=True/False when your application knows how things went.
When the user keeps pushing back on answers a principle shaped, that principle comes into
doubt and is reviewed. on_event shows each step as it happens.

The review decision here comes from the toy model in offline_model.py, which always removes a
principle in doubt and treats "No, that's wrong" as pushback. A real model reads the exchange
and may instead revise the principle or decide that it stands.
"""

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]   # offline_model.py, and wisdom_core.py in a source checkout
from offline_model import offline_model  # noqa: E402

from wisdom_core import Agent  # noqa: E402

seen = Counter()


def watch(event: dict) -> None:
    """Called with a dict for every step: recall, model calls, synthesis stages, upkeep."""
    seen[event["type"]] += 1


agent = Agent(core_axiom="Help people make sound money decisions.", model=offline_model(),
              path=":memory:", on_event=watch)

messages = [
    "Which debt should I pay first, the 22% card or the 7% loan?",
    "And my brother: 18% card or 6% car loan?",               # we report this answer went well
    "No, that's wrong. My card has a 0% promo for a year.",
    "No, that's wrong again. The loan has a prepayment bonus.",
    "No, that's wrong. You keep ignoring my card's promo rate.",
    "No, that's wrong. The promo matters more than the rate.",
]
for n, message in enumerate(messages, 1):
    agent.respond(message)                                  # answer first...
    report = agent.synthesize(success=True if n == 2 else None)   # ...then learn
    print(f"[{n}] you> {message}")
    for line in report.lines():
        print("      ", line)

print("\nEvery principle ever held (include_ended=True):")
for p in agent.core.principles(include_ended=True):
    print(f"  [{p.status}] {p.name}" + (f"  <- {p.end_reason}" if p.end_reason else ""))
print("\nstats:", agent.core.stats())
print("events seen:", dict(seen))
agent.close()
