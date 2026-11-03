"""Use the Wisdom Core inside an agent you wrote yourself.

This agent answers in two decisions: first it plans, then it writes the
answer. Before EACH decision it asks the Wisdom Core for the Core Axiom and the
principles relevant to that decision. After the task it runs Wisdom Synthesis
once, with anything known about how the task went.

    python examples/custom_agent.py "How should I prepare for a job interview?"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import wisdom_core as wc  # noqa: E402

core = wc.WisdomCore(
    core_axiom="Help people get and keep good work.",
    model=wc.Ollama(),               # or wc.OpenAICompatible(...), or wc.FunctionModel(...)
    path="career_agent.db",
)


def decide(role: str, step: str, material: str) -> str:
    """One decision: the Core Axiom and relevant principles go in, as written."""
    system = f"{role}\n\n{core.inject(step)}"
    return core.model.chat([{"role": "system", "content": system},
                            {"role": "user", "content": material}])


def run(task: str) -> str:
    plan = decide("You plan answers. Reply with a short numbered plan only.",
                  f"Plan how to answer: {task}", task)
    answer = decide("You write clear, practical answers that follow the plan.",
                    f"Write the answer to: {task}", f"Task: {task}\n\nPlan:\n{plan}")
    report = core.synthesize(task, answer, outcome=None)   # pass feedback here when you have it
    print(answer)
    print("\n" + report.summary())
    return answer


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "How should I prepare for a job interview?")
