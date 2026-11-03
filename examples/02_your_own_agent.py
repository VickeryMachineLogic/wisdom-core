"""Use WisdomCore inside an agent you already have: inject before, synthesize after.

    python examples/02_your_own_agent.py

`my_agent` stands in for your own code. The Wisdom Core only needs two calls from you:
core.inject(task) before the agent decides, and core.synthesize(...) after the task.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]   # offline_model.py, and wisdom_core.py in a source checkout
from offline_model import offline_model  # noqa: E402

from wisdom_core import WisdomCore  # noqa: E402

core = WisdomCore(core_axiom="Help people make sound money decisions.",
                  model=offline_model(), path=":memory:")   # ":memory:" keeps nothing on disk


def my_agent(system_prompt: str, task: str) -> str:
    """Your own agent: any model call, tool use or workflow. Here, the same toy model."""
    return core.model.chat([{"role": "system", "content": system_prompt},
                            {"role": "user", "content": task}])


tasks = [
    ("I have $5,000 saved. Should I put it all into one stock?", "The user said it helped."),
    ("Can you add up my bills? Rent 1,250, phone 85, power 140.", None),
    ("Which is first: the 24% credit card or the 4% student loan?", None),
]
for task, outcome in tasks:
    principles = core.recall(task)                                   # the relevant principles themselves
    system_prompt = "You are a careful assistant.\n\n" + core.format_context(principles)
    answer = my_agent(system_prompt, task)
    report = core.synthesize(task, answer, outcome=outcome, used=principles)
    print(f"task:   {task}\nanswer: {answer}\n" + "\n".join("        " + l for l in report.lines()) + "\n")

print("What gets injected for a new debt question:\n")
print(core.inject("Should I pay my credit card or my car loan first?", expand=False))
print("\nstats:", core.stats())
core.close()
