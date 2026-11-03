"""Quickstart with no model server: an Agent learns a principle, then uses it.

    python examples/01_quickstart_offline.py

Uses the rule-based stand-in in offline_model.py, so the answers are canned. With Ollama
running, replace `model=offline_model()` with nothing at all (Ollama is the default).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]   # offline_model.py, and wisdom_core.py in a source checkout
from offline_model import offline_model  # noqa: E402

from wisdom_core import Agent  # noqa: E402

path = os.path.join(tempfile.mkdtemp(prefix="wisdom-quickstart-"), "agent.db")

with Agent(core_axiom="Help people make sound money decisions.", model=offline_model(), path=path) as agent:
    for message in [
        "I have a car loan at 6% and a credit card at 24%. Which should I pay off first?",
        "My cousin has a 19% store card and a 5% loan. What goes first?",
    ]:
        print("you>  ", message)
        print("agent>", agent.ask(message))
        print("       principles injected:", len(agent.last_principles))
        for line in agent.last_report.lines():          # what Wisdom Synthesis did
            print("       wisdom core", line)
        print()

    print("Principles held:")
    for p in agent.principles():
        print(" -", p.name)

print(f"\nThe agent's memory is one SQLite file: {path}")
print(f"Inspect it with:  wisdom-core --db {path} --principles --history")
