"""Look inside an agent's memory: principles, statuses, history, removal, backup and restore.

    python examples/03_inspect_and_manage.py

Nothing is ever deleted. A principle that is merged or removed stays in the database with
status "merged" or "removed", the reason, and (for merges) the principle it became.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]   # offline_model.py, and wisdom_core.py in a source checkout
from offline_model import offline_model  # noqa: E402

from wisdom_core import WisdomCore  # noqa: E402

core = WisdomCore(core_axiom="Help people make sound money decisions.", model=offline_model(), path=":memory:")
for task in ["Should I put my savings into one stock?", "Add up my rent and bills.", "Which debt first, 24% or 6%?"]:
    core.synthesize(task, "(an answer)")

print("Active principles:")
for p in core.principles():
    print(f"  [{p.status}] {p.id[:8]}  {p.name}")

# Remove one by hand. It is retired, not deleted: it keeps its text, gets status "removed" and the reason.
target = core.principles()[1]
core.remove(target.id, "the business no longer gives budgeting help")

print("\nAll principles, including ended ones (include_ended=True):")
for p in core.principles(include_ended=True):
    extra = f"  <- {p.end_reason}" if p.status != "active" else ""
    print(f"  [{p.status}] {p.name}{extra}")

print("\nHistory, newest first:")
for e in core.history(limit=10):
    print(f"  {e.kind:<8} {e.detail[:80]}")

# How the answers a principle shaped were received (all zero here: no feedback was given).
print("\nStanding of the first principle:", core.standing(core.principles()[0]))

# Back up the learning as plain data, then rebuild it in a fresh database.
state = core.export_state()
print(f"\nExported {len(json.dumps(state))} bytes of JSON-ready data: axiom, principles, log, calibration.")
# A new database still needs a Core Axiom to open; the import then replaces it with the backup's axiom history.
copy = WisdomCore(core_axiom=state["axiom"], model=offline_model(), path=":memory:")
print("Imported:", copy.import_state(state))
print("Copy's axiom:", copy.core_axiom)
print("Copy's active principles:", [p.name for p in copy.principles()])

# Changing the Core Axiom keeps every older one.
core.set_core_axiom("Help small businesses make sound money decisions.")
print("\nAxiom history:", [text for text, _ in core.axiom_history()])
core.close()
copy.close()
