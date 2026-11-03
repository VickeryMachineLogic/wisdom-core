"""The Wisdom Core itself: axiom, storage, recall, injection, synthesis, the agent."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

from typing import Any

import wisdom_core as wc
from tests.fakes import ScriptedModel, bow_embed, close_stores, many, principle, seed

tearDownModule = close_stores          # close every database this module's tests opened


AXIOM = "Help people make sound money decisions."

SAVINGS = principle("Ask when the money will be needed before suggesting where to keep savings",
                    "Advice that ignores the time horizon can put needed money at risk",
                    "Savings and investment questions", "someone asks where to keep savings",
                    ["time horizon", "savings", "risk"])
TOTALS = principle("Add the figures twice before reporting a total",
                   "A wrong total misleads the person", "Any sum given to a person",
                   "adding up amounts, budgets or bills", ["arithmetic", "budget"])
DEBT = principle("List debts by interest rate before suggesting which to repay first",
                 "Paying the costliest debt first saves the most money", "Debt repayment questions",
                 "someone has several debts", ["interest rate", "loans", "credit cards"])


class TempDir(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="wisdom-test-")
        self.path = os.path.join(self.dir, "wisdom.db")

    def tearDown(self) -> None:
        close_stores()
        shutil.rmtree(self.dir, ignore_errors=True)


def memory_core(model: ScriptedModel | None = None, **kw) -> tuple[wc.WisdomCore, ScriptedModel]:
    model = model or ScriptedModel()
    return wc.WisdomCore(AXIOM, model, path=":memory:", **kw), model


def careful_core(model: ScriptedModel | None = None, **kw) -> tuple[wc.WisdomCore, ScriptedModel]:
    """A core with the force level off: the model decides for itself whether a task taught anything.
    The default (a principle from every task, decided by measurement) is covered by LevelsTests."""
    return memory_core(model, levels=wc.Levels(force=False), **kw)


# ---------------------------------------------------------------------------
class AxiomTests(TempDir):
    def test_a_new_database_needs_an_axiom(self):
        with self.assertRaises(wc.WisdomCoreError):
            wc.WisdomCore(None, ScriptedModel(), path=self.path)
        with self.assertRaises(wc.WisdomCoreError):
            wc.WisdomCore("   ", ScriptedModel(), path=self.path)

    def test_axiom_is_stored_and_read_back(self):
        wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path).close()
        with wc.WisdomCore(None, ScriptedModel(), path=self.path) as core:
            self.assertEqual(core.core_axiom, AXIOM)
            self.assertEqual(len(core.axiom_history()), 1)

    def test_same_axiom_again_adds_no_history(self):
        wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path).close()
        with wc.WisdomCore("  " + AXIOM + "\n", ScriptedModel(), path=self.path) as core:
            self.assertEqual(len(core.axiom_history()), 1)

    def test_changing_the_axiom_keeps_principles_and_records_the_change(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path) as core:
            seed(core, SAVINGS)
        with self.assertLogs("wisdom_core", level="WARNING"):
            core = wc.WisdomCore("Protect people from financial scams.", ScriptedModel(), path=self.path)
        with core:
            self.assertEqual(core.core_axiom, "Protect people from financial scams.")
            self.assertEqual([t for t, _ in core.axiom_history()], [AXIOM, "Protect people from financial scams."])
            self.assertEqual(len(core.principles()), 1)
            axiom_events = [e for e in core.history() if e.kind == "axiom"]
            self.assertIn(f"changed from: {AXIOM}", axiom_events[0].detail)

    def test_set_core_axiom(self):
        core, _ = memory_core()
        core.set_core_axiom("New purpose.")
        self.assertEqual(core.core_axiom, "New purpose.")
        with self.assertRaises(ValueError):
            core.set_core_axiom("  ")

    def test_new_principles_record_the_axiom_they_were_learned_under(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path) as core:
            first = seed(core, SAVINGS)[0]
            core.set_core_axiom("Second axiom.")
            second = seed(core, TOTALS)[0]
        con = sqlite3.connect(self.path)
        rows = dict(con.execute("SELECT p.id, a.text FROM principles p JOIN axioms a ON a.id = p.axiom_id"))
        con.close()
        self.assertEqual(rows[first.id], AXIOM)
        self.assertEqual(rows[second.id], "Second axiom.")

    def test_context_must_be_positive(self):
        with self.assertRaises(ValueError):
            wc.WisdomCore(AXIOM, ScriptedModel(), path=":memory:", context=0)


# ---------------------------------------------------------------------------
class StorageTests(TempDir):
    def test_principles_survive_reopening(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path) as core:
            seed(core, SAVINGS, TOTALS)
        with wc.WisdomCore(None, ScriptedModel(), path=self.path) as core:
            got = core.principles()
            self.assertEqual([p.name for p in got], [SAVINGS["name"], TOTALS["name"]])
            self.assertEqual(got[0].importance, SAVINGS["importance"])
            self.assertEqual(got[0].extent, SAVINGS["extent"])
            self.assertEqual(got[0].use_case, SAVINGS["use_case"])
            self.assertEqual(got[0].related, SAVINGS["related"])
            self.assertEqual(got[0].status, "active")

    def test_database_in_a_new_folder(self):
        path = os.path.join(self.dir, "a", "b", "agent.db")
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=path):
            pass
        self.assertTrue(os.path.exists(path))

    def test_database_from_a_newer_version_is_refused(self):
        wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path).close()
        con = sqlite3.connect(self.path)
        con.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        con.commit()
        con.close()
        with self.assertRaises(wc.WisdomCoreError):
            wc.WisdomCore(None, ScriptedModel(), path=self.path)

    def test_a_database_from_something_else_is_refused(self):
        con = sqlite3.connect(self.path)
        con.execute("CREATE TABLE invoices (id INTEGER)")
        con.commit()
        con.close()
        with self.assertRaises(wc.WisdomCoreError) as err:
            wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path)
        self.assertIn("did not create", str(err.exception))
        con = sqlite3.connect(self.path)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        con.close()
        self.assertEqual(tables, {"invoices"})           # left exactly as it was

    def test_a_file_that_is_not_a_database_is_refused(self):
        with open(self.path, "w") as fh:
            fh.write("just some text, definitely not SQLite " * 50)
        with self.assertRaises(wc.WisdomCoreError) as err:
            wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path)
        self.assertIn("not a Wisdom Core database", str(err.exception))

    def test_event_times_are_distinct_and_ordered(self):
        core, _ = memory_core()
        seed(core, SAVINGS, TOTALS, DEBT)
        stamps = [e.at for e in reversed(core.history())]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len(set(stamps)), len(stamps))

    def test_a_second_process_sees_new_principles(self):
        a = wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path)
        b = wc.WisdomCore(None, ScriptedModel(), path=self.path)
        try:
            seed(a, SAVINGS)
            self.assertEqual([p.name for p in b.recall("where to keep my savings", expand=False)], [SAVINGS["name"]])
            seed(a, DEBT)
            names = [p.name for p in b.recall("which debt to repay first", expand=False)]
            self.assertEqual(names[0], DEBT["name"])
        finally:
            a.close()
            b.close()


# ---------------------------------------------------------------------------
class RecallTests(unittest.TestCase):
    def test_an_empty_core_calls_no_model(self):
        core, model = memory_core()
        self.assertEqual(core.recall("anything"), [])
        self.assertEqual(model.calls, [])
        self.assertEqual(model.embed_calls, [])

    def test_ranked_by_meaning(self):
        core, _ = memory_core()
        seed(core, TOTALS, DEBT, SAVINGS)
        got = core.recall("Where should I keep my savings for a house?", expand=False)
        self.assertEqual(got[0].name, SAVINGS["name"])
        scores = [p.score for p in got]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_context_limits_how_many(self):
        core, _ = memory_core(context=2)
        seed(core, TOTALS, DEBT, SAVINGS)
        self.assertEqual(len(core.recall("savings debt totals", expand=False)), 2)
        self.assertEqual(len(core.recall("savings debt totals", expand=False, k=1)), 1)
        self.assertEqual(core.recall("savings", expand=False, k=0), [])

    def test_default_context_is_twelve(self):
        core, _ = memory_core()
        self.assertEqual(core.context, 12)
        seed(core, *many(15, "budgeting rule"))
        self.assertEqual(len(core.recall("budgeting", expand=False)), 12)

    def test_no_situation_call_when_every_principle_fits(self):
        core, model = memory_core()
        seed(core, TOTALS, DEBT)
        self.assertEqual(len(core.recall("my budget")), 2)
        self.assertNotIn("situation", model.kinds())

    def test_situation_description_finds_principles_the_task_words_miss(self):
        core, model = memory_core(context=1)
        seed(core, TOTALS, DEBT, SAVINGS)
        task = "Help me sort out this savings mess please"
        self.assertEqual(core.recall(task, expand=False)[0].name, SAVINGS["name"])   # the words alone
        model.queue("situation", "The person has several debts: loans and credit cards with different "
                                 "interest rates, and wants a repayment order.")
        self.assertEqual(core.recall(task)[0].name, DEBT["name"])                    # with the description
        prompt = model.prompts("situation")[0]
        self.assertEqual(prompt[0]["content"], wc.SITUATION_SYSTEM)
        self.assertIn(AXIOM, prompt[1]["content"])
        self.assertIn(task, prompt[1]["content"])

    def test_expand_false_skips_the_situation_call(self):
        core, model = memory_core(context=1)
        seed(core, TOTALS, DEBT, SAVINGS)
        core.recall("savings", expand=False)
        self.assertNotIn("situation", model.kinds())

    def test_situation_sees_the_conversation(self):
        core, model = memory_core(context=1)
        seed(core, TOTALS, DEBT, SAVINGS)
        core.recall("and the second one?", conversation=[
            {"role": "user", "content": "I have a car loan and a credit card."},
            {"role": "assistant", "content": "Which would you like to discuss?"},
        ])
        self.assertIn("I have a car loan and a credit card.", model.last_user("situation"))

    def test_merged_and_removed_principles_are_not_recalled(self):
        core, model = memory_core()
        s, t, d = seed(core, SAVINGS, TOTALS, DEBT)
        core.remove(t.id, "wrong")
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P2"],
                                             "principle": principle("Combined savings and debt rule")}]})
        core.synthesize("task", "result", used=[s, d])
        names = [p.name for p in core.recall("savings debt totals budget", expand=False)]
        self.assertEqual(names, ["Combined savings and debt rule"])

    def test_merged_principles_do_not_crowd_out_live_ones(self):
        core, model = memory_core()
        a, b, c = seed(core, SAVINGS, principle("Ask when savings will be needed before choosing an account"), DEBT)
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P2"],
                                             "principle": principle("Match the account to the timeline")}]})
        core.synthesize("task", "result", used=[a, b])
        got = core.recall("Where should I keep my savings?", expand=False, k=1)
        self.assertEqual(len(got), 1)
        self.assertNotIn(got[0].id, {a.id, b.id})

    def test_times_injected_is_counted(self):
        core, _ = memory_core()
        (s,) = seed(core, SAVINGS)
        core.recall("savings", expand=False)
        got = core.recall("savings", expand=False)
        self.assertEqual(got[0].times_injected, 2)
        self.assertEqual(core.get(s.id).times_injected, 2)

    def test_injection_carries_the_axiom_and_principles_as_written(self):
        core, _ = memory_core()
        seed(core, SAVINGS, DEBT)
        recalled = core.recall("where to keep savings", expand=False)
        block = core.format_context(recalled)
        self.assertTrue(block.startswith(f"CORE AXIOM\n{AXIOM}\n\nPRINCIPLES\n{wc.PRINCIPLES_PREAMBLE}\n"))
        for i, p in enumerate(recalled, 1):
            self.assertIn(f"{i}. {p.render()}", block)
        for text in (SAVINGS["name"], SAVINGS["importance"], SAVINGS["extent"]):
            self.assertIn(text, block)

    def test_inject_with_no_principles(self):
        core, _ = memory_core()
        self.assertEqual(core.inject("anything"), f"CORE AXIOM\n{AXIOM}\n\nPRINCIPLES\n{wc.NO_PRINCIPLES}")

    def test_pure_python_and_numpy_rank_the_same(self):
        core, _ = memory_core()
        seed(core, TOTALS, DEBT, SAVINGS, principle("Explain fees in plain words"))
        with_np = [(p.id, round(p.score, 5)) for p in core.recall("fees savings debt", expand=False)]
        core._index = None
        with mock.patch.object(wc, "_np", None):
            without = [(p.id, round(p.score, 5)) for p in core.recall("fees savings debt", expand=False)]
        self.assertEqual(with_np, without)


# ---------------------------------------------------------------------------
class SynthesisTests(unittest.TestCase):
    def test_the_wisdom_core_reads_the_task_it_just_finished(self):
        core, model = careful_core()
        core.synthesize("TASK-7f3 How much should I save each month?", "RESULT-9c1 About 15 percent of pay.")
        system, user = model.prompts("reflect")[0]
        self.assertEqual(system["content"], wc.REFLECT_SYSTEM)
        self.assertIn(AXIOM, user["content"])
        self.assertIn("TASK-7f3 How much should I save each month?", user["content"])
        self.assertIn("RESULT-9c1 About 15 percent of pay.", user["content"])

    def test_first_task_can_write_the_first_principle(self):
        core, model = careful_core()
        model.queue("execute", {"actions": [{"type": "add", "principle": TOTALS}]})
        report = core.synthesize("Add up my bills", "They come to 420.")
        self.assertEqual([p.name for p in report.added], [TOTALS["name"]])
        (p,) = core.principles()
        self.assertEqual(p.learned_from, "Add up my bills")
        self.assertEqual(p.related, TOTALS["related"])
        self.assertIn("(none: the agent holds no principles yet, so there are no labels to refer to)",
                      model.last_user("reflect"))
        self.assertEqual(core.history(1)[0].kind, "added")
        self.assertTrue(report.changed)

    def test_covered_changes_nothing(self):
        core, model = careful_core()
        (t,) = seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "covered", "id": "P1"}]})
        report = core.synthesize("Add up my bills", "420")
        self.assertEqual([p.id for p in report.covered], [t.id])
        self.assertFalse(report.changed)
        self.assertEqual([p.id for p in core.principles()], [t.id])
        self.assertEqual(core.history(1)[0].kind, "covered")

    def test_injected_principles_are_listed_first(self):
        core, model = careful_core()
        s, t, d = seed(core, SAVINGS, TOTALS, DEBT)
        core.synthesize("Add up my bills", "420", used=[d])
        listing = model.last_user("execute")
        self.assertIn(f"P1. {d.name}", listing)

    def test_the_wisdom_core_sees_at_most_twenty_principles(self):
        core, model = careful_core()
        held = seed(core, *many(30, "budget money rule"))
        core.synthesize("budget money", "result about budget money", used=held)
        listing = model.last_user("execute").split("LISTED PRINCIPLES\n")[1].split("\n\nMEMO")[0]
        self.assertEqual(len(listing.splitlines()), wc.MAX_CANDIDATES)
        self.assertEqual(listing.splitlines()[0], f"P1. {held[0].name}")

    def test_the_executor_sees_the_labels_and_the_memo(self):
        core, model = careful_core()
        (t,) = seed(core, TOTALS)
        model.queue("reflect", "MEMO-5d2: P1 already covers this.")
        core.synthesize("Add up my bills", "420")
        user = model.last_user("execute")
        self.assertIn(f"P1. {t.name}", user)
        self.assertIn("MEMO-5d2: P1 already covers this.", user)
        self.assertIn(AXIOM, user)
        kwargs = [kw for k, _, kw in model.calls if k == "execute"][0]
        self.assertEqual(kwargs, {"json_mode": True, "temperature": 0.0})

    def test_merge_replaces_its_sources(self):
        core, model = careful_core()
        a, b = seed(core, TOTALS, principle("Double-check every sum before giving it"))
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P2"],
                                             "principle": principle("Check every total twice before giving it",
                                                                    "Wrong sums mislead")}]})
        report = core.synthesize("Add these up", "Total 12", used=[a, b])
        ((sources, merged),) = report.merged
        self.assertEqual({s.id for s in sources}, {a.id, b.id})
        self.assertEqual([p.name for p in core.principles()], ["Check every total twice before giving it"])
        for old in (a, b):
            stored = core.get(old.id)
            self.assertEqual(stored.status, "merged")
            self.assertEqual(stored.merged_into, merged.id)
        event = core.history(1)[0]
        self.assertEqual((event.kind, set(event.sources)), ("merged", {a.id, b.id}))

    def test_merge_naming_an_unknown_label_changes_nothing(self):
        core, model = careful_core()
        a, b = seed(core, TOTALS, DEBT)
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P9"],
                                             "principle": principle("Merged rule")}]})
        report = core.synthesize("task", "result", used=[a, b])
        self.assertEqual(report.merged, [])
        self.assertEqual(len(report.skipped), 1)
        self.assertEqual({p.id for p in core.principles()}, {a.id, b.id})

    def test_merge_with_one_unknown_label_among_good_ones_changes_nothing(self):
        core, model = careful_core()
        a, b, c = seed(core, TOTALS, DEBT, SAVINGS)
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P2", "P7"],
                                             "principle": principle("Merged rule")}]})
        report = core.synthesize("task", "result", used=[a, b, c])
        self.assertEqual(report.merged, [])
        self.assertIn("P7", report.skipped[0])
        self.assertEqual({p.id for p in core.principles()}, {a.id, b.id, c.id})

    def test_a_merge_needs_a_listed_principle_and_a_new_one(self):
        core, model = careful_core()
        a, b = seed(core, TOTALS, DEBT)
        model.queue("execute", {"actions": [
            {"type": "merge", "ids": ["P9"], "principle": principle("One")},
            {"type": "merge", "ids": [], "principle": principle("Two")},
            {"type": "merge", "ids": ["P1", "P2"]},
        ]})
        report = core.synthesize("task", "result", used=[a, b])
        self.assertEqual(report.merged, [])
        self.assertEqual(len(report.skipped), 3)
        self.assertEqual({p.id for p in core.principles()}, {a.id, b.id})

    def test_one_listed_principle_in_a_merge_is_revised(self):
        core, model = careful_core()
        a, b = seed(core, TOTALS, DEBT)
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P1"],
                                             "principle": principle("Add every figure twice and show the sum")}]})
        report = core.synthesize("Add up my bills", "420", used=[a, b])
        ((sources, revised),) = report.merged
        self.assertEqual(([s.id for s in sources], revised.name), ([a.id], "Add every figure twice and show the sum"))
        self.assertEqual(report.lines(), [f"= revised: {TOTALS['name']} -> Add every figure twice and show the sum"])
        old = core.get(a.id)
        self.assertEqual((old.status, old.merged_into, old.end_reason),
                         ("merged", revised.id, "revised as: Add every figure twice and show the sum"))
        self.assertEqual(sorted(p.name for p in core.principles()), sorted([DEBT["name"], revised.name]))

    def test_remove_keeps_the_principle_in_the_log_with_its_reason(self):
        core, model = careful_core()
        a, b = seed(core, TOTALS, DEBT)
        model.queue("execute", {"actions": [{"type": "remove", "id": "P2", "reason": "It conflicts with the axiom"}]})
        report = core.synthesize("task", "result", used=[a, b])
        ((gone, reason),) = report.removed
        self.assertEqual((gone.id, reason), (b.id, "It conflicts with the axiom"))
        self.assertEqual([p.id for p in core.principles()], [a.id])
        ended = {p.id: p for p in core.principles(include_ended=True)}
        self.assertEqual(ended[b.id].status, "removed")
        self.assertEqual(ended[b.id].end_reason, "It conflicts with the axiom")
        self.assertEqual(core.history(1)[0].kind, "removed")

    def test_a_conflict_is_removed_not_just_flagged(self):
        core, model = careful_core()
        a, b = seed(core, TOTALS, DEBT)
        model.queue("execute", {"actions": [{"type": "conflict", "id": "P1", "reason": "contradicts P2"}]})
        report = core.synthesize("task", "result", used=[a, b])
        self.assertEqual([p.id for p, _ in report.removed], [a.id])
        self.assertEqual([p.id for p in core.principles()], [b.id])

    def test_remove_needs_a_real_reason(self):
        core, model = careful_core()
        (a,) = seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "remove", "id": "P1"},
                                            {"type": "remove", "id": "P1", "reason": "..."},
                                            {"type": "remove", "id": "P1", "reason": "<reason>"}]})
        report = core.synthesize("task", "result", used=[a])
        self.assertEqual(report.removed, [])
        self.assertEqual(len(report.skipped), 3)
        self.assertEqual(core.get(a.id).status, "active")

    def test_a_model_that_copies_the_example_changes_nothing(self):
        core, model = careful_core()
        held = seed(core, TOTALS, DEBT, SAVINGS, principle("Explain every fee in plain words"))
        example = wc.EXECUTE_SYSTEM.split("in this form:\n", 1)[1].split("\n\nAction types", 1)[0]
        model.queue("execute", example)
        report = core.synthesize("task", "result", used=held)
        self.assertFalse(report.changed, report.lines())
        self.assertEqual({p.id for p in core.principles()}, {p.id for p in held})
        self.assertEqual(len(report.skipped), 4)            # add, merge, remove and stands refused

    def test_placeholder_principles_are_not_added(self):
        core, model = careful_core()
        model.queue("execute", {"actions": [{"type": "add", "principle": {"name": "..."}},
                                            {"type": "add", "principle": {"name": "<name>"}},
                                            {"type": "add", "principle": {"name": "N/A"}}]})
        report = core.synthesize("task", "result")
        self.assertEqual(report.added, [])
        self.assertEqual(len(report.skipped), 3)

    def test_principles_that_were_not_listed_cannot_be_touched(self):
        core, model = careful_core()
        related = seed(core, *many(20, "budget spending groceries rent"))
        (victim,) = seed(core, principle("Quantum chromodynamics lattice gluon"))
        model.queue("execute", {"actions": [
            {"type": "remove", "id": victim.id, "reason": "not listed"},
            {"type": "remove", "id": "P99", "reason": "no such label"},
        ]})
        report = core.synthesize("budget spending groceries rent", "budget spending plan for groceries and rent")
        listing = model.last_user("execute")
        self.assertNotIn(victim.name, listing)
        self.assertEqual(report.removed, [])
        self.assertEqual(len(report.skipped), 2)
        self.assertEqual(core.get(victim.id).status, "active")
        self.assertTrue(all(core.get(p.id).status == "active" for p in related))

    def test_a_near_identical_new_principle_counts_as_covered(self):
        core, model = careful_core()
        (t,) = seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "add", "principle": TOTALS}]})
        report = core.synthesize("Add up my bills", "420")
        self.assertEqual(report.added, [])
        self.assertEqual([p.id for p in report.covered], [t.id])
        self.assertEqual(len(core.principles()), 1)

    def test_one_new_principle_per_task_by_default(self):
        core, model = careful_core()
        model.queue("execute", {"actions": [{"type": "add", "principle": principle(f"Distinct rule {w}")}
                                            for w in ("alpha", "bravo", "charlie", "delta", "echo")]})
        report = core.synthesize("task", "result")
        self.assertEqual(wc.MAX_ADDS_PER_TASK, 1)
        self.assertEqual([p.name for p in report.added], ["Distinct rule alpha"])
        self.assertEqual(len(report.skipped), 4)
        self.assertIn("Add at most 1 new principle from this task.", model.last_user("execute"))

    def test_a_principle_merged_in_this_pass_cannot_also_be_removed(self):
        core, model = careful_core()
        a, b = seed(core, TOTALS, DEBT)
        model.queue("execute", {"actions": [
            {"type": "merge", "ids": ["P1", "P2"], "principle": principle("Combined")},
            {"type": "remove", "id": "P1", "reason": "late"},
            {"type": "covered", "id": "P2"},
        ]})
        report = core.synthesize("task", "result", used=[a, b])
        self.assertEqual(len(report.merged), 1)
        self.assertEqual(report.removed, [])
        self.assertEqual(report.covered, [])
        self.assertEqual(len(report.skipped), 2)

    def test_invalid_json_is_asked_for_again(self):
        core, model = careful_core()
        model.queue("execute", "Sure! Here are the actions I would take.",
                    {"actions": [{"type": "add", "principle": TOTALS}]})
        report = core.synthesize("task", "result")
        self.assertEqual(len(report.added), 1)
        retry = model.prompts("execute")[1]
        self.assertEqual(retry[-1]["content"], wc.JSON_RETRY)
        self.assertEqual(retry[-2]["role"], "assistant")

    def test_invalid_json_twice_is_reported_not_raised(self):
        core, model = careful_core()
        model.queue("execute", "no", "still no")
        report = core.synthesize("task", "result")
        self.assertIsNotNone(report.error)
        self.assertFalse(report.changed)
        self.assertEqual(core.history(1)[0].kind, "failed")
        self.assertIn("learning failed", report.summary())

    def test_a_model_failure_is_reported_not_raised(self):
        core, model = careful_core()
        model.queue("reflect", wc.ModelError("model is down"))
        report = core.synthesize("task", "result")
        self.assertIn("model is down", report.error)
        self.assertEqual(core.stats()["failed"], 1)

    def test_an_empty_memo_is_a_failure(self):
        core, model = careful_core()
        model.queue("reflect", "<think>hmm</think>   ")
        report = core.synthesize("task", "result")
        self.assertIn("empty memo", report.error)
        self.assertNotIn("execute", model.kinds())

    def test_nothing_to_learn(self):
        core, _ = careful_core()
        report = core.synthesize("hi", "Hello!")
        self.assertEqual(report.lines(), ["~ nothing new to learn"])
        self.assertEqual(core.history(1)[0].kind, "nothing")

    def test_unknown_action_is_ignored(self):
        core, model = careful_core()
        model.queue("execute", {"actions": [{"type": "flag", "id": "P1"}, {"type": "none"}]})
        report = core.synthesize("task", "result")
        self.assertEqual(report.skipped, ["unknown action 'flag'"])

    def test_outcome_and_conversation_reach_the_wisdom_core(self):
        core, model = careful_core()
        core.synthesize("Which card first?", "Pay the 24 percent card first.",
                        outcome="OUTCOME-3b: the person said this saved them money",
                        conversation=[{"role": "user", "content": "EARLIER-8e: I have two cards."},
                                      {"role": "assistant", "content": "Tell me their rates."}])
        user = model.last_user("reflect")
        self.assertIn("WHAT HAPPENED AFTERWARDS\nOUTCOME-3b", user)
        self.assertIn("EARLIER IN THE CONVERSATION\nUser: EARLIER-8e: I have two cards.", user)

    def test_loosely_written_principles_are_understood(self):
        core, model = careful_core()
        model.queue("execute", """```json
{"actions": [{"type": "Add", "principle": {"Name": "Confirm the currency before converting amounts",
  "why": "Mixing currencies gives wrong totals", "Scope": "Any amount in two currencies",
  "useCase": "When amounts come in different currencies", "edges": "currency, exchange rate"},},]}
```""")
        report = core.synthesize("task", "result")
        (p,) = report.added
        self.assertEqual(p.importance, "Mixing currencies gives wrong totals")
        self.assertEqual(p.extent, "Any amount in two currencies")
        self.assertEqual(p.use_case, "When amounts come in different currencies")
        self.assertEqual(p.related, ["currency", "exchange rate"])

    def test_a_bug_in_your_own_model_code_is_raised(self):
        def chat(messages):
            if messages[0]["content"] == wc.REFLECT_SYSTEM:
                raise RuntimeError("bug in my model code")
            return "ok"
        core = wc.WisdomCore(AXIOM, wc.FunctionModel(chat, lambda texts: [[1.0, 0.0]] * len(texts)), path=":memory:")
        with self.assertRaises(RuntimeError):
            core.synthesize("task", "result")

    def test_principles_in_other_scripts_are_kept(self):
        core, model = careful_core()
        core.levels.max_adds = 2
        model.queue("execute", {"actions": [
            {"type": "add", "principle": {"name": "Перед советом уточняйте срок вложений"}},
            {"type": "add", "principle": {"name": "先确认期限再给投资建议"}},
        ]})
        report = core.synthesize("task", "result")
        self.assertEqual(len(report.added), 2, report.lines())

    def test_empty_text_is_never_sent_for_embedding(self):
        core, model = careful_core()
        seed(core, TOTALS)
        core.synthesize("", "")
        self.assertNotIn("", [t for batch in model.embed_calls for t in batch])

    def test_syntheses_are_counted(self):
        core, _ = careful_core()
        core.synthesize("a", "b")
        core.synthesize("c", "d")
        self.assertEqual(core.stats()["syntheses"], 2)


# ---------------------------------------------------------------------------
class AgentTests(unittest.TestCase):
    def make(self, **kw) -> tuple[wc.Agent, ScriptedModel]:
        model = ScriptedModel()
        return wc.Agent(AXIOM, model, path=":memory:", **kw), model

    def test_the_answer_prompt_carries_the_axiom_and_principles_as_written(self):
        agent, model = self.make()
        seed(agent.core, SAVINGS, DEBT)
        agent.ask("Where should I keep my savings?")
        system = model.prompts("answer")[0][0]["content"]
        self.assertTrue(system.startswith(wc.AGENT_PREAMBLE))
        self.assertIn(f"CORE AXIOM\n{AXIOM}", system)
        self.assertEqual(len(agent.last_principles), 2)
        for i, p in enumerate(agent.last_principles, 1):
            self.assertIn(f"{i}. {p.render()}", system)

    def test_principles_are_recalled_again_for_every_decision(self):
        agent, model = self.make(context=1)
        seed(agent.core, SAVINGS, DEBT)
        agent.ask("Where should I keep my savings?")
        agent.ask("Which debt should I repay first, the loan or the credit card?")
        first, second = (m[0]["content"] for m in model.prompts("answer"))
        self.assertIn(SAVINGS["name"], first)
        self.assertIn(DEBT["name"], second)
        self.assertNotIn(DEBT["name"], first)

    def test_each_task_is_learned_from_right_after_it_is_done(self):
        agent, model = self.make()
        model.queue("answer", "ANSWER-ONE", "ANSWER-TWO")
        agent.ask("QUESTION-ONE about savings")
        self.assertIn("QUESTION-ONE about savings", model.last_user("reflect"))
        self.assertIn("ANSWER-ONE", model.last_user("reflect"))
        agent.ask("QUESTION-TWO about debt")
        self.assertIn("QUESTION-TWO about debt", model.last_user("reflect"))
        self.assertIn("ANSWER-TWO", model.last_user("reflect"))
        # two calls to execute per task: the second is the forced principle, since neither task drafted one
        self.assertEqual(model.kinds(), ["answer", "reflect", "execute", "execute"] * 2)

    def test_the_agent_learns_and_uses_what_it_learned(self):
        agent, model = self.make()
        model.queue("execute", {"actions": [{"type": "add", "principle": DEBT}]})
        agent.ask("I have a car loan at 6% and a card at 24%. Which first?")
        self.assertEqual([p.name for p in agent.last_report.added], [DEBT["name"]])
        agent.ask("Now I also have a student loan. Which debt to repay first?")
        self.assertIn(DEBT["name"], model.prompts("answer")[1][0]["content"])
        self.assertEqual([p.name for p in agent.last_principles], [DEBT["name"]])

    def test_feedback_in_the_next_message_reaches_the_wisdom_core(self):
        agent, model = self.make()
        model.queue("answer", "Pay the car loan first.", "Understood, the card first.")
        agent.ask("Car loan 6% or card 24%, which first?")
        agent.ask("That was wrong, the card costs more.")
        user = model.last_user("reflect")
        self.assertIn("EARLIER IN THE CONVERSATION\nUser: Car loan 6% or card 24%, which first?", user)
        self.assertIn("THE AGENT'S PREVIOUS ANSWER, WHICH THE USER IS RESPONDING TO\nPay the car loan first.", user)
        self.assertIn("WHAT THE USER SAID\nThat was wrong, the card costs more.", user)
        self.assertIn("THE AGENT'S ANSWER\nUnderstood, the card first.", user)
        self.assertNotIn("Agent: Pay the car loan first.", user)          # shown once, as what is being answered

    def test_respond_now_learn_later_with_an_outcome(self):
        agent, model = self.make()
        reply = agent.respond("Which card should I pay first?")
        self.assertEqual(model.kinds(), ["answer"])
        self.assertIsNone(agent.last_report)
        report = agent.synthesize(outcome="OUTCOME-71: the person paid the card and saved money")
        self.assertIs(agent.last_report, report)
        user = model.last_user("reflect")
        self.assertIn("Which card should I pay first?", user)
        self.assertIn(reply, user)
        self.assertIn("OUTCOME-71", user)
        with self.assertRaises(wc.WisdomCoreError):
            agent.synthesize()                     # already learned from that task
        agent.respond("again")
        agent.reset()
        with self.assertRaises(wc.WisdomCoreError):
            agent.synthesize()                     # reset forgets the pending task too

    def test_learn_false(self):
        agent, model = self.make(learn=False)
        agent.ask("hello")
        self.assertEqual(model.kinds(), ["answer"])
        self.assertIsNone(agent.last_report)

    def test_history_window(self):
        agent, model = self.make(history=1)
        for q in ("one", "two", "three"):
            agent.ask(q)
        messages = model.prompts("answer")[-1]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "two")
        agent_none, model_none = self.make(history=0)
        agent_none.ask("one")
        agent_none.ask("two")
        self.assertEqual(len(model_none.prompts("answer")[-1]), 2)

    def test_reset_forgets_the_conversation_but_keeps_principles(self):
        agent, _ = self.make()
        seed(agent.core, SAVINGS)
        agent.ask("hello")
        agent.reset()
        self.assertEqual(agent.messages, [])
        self.assertEqual(len(agent.principles()), 1)

    def test_thinking_is_stripped_from_answers(self):
        agent, _ = self.make()
        agent.core.model.queue("answer", "<think>let me think</think>\nFinal answer.")
        self.assertEqual(agent.ask("q"), "Final answer.")
        self.assertEqual(agent.messages[-1]["content"], "Final answer.")

    def test_empty_message(self):
        agent, _ = self.make()
        with self.assertRaises(ValueError):
            agent.ask("   ")

    def test_instructions_come_after_the_principles(self):
        agent, model = self.make(instructions="Answer in French.")
        agent.ask("q")
        system = model.prompts("answer")[0][0]["content"]
        self.assertTrue(system.endswith("INSTRUCTIONS\nAnswer in French."))
        self.assertLess(system.index("PRINCIPLES"), system.index("INSTRUCTIONS"))

    def test_a_model_error_while_answering_is_raised_and_nothing_is_recorded(self):
        agent, model = self.make()
        model.queue("answer", wc.ModelError("down"))
        with self.assertRaises(wc.ModelError):
            agent.ask("q")
        self.assertEqual(agent.messages, [])
        self.assertNotIn("reflect", model.kinds())

    def test_an_unexpected_error_while_learning_never_costs_the_answer(self):
        agent, model = self.make()
        model.queue("reflect", RuntimeError("boom"))
        with self.assertLogs("wisdom_core", level="WARNING"):
            reply = agent.ask("q")
        self.assertEqual(reply, "OK.")
        self.assertIn("RuntimeError: boom", agent.last_report.error)

    def test_a_string_model_means_ollama(self):
        agent = wc.Agent(AXIOM, "llama3.2", path=":memory:")
        self.assertIsInstance(agent.core.model, wc.Ollama)
        self.assertEqual(agent.core.model.model, "llama3.2")

    def test_a_bad_model_argument(self):
        with self.assertRaises(TypeError):
            wc.Agent(AXIOM, 42, path=":memory:")


# ---------------------------------------------------------------------------
class EmbeddingChangeTests(TempDir):
    def test_a_new_embedding_model_re_embeds_every_principle(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(embed_name="space-a"), path=self.path) as core:
            seed(core, SAVINGS, DEBT)
        model = ScriptedModel(embed_name="space-b")
        with wc.WisdomCore(None, model, path=self.path) as core:
            got = core.recall("where to keep savings", expand=False)
            self.assertEqual(got[0].name, SAVINGS["name"])
        embedded = [t for batch in model.embed_calls for t in batch]
        self.assertTrue(any(SAVINGS["name"] in t for t in embedded))
        con = sqlite3.connect(self.path)
        spaces = {r[0] for r in con.execute("SELECT embed_name FROM principles")}
        con.close()
        self.assertEqual(spaces, {f"space-b (layout {wc.EMBED_LAYOUT})"})

    def test_a_different_vector_size_under_the_same_name_is_handled(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(dims=256), path=self.path) as core:
            seed(core, SAVINGS, DEBT)
        with self.assertLogs("wisdom_core", level="WARNING"):
            with wc.WisdomCore(None, ScriptedModel(dims=64), path=self.path) as core:
                got = core.recall("which debt first", expand=False)
                self.assertEqual(got[0].name, DEBT["name"])
        con = sqlite3.connect(self.path)
        dims = {r[0] for r in con.execute("SELECT dims FROM principles")}
        con.close()
        self.assertEqual(dims, {64})

    def test_a_broken_embedder_is_a_model_error(self):
        model = wc.FunctionModel(lambda m: "ok", lambda texts: [[1.0, 2.0]])   # one vector for many texts
        core = wc.WisdomCore(AXIOM, model, path=":memory:")
        with self.assertRaises(wc.ModelError):
            core._embed(["a", "b"])
        sizes = wc.FunctionModel(lambda m: "ok", lambda texts: [[1.0] * (i + 1) for i in range(len(texts))])
        with self.assertRaises(wc.ModelError):
            wc.WisdomCore(AXIOM, sizes, path=":memory:")._embed(["a", "b"])
        bad = wc.WisdomCore(AXIOM, wc.FunctionModel(lambda m: "ok", lambda texts: [["x"] for _ in texts]),
                            path=":memory:")
        with self.assertRaises(wc.ModelError):
            bad._embed(["a"])


# ---------------------------------------------------------------------------
class ManualAndStatsTests(unittest.TestCase):
    def test_remove_by_hand(self):
        core, _ = memory_core()
        (a,) = seed(core, SAVINGS)
        core.remove(a.id, "outdated")
        self.assertEqual(core.get(a.id).end_reason, "outdated")
        with self.assertRaises(wc.WisdomCoreError):
            core.remove(a.id, "again")
        with self.assertRaises(KeyError):
            core.remove("nope")

    def test_stats(self):
        core, model = memory_core()
        a, b, c = seed(core, SAVINGS, TOTALS, DEBT)
        model.queue("execute", {"actions": [
            {"type": "merge", "ids": ["P1", "P2"], "principle": principle("Merged savings and totals rule")},
            {"type": "remove", "id": "P3", "reason": "wrong"},
        ]})
        core.synthesize("task", "result", used=[a, b, c])
        stats = core.stats()
        self.assertEqual((stats["active"], stats["merged"], stats["removed"]), (1, 2, 1))
        self.assertEqual((stats["added"], stats["merges"], stats["removals"]), (3, 1, 1))
        self.assertEqual(stats["syntheses"], 1)

    def test_history_is_newest_first_and_limited(self):
        core, _ = memory_core()
        seed(core, SAVINGS, TOTALS)
        events = core.history()
        self.assertEqual([e.kind for e in events], ["added", "added", "calibrated", "axiom"])
        self.assertEqual(len(core.history(1)), 1)
        self.assertEqual(len(core.history(None)), 4)


class LevelsTests(unittest.TestCase):
    """The semantic levels: what forces a principle, and what counts as already held."""

    def test_a_principle_is_written_for_every_task(self):
        core, model = memory_core()
        (t,) = seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "covered", "id": "P1"}]},   # the model would write nothing
                               {"actions": [{"type": "add", "principle": SAVINGS}]})
        report = core.synthesize("Where should I keep my savings?", "In a savings account.")
        self.assertEqual([p.name for p in report.added], [SAVINGS["name"]])
        self.assertEqual(model.kinds(), ["lesson", "reflect", "execute", "execute"])   # asked again for one
        self.assertIn(wc.FORCE_MEMO, model.last_user("reflect"))
        self.assertIn(wc.FORCE_ACTIONS, model.prompts("execute")[0][-1]["content"])
        self.assertIn(wc.FORCE_RETRY, model.last_user("execute"))
        self.assertEqual([p.id for p in core.principles()], [t.id, report.added[0].id])

    def test_a_new_principle_says_how_close_the_nearest_held_one_was(self):
        core, model = memory_core()
        seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "add", "principle": SAVINGS}]})
        report = core.synthesize("Where should I keep my savings?", "In a savings account.")
        (added,) = report.added
        self.assertIsNotNone(added.score)
        self.assertLess(added.score, 0.95)
        self.assertIn(f"(nearest held: {added.score:.2f})", report.lines()[0])
        self.assertEqual(report.to_dict()["added"][0]["score"], added.score)

    def test_a_draft_measured_as_already_held_is_not_asked_for_again(self):
        core, model = memory_core()
        seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "add", "principle": TOTALS}]})
        report = core.synthesize("Add up my bills", "420")
        self.assertEqual(model.kinds(), ["lesson", "reflect", "execute"])   # the lesson was drafted; that is enough
        self.assertEqual((report.added, report.skipped), ([], []))
        self.assertEqual(len(report.covered), 1)

    def test_a_principle_it_already_holds_is_measured_not_judged(self):
        core, model = memory_core()
        (t,) = seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "add", "principle": TOTALS}]})   # the same lesson again
        report = core.synthesize("Add up my bills", "420")
        self.assertEqual(report.added, [])
        self.assertEqual([p.id for p in report.covered], [t.id])
        self.assertGreaterEqual(report.covered[0].score, core.effective_levels().covered)
        self.assertIn(f"~ already covered by: {t.name} ({report.covered[0].score:.2f})", report.lines())
        event = core.history(1)[0]
        self.assertEqual(event.kind, "covered")
        self.assertIn("measured 1.00", event.detail)
        self.assertIn("covered level of 0.95", event.detail)

    def test_the_covered_level_decides_how_close_is_close_enough(self):
        for level, added in ((0.05, 0), (0.95, 1)):
            with self.subTest(level=level):
                core, model = memory_core(levels=wc.Levels(covered=level))
                seed(core, TOTALS)
                model.queue("execute", {"actions": [{"type": "add", "principle": SAVINGS}]})
                report = core.synthesize("Where should I keep my savings?", "In a savings account.")
                self.assertEqual(len(report.added), added)
                self.assertEqual(len(core.principles()), 1 + added)
                core.close()

    def test_with_force_off_the_model_decides(self):
        core, model = careful_core()
        (t,) = seed(core, TOTALS)
        model.queue("execute", {"actions": [{"type": "covered", "id": "P1"}]})
        report = core.synthesize("Add up my bills", "420")
        self.assertEqual((report.added, [p.id for p in report.covered]), ([], [t.id]))
        self.assertIsNone(report.covered[0].score)                    # nothing was measured
        self.assertEqual(model.kinds(), ["lesson", "reflect", "execute"])   # and it was not asked again
        self.assertIn(wc.CAREFUL_MEMO, model.last_user("reflect"))
        self.assertNotIn(wc.FORCE_ACTIONS, model.last_user("execute"))

    def test_a_model_that_still_writes_nothing_is_reported(self):
        core, model = memory_core()
        seed(core, TOTALS)
        report = core.synthesize("Add up my bills", "420")            # the default reply changes nothing
        self.assertEqual(report.added, [])
        self.assertIsNone(report.error)
        self.assertEqual(report.skipped, ["forced: the model drafted no principle for this task"])
        self.assertEqual(model.kinds(), ["lesson", "reflect", "execute", "execute"])

    def test_a_forced_ask_that_fails_costs_nothing(self):
        core, model = memory_core()
        seed(core, TOTALS)
        model.queue("execute", {"actions": []}, wc.ModelError("no reply"))
        report = core.synthesize("Add up my bills", "420")
        self.assertIsNone(report.error)
        self.assertEqual(report.skipped, ["forced: the model drafted no principle for this task"])

    def test_how_many_principles_one_task_may_add(self):
        core, model = memory_core(levels=wc.Levels(max_adds=1))
        model.queue("execute", {"actions": [{"type": "add", "principle": SAVINGS},
                                            {"type": "add", "principle": DEBT}]})
        report = core.synthesize("money", "answer")
        self.assertEqual([p.name for p in report.added], [SAVINGS["name"]])
        self.assertEqual(report.skipped, [f"add: only 1 new principle per task ({DEBT['name']})"])
        core.levels.max_adds = 3
        self.assertIn("Add at most 3 new principles from this task.", core._execute_messages("memo", {})[-1]["content"])

    def test_the_merge_level_refuses_a_merge_of_principles_that_are_not_alike(self):
        core, model = memory_core(levels=wc.Levels(force=False, merge=0.9))
        a, b = seed(core, SAVINGS, TOTALS)
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P2"],
                                             "principle": principle("A combined rule")}]})
        report = core.synthesize("money", "answer", used=[a, b])
        self.assertEqual(report.merged, [])
        self.assertEqual(len(report.skipped), 1)
        self.assertIn("below the merge level of 0.90", report.skipped[0])
        self.assertIn(SAVINGS["name"], report.skipped[0])
        self.assertEqual([p.id for p in core.principles()], [a.id, b.id])

    def test_the_recall_floor_leaves_out_what_is_not_relevant_enough(self):
        core, model = memory_core()
        seed(core, SAVINGS, TOTALS, DEBT)
        everything = core.recall("Which debt should I repay first?", expand=False)
        self.assertEqual(len(everything), 3)
        scores = [p.score for p in everything]
        core.levels.recall_floor = (scores[0] + scores[1]) / 2
        events = []
        core.on_event = events.append
        kept = core.recall("Which debt should I repay first?", expand=False)
        self.assertEqual([p.id for p in kept], [everything[0].id])
        done = [e for e in events if e["type"] == "recall.done"][-1]
        self.assertEqual((done["below"], round(done["floor"], 6)), (2, round(core.levels.recall_floor, 6)))

    def test_the_levels_are_checked_and_can_be_saved(self):
        self.assertEqual(wc.Levels().to_dict(), {"force": True, "covered": None, "combine": None, "merge": 0.0,
                                                 "recall_floor": 0.0, "context": 12, "max_adds": 1,
                                                 "new_ground": None})        # None: measured for the model
        self.assertEqual(wc.FALLBACK_LEVELS, {"new_ground": 0.85, "covered": 0.95, "combine": 0.85})
        self.assertEqual(wc.Levels().resolved().to_dict()["covered"], 0.95)
        self.assertEqual(wc.Levels.from_dict({"covered": 0.5, "unknown": 1, "force": None}),
                         wc.Levels(covered=0.5))
        self.assertEqual(wc.Levels().replace(context=3).context, 3)
        for bad in ({"covered": 1.5}, {"covered": "x"}, {"merge": -0.1}, {"combine": 2}, {"context": 0}, {"max_adds": 0},
                    {"new_ground": 1.2}, {"new_ground": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                wc.Levels(**bad)
        core, _ = memory_core(levels={"context": 4, "force": False})
        self.assertEqual((core.context, core.levels.force), (4, False))
        core.levels = wc.Levels(context=2)
        self.assertEqual(core.context, 2)
        with self.assertRaises(ValueError):
            core.context = 0


class FirstLookTests(unittest.TestCase):
    """The quick first look: Wisdom Synthesis runs in full only when the exchange is new ground."""

    def test_a_lesson_already_held_stops_the_synthesis_there(self):
        core, model = memory_core()
        (t,) = seed(core, TOTALS)
        model.queue("lesson", {"principle": TOTALS, "corrects": False})
        report = core.synthesize("Add up my bills", "420")
        self.assertEqual(model.kinds(), ["lesson"])                  # no memo, no changes call
        self.assertEqual([p.id for p in report.covered], [t.id])
        look = report.first_look
        self.assertTrue(look["held"])
        self.assertGreaterEqual(look["score"], core.effective_levels().new_ground)
        self.assertEqual((look["lesson"], look["nearest"], look["level"]), (TOTALS["name"], t.name, 0.85))
        self.assertIsNone(report.error)
        self.assertFalse(report.changed)
        self.assertTrue(any(line.startswith("~ already held, so nothing more ran") for line in report.lines()))
        self.assertEqual(report.to_dict()["first_look"]["held"], True)
        event = core.history(1)[0]
        self.assertEqual((event.kind, event.principle_id), ("covered", t.id))
        self.assertIn("first look", event.detail)
        self.assertIn("new-ground level of 0.85", event.detail)
        stats = core.stats()
        self.assertEqual((stats["held_at_first_look"], stats["syntheses"]), (1, 1))
        self.assertEqual([p.id for p in core.principles()], [t.id])

    def test_new_ground_runs_the_whole_synthesis(self):
        core, model = memory_core()
        seed(core, TOTALS)
        model.queue("lesson", {"principle": SAVINGS, "corrects": False})
        model.queue("execute", {"actions": [{"type": "add", "principle": DEBT}]})
        report = core.synthesize("Which debt first?", "The card.")
        self.assertEqual(model.kinds(), ["lesson", "reflect", "execute"])
        self.assertEqual([p.name for p in report.added], [DEBT["name"]])    # the memo's own principle wins
        self.assertFalse(report.first_look["held"])
        self.assertEqual(report.first_look["why"], "new ground")
        self.assertLess(report.first_look["score"], 0.85)
        self.assertEqual(core.stats()["held_at_first_look"], 0)

    def test_on_new_ground_a_memo_that_writes_nothing_gets_the_first_look_lesson(self):
        core, model = memory_core()
        seed(core, TOTALS)
        model.queue("lesson", {"principle": SAVINGS, "corrects": False})   # the changes call then adds nothing
        events = []
        core.on_event = events.append
        report = core.synthesize("Where should I keep my savings?", "In a savings account.")
        self.assertEqual(model.kinds(), ["lesson", "reflect", "execute"])  # not asked again: the lesson is already named
        self.assertEqual([p.name for p in report.added], [SAVINGS["name"]])
        self.assertEqual(report.skipped, [])
        forced = [e for e in events if e["type"] == "synthesis.actions" and e.get("forced")]
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0]["actions"][0]["type"], "add")

    def test_with_force_off_new_ground_still_leaves_the_choice_to_the_model(self):
        core, model = careful_core()
        seed(core, TOTALS)
        model.queue("lesson", {"principle": SAVINGS, "corrects": False})
        report = core.synthesize("Where should I keep my savings?", "In a savings account.")
        self.assertEqual(model.kinds(), ["lesson", "reflect", "execute"])
        self.assertEqual(report.added, [])

    def test_push_back_from_the_user_always_runs_the_whole_synthesis(self):
        for corrects in (True, "yes", "true"):
            with self.subTest(corrects=corrects):
                core, model = memory_core()
                (t,) = seed(core, TOTALS)
                model.queue("lesson", {"principle": TOTALS, "corrects": corrects})
                model.queue("execute", {"actions": [{"type": "remove", "id": "P1", "reason": "the user showed it wrong"}]})
                report = core.synthesize("No, that total is wrong.", "Sorry, it is 410.")
                self.assertEqual(model.kinds()[:3], ["lesson", "reflect", "execute"])
                self.assertFalse(report.first_look["held"])
                self.assertTrue(report.first_look["corrects"])
                self.assertIn("pushed back", report.first_look["why"])
                self.assertEqual([p.id for p, _ in report.removed], [t.id])
                core.close()

    def test_no_first_look_while_nothing_is_held_or_an_outcome_is_given_or_it_is_off(self):
        cases = (("nothing held", {}, False, None, "no principles are held yet"),
                 ("outcome", {}, True, "The person paid it off.", "an outcome was given"),
                 ("off", {"levels": wc.Levels(new_ground=0)}, True, None, "new ground is set to 0"))
        for name, kw, held, outcome, why in cases:
            with self.subTest(name):
                core, model = memory_core(**kw)
                if held:
                    seed(core, TOTALS)
                model.queue("lesson", {"principle": TOTALS, "corrects": False})
                report = core.synthesize("Add up my bills", "420", outcome=outcome)
                self.assertNotIn("lesson", model.kinds())
                self.assertEqual(model.kinds()[:2], ["reflect", "execute"])
                self.assertFalse(report.first_look["ran"])
                self.assertIn(why, report.first_look["why"])
                core.close()

    def test_a_first_look_that_cannot_be_read_falls_back_to_the_whole_synthesis(self):
        for name, replies, calls in (("unreadable twice", ("not json", "still not json"), ["lesson", "lesson"]),
                                     ("model error", (wc.ModelError("down"),), ["lesson"]),
                                     ("no lesson named", ({"principle": "", "corrects": False},), ["lesson"]),
                                     ("a label instead of a lesson", ({"principle": "P1"},), ["lesson"])):
            with self.subTest(name):
                core, model = memory_core()
                seed(core, TOTALS)
                model.queue("lesson", *replies)
                model.queue("execute", {"actions": [{"type": "add", "principle": SAVINGS}]})
                report = core.synthesize("Where should I keep my savings?", "In a savings account.")
                self.assertEqual(model.kinds(), calls + ["reflect", "execute"])
                self.assertIsNone(report.error)
                self.assertEqual([p.name for p in report.added], [SAVINGS["name"]])
                self.assertTrue(report.first_look["ran"])
                self.assertIn("whole synthesis runs", report.first_look["why"])
                core.close()

    def test_one_unreadable_reply_is_asked_again(self):
        core, model = memory_core()
        seed(core, TOTALS)
        model.queue("lesson", "Sure! Here it is.", {"principle": TOTALS})
        report = core.synthesize("Add up my bills", "420")
        self.assertEqual(model.kinds(), ["lesson", "lesson"])
        self.assertIn(wc.JSON_RETRY, model.last_user("lesson"))
        self.assertTrue(report.first_look["held"])

    def test_the_first_look_reads_the_exchange_but_never_the_principles(self):
        core, model = memory_core()
        seed(core, TOTALS)
        conversation = [{"role": "user", "content": "I owe on two cards."},
                        {"role": "assistant", "content": "Pay the higher rate first."}]
        model.queue("lesson", {"principle": SAVINGS})
        core.synthesize("That makes sense, thanks.", "Glad it helps.", conversation=conversation,
                        reasoning="The user accepted the advice about rates.")
        (messages,) = model.prompts("lesson")
        self.assertEqual(messages[0]["content"], wc.LESSON_SYSTEM)
        user = messages[1]["content"]
        for part in (AXIOM, "WHAT THE USER SAID\nThat makes sense, thanks.",
                     "THE AGENT'S PREVIOUS ANSWER, WHICH THE USER IS RESPONDING TO\nPay the higher rate first.",
                     "THE AGENT'S REASONING BEFORE IT ANSWERED\nThe user accepted the advice about rates.",
                     "THE AGENT'S ANSWER\nGlad it helps."):
            self.assertIn(part, user)
        self.assertNotIn(TOTALS["name"], user)
        self.assertNotIn("PRINCIPLES", user)
        _, _, options = model.calls[0]
        self.assertEqual((options["json_mode"], options["temperature"]), (True, 0.0))

    def test_the_new_ground_level_decides_how_close_is_already_held(self):
        near = principle("Add the figures twice before reporting the final sum", TOTALS["importance"],
                         TOTALS["extent"], TOTALS["use_case"], TOTALS["related"])
        for level, held in ((0.5, True), (0.999, False)):
            with self.subTest(level=level):
                core, model = memory_core(levels=wc.Levels(new_ground=level))
                seed(core, TOTALS)
                model.queue("lesson", {"principle": near})
                report = core.synthesize("Add up my bills", "420")
                self.assertEqual(report.first_look["held"], held)
                self.assertEqual("reflect" in model.kinds(), not held)
                core.close()

    def test_the_agent_skips_the_synthesis_on_ground_it_already_holds(self):
        model = ScriptedModel()
        agent = wc.Agent(AXIOM, model, path=":memory:")
        seed(agent.core, TOTALS)
        model.queue("lesson", {"principle": TOTALS})
        agent.ask("Add up my bills: 200 and 220.")
        self.assertTrue(agent.last_report.first_look["held"])
        self.assertEqual(model.kinds(), ["answer", "lesson"])


class EmbeddingLayoutTests(TempDir):
    """What is embedded for a principle, and how each kind of text is written for the embedding model."""

    def test_two_texts_per_principle_without_shared_labels(self):
        p = wc.Principle(id="x", **DEBT)
        self.assertEqual(p.identity_text(), DEBT["name"] + ".")
        recall = p.recall_text()
        self.assertEqual(recall, f"{DEBT['name']}. Someone has several debts. interest rate, loans, credit cards.")
        for label in ("Why:", "Scope:", "Use when:", "Related:"):
            self.assertNotIn(label, recall)
        self.assertNotIn(DEBT["importance"], recall)          # why it matters is not what it is about
        self.assertNotIn(DEBT["extent"], recall)

    def test_every_prompt_that_writes_a_principle_asks_for_words_that_will_be_found(self):
        for prompt in (wc.EXECUTE_SYSTEM, wc.COMBINE_SYSTEM, wc.LESSON_SYSTEM):
            with self.subTest(prompt=prompt.splitlines()[0]):
                self.assertIn("The agent finds its principles by meaning", prompt)
                self.assertIn("naming its subject plainly", prompt)
                self.assertIn("described the way a person would bring it up", prompt)
                self.assertIn("the subject itself first", prompt)
        self.assertIn("the way a principle's \"use when\" line describes", wc.SITUATION_SYSTEM)

    def test_each_model_gets_its_own_markers(self):
        cases = {
            "embeddinggemma:latest": ("task: search result | query: x", "title: none | text: x",
                                      "task: sentence similarity | query: x"),
            "hf.co/unsloth/embeddinggemma-300m-GGUF:Q8_0": ("task: search result | query: x",
                                                             "title: none | text: x",
                                                             "task: sentence similarity | query: x"),
            "nomic-embed-text:v1.5": ("search_query: x", "search_document: x", "clustering: x"),
            "mxbai-embed-large": ("Represent this sentence for searching relevant passages: x", "x", "x"),
            "bge-m3:latest": ("x", "x", "x"),
            "bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: x", "x", "x"),
            "jeffh/intfloat-multilingual-e5-large:f16": ("query: x", "passage: x", "query: x"),
            "multilingual-e5-large": ("query: x", "passage: x", "query: x"),
            "text-embedding-3-small": ("x", "x", "x"),
        }
        for name, (query, document, similarity) in cases.items():
            with self.subTest(name=name):
                prompts = wc.embed_prompts_for(ScriptedModel(embed_name=name))
                got = tuple(prompts.get(role, "{}").replace("{}", "x") for role in ("query", "document", "similarity"))
                self.assertEqual(got, (query, document, similarity))

    def test_a_backend_can_bring_its_own_markers(self):
        model = ScriptedModel(embed_name="embeddinggemma")
        model.embed_prompts = {"query": "Q: {}", "document": "D: {}", "ignored": "no placeholder"}
        self.assertEqual(wc.embed_prompts_for(model), {"query": "Q: {}", "document": "D: {}"})

    def test_searches_stored_principles_and_comparisons_are_written_for_the_model(self):
        model = ScriptedModel(embed_name="embeddinggemma:latest")
        core = wc.WisdomCore(AXIOM, model, path=":memory:")
        seed(core, DEBT)
        calibration, fingerprint, *rest = model.embed_calls     # the model is measured before its first principle
        self.assertEqual(len(calibration), 5 * len(wc._CHECK_SET))
        self.assertEqual(fingerprint, ["task: sentence similarity | query: " + wc.FINGERPRINT_TEXT])
        stored = [t for batch in rest for t in batch]
        self.assertEqual(stored, ["title: none | text: " + wc.Principle(id="x", **DEBT).recall_text(),
                                  "task: sentence similarity | query: " + DEBT["name"] + "."])
        model.embed_calls.clear()
        core.recall("Which debt first?", expand=False)
        self.assertEqual(model.embed_calls, [["task: search result | query: Which debt first?"]])
        model.embed_calls.clear()
        model.queue("lesson", {"principle": DEBT})
        core.synthesize("Which debt first?", "The card.")
        self.assertEqual(model.embed_calls[0], ["task: sentence similarity | query: " + wc.FINGERPRINT_TEXT])
        self.assertEqual(model.embed_calls[1], ["task: sentence similarity | query: " + DEBT["name"] + "."])
        core.close()

    def test_duplicates_are_told_by_what_a_principle_says(self):
        core, model = memory_core()
        (t,) = seed(core, DEBT)
        same_lesson = principle(DEBT["name"], "Other words about why", "Other limits", "a different situation",
                                ["other", "words"])
        model.queue("execute", {"actions": [{"type": "add", "principle": same_lesson}]})
        report = core.synthesize("Which debt first?", "The card.")
        self.assertEqual([p.id for p in report.covered], [t.id])     # the same statement is the same lesson
        self.assertAlmostEqual(report.covered[0].score, 1.0, places=5)
        other_lesson = principle("Ask about penalties for paying a loan off early", DEBT["importance"],
                                 DEBT["extent"], DEBT["use_case"], DEBT["related"])
        model.queue("execute", {"actions": [{"type": "add", "principle": other_lesson}]})
        report = core.synthesize("Can I pay my loan off early?", "Check for penalties.")
        self.assertEqual([p.name for p in report.added], [other_lesson["name"]])   # the same fields do not make it a duplicate
        self.assertEqual(len(core.principles()), 2)

    def test_principles_that_share_a_situation_but_say_different_things_are_kept_apart(self):
        situation = ("someone has several debts, a mortgage, two credit cards and a car loan, some spare money each "
                     "month, and asks what to do with that spare money")
        words = ["debt", "interest rate", "spare money", "repayment", "loans", "credit cards", "mortgage", "budget"]
        first = principle("Pay the highest-interest debt first", "", "", situation, words)
        second = principle("Ask whether the loan has an early repayment penalty", "", "", situation, words)
        core, model = memory_core()
        a, b = seed(core, first, second)
        index = core._ensure_index()
        self.assertGreater(wc._dot(index.vector(a.id, "recall"), index.vector(b.id, "recall")), 0.85)   # found together
        self.assertLess(wc._dot(index.vector(a.id), index.vector(b.id)), 0.5)                          # but not alike
        report = core.synthesize("I have spare money each month. What should I do?", "Pay the card first.",
                                 used=[a, b])
        self.assertEqual(report.merged, [])
        self.assertNotIn("combine", model.kinds())
        self.assertEqual(len(core.principles()), 2)

    def test_a_principle_missing_its_identity_vector_is_embedded_again(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path) as core:
            seed(core, SAVINGS, DEBT)
        con = sqlite3.connect(self.path)
        con.execute("UPDATE principles SET identity = NULL WHERE name = ?", (DEBT["name"],))
        con.commit()
        con.close()
        events = []
        with wc.WisdomCore(None, ScriptedModel(), path=self.path, on_event=events.append) as core:
            core.recall("which debt", expand=False)
        self.assertEqual([e["count"] for e in events if e["type"] == "reindex"], [1])
        con = sqlite3.connect(self.path)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM principles WHERE identity IS NULL").fetchone()[0], 0)
        con.close()

    def test_a_database_from_before_two_vectors_is_upgraded_once(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path) as core:
            seed(core, SAVINGS, DEBT)
        con = sqlite3.connect(self.path)           # as 0.4 left it: one vector, stored under the bare model name
        con.execute("ALTER TABLE principles DROP COLUMN identity")
        con.execute("UPDATE principles SET embed_name = 'bow'")
        con.commit()
        con.close()
        model = ScriptedModel()
        events = []
        with wc.WisdomCore(None, model, path=self.path, on_event=events.append) as core:
            got = core.recall("where to keep savings", expand=False)
            self.assertEqual(got[0].name, SAVINGS["name"])
            self.assertEqual([e["count"] for e in events if e["type"] == "reindex"], [2])
            core.recall("which debt", expand=False)
            self.assertEqual(len([e for e in events if e["type"] == "reindex"]), 1)   # once, not every time
        con = sqlite3.connect(self.path)
        rows = con.execute("SELECT embed_name, identity IS NOT NULL FROM principles").fetchall()
        con.close()
        self.assertEqual(set(rows), {(f"bow (layout {wc.EMBED_LAYOUT})", 1)})


def _check_model(reworded: Any, same: Any, misplaced: bool = False) -> wc.FunctionModel:
    """An embedder for calibration: each principle of the check set on its own axis, its rewording at ``reworded``
    to it, its related lesson at ``same`` (a number, or one per principle), the principle it is set beside from
    another subject on an axis of its own, and each question near its principle (or, ``misplaced``, the next one).
    Any other text gets the bag-of-words vector the other tests use."""
    n = len(wc._CHECK_SET)
    reworded = list(reworded) if isinstance(reworded, (list, tuple)) else [reworded] * n
    same = list(same) if isinstance(same, (list, tuple)) else [same] * n
    dims = 1024
    axes: dict[str, list[float]] = {}

    def axis(key: str) -> list[float]:
        if key not in axes:
            vec = [0.0] * dims
            vec[len(axes)] = 1.0
            axes[key] = vec
        return axes[key]

    def at(base: list[float], cos: float, key: str) -> list[float]:
        other = axis("other:" + key)
        sin = (1 - cos * cos) ** 0.5
        return [cos * b + sin * o for b, o in zip(base, other, strict=True)]

    table: dict[str, list[float]] = {}
    for i, entry in enumerate(wc._CHECK_SET):
        base = table[entry["principle"].identity_text()] = axis(f"base{i}")
        table[entry["reworded"]] = at(base, reworded[i], f"reworded{i}")
        table[entry["related"]] = at(base, same[i], f"related{i}")
        table[entry["principle"].recall_text()] = axis(f"doc{i}")
    for i, entry in enumerate(wc._CHECK_SET):
        j = (i + 1) % n if misplaced else i
        table[entry["question"]] = at(table[wc._CHECK_SET[j]["principle"].recall_text()], 0.7, f"question{i}")
    return wc.FunctionModel(lambda messages: "",
                            lambda texts: [table[t] if t in table else bow_embed([t], dims)[0] for t in texts],
                            embed_name="check")


class CheckEmbeddingsTests(unittest.TestCase):
    """check_embeddings(): measuring an embedding model, and the levels that fit it."""

    def test_a_model_that_tells_them_apart_clearly(self):
        result = wc.check_embeddings(_check_model(0.9, 0.6))
        self.assertEqual(result["count"], 15)
        self.assertAlmostEqual(result["lowest_reworded"], 0.9, places=5)
        self.assertAlmostEqual(result["highest_same_subject"], 0.6, places=5)
        self.assertEqual(result["separation"], "clear")
        self.assertTrue(result["separates"])
        self.assertEqual(result["calibrated"], {"new_ground": 0.75, "combine": 0.75, "covered": 0.9})
        self.assertEqual(result["suggested"], 0.75)
        self.assertIn("New ground and Combine fit it at 0.75, and Already covered at 0.90", result["verdict"])
        self.assertEqual((result["recall"]["found"], result["recall"]["of"]), (15, 15))
        self.assertIn("15 of 15 questions", result["recall_verdict"])
        self.assertTrue(all(abs(x) < 1e-6 for x in result["unrelated"]))
        self.assertEqual(result["levels"], {"new_ground": None, "covered": None, "combine": None})
        self.assertFalse(result["prompts"])

    def test_already_covered_stays_above_combine(self):
        result = wc.check_embeddings(_check_model(0.8, 0.7))
        self.assertEqual(result["calibrated"], {"new_ground": 0.75, "combine": 0.75, "covered": 0.8})
        result = wc.check_embeddings(_check_model(0.8, 0.79))      # the closest rewordings sit just above combine
        self.assertEqual(result["calibrated"]["covered"], round(result["calibrated"]["combine"] + 0.03, 2))

    def test_a_model_that_tells_most_of_them_apart(self):
        reworded = [0.9] * 13 + [0.55, 0.5]                     # two rewordings it misses...
        same = [0.6] * 13 + [0.92, 0.95]                        # ...and two related lessons it thinks are the same
        result = wc.check_embeddings(_check_model(reworded, same))
        self.assertEqual(result["separation"], "partial")
        self.assertEqual(result["calibrated"]["combine"], 0.75)
        self.assertIn("some of each overlap", result["verdict"])

    def test_a_model_that_cannot_tell_them_apart(self):
        result = wc.check_embeddings(_check_model(0.8, 0.85))
        self.assertEqual(result["separation"], "none")
        self.assertIsNone(result["calibrated"])
        self.assertIsNone(result["suggested"])
        self.assertIn("no level tells them apart", result["verdict"])

    def test_questions_that_find_the_wrong_principle_are_counted(self):
        result = wc.check_embeddings(_check_model(0.9, 0.6, misplaced=True))
        self.assertEqual(result["recall"]["found"], 0)
        self.assertIn("0 of 15 questions", result["recall_verdict"])
        self.assertTrue(all(r < n for r, n in zip(result["recall"]["right"], result["recall"]["next_best"], strict=True)))

    def test_one_request_written_for_the_model_and_nothing_stored(self):
        model = ScriptedModel(embed_name="embeddinggemma:latest")
        core = wc.WisdomCore(AXIOM, model, path=":memory:")
        result = core.check_embeddings()
        self.assertEqual(len(model.embed_calls), 1)
        sent = model.embed_calls[0]
        self.assertEqual(len(sent), 75)
        self.assertTrue(sent[0].startswith("task: sentence similarity | query: "))
        self.assertTrue(sent[-1].startswith("task: search result | query: "))
        self.assertIn("title: none | text: " + wc._CHECK_SET[0]["principle"].recall_text(), sent)
        self.assertTrue(result["prompts"])
        self.assertEqual(result["model"], "embeddinggemma:latest")
        self.assertEqual(core.principles(), [])
        self.assertIsNone(core.calibration(measure=False))       # checking is not calibrating
        core.close()

    def test_the_check_set_spans_subjects_and_sets_each_principle_beside_another_subject(self):
        names = [e["principle"].name for e in wc._CHECK_SET]
        self.assertEqual(len(set(names)), 15)
        for e in wc._CHECK_SET:
            self.assertTrue(e["principle"].use_case and len(e["principle"].related) >= 3)
            self.assertNotEqual(e["reworded"], e["related"])
        self.assertNotEqual(wc._CHECK_UNRELATED_OFFSET % 15, 0)


class CalibrationTests(TempDir):
    """The levels that depend on the embedding model are measured for it automatically."""

    def test_the_first_use_of_a_model_measures_it_once_and_keeps_it(self):
        model = _check_model(0.9, 0.6)
        calls: list[int] = []
        embed = model.embed
        model.embed = lambda texts: (calls.append(len(texts)), embed(texts))[1]   # type: ignore[method-assign]
        events = []
        with wc.WisdomCore(AXIOM, model, path=self.path, on_event=events.append) as core:
            levels = core.effective_levels()
            self.assertEqual((levels.new_ground, levels.combine, levels.covered), (0.75, 0.75, 0.9))
            self.assertEqual(core.levels.to_dict()["combine"], None)                # the setting stays automatic
            self.assertEqual(calls, [75, 1])                                        # and its fingerprint kept
            core.effective_levels()
            self.assertEqual(calls, [75, 1])                                        # measured once
            calib = core.calibration()
            self.assertEqual((calib["levels"]["combine"], calib["separation"]), (0.75, "clear"))
            self.assertEqual(calib["space"], f"check (layout {wc.EMBED_LAYOUT})")
            done = [e for e in events if e["type"] == "calibrated"]
            self.assertEqual(len(done), 1)
            self.assertTrue(done[0]["ok"])
            event = core.history(1)[0]
            self.assertEqual(event.kind, "calibrated")
            self.assertIn("new ground and combine 0.75, already covered 0.90", event.detail)
        with wc.WisdomCore(None, model, path=self.path) as core:              # kept in the database
            self.assertEqual(core.effective_levels().combine, 0.75)
            self.assertEqual(calls, [75, 1])

    def test_a_new_embedding_model_is_measured_again(self):
        with wc.WisdomCore(AXIOM, _check_model(0.9, 0.6), path=self.path) as core:
            self.assertEqual(core.effective_levels().combine, 0.75)
        other = _check_model(0.8, 0.5)
        other.embed_name = "check-two"
        with wc.WisdomCore(None, other, path=self.path) as core:
            self.assertEqual(core.effective_levels().combine, 0.65)
            self.assertEqual(core.calibration()["model"], "check-two")

    def test_a_level_set_by_hand_is_used_as_given(self):
        core = wc.WisdomCore(AXIOM, _check_model(0.9, 0.6), path=":memory:", levels={"combine": 0.8})
        levels = core.effective_levels()
        self.assertEqual((levels.new_ground, levels.combine, levels.covered), (0.75, 0.8, 0.9))
        core.close()

    def test_with_every_level_set_by_hand_nothing_is_measured(self):
        model = ScriptedModel()
        core = wc.WisdomCore(AXIOM, model, path=":memory:",
                             levels=wc.Levels(new_ground=0.7, covered=0.9, combine=0.8))
        seed(core, DEBT)
        self.assertEqual(core.effective_levels().to_dict(), core.levels.to_dict())
        self.assertTrue(all(len(batch) != 75 for batch in model.embed_calls))
        self.assertIsNone(core.calibration(measure=False))
        core.close()

    def test_a_model_that_cannot_be_calibrated_keeps_the_fallbacks_and_says_so(self):
        core = wc.WisdomCore(AXIOM, _check_model(0.8, 0.85), path=":memory:")
        levels = core.effective_levels()
        self.assertEqual((levels.new_ground, levels.combine, levels.covered), (0.85, 0.85, 0.95))
        calib = core.calibration()
        self.assertIsNone(calib["levels"])
        self.assertEqual(calib["separation"], "none")
        self.assertIn("could not be calibrated; the fallback levels are used", core.history(1)[0].detail)
        core.close()

    def test_an_unreachable_model_uses_the_fallbacks_and_tries_again_later(self):
        model = _check_model(0.9, 0.6)
        working = model.embed
        model.embed = lambda texts: (_ for _ in ()).throw(wc.ModelError("down"))   # type: ignore[method-assign]
        events = []
        core = wc.WisdomCore(AXIOM, model, path=":memory:", on_event=events.append)
        self.assertEqual(core.effective_levels().combine, 0.85)
        self.assertIsNone(core.calibration(measure=False))
        self.assertEqual([e["ok"] for e in events if e["type"] == "calibrated"], [False])
        model.embed = working                                                      # type: ignore[method-assign]
        self.assertEqual(core.effective_levels().combine, 0.75)                     # measured once it answers
        core.close()

    def test_the_measured_levels_decide_what_is_combined(self):
        near = principle("List debts by interest rate before advising which loan to repay first",   # 0.83 alike
                         DEBT["importance"], DEBT["extent"], DEBT["use_case"], DEBT["related"])
        for name, embedder, combined in (("measured at 0.75", _check_model(0.9, 0.6), True),
                                         ("fallback 0.85", ScriptedModel(), False)):
            with self.subTest(name):
                chat = ScriptedModel()
                core = wc.WisdomCore(AXIOM, wc.Combined(chat, embedder), path=":memory:")
                seed(core, DEBT)
                chat.queue("execute", {"actions": [{"type": "add", "principle": near}]})
                report = core.synthesize("Which debt first?", "The card.")
                self.assertEqual(bool(report.merged), combined, report.lines())
                self.assertEqual(len(core.principles()), 1 if combined else 2)
                core.close()

    def test_a_live_core_given_another_embedding_model_measures_it(self):
        core = wc.WisdomCore(AXIOM, _check_model(0.9, 0.6), path=":memory:")
        self.assertEqual(core.effective_levels().combine, 0.75)
        other = _check_model(0.8, 0.5)
        other.embed_name = "check-two"
        core.model = other
        self.assertEqual(core.effective_levels().combine, 0.65)
        core.close()

    def test_the_measured_levels_decide_the_first_look_and_distillation(self):
        near = principle("List debts by interest rate before advising which loan to repay first",   # 0.83 alike
                         DEBT["importance"], DEBT["extent"], DEBT["use_case"], DEBT["related"])
        for name, embedder, held in (("measured at 0.75", _check_model(0.9, 0.6), True),
                                     ("fallback 0.85", ScriptedModel(), False)):
            with self.subTest(name):
                chat = ScriptedModel()
                core = wc.WisdomCore(AXIOM, wc.Combined(chat, embedder), path=":memory:")
                seed(core, DEBT)
                chat.queue("lesson", {"principle": near})
                report = core.synthesize("So the card first?", "Yes.")
                self.assertEqual(report.first_look["held"], held)
                core.close()
                chat = ScriptedModel()
                core = wc.WisdomCore(AXIOM, wc.Combined(chat, embedder), path=":memory:",
                                     levels={"combine": 0.95, "covered": 0.99})     # both kept while seeding
                a, b = seed(core, DEBT, near)
                core.levels = core.levels.replace(combine=None, covered=None)         # measured from here on
                chat.queue("execute", {"actions": []})
                report = core.synthesize("Which debt first?", "The card.", used=[a, b])
                self.assertEqual(bool(report.merged), held, report.lines())
                core.close()

    def test_the_settings_can_go_back_to_automatic(self):
        levels = wc.Levels(combine=0.8)
        self.assertEqual(levels.automatic(), ["new_ground", "covered"])
        self.assertEqual(levels.replace(combine=None).automatic(), ["new_ground", "covered", "combine"])
        self.assertEqual(wc.Levels.from_dict({"combine": None}).combine, None)


def _shifting_model() -> tuple[wc.FunctionModel, dict[str, bool]]:
    """An embedder whose weights can change underneath its name: with ``state["shift"]`` set, every vector comes
    out reversed (the same size, different numbers)."""
    state = {"shift": False}

    def embed(texts):
        vecs = bow_embed(list(texts), 1024)
        return [v[::-1] if state["shift"] else v for v in vecs]
    return wc.FunctionModel(lambda messages: "", embed, embed_name="same-name"), state


class FingerprintTests(TempDir):
    """Is the embedding model still the one that was measured? One fixed sentence says."""

    def fingerprint_calls(self, model: ScriptedModel) -> int:
        return sum(1 for batch in model.embed_calls if batch == [wc.FINGERPRINT_TEXT])

    def test_the_first_check_keeps_the_fingerprint_and_a_reopened_core_finds_it_unchanged(self):
        model = ScriptedModel()
        with wc.WisdomCore(AXIOM, model, path=self.path) as core:
            seed(core, DEBT)
            self.assertFalse(core.check_fingerprint())
            saved = json.loads(core._store.meta("fingerprint"))
            self.assertEqual(saved["space"], f"bow (layout {wc.EMBED_LAYOUT})")
            self.assertEqual(len(saved["vector"]), 1024)
        events = []
        with wc.WisdomCore(None, model, path=self.path, on_event=events.append) as core:
            core.recall("Which debt first?", expand=False)                 # the first search checks it
            self.assertEqual(self.fingerprint_calls(model), 3)          # measured (seed), checked, checked
            self.assertFalse([e for e in events if e["type"] in ("model_changed", "reindex")])

    def test_a_model_that_changed_under_the_same_name_is_embedded_and_measured_again(self):
        model, state = _shifting_model()
        with wc.WisdomCore(AXIOM, model, path=self.path) as core:
            seed(core, DEBT, SAVINGS)
            core.check_fingerprint()
            first = core.calibration(measure=False)
            self.assertIsNotNone(first)
        state["shift"] = True                                                 # new weights, same tag
        events = []
        with wc.WisdomCore(None, model, path=self.path, on_event=events.append) as core:
            got = core.recall("which debt to repay first", expand=False)
            self.assertEqual(got[0].name, DEBT["name"])                       # found with the new vectors
            changed = [e for e in events if e["type"] == "model_changed"]
            self.assertEqual(len(changed), 1)
            self.assertLess(changed[0]["score"], wc.FINGERPRINT_MATCH)
            self.assertEqual(changed[0]["reembedded"], 2)
            self.assertEqual([e["count"] for e in events if e["type"] == "reindex"], [2])
            self.assertEqual(len([e for e in events if e["type"] == "calibrated"]), 1)   # measured again
            self.assertNotEqual(core.calibration(measure=False)["at"], first["at"])
            self.assertIn("changed underneath", [e.detail for e in core.history() if e.kind == "calibrated"][1])
            self.assertFalse(core.check_fingerprint())                        # and now it matches again

    def test_it_is_checked_again_every_few_lessons(self):
        model = ScriptedModel()
        core = wc.WisdomCore(AXIOM, model, path=":memory:")
        seed(core, DEBT)
        with mock.patch.object(wc, "FINGERPRINT_EVERY", 2):
            for _ in range(5):
                core.synthesize("Which debt first?", "The card.")
        self.assertEqual(self.fingerprint_calls(model), 3)                  # the first lesson, then every second
        core.close()

    def test_a_fingerprint_that_cannot_be_taken_costs_nothing(self):
        model = ScriptedModel()
        embed = model.embed

        def failing(texts):
            if list(texts) == [wc.FINGERPRINT_TEXT]:
                raise wc.ModelError("down for a moment")
            return embed(texts)
        model.embed = failing                                                 # type: ignore[method-assign]
        core = wc.WisdomCore(AXIOM, model, path=":memory:")
        seed(core, DEBT)                                                      # measured, but no fingerprint
        report = core.synthesize("Which debt first?", "The card.")
        self.assertIsNone(report.error)
        self.assertIsNone(core._fp_lessons)                                   # tried again next time
        core.close()


def _domain_model(pairs: dict[str, tuple[float, float]]) -> wc.FunctionModel:
    """The calibration embedder of _check_model (rewordings 0.9, related lessons 0.6: levels 0.75), plus each
    principle in ``pairs`` with its rewording and its related lesson at the two scores given."""
    base = _check_model(0.9, 0.6)
    extra: dict[str, list[float]] = {}
    for i, (name, (rw, rel)) in enumerate(pairs.items()):
        axis = [0.0] * 1024
        axis[700 + 3 * i] = 1.0
        extra[f"{name}."] = axis
        for text, cos, off in ((f"Reworded: {name}", rw, 1), (f"Related to: {name}", rel, 2)):
            other = [0.0] * 1024
            other[700 + 3 * i + off] = 1.0
            extra[text] = [cos * a + (1 - cos * cos) ** 0.5 * b for a, b in zip(axis, other, strict=True)]
    return wc.FunctionModel(lambda messages: "",
                            lambda texts: [extra[t] if t in extra else base.embed([t])[0] for t in texts],
                            embed_name="check")


OWN = ["Pay the card with the highest rate first", "Keep three months of costs in savings",
       "Check a loan for early repayment fees", "Write the budget down before the month starts",
       "Compare the total cost of a loan, not the monthly payment"]


def _pairs_reply(names: list[str], count: int | None = None) -> dict[str, Any]:
    return {"pairs": [{"id": f"P{i}", "reworded": f"Reworded: {n}", "related": f"Related to: {n}"}
                      for i, n in enumerate(names[:count] if count else names, 1)]}


class OwnPrinciplesCalibrationTests(unittest.TestCase):
    """Calibrating on the agent's own principles once it holds enough, and again as it learns."""

    def setUp(self) -> None:
        for name, value in (("DOMAIN_MIN", 4), ("DOMAIN_EVERY", 3)):
            patcher = mock.patch.object(wc, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def core(self, scores: tuple[float, float], **kw) -> tuple[wc.WisdomCore, ScriptedModel]:
        chat = ScriptedModel()
        core = wc.WisdomCore(AXIOM, wc.Combined(chat, _domain_model({n: scores for n in OWN})), path=":memory:", **kw)
        self.addCleanup(core.close)
        seed(core, *OWN[:3])
        return core, chat

    def learn(self, core: wc.WisdomCore, chat: ScriptedModel, name: str) -> wc.SynthesisReport:
        chat.queue("execute", {"actions": [{"type": "add", "principle": principle(name)}]})
        return core.synthesize("A money question", "An answer")

    def test_its_own_principles_set_the_levels_once_it_holds_enough(self):
        core, chat = self.core((0.8, 0.4))
        events = []
        core.on_event = events.append
        self.assertEqual(core.effective_levels().combine, 0.75)               # the general check, for now
        chat.queue("calibrate", _pairs_reply(OWN[:4]))
        report = self.learn(core, chat, OWN[3])                                # the fourth principle
        self.assertIsNone(report.error)
        calib = core.calibration(measure=False)
        self.assertEqual(calib["source"], "agent")
        self.assertEqual(calib["levels"], {"new_ground": 0.6, "combine": 0.6, "covered": 0.8})
        self.assertEqual(calib["general"]["levels"]["combine"], 0.75)         # what the general check found
        self.assertEqual((calib["domain"]["separation"], calib["domain"]["held"]), ("clear", 4))
        self.assertEqual(len(calib["domain"]["pairs"]), 4)
        self.assertEqual(core.effective_levels().combine, 0.6)
        prompt = chat.prompts("calibrate")[0]
        self.assertEqual(prompt[0]["content"], wc.DOMAIN_SYSTEM)
        self.assertEqual(prompt[1]["content"], "PRINCIPLES\n" + "\n".join(f"P{i}. {n}" for i, n in enumerate(OWN[:4], 1)))
        done = [e for e in events if e["type"] == "calibrated"]
        self.assertEqual((done[-1]["ok"], done[-1]["source"]), (True, "agent"))
        self.assertIn("own principles: On 4 of the agent's own principles", core.history(1)[0].detail)

    def test_it_is_done_again_every_few_new_principles(self):
        core, chat = self.core((0.8, 0.4))
        chat.queue("calibrate", _pairs_reply(OWN[:4]), _pairs_reply(OWN[:5]))
        self.learn(core, chat, OWN[3])
        self.assertEqual(chat.kinds().count("calibrate"), 1)
        self.learn(core, chat, OWN[4])
        self.learn(core, chat, "Read the contract before signing it")
        self.assertEqual(chat.kinds().count("calibrate"), 1)                 # 2 new principles: not yet
        self.learn(core, chat, "Ask what a fee is for before paying it")
        self.assertEqual(chat.kinds().count("calibrate"), 2)                 # the third
        self.assertEqual(core.calibration(measure=False)["domain"]["held"], 7)

    def test_own_principles_that_overlap_leave_the_general_calibration_in_use(self):
        core, chat = self.core((0.5, 0.7))
        chat.queue("calibrate", _pairs_reply(OWN[:4]))
        self.learn(core, chat, OWN[3])
        calib = core.calibration(measure=False)
        self.assertEqual((calib["source"], calib["domain"]["separation"]), ("general", "none"))
        self.assertEqual(core.effective_levels().combine, 0.75)
        self.assertIn("overlap, so the general calibration stays in use", calib["domain"]["verdict"])

    def test_too_few_usable_pairs_change_nothing_and_wait(self):
        core, chat = self.core((0.8, 0.4))
        copies = {"pairs": [{"id": "P1", "reworded": OWN[0], "related": "Related to: x"},     # a copy is no test
                            {"id": "P9", "reworded": "a", "related": "b"},                    # not listed
                            {"id": "P2", "reworded": "Reworded: " + OWN[1], "related": "..."}]}  # placeholder
        chat.queue("calibrate", copies)
        self.learn(core, chat, OWN[3])
        calib = core.calibration(measure=False)
        self.assertEqual((calib["source"], calib["domain"]), ("general", None))
        self.learn(core, chat, OWN[4])
        self.assertEqual(chat.kinds().count("calibrate"), 1)                 # not asked again straight away

    def test_a_model_that_fails_costs_the_lesson_nothing(self):
        core, chat = self.core((0.8, 0.4))
        chat.queue("calibrate", "not json", "still not json")
        report = self.learn(core, chat, OWN[3])
        self.assertIsNone(report.error)
        self.assertEqual([p.name for p in report.added], [OWN[3]])
        self.assertEqual(core.calibration(measure=False)["source"], "general")
        chat.queue("calibrate", wc.ModelError("down"))
        for name in OWN[4:] + ["One more", "And another", "A last one"]:
            self.assertIsNone(self.learn(core, chat, name).error)

    def test_levels_set_by_hand_are_never_calibrated(self):
        core, chat = self.core((0.8, 0.4), levels=wc.Levels(new_ground=0.7, covered=0.9, combine=0.8))
        core.calibrate()                                                       # even with a calibration there
        self.learn(core, chat, OWN[3])
        self.assertNotIn("calibrate", chat.kinds())

    def test_a_rewording_that_only_copies_the_principle_does_not_count(self):
        core, chat = self.core((0.8, 0.4))
        reply = _pairs_reply(OWN[:4])
        reply["pairs"][3]["reworded"] = OWN[3] + "."                           # a copy: no test of anything
        chat.queue("calibrate", reply)
        self.learn(core, chat, OWN[3])
        self.assertEqual((core.calibration(measure=False)["source"], core.calibration(measure=False)["domain"]),
                         ("general", None))                                     # 3 usable pairs: too few

    def test_reading_the_calibration_never_needs_the_model(self):
        core, _ = self.core((0.8, 0.4))
        self.assertIsNotNone(core.calibration(measure=False))

        class Unreachable:
            name = "chat"

            @property
            def embed_name(self):
                raise wc.ModelError("the service is down")
        core.model = Unreachable()
        self.assertIsNone(core.calibration(measure=False))

    def test_measuring_the_model_again_starts_over_from_the_general_check(self):
        core, chat = self.core((0.8, 0.4))
        chat.queue("calibrate", _pairs_reply(OWN[:4]), _pairs_reply(OWN[:5]))
        self.learn(core, chat, OWN[3])
        self.assertEqual(core.effective_levels().combine, 0.6)
        core.calibrate()
        self.assertEqual((core.calibration(measure=False)["source"], core.effective_levels().combine), ("general", 0.75))
        self.learn(core, chat, OWN[4])                                         # due again at once
        self.assertEqual(core.effective_levels().combine, 0.6)


class StandingTests(unittest.TestCase):
    """Principles earn their place: how the answers they shaped were received, and review when in doubt."""

    def setUp(self) -> None:
        self.core, self.model = careful_core()                            # the model decides what to write
        self.held = seed(self.core, *many(6, "money rule"))           # P1..P6, clearly different lessons
        self.turns = 0

    def tearDown(self) -> None:
        self.core.close()

    def react(self, previous, corrects: bool, **kw) -> wc.SynthesisReport:
        """One turn whose first look reads the user's message as a pushback (or not) on the previous answer. Its
        lesson is new ground every time, so the whole synthesis runs and the memo sees the listing."""
        self.turns += 1
        lesson = principle(f"fresh lesson word{self.turns}a word{self.turns}b word{self.turns}c")
        self.model.queue("lesson", {"principle": lesson, "corrects": corrects})
        return self.core.synthesize("Thanks, but what about fees?" if not corrects else "No, that is wrong.",
                                    "An answer.", previous=previous, **kw)

    def fresh(self, p: wc.Principle) -> wc.Principle:
        return self.core.get(p.id)

    def test_a_pushback_counts_against_the_principles_behind_the_previous_answer(self):
        report = self.react(self.held, corrects=True)
        shaped = [self.fresh(p) for p in self.held]
        self.assertEqual([(p.shaped, p.questioned) for p in shaped], [(1, 1)] * 4 + [(0, 0)] * 2)   # the top 4
        self.assertEqual([p.id for p in report.questioned], [p.id for p in self.held[:4]])
        self.assertEqual(report.reaction["kind"], "pushback")
        self.assertEqual(report.supported, [])
        self.assertIn("v the user pushed back: questioned 4 principle(s) behind the answer: " + self.held[0].name,
                      "\n".join(report.lines()))
        event = next(e for e in self.core.history() if e.kind == "questioned")
        self.assertEqual(event.sources, [p.id for p in self.held[:4]])
        self.assertIn(self.held[0].name, event.detail)
        stats = self.core.stats()
        self.assertEqual((stats["answers_reacted"], stats["answers_questioned"], stats["in_doubt"]), (1, 1, 0))
        listed = self.model.last_user("reflect")
        for p in self.held[:4]:                                               # listed for the memo to review
            self.assertIn(p.name, listed)

    def test_a_questioned_principle_is_listed_for_review_even_when_the_task_would_not_find_it(self):
        seed(self.core, *many(24, "budget rule"))                        # more than the memo can list by similarity
        (x,) = seed(self.core, principle("Zebras zoom through zoos", use_case="zoo visits", related=["zebra"]))
        self.model.queue("lesson", {"principle": principle("fresh budget lesson"), "corrects": True})
        self.model.queue("execute", {"actions": []})
        self.core.synthesize("A budget rule question", "An answer.", previous=[x])
        listing = self.model.last_user("reflect").split("PRINCIPLES THAT MAY RELATE")[1]
        self.assertIn(x.name, listing)
        self.assertEqual(listing.count("\nP"), wc.CANDIDATES_PER_QUERY + 1)          # the search's own, plus x

    def test_carrying_on_supports_them_quietly(self):
        report = self.react(self.held[:2], corrects=False)
        self.assertEqual([(self.fresh(p).shaped, self.fresh(p).questioned) for p in self.held[:2]], [(1, 0), (1, 0)])
        self.assertEqual([p.id for p in report.supported], [p.id for p in self.held[:2]])
        self.assertEqual((report.questioned, report.reaction["kind"]), ([], "accepted"))
        self.assertNotIn("questioned", [e.kind for e in self.core.history()])
        self.assertIn("^ the user carried on: supported 2 principle(s)", "\n".join(report.lines()))

    def test_an_outcome_counts_this_answers_own_principles(self):
        used = self.held[2:]
        self.model.queue("execute", {"actions": []})
        report = self.core.synthesize("task", "result", outcome="The customer's problem was not solved.",
                                      used=used, success=False)
        self.assertFalse(report.first_look["ran"])                          # an outcome skips the first look...
        self.assertEqual([(self.fresh(p).shaped, self.fresh(p).questioned) for p in used], [(1, 1)] * 4)
        self.assertEqual((report.reaction["kind"], len(report.questioned)), ("outcome", 4))
        self.assertIn("v the outcome was poor", "\n".join(report.lines()))
        report = self.core.synthesize("task", "result", used=self.held[:1], success=True)   # ...and alone it does not
        self.assertEqual((self.fresh(self.held[0]).shaped, self.fresh(self.held[0]).questioned), (1, 0))
        self.assertIn("^ the outcome was good", "\n".join(report.lines()))
        self.assertEqual(self.core.stats()["answers_reacted"], 2)

    def test_nothing_is_counted_without_a_reading(self):
        self.model.queue("lesson", "not json", "still not json")              # the first look could not be read
        self.core.synthesize("No, wrong.", "answer", previous=self.held)
        self.assertEqual(self.fresh(self.held[0]).shaped, 0)
        self.core.levels.new_ground = 0                                        # the first look is off
        self.core.synthesize("No, wrong.", "answer", previous=self.held)
        self.assertEqual(self.fresh(self.held[0]).shaped, 0)
        self.assertEqual(self.core.stats()["answers_reacted"], 0)

    def test_a_pushback_the_first_look_read_counts_even_when_it_named_no_lesson(self):
        self.model.queue("lesson", {"principle": "", "corrects": True})
        self.core.synthesize("No, wrong.", "answer", previous=self.held[:1])
        self.assertEqual((self.fresh(self.held[0]).shaped, self.fresh(self.held[0]).questioned), (1, 1))

    def calm_agent(self, accepted: int = 6) -> None:
        for _ in range(accepted):
            self.react(self.held[5:], corrects=False)                          # P6 keeps serving well

    def test_a_principle_questioned_far_more_than_usual_comes_into_doubt(self):
        self.calm_agent()
        x = self.held[0]
        reports = [self.react([x], corrects=True) for _ in range(3)]
        self.assertEqual([len(r.doubt) for r in reports], [0, 0, 1])          # on the third pushback
        self.assertEqual(reports[2].doubt[0].id, x.id)
        standing = self.core.standing(self.fresh(x))
        self.assertTrue(standing["in_doubt"])
        self.assertEqual((standing["shaped"], standing["questioned"], standing["reviews"]), (3, 3, 0))
        self.assertAlmostEqual(standing["usual"], 1 / 10)                      # none of the other 6 answers, with the prior
        self.assertAlmostEqual(standing["surprise"], 0.001)                    # 3 of 3 at a tenth: very unlikely
        self.assertIn("! in doubt: " + x.name, "\n".join(reports[2].lines()))
        event = next(e for e in self.core.history() if e.kind == "doubt")
        self.assertEqual(event.principle_id, x.id)
        self.assertIn("3 of the 3 answers it shaped", event.detail)
        self.assertEqual(self.core.stats()["in_doubt"], 1)
        self.assertFalse(self.core.standing(self.fresh(self.held[5]))["in_doubt"])

    def test_what_counts_as_doubt_depends_on_how_often_this_user_pushes_back(self):
        x = self.held[0]
        for i in range(6):                                                     # x: questioned 4 of 6
            self.react([x], corrects=i < 4)
        self.calm_agent(10)                                                     # a calm user: 0 of 10 other answers
        standing = self.core.standing(self.fresh(x))
        self.assertAlmostEqual(standing["usual"], 1 / 14)
        self.assertTrue(standing["in_doubt"])                                   # 4 of 6 at 7%: very unlikely
        for _ in range(12):                                                     # a demanding user: 12 of 22 others
            self.react(self.held[5:], corrects=True)
        standing = self.core.standing(self.fresh(x))
        self.assertAlmostEqual(standing["usual"], 13 / 26)
        self.assertGreater(standing["surprise"], 0.3)                          # 4 of 6 at a half: ordinary
        self.assertFalse(standing["in_doubt"])

    def test_three_pushbacks_are_needed_however_unlikely(self):
        self.calm_agent(20)
        x = self.held[0]
        self.react([x], corrects=True)
        self.react([x], corrects=True)
        standing = self.core.standing(self.fresh(x))
        self.assertLess(standing["surprise"], wc.DOUBT_ALPHA)
        self.assertFalse(standing["in_doubt"])

    def test_an_in_doubt_principle_is_listed_for_review_and_can_stand(self):
        self.calm_agent()
        x = self.held[0]
        for _ in range(3):
            self.react([x], corrects=True)
        self.model.queue("execute", {"actions": [{"type": "stands", "id": "P1",
                                                  "reason": "the pushbacks were about tone, not the rule"}]})
        report = self.react(self.held[5:], corrects=False, used=[x])          # x comes up again
        memo_prompt = self.model.last_user("reflect")
        self.assertIn("[IN DOUBT: the user pushed back on 3 of the 3 answers it shaped, against about 9% of this "
                      "agent's other answers. Review it.]", memo_prompt)       # 0 of 7 others, with the prior
        self.assertIn("P1. " + x.name + " (in doubt)", self.model.last_user("execute"))
        self.assertEqual([(p.id, reason) for p, reason in report.stands],
                         [(x.id, "the pushbacks were about tone, not the rule")])
        self.assertIn("= stands: " + x.name, "\n".join(report.lines()))
        x = self.fresh(x)
        self.assertEqual((x.review["count"], x.review["shaped"], x.review["questioned"]), (1, 3, 3))
        standing = self.core.standing(x)
        self.assertEqual((standing["in_doubt"], standing["shaped"], standing["questioned"], standing["reviews"]),
                         (False, 0, 0, 1))                                      # a clean start since the review
        self.assertEqual((x.shaped, x.questioned), (3, 3))                     # the whole record is kept
        self.assertEqual(next(e for e in self.core.history() if e.kind == "stands").principle_id, x.id)
        self.react(self.held[5:], corrects=False, used=[x])
        self.assertNotIn("IN DOUBT", self.model.last_user("reflect"))
        self.assertEqual(report.to_dict()["stands"][0]["reason"], "the pushbacks were about tone, not the rule")

    def test_a_review_is_remembered_if_it_comes_into_doubt_again(self):
        self.calm_agent()
        x = self.held[0]
        for _ in range(3):
            self.react([x], corrects=True)
        self.model.queue("execute", {"actions": [{"type": "stands", "id": "P1", "reason": "sound"}]})
        self.react(self.held[5:], corrects=False, used=[x])
        for _ in range(3):
            self.react([x], corrects=True)
        self.assertTrue(self.core.standing(self.fresh(x))["in_doubt"])
        self.model.queue("execute", {"actions": []})
        self.react(self.held[5:], corrects=False, used=[x])
        self.assertIn("reviewed and found to stand 1 time(s) before", self.model.last_user("reflect"))

    def test_stands_applies_only_to_a_principle_in_doubt(self):
        self.model.queue("execute", {"actions": [{"type": "stands", "id": "P1", "reason": "fine"},
                                                 {"type": "stands", "id": "P9"}, {"type": "stands"}]})
        report = self.react(self.held[5:], corrects=False, used=self.held[:1])
        self.assertEqual(report.stands, [])
        self.assertEqual(report.skipped, [f"stands: {self.held[0].name} was not in doubt, so there was nothing to review",
                                          "stands: P9 is not a listed principle", "stands: no principle was named"])

    def test_a_proven_principle_is_noted_for_the_memo(self):
        y = self.held[5]
        self.calm_agent(8)
        self.model.queue("execute", {"actions": []})
        self.react(self.held[5:], corrects=False, used=[y])
        self.assertIn(f"P1. {y.render()} [PROVEN: shaped 9 answers, none questioned.]",   # counted before the memo
                      self.model.last_user("reflect"))
        self.assertTrue(self.core.standing(self.fresh(y))["proven"])
        self.assertNotIn("[", self.model.last_user("reflect").split("P2. ")[1].split("\n")[0])   # others: no note

    def test_removing_or_revising_a_principle_resolves_its_doubt(self):
        self.calm_agent()
        x = self.held[0]
        for _ in range(3):
            self.react([x], corrects=True)
        self.model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1"],
                                                  "principle": principle("A better rule about money")}]})
        report = self.react(self.held[5:], corrects=False, used=[x])
        (sources, revised), = report.merged
        self.assertEqual((revised.shaped, revised.questioned), (0, 0))        # a fresh start
        self.assertEqual(self.core.stats()["in_doubt"], 0)

    def test_the_record_survives_a_backup(self):
        self.calm_agent(2)
        self.react(self.held[:1], corrects=True)
        self.model.queue("execute", {"actions": []})
        state = self.core.export_state()
        other = wc.WisdomCore(AXIOM, ScriptedModel(), path=":memory:")
        other.import_state(state)
        copied = {p.name: p for p in other.principles()}
        self.assertEqual((copied[self.held[0].name].shaped, copied[self.held[0].name].questioned), (1, 1))
        self.assertEqual(copied[self.held[5].name].shaped, 2)
        self.assertEqual(other.stats()["answers_reacted"], 3)
        self.assertAlmostEqual(other.usual_pushback(), self.core.usual_pushback())
        other.close()
        old = wc.WisdomCore(AXIOM, ScriptedModel(), path=":memory:")          # a backup from before standing
        stripped = {**state, "principles": [{k: v for k, v in p.items() if k not in ("shaped", "questioned", "review")}
                                            for p in state["principles"]]}
        old.import_state(stripped)
        self.assertEqual([(p.shaped, p.questioned, p.review) for p in old.principles()][0], (0, 0, None))
        old.close()


class StandingUpgradeTests(TempDir):
    def test_a_database_from_before_standing_is_upgraded(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.path) as core:
            seed(core, DEBT)
        con = sqlite3.connect(self.path)
        for column in ("shaped", "questioned", "review"):
            con.execute(f"ALTER TABLE principles DROP COLUMN {column}")
        con.commit()
        con.close()
        model = ScriptedModel()
        with wc.WisdomCore(None, model, path=self.path) as core:
            (p,) = core.principles()
            self.assertEqual((p.shaped, p.questioned, p.review), (0, 0, None))
            model.queue("lesson", {"principle": principle("x"), "corrects": True})
            core.synthesize("No.", "answer", previous=[p])
            self.assertEqual(core.get(p.id).questioned, 1)


class AgentStandingTests(unittest.TestCase):
    def test_the_agent_hands_over_the_principles_behind_the_previous_answer(self):
        model = ScriptedModel()
        agent = wc.Agent(AXIOM, model, path=":memory:")
        a, b = seed(agent.core, DEBT, SAVINGS)
        model.queue("lesson", {"principle": principle("x"), "corrects": False})
        agent.ask("Which debt first?")                                         # nothing before this answer
        self.assertEqual([p.id for p in agent.previous_principles], [p.id for p in agent.last_principles])
        self.assertEqual(agent.core.stats()["answers_reacted"], 0)
        model.queue("lesson", {"principle": principle("y"), "corrects": True})
        agent.ask("No, that is wrong.")
        first = [p.id for p in agent.last_report.questioned]
        self.assertEqual(set(first), {a.id, b.id})                             # what shaped the first answer
        self.assertEqual(agent.core.get(a.id).questioned, 1)
        agent.reset()
        self.assertEqual(agent.previous_principles, [])
        model.queue("lesson", {"principle": principle("z"), "corrects": True})
        agent.ask("Still wrong.")                                              # a new conversation: nothing before
        self.assertEqual(agent.core.get(a.id).questioned, 1)

    def test_an_application_can_say_how_the_answer_went(self):
        model = ScriptedModel()
        agent = wc.Agent(AXIOM, model, path=":memory:")
        (a,) = seed(agent.core, DEBT)
        agent.respond("Which debt first?")
        model.queue("execute", {"actions": []})
        report = agent.synthesize(outcome="The person paid the card off.", success=True)
        self.assertEqual(report.reaction["kind"], "outcome")
        self.assertEqual((agent.core.get(a.id).shaped, agent.core.get(a.id).questioned), (1, 0))

    def test_principles_behind_an_earlier_answer_can_be_given_as_ids(self):
        model = ScriptedModel()
        agent = wc.Agent(AXIOM, model, path=":memory:")
        (a,) = seed(agent.core, DEBT)
        agent.previous_principles = [a.id]                                     # restored from a record, say
        model.queue("lesson", {"principle": principle("x"), "corrects": True})
        agent.ask("No, that is wrong.")
        self.assertEqual(agent.core.get(a.id).questioned, 1)


class BackupTests(TempDir):
    """export_state() and import_state(): the whole of one agent's learning, moved somewhere else."""

    def furnished(self) -> tuple[wc.WisdomCore, ScriptedModel]:
        """A core that has learned, merged, removed and changed its axiom."""
        model = ScriptedModel()
        core = wc.WisdomCore(AXIOM, model, path=self.path)
        a, b, c = seed(core, SAVINGS, TOTALS, DEBT)
        model.queue("execute", {"actions": [{"type": "merge", "ids": ["P1", "P2"],
                                             "principle": principle("Check the figures and the time horizon")},
                                            {"type": "remove", "id": "P3", "reason": "wrong for this axiom"}]})
        core.synthesize("A task", "A result", used=[a, b, c])
        core.set_core_axiom("Help people keep what they earn.")
        core.recall("money", expand=False)
        return core, model

    def test_a_backup_rebuilds_everything_somewhere_else(self):
        core, _ = self.furnished()
        state = core.export_state()
        before = [p.to_dict() for p in core.principles(include_ended=True)]
        history = core.axiom_history()
        events = [(e.kind, e.detail, e.task, e.sources) for e in core.history(None)]
        stats = core.stats()
        core.close()
        self.assertEqual(json.loads(json.dumps(state)), state)          # it is plain data
        self.assertEqual((state["state_version"], state["wisdom_core"]), (wc.STATE_VERSION, wc.__version__))
        self.assertEqual(state["axiom"], "Help people keep what they earn.")

        other = os.path.join(self.dir, "elsewhere.db")
        into = wc.WisdomCore("A placeholder axiom.", ScriptedModel(), path=other)
        self.addCleanup(into.close)
        written = into.import_state(state, replace=True)
        self.assertEqual(written["principles"], len(before))
        self.assertEqual([p.to_dict() for p in into.principles(include_ended=True)], before)
        self.assertEqual(into.axiom_history(), history)
        self.assertEqual(into.core_axiom, "Help people keep what they earn.")
        self.assertEqual([(e.kind, e.detail, e.task, e.sources) for e in into.history(None)], events)
        self.assertEqual(into.stats(), stats)
        recalled = into.recall("Which debt should I repay first?", expand=False)   # embedded again on the way
        self.assertEqual([p.name for p in recalled], [p.name for p in into.principles()])
        self.assertTrue(all(p.score is not None for p in recalled))

    def test_an_imported_agent_goes_on_learning(self):
        core, _ = self.furnished()
        state = core.export_state()
        core.close()
        into = wc.WisdomCore("A placeholder axiom.", ScriptedModel(), path=os.path.join(self.dir, "two.db"))
        self.addCleanup(into.close)
        into.import_state(state, replace=True)
        into.model.queue("execute", {"actions": [{"type": "add", "principle": DEBT}]})
        report = into.synthesize("Which debt first?", "The card.")
        self.assertEqual([p.name for p in report.added], [DEBT["name"]])
        self.assertEqual(len(into.principles()), 2)

    def test_a_database_that_already_holds_principles_is_not_overwritten(self):
        core, _ = self.furnished()
        state = core.export_state()
        core.close()
        into = wc.WisdomCore("Another axiom.", ScriptedModel(), path=os.path.join(self.dir, "three.db"))
        self.addCleanup(into.close)
        seed(into, SAVINGS)
        with self.assertRaises(wc.WisdomCoreError) as err:
            into.import_state(state)
        self.assertIn("already holds principles", str(err.exception))
        self.assertEqual([p.name for p in into.principles()], [SAVINGS["name"]])
        into.import_state(state, replace=True)                        # asked for, it replaces
        self.assertEqual(len(into.principles()), 1)
        self.assertEqual(into.principles()[0].name, "Check the figures and the time horizon")
        found = into.recall("money", expand=False)                    # and the search sees the new ones only
        self.assertEqual([p.name for p in found], ["Check the figures and the time horizon"])

    def test_what_cannot_be_imported(self):
        core, _ = memory_core()
        state = core.export_state()
        for bad, why in ((None, "not an exported"), ("text", "not an exported"), ({}, "not an exported"),
                         ({"state_version": "x"}, "not an exported"),
                         ({**state, "state_version": wc.STATE_VERSION + 1}, "newer Wisdom Core")):
            with self.subTest(bad=bad), self.assertRaises(wc.WisdomCoreError) as err:
                core.import_state(bad, replace=True)
            self.assertIn(why, str(err.exception))
        with self.assertRaises(wc.WisdomCoreError) as err:
            core.import_state({"state_version": wc.STATE_VERSION, "principles": []}, replace=True)
        self.assertIn("no Core Axiom", str(err.exception))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
def similarity(a: dict, b: dict) -> float:
    """How alike two principles measure with the test embedder, as the Wisdom Core measures them."""
    ta, tb = (wc.Principle(id="x", **p).identity_text() for p in (a, b))
    va, vb = (wc._unit(v) for v in ScriptedModel().embed([ta, tb]))
    return wc._dot(va, vb)


NEAR_DEBT = principle("Always list debts by interest rate before suggesting which debt to repay first",   # 0.89
                      "Paying the costliest debt first saves the most money", "Debt repayment questions",
                      "someone has several debts", ["interest rate", "loans"])


class CombineTests(unittest.TestCase):
    """Principles nearly the same are combined into one by the Wisdom Core itself (the combine level)."""

    def setUp(self) -> None:
        self.alike = similarity(DEBT, NEAR_DEBT)
        self.assertGreater(self.alike, 0.85)            # the pair the tests rely on is nearly the same...
        self.assertLess(self.alike, 0.95)               # ...but not identical
        self.assertLess(similarity(DEBT, SAVINGS), 0.85)

    def test_the_defaults(self):
        self.assertEqual((wc.Levels().resolved().combine, wc.Levels().max_adds, wc.Levels().resolved().covered),
                         (0.85, 1, 0.95))

    def test_a_draft_nearly_the_same_as_a_held_principle_is_combined_with_it(self):
        core, model = memory_core()
        (held,) = seed(core, DEBT)
        model.queue("execute", {"actions": [{"type": "add", "principle": NEAR_DEBT}]})
        model.queue("combine", {"principle": principle("Rank debts by interest rate, then repay the costliest first",
                                                       "It saves the most money", "Debt repayment",
                                                       "someone has several debts", ["debt"])})
        report = core.synthesize("Which debt first?", "The card.")
        self.assertEqual(report.added, [])
        ((sources, merged),) = report.merged
        self.assertEqual([s.name for s in sources], [DEBT["name"], NEAR_DEBT["name"]])
        self.assertEqual(merged.name, "Rank debts by interest rate, then repay the costliest first")
        self.assertAlmostEqual(merged.score, self.alike, places=5)
        self.assertEqual([p.name for p in core.principles()], [merged.name])       # held once, never twice
        old = core.get(held.id)
        self.assertEqual((old.status, old.merged_into), ("merged", merged.id))
        self.assertIn(f"({self.alike:.2f} alike)", report.lines()[0])
        self.assertEqual(report.to_dict()["merged"][0]["score"], merged.score)
        listing = model.last_user("combine")
        self.assertIn(DEBT["name"], listing)
        self.assertIn(NEAR_DEBT["name"], listing)
        self.assertIn(AXIOM, listing)
        (event,) = [e for e in core.history() if e.kind == "merged"]
        self.assertEqual(event.sources, [held.id])
        self.assertIn("combined automatically", event.detail)

    def test_a_draft_at_the_covered_level_is_simply_already_held(self):
        core, model = memory_core()
        seed(core, DEBT)
        model.queue("execute", {"actions": [{"type": "add", "principle": DEBT}]})
        report = core.synthesize("Which debt first?", "The card.")
        self.assertEqual(([p.name for p in report.covered], report.merged), ([DEBT["name"]], []))
        self.assertNotIn("combine", model.kinds())

    def test_a_draft_below_the_combine_level_is_added(self):
        core, model = memory_core()
        seed(core, DEBT)
        model.queue("execute", {"actions": [{"type": "add", "principle": SAVINGS}]})
        report = core.synthesize("Where do I keep savings?", "A savings account.")
        self.assertEqual([p.name for p in report.added], [SAVINGS["name"]])
        self.assertNotIn("combine", model.kinds())

    def test_a_combine_level_of_zero_turns_it_off(self):
        core, model = memory_core(levels=wc.Levels(combine=0))
        seed(core, DEBT)
        model.queue("execute", {"actions": [{"type": "add", "principle": NEAR_DEBT}]})
        report = core.synthesize("Which debt first?", "The card.")
        self.assertEqual([p.name for p in report.added], [NEAR_DEBT["name"]])
        self.assertEqual(len(core.principles()), 2)
        self.assertNotIn("combine", model.kinds())

    def test_near_duplicates_already_held_are_combined_when_a_task_brings_them_up(self):
        core, model = memory_core(levels=wc.Levels(combine=0))
        a, b = seed(core, DEBT, NEAR_DEBT)              # held twice, from before the combine level
        t = seed(core, TOTALS)[0]
        core.levels.combine = 0.85
        model.queue("execute", {"actions": [{"type": "covered", "id": "P1"}]})
        report = core.synthesize("Which debt should I repay first?", "The card.", used=[a, b])
        ((sources, merged),) = report.merged
        self.assertEqual({s.id for s in sources}, {a.id, b.id})
        self.assertEqual(sorted(p.name for p in core.principles()), sorted([merged.name, t.name]))
        self.assertEqual(model.kinds().count("combine"), 1)

    def test_at_most_two_combines_per_task(self):
        core, model = memory_core(levels=wc.Levels(combine=0, covered=1.0, force=False))
        pairs = [(principle(f"Check rule {w} before advising about money matters", related=[w]),
                  principle(f"Check rule {w} before advising about money matters carefully", related=[w]))
                 for w in ("alpha", "bravo", "charlie")]
        held = seed(core, *[p for pair in pairs for p in pair])
        core.levels.combine = 0.85
        self.assertTrue(all(similarity(x, y) >= 0.85 for x, y in pairs))
        report = core.synthesize("money matters advice", "ok", used=held)
        self.assertEqual(len(report.merged), wc.MAX_COMBINES_PER_TASK)
        self.assertEqual(len(core.principles()), 6 - wc.MAX_COMBINES_PER_TASK)

    def test_when_the_model_cannot_combine_them_nothing_is_lost(self):
        core, model = memory_core()
        (held,) = seed(core, DEBT)
        model.queue("execute", {"actions": [{"type": "add", "principle": NEAR_DEBT}]})
        model.queue("combine", "not json", "still not json")
        report = core.synthesize("Which debt first?", "The card.")
        self.assertIsNone(report.error)
        self.assertEqual(([p.name for p in core.principles()], [p.name for p in report.covered]),
                         ([DEBT["name"]], [DEBT["name"]]))
        self.assertTrue(any(s.startswith("combine: the model wrote no combined principle") for s in report.skipped))
        self.assertEqual(model.kinds().count("combine"), 2)                       # asked once more for JSON

    def test_a_model_error_while_combining_keeps_both_sides_safe(self):
        core, model = memory_core(levels=wc.Levels(combine=0))
        a, b = seed(core, DEBT, NEAR_DEBT)
        core.levels.combine = 0.85
        model.queue("execute", {"actions": [{"type": "covered", "id": "P1"}]})
        model.queue("combine", wc.ModelError("the server went away"))
        report = core.synthesize("Which debt first?", "The card.", used=[a, b])
        self.assertIsNone(report.error)
        self.assertEqual(report.merged, [])
        self.assertEqual(len(core.principles()), 2)


# ---------------------------------------------------------------------------
class LearningFromTheUserTests(unittest.TestCase):
    """Wisdom Synthesis learns from what the user said first, and from the agent's own answer only second."""

    def test_the_memo_is_told_to_learn_from_the_user_first(self):
        for words in ("Learn from the user and from the substance of the thinking, never from the form of the answer",
                      "WHAT THE USER SHOWED", "a correction, a complaint, approval",
                      "WHAT THE THINKING ESTABLISHED", "what they worked out about the subject itself",
                      "pieces of wisdom", "Work it out from this exchange before you compare it with the listed principles",
                      "If your lesson only restates a listed principle, it is not what this exchange added",
                      "fact, value or goal they stated", "counts only if the user asked for it or reacted to it",
                      "only to labels that are listed"):
            self.assertIn(words, wc.REFLECT_SYSTEM)
        self.assertIn("Take it from what the user showed or from what the thinking established about the subject",
                      wc.FORCE_MEMO)
        self.assertIn("never from how the answer was laid out", wc.FORCE_MEMO)
        self.assertIn("stating the truth or the way of acting it teaches", wc.EXECUTE_SYSTEM)

    def test_the_answer_being_responded_to_is_named_and_the_user_comes_before_the_agent(self):
        core, model = memory_core()
        conversation = [{"role": "user", "content": "What is good?"},
                        {"role": "assistant", "content": "Three frameworks: consequences, duty and virtue."},
                        {"role": "user", "content": "Too long. Just tell me your view."},
                        {"role": "assistant", "content": "Good is what helps people flourish."}]
        core.synthesize("I prefer short, direct answers. What is bad?", "Bad is what harms people.",
                        conversation=conversation)
        user = model.last_user("reflect")
        self.assertIn("EARLIER IN THE CONVERSATION\nUser: What is good?\nAgent: Three frameworks", user)
        self.assertIn("User: Too long. Just tell me your view.", user)
        self.assertIn("THE AGENT'S PREVIOUS ANSWER, WHICH THE USER IS RESPONDING TO\nGood is what helps people flourish.",
                      user)
        order = [user.index(h) for h in ("EARLIER IN THE CONVERSATION", "THE AGENT'S PREVIOUS ANSWER",
                                         "WHAT THE USER SAID\nI prefer short, direct answers.", "THE AGENT'S ANSWER\n",
                                         "PRINCIPLES THAT MAY RELATE")]
        self.assertEqual(order, sorted(order))

    def test_without_an_earlier_answer_there_is_no_previous_answer_section(self):
        core, model = memory_core()
        core.synthesize("What is sophistry?", "Clever but fallacious argument.",
                        conversation=[{"role": "user", "content": "hello"}])
        user = model.last_user("reflect")
        self.assertNotIn("PREVIOUS ANSWER", user)
        self.assertIn("User: hello", user)

    def test_the_agents_own_long_answer_is_cut_shorter_than_before(self):
        core, model = memory_core()
        core.synthesize("Explain everything.", "word " * 4000)
        answer = model.last_user("reflect").split("THE AGENT'S ANSWER\n", 1)[1].split("\n\nPRINCIPLES", 1)[0]
        self.assertLess(len(answer), wc.ANSWER_CHARS + 100)
        self.assertLess(wc.ANSWER_CHARS, wc.RESULT_CHARS)

    def test_a_long_principle_is_cut_at_a_word_never_in_the_middle(self):
        core, model = careful_core()
        long_name = "When addressing profound questions " + "structure the response carefully " * 20 + "always."
        model.queue("execute", {"actions": [{"type": "add", "principle": principle(long_name, "why " * 200)}]})
        (p,) = core.synthesize("q", "a").added
        for text, limit in ((p.name, wc.NAME_CHARS), (p.importance, wc.FIELD_CHARS)):
            self.assertLessEqual(len(text), limit)
            self.assertTrue(text.endswith("…"), text)
            self.assertNotIn("characters cut", text)
            self.assertNotIn("\n", text)
        self.assertTrue(long_name.startswith(p.name[:-1]))
        self.assertIn("(at most 25 words)", wc.EXECUTE_SYSTEM)
        self.assertIn("(at most 25 words)", wc.COMBINE_SYSTEM)

    def test_short_fields_are_left_alone(self):
        self.assertEqual(wc._shorten("  Add   the figures twice.  ", 240), "Add the figures twice.")
        self.assertEqual(wc._shorten("x" * 300, 240), "x" * 239 + "…")


class ReasoningTests(unittest.TestCase):
    """The thinking behind an answer reaches Wisdom Synthesis, so the lesson can come from what it worked out."""

    def test_reasoning_given_to_synthesize_sits_between_the_user_and_the_answer(self):
        core, model = memory_core()
        core.synthesize("What is loyalty?", "Loyalty is fidelity that keeps the right to object.",
                        reasoning="Genuine loyalty survives disagreement; coerced loyalty demands silence.")
        user = model.last_user("reflect")
        block = "THE AGENT'S REASONING BEFORE IT ANSWERED\nGenuine loyalty survives disagreement"
        self.assertIn(block, user)
        self.assertLess(user.index("WHAT THE USER SAID"), user.index(block))
        self.assertLess(user.index(block), user.index("THE AGENT'S ANSWER"))

    def test_no_reasoning_no_section(self):
        core, model = memory_core()
        core.synthesize("What is loyalty?", "Fidelity.")
        core.synthesize("What is honor?", "Integrity.", reasoning="   ")
        for prompt in model.prompts("reflect"):
            self.assertNotIn("REASONING", prompt[-1]["content"])

    def test_long_reasoning_is_cut(self):
        core, model = memory_core()
        core.synthesize("q", "a", reasoning="thought " * 3000)
        part = model.last_user("reflect").split("BEFORE IT ANSWERED\n", 1)[1].split("\n\nTHE AGENT'S ANSWER", 1)[0]
        self.assertLess(len(part), wc.REASONING_CHARS + 100)

    def test_the_agent_hands_on_reasoning_written_inline(self):
        model = ScriptedModel()
        model.queue("answer", "<think>Honor is integrity when unobserved.</think>Honor is acting on your values.")
        agent = wc.Agent(AXIOM, model, path=":memory:")
        self.assertEqual(agent.ask("What is honor?"), "Honor is acting on your values.")
        self.assertEqual(agent.last_reasoning, "Honor is integrity when unobserved.")
        self.assertIn("BEFORE IT ANSWERED\nHonor is integrity when unobserved.", model.last_user("reflect"))

    def test_the_agent_hands_on_a_reasoning_models_separate_thinking(self):
        class Thinker(ScriptedModel):
            def chat(self, messages, *, json_mode=False, temperature=None, stats=None):
                reply = super().chat(messages, json_mode=json_mode, temperature=temperature)
                if stats is not None and self.kind_of(messages) == "answer":
                    stats["thinking"] = "Loyalty that forbids dissent is control, not loyalty."
                return reply
        model = Thinker()
        agent = wc.Agent(AXIOM, model, path=":memory:")
        agent.ask("What is loyalty?")
        self.assertEqual(agent.last_reasoning, "Loyalty that forbids dissent is control, not loyalty.")
        self.assertIn("BEFORE IT ANSWERED\nLoyalty that forbids dissent is control", model.last_user("reflect"))

    def test_an_answer_without_reasoning_hands_on_nothing(self):
        model = ScriptedModel()
        agent = wc.Agent(AXIOM, model, path=":memory:")
        agent.ask("hello")
        self.assertEqual(agent.last_reasoning, "")
        self.assertNotIn("REASONING", model.last_user("reflect"))

    def test_respond_now_learn_later_keeps_the_reasoning(self):
        model = ScriptedModel()
        model.queue("answer", "<think>weigh the rates</think>The card.")
        agent = wc.Agent(AXIOM, model, path=":memory:")
        agent.respond("Which debt first?")
        agent.respond("And then?")                       # a second answer before learning: its own reasoning, none
        agent.synthesize()
        self.assertNotIn("weigh the rates", model.last_user("reflect"))

