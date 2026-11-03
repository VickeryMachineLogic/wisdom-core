"""The command line: python -m wisdom_core."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

import wisdom_core as wc
from tests.fakes import ScriptedModel, close_stores, principle, seed
from tests.httpfake import ollama_server

tearDownModule = close_stores          # close every database this module's tests opened


AXIOM = "Help people make sound money decisions."
MODULE_DIR = os.path.dirname(os.path.abspath(wc.__file__))


def run_cli(args: list[str], stdin: str = "", env: dict[str, str] | None = None, cwd: str | None = None):
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = MODULE_DIR + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env.pop("OLLAMA_HOST", None)
    full_env.update(env or {})
    return subprocess.run([sys.executable, "-m", "wisdom_core", *args], input=stdin, capture_output=True,
                          text=True, env=full_env, cwd=cwd, timeout=60)


class ListingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="wisdom-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.db = os.path.join(self.dir, "agent.db")
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.db) as core:
            seed(core, principle("Add the figures twice before reporting a total", "A wrong total misleads",
                                 "Any sum", "adding up bills"))

    def test_principles_without_any_model(self):
        out = run_cli(["--db", self.db, "--principles"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(f"Core Axiom: {AXIOM}", out.stdout)
        self.assertIn("1. Add the figures twice before reporting a total", out.stdout)
        self.assertIn("why: A wrong total misleads", out.stdout)

    def test_history(self):
        out = run_cli(["--db", self.db, "--history"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("added", out.stdout)
        self.assertIn("axiom", out.stdout)

    def test_missing_database(self):
        out = run_cli(["--db", os.path.join(self.dir, "nope.db"), "--principles"])
        self.assertEqual(out.returncode, 1)
        self.assertIn("No principle database", out.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "nope.db")))

    def test_version(self):
        out = run_cli(["--version"])
        self.assertEqual(out.stdout.strip(), f"wisdom-core {wc.__version__}")


class ChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="wisdom-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.db = os.path.join(self.dir, "agent.db")

    def test_a_session(self):
        brain = ScriptedModel()
        brain.queue("answer", "Pay the 24% card first.")
        brain.queue("execute", {"actions": [{"type": "add", "principle": principle(
            "Repay the debt with the highest interest rate first", "It saves the most money")}]})
        server, _ = ollama_server(brain)
        self.addCleanup(server.close)
        script = "\n".join(["Car loan at 6% or card at 24%, which first?", "/principles", "/stats", "/axiom",
                            "/history", "/reset", "/help", "/bogus", "", "/quit"]) + "\n"
        out = run_cli(["--db", self.db, "--axiom", AXIOM, "--host", server.url], stdin=script)
        self.assertEqual(out.returncode, 0, out.stderr)
        text = out.stdout
        self.assertIn(f"Core Axiom: {AXIOM}", text)
        self.assertIn("ollama:llama3.2:latest (chat), ollama:nomic-embed-text:latest (embeddings)", text)
        self.assertIn("agent> Pay the 24% card first.", text)
        self.assertIn("wisdom core + added: Repay the debt with the highest interest rate first", text)
        self.assertIn("1. Repay the debt with the highest interest rate first", text)
        self.assertIn("active: 1", text)
        self.assertIn("Conversation cleared. Principles are kept.", text)
        self.assertIn("/principles", text)            # the help text
        self.assertIn("Unknown command", text)

    def test_the_axiom_is_asked_for_when_missing(self):
        server, _ = ollama_server()
        self.addCleanup(server.close)
        out = run_cli(["--db", self.db, "--host", server.url], stdin=f"{AXIOM}\n/axiom\n")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Core Axiom (the one sentence this agent serves)", out.stdout)
        with wc.WisdomCore(None, ScriptedModel(), path=self.db) as core:
            self.assertEqual(core.core_axiom, AXIOM)
        again = run_cli(["--db", self.db, "--host", server.url], stdin="/quit\n")   # read from the database now
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertNotIn("the one sentence this agent serves", again.stdout)

    def test_changing_the_axiom_is_announced(self):
        with wc.WisdomCore(AXIOM, ScriptedModel(), path=self.db):
            pass
        server, _ = ollama_server()
        self.addCleanup(server.close)
        out = run_cli(["--db", self.db, "--axiom", "Protect people from scams.", "--host", server.url],
                      stdin="/quit\n")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(f"Changing this agent's Core Axiom. It was: {AXIOM}", out.stdout)
        self.assertIn("Core Axiom: Protect people from scams.", out.stdout)

    def test_a_foreign_database_is_refused(self):
        import sqlite3
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE invoices (id INTEGER)")
        con.commit()
        con.close()
        out = run_cli(["--db", self.db, "--principles"])
        self.assertEqual(out.returncode, 1)
        self.assertIn("did not create", out.stderr)
        out = run_cli(["--db", self.db, "--axiom", AXIOM], stdin="/quit\n")
        self.assertEqual(out.returncode, 1)
        self.assertIn("did not create", out.stderr)

    def test_no_axiom_given(self):
        out = run_cli(["--db", self.db], stdin="\n")
        self.assertEqual(out.returncode, 1)
        self.assertIn("A Core Axiom is needed", out.stderr)

    def test_ollama_not_running(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        out = run_cli(["--db", self.db, "--axiom", AXIOM, "--host", f"127.0.0.1:{port}"], stdin="hi\n")
        self.assertEqual(out.returncode, 1)
        self.assertIn("Is Ollama running?", out.stderr)

    def test_a_model_error_mid_session_is_shown_and_the_session_goes_on(self):
        brain = ScriptedModel()
        brain.queue("answer", wc.ModelError("model crashed"), "Second try worked.")
        server, _ = ollama_server(brain)
        self.addCleanup(server.close)
        brain_errors = run_cli(["--db", self.db, "--axiom", AXIOM, "--host", server.url],
                               stdin="first\nsecond\n/quit\n")
        # the fake server turns the ModelError into an HTTP 500 for the first message
        self.assertEqual(brain_errors.returncode, 0, brain_errors.stderr)
        self.assertIn("[model error]", brain_errors.stdout)
        self.assertIn("agent> Second try worked.", brain_errors.stdout)

    def test_openai_needs_a_model_name(self):
        out = run_cli(["--db", self.db, "--axiom", AXIOM, "--openai-base-url", "http://127.0.0.1:1/v1"])
        self.assertEqual(out.returncode, 1)
        self.assertIn("--model is required", out.stderr)

    def test_no_learn(self):
        brain = ScriptedModel()
        server, _ = ollama_server(brain)
        self.addCleanup(server.close)
        out = run_cli(["--db", self.db, "--axiom", AXIOM, "--host", server.url, "--no-learn"], stdin="hi\n")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("wisdom core", out.stdout)
        self.assertEqual(brain.kinds(), ["answer"])


if __name__ == "__main__":
    unittest.main()
