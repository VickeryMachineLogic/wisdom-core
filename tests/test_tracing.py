"""Framework 0.2: watching it work (on_event), streaming, the think switch, context windows."""

from __future__ import annotations

import json
import threading
import time
import unittest

import wisdom_core as wc
from tests.fakes import ScriptedModel, bow_embed, close_stores, principle, seed
from tests.httpfake import ollama_server, openai_server

tearDownModule = close_stores          # close every database this module's tests opened


AXIOM = "Help people make sound money decisions."
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
DEBT = principle("List debts by interest rate before suggesting which to repay first",
                 "Paying the costliest debt first saves the most money", "Debt repayment questions",
                 "someone has several debts", ["interest rate", "loans"])


class Stop(BaseException):
    """What a caller raises from on_event to stop a run."""


def streaming_function_model(replies: dict[str, str] | None = None):
    """A FunctionModel whose chat streams its reply in small pieces."""
    replies = replies or {}

    def chat(messages, json_mode=False, temperature=None, on_token=None, stats=None, think=None):
        system = messages[0]["content"]
        kind = {wc.REFLECT_SYSTEM: "reflect", wc.EXECUTE_SYSTEM: "execute",
                wc.SITUATION_SYSTEM: "situation", wc.LESSON_SYSTEM: "lesson"}.get(system, "answer")
        chat.seen.append((kind, think))
        text = replies.get(kind, '{"actions": []}' if kind == "execute" else f"{kind} reply text")
        if on_token:
            for i in range(0, len(text), 4):
                on_token(text[i:i + 4], "content")
        if stats is not None:
            stats["output_tokens"] = len(text)
        return text
    chat.seen = []
    return wc.FunctionModel(chat, lambda texts: bow_embed(texts), name="streamer", embed_name="bow"), chat


class EventTests(unittest.TestCase):
    def test_a_task_already_held_reports_the_first_look_and_stops(self):
        events = []
        model = ScriptedModel()
        agent = wc.Agent(AXIOM, model, path=":memory:", on_event=events.append)
        seed(agent.core, DEBT)
        model.queue("answer", "Pay the 24% card first.")
        model.queue("lesson", {"principle": DEBT, "corrects": False})
        agent.ask("Car loan 6% or card 24%?")
        kinds = [(e["type"], e.get("purpose")) for e in events if e["type"] not in ("embed", "model.token")]
        self.assertEqual(kinds[kinds.index(("synthesis.start", None)):], [
            ("synthesis.start", None), ("model.start", "lesson"), ("model.done", "lesson"),
            ("synthesis.first_look", None), ("synthesis.done", None),
        ])
        look = next(e for e in events if e["type"] == "synthesis.first_look")
        self.assertTrue(look["held"])
        self.assertEqual((look["lesson"], look["nearest"]), (DEBT["name"], DEBT["name"]))
        done = events[-1]
        self.assertTrue(done["report"]["first_look"]["held"])
        self.assertEqual([p["name"] for p in done["report"]["covered"]], [DEBT["name"]])

    def test_one_task_reports_every_step_in_order(self):
        events = []
        model = ScriptedModel()
        model.queue("answer", "Pay the 24% card first.")
        model.queue("execute", {"actions": [{"type": "add", "principle": DEBT}]})
        agent = wc.Agent(AXIOM, model, path=":memory:", on_event=events.append)
        agent.ask("Car loan 6% or card 24%?")
        kinds = [(e["type"], e.get("purpose")) for e in events if e["type"] != "embed"]
        self.assertEqual(kinds, [
            ("recall.start", None), ("recall.done", None),
            ("model.start", "answer"), ("model.done", "answer"),
            ("synthesis.start", None), ("synthesis.first_look", None), ("synthesis.candidates", None),
            ("model.start", "reflect"), ("model.done", "reflect"), ("synthesis.memo", None),
            ("model.start", "execute"), ("model.done", "execute"), ("synthesis.actions", None),
            ("calibrated", None),                       # the embedding model measured at the first principle
            ("principles.changed", None), ("synthesis.done", None),
        ])
        look = next(e for e in events if e["type"] == "synthesis.first_look")
        self.assertEqual((look["ran"], look["why"]), (False, "skipped: no principles are held yet"))
        start = next(e for e in events if e["type"] == "model.start" and e["purpose"] == "answer")
        self.assertIn(f"CORE AXIOM\n{AXIOM}", start["messages"][0]["content"])
        self.assertEqual(start["messages"][-1], {"role": "user", "content": "Car loan 6% or card 24%?"})
        done = next(e for e in events if e["type"] == "model.done" and e["purpose"] == "answer")
        self.assertEqual(done["reply"], "Pay the 24% card first.")
        self.assertIsNone(done["error"])
        self.assertEqual(start["call"], done["call"])
        report = events[-1]["report"]
        self.assertEqual(report["added"][0]["name"], DEBT["name"])
        json.dumps(events)                      # everything a watcher gets can be sent on as JSON

    def test_recall_reports_the_principles_and_their_scores(self):
        events = []
        core = wc.WisdomCore(AXIOM, ScriptedModel(), path=":memory:", on_event=events.append)
        seed(core, DEBT, principle("Add the figures twice before reporting a total"))
        events.clear()
        core.recall("Which debt should I repay first?", expand=False)
        done = next(e for e in events if e["type"] == "recall.done")
        self.assertEqual(done["held"], 2)
        self.assertEqual(done["principles"][0]["name"], DEBT["name"])
        self.assertGreater(done["principles"][0]["score"], done["principles"][1]["score"])
        self.assertEqual(done["situation"], "")

    def test_the_situation_description_is_reported(self):
        events = []
        model = ScriptedModel()
        core = wc.WisdomCore(AXIOM, model, path=":memory:", context=1, on_event=events.append)
        seed(core, DEBT, principle("Add the figures twice before reporting a total"))
        model.queue("situation", "Several debts at different interest rates.")
        core.recall("help me")
        done = next(e for e in events if e["type"] == "recall.done")
        self.assertEqual(done["situation"], "Several debts at different interest rates.")
        purposes = [e["purpose"] for e in events if e["type"] == "model.start"]
        self.assertEqual(purposes, ["situation"])

    def test_a_failed_model_call_is_reported_then_raised(self):
        events = []
        model = ScriptedModel()
        model.queue("answer", wc.ModelError("model is down"))
        agent = wc.Agent(AXIOM, model, path=":memory:", on_event=events.append)
        with self.assertRaises(wc.ModelError):
            agent.ask("hello")
        done = next(e for e in events if e["type"] == "model.done")
        self.assertIn("model is down", done["error"])
        self.assertIsNone(done["reply"])

    def test_a_broken_watcher_does_not_break_the_agent(self):
        def watcher(event):
            raise ValueError("bug in the watcher")
        agent = wc.Agent(AXIOM, ScriptedModel(), path=":memory:", on_event=watcher)
        with self.assertLogs("wisdom_core", level="WARNING"):
            self.assertEqual(agent.ask("hello"), "OK.")

    def test_a_watcher_can_stop_a_run(self):
        model, _ = streaming_function_model()

        def watcher(event):
            if event["type"] == "model.token":
                raise Stop()
        agent = wc.Agent(AXIOM, model, path=":memory:", on_event=watcher)
        with self.assertRaises(Stop):
            agent.ask("hello")
        self.assertEqual(agent.messages, [])         # nothing half-finished is kept

    def test_the_watcher_can_be_attached_later(self):
        events = []
        agent = wc.Agent(AXIOM, ScriptedModel(), path=":memory:")
        watcher = events.append
        agent.on_event = watcher
        self.assertIs(agent.core.on_event, watcher)
        self.assertIs(agent.on_event, watcher)
        agent.ask("hello")
        self.assertTrue(any(e["type"] == "synthesis.done" for e in events))

    def test_reports_serialise(self):
        a = wc.Principle(id="a", name="A")
        report = wc.SynthesisReport(task="t", added=[a], merged=[([a], a)], removed=[(a, "why")], covered=[a])
        data = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(data["merged"][0]["result"]["name"], "A")
        self.assertEqual(data["removed"][0]["reason"], "why")
        self.assertTrue(data["changed"])


class StreamingTests(unittest.TestCase):
    def test_tokens_arrive_in_order_and_add_up_to_the_reply(self):
        events = []
        model, _ = streaming_function_model({"answer": "The card at 24% costs the most, so pay it first."})
        agent = wc.Agent(AXIOM, model, path=":memory:", on_event=events.append)
        reply = agent.ask("Which first?")
        answer_call = next(e["call"] for e in events if e["type"] == "model.start" and e["purpose"] == "answer")
        tokens = [e["text"] for e in events if e["type"] == "model.token" and e["call"] == answer_call]
        self.assertGreater(len(tokens), 5)
        self.assertEqual("".join(tokens), reply)
        stats = next(e["stats"] for e in events if e["type"] == "model.done" and e["call"] == answer_call)
        self.assertEqual(stats["output_tokens"], len(reply))

    def test_a_function_that_cannot_stream_still_shows_its_reply(self):
        events = []
        model = wc.FunctionModel(lambda messages: "whole reply", lambda texts: bow_embed(texts))
        agent = wc.Agent(AXIOM, model, path=":memory:", on_event=events.append, learn=False)
        agent.ask("hi")
        self.assertEqual([e["text"] for e in events if e["type"] == "model.token"], ["whole reply"])

    def test_no_watcher_means_no_streaming(self):
        given = []

        def chat(messages, on_token=None):
            given.append(on_token)
            return "reply"
        agent = wc.Agent(AXIOM, wc.FunctionModel(chat, lambda texts: bow_embed(texts)), path=":memory:",
                         learn=False)
        agent.ask("hi")
        self.assertEqual(given, [None])

    def test_internal_calls_switch_thinking_off_and_the_answer_does_not(self):
        model, chat = streaming_function_model()
        agent = wc.Agent(AXIOM, model, path=":memory:", context=1)
        seed(agent.core, DEBT, principle("Add the figures twice before reporting a total"))
        agent.ask("hello")
        self.assertEqual(dict(chat.seen), {"situation": False, "answer": None, "lesson": False, "reflect": False,
                                           "execute": False})

    def test_ollama_streams_content_and_thinking(self):
        brain = ScriptedModel()
        brain.queue("answer", "<think>weighing the rates</think>Pay the card first.")
        server, _ = ollama_server(brain, split_thinking=True)
        self.addCleanup(server.close)
        got = []
        stats = {}
        o = wc.Ollama("llama3.2", "nomic-embed-text", host=server.url)
        reply = o.chat(MSGS, on_token=lambda text, kind: got.append((kind, text)), stats=stats)
        self.assertEqual(reply, "Pay the card first.")
        self.assertEqual("".join(t for k, t in got if k == "thinking"), "weighing the rates")
        self.assertEqual("".join(t for k, t in got if k == "content"), "Pay the card first.")
        self.assertEqual(stats["thinking"], "weighing the rates")
        self.assertEqual(stats["done_reason"], "stop")
        self.assertGreater(stats["output_tokens"], 0)
        self.assertEqual(stats["num_ctx"], 8192)
        self.assertIs(server.bodies("/api/chat")[0]["stream"], True)

    def test_ollama_without_streaming_still_fills_stats(self):
        brain = ScriptedModel()
        brain.queue("answer", "<think>hmm</think>Answer.")
        server, _ = ollama_server(brain, split_thinking=True)
        self.addCleanup(server.close)
        stats = {}
        reply = wc.Ollama("llama3.2", "nomic-embed-text", host=server.url).chat(MSGS, stats=stats)
        self.assertEqual((reply, stats["thinking"], stats["done_reason"]), ("Answer.", "hmm", "stop"))
        self.assertIs(server.bodies("/api/chat")[0]["stream"], False)

    def test_openai_streams_content_and_reasoning(self):
        brain = ScriptedModel()
        brain.queue("answer", "<think>because</think>Final words here.")
        server, _ = openai_server(brain)
        self.addCleanup(server.close)
        got = []
        stats = {}
        o = wc.OpenAICompatible("m", "e", base_url=server.url + "/v1")
        reply = o.chat(MSGS, on_token=lambda text, kind: got.append((kind, text)), stats=stats)
        self.assertEqual(reply, "Final words here.")
        self.assertEqual("".join(t for k, t in got if k == "thinking"), "because")
        self.assertEqual((stats["done_reason"], stats["output_tokens"]), ("stop", 7))
        self.assertIs(server.bodies("/v1/chat/completions")[0]["stream"], True)

    def test_openai_non_streaming_stats(self):
        server, brain = openai_server()
        self.addCleanup(server.close)
        stats = {}
        wc.OpenAICompatible("m", base_url=server.url + "/v1").chat(MSGS, stats=stats)
        self.assertEqual((stats["prompt_tokens"], stats["output_tokens"], stats["done_reason"]), (11, 7, "stop"))

    def test_a_stream_that_reports_an_error(self):
        server, _ = ollama_server()
        self.addCleanup(server.close)
        from tests.httpfake import Stream
        server.routes[("POST", "/api/chat")] = lambda body, headers: (200, Stream([
            {"message": {"content": "partial"}, "done": False}, {"error": "out of memory"}]))
        with self.assertRaises(wc.ModelError) as err:
            wc.Ollama("llama3.2", "e", host=server.url).chat(MSGS, on_token=lambda t, k: None)
        self.assertIn("out of memory", str(err.exception))

    def test_a_stream_errors_before_it_starts(self):
        server, _ = ollama_server()
        self.addCleanup(server.close)
        with self.assertRaises(wc.ModelError) as err:
            wc.Ollama("not-there", "e", host=server.url).chat(MSGS, on_token=lambda t, k: None)
        self.assertIn("ollama pull not-there", str(err.exception))

    def test_stopping_mid_stream_closes_the_connection(self):
        brain = ScriptedModel()
        brain.queue("answer", "word " * 400)
        server, _ = ollama_server(brain, delay=0.002)
        self.addCleanup(server.close)
        seen = []

        def on_token(text, kind):
            seen.append(text)
            if len(seen) == 5:
                raise Stop()
        o = wc.Ollama("llama3.2", "e", host=server.url)
        started = time.monotonic()
        with self.assertRaises(Stop):
            o.chat(MSGS, on_token=on_token)
        self.assertLess(time.monotonic() - started, 5)
        deadline = time.monotonic() + 3
        while server.disconnects == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(server.disconnects, 1)
        self.assertLess(server.streamed, 300)


class ThinkSwitchTests(unittest.TestCase):
    def test_think_false_is_sent_when_asked(self):
        server, _ = ollama_server()
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "e", host=server.url)
        o.chat(MSGS, think=False)
        o.chat(MSGS)
        bodies = server.bodies("/api/chat")
        self.assertIs(bodies[0]["think"], False)
        self.assertNotIn("think", bodies[1])

    def test_a_server_that_refuses_the_switch_is_remembered(self):
        server, _ = ollama_server(reject_think=True)
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "e", host=server.url)
        self.assertEqual(o.chat(MSGS, think=False), "OK.")
        o.chat(MSGS, think=False)
        bodies = server.bodies("/api/chat")
        self.assertEqual(["think" in b for b in bodies], [True, False, False])

    def test_a_model_that_only_thinks_is_asked_again_with_thinking_off(self):
        server, _ = ollama_server(starve=True)
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "e", host=server.url)
        stats = {}
        self.assertEqual(o.chat(MSGS, stats=stats), "OK.")
        self.assertIn("asked again with thinking off", stats["note"])
        o.chat(MSGS)                                    # remembered: thinking is off from the start
        bodies = server.bodies("/api/chat")
        self.assertEqual([b.get("think") for b in bodies], [None, False, False])

    def test_streaming_a_model_that_only_thinks(self):
        server, _ = ollama_server(starve=True)
        self.addCleanup(server.close)
        got = []
        reply = wc.Ollama("llama3.2", "e", host=server.url).chat(MSGS, on_token=lambda t, k: got.append(k))
        self.assertEqual(reply, "OK.")
        self.assertIn("thinking", got)
        self.assertEqual(got[-1], "content")


class ContextWindowTests(unittest.TestCase):
    def window(self, **kw) -> int:
        server, _ = ollama_server(**kw)
        self.addCleanup(server.close)
        return wc.Ollama("llama3.2", "e", host=server.url).context_window

    def test_read_from_the_model_and_capped(self):
        self.assertEqual(self.window(context_length=131072), wc.DEFAULT_NUM_CTX)
        self.assertEqual(self.window(context_length=4096), 4096)
        self.assertEqual(self.window(), wc.FALLBACK_NUM_CTX)

    def test_given_explicitly(self):
        server, _ = ollama_server(context_length=4096)
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "e", host=server.url, num_ctx=32768)
        o.chat(MSGS)
        self.assertEqual(server.bodies("/api/chat")[0]["options"]["num_ctx"], 32768)
        self.assertNotIn("/api/show", server.paths())

    def test_asked_once_per_model_and_sent_with_every_call(self):
        server, _ = ollama_server(context_length=12000)
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "e", host=server.url)
        o.chat(MSGS)
        o.chat(MSGS, on_token=lambda t, k: None)
        self.assertEqual([b["options"]["num_ctx"] for b in server.bodies("/api/chat")], [12000, 12000])
        self.assertEqual(server.paths().count("/api/show"), 1)

    def test_switching_models(self):
        server, _ = ollama_server(context_length=12000)
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "nomic-embed-text", host=server.url)
        o.model = "nomic-embed-text"
        self.assertEqual(o.model, "nomic-embed-text")
        o.embed_model = "llama3.2"
        self.assertEqual(o.embed_name, "ollama:llama3.2:latest")


class FitHistoryTests(unittest.TestCase):
    def convo(self, n: int, size: int = 400) -> list[dict[str, str]]:
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * size} for i in range(n)]

    def test_no_window_keeps_everything(self):
        kept, dropped = wc.fit_history("sys", self.convo(10), "hi", None)
        self.assertEqual((len(kept), dropped), (10, 0))

    def test_oldest_go_first_and_the_rest_fit(self):
        system = "S" * 2000
        kept, dropped = wc.fit_history(system, self.convo(12), "question", 1500)
        self.assertGreater(dropped, 0)
        self.assertEqual(kept[-1]["content"][:3], "m11")
        self.assertEqual(kept[0]["role"], "user")
        used = len(system) + len("question") + sum(len(m["content"]) + 16 for m in kept) + 32
        self.assertLessEqual(used, (1500 - min(wc.REPLY_ROOM, 1500 // 4)) * wc.CHARS_PER_TOKEN)

    def test_the_system_prompt_is_never_trimmed(self):
        kept, dropped = wc.fit_history("S" * 100000, self.convo(4), "question", 1000)
        self.assertEqual((kept, dropped), ([], 4))

    def test_the_agent_trims_to_the_models_window_and_says_so(self):
        events = []

        class Small(ScriptedModel):
            context_window = 900
        model = Small()
        model.defaults["answer"] = "A fairly long answer. " * 40
        agent = wc.Agent(AXIOM, model, path=":memory:", on_event=events.append, learn=False)
        for q in ("one", "two", "three", "four"):
            agent.ask(q)
        trimmed = [e for e in events if e["type"] == "history.trimmed"]
        self.assertTrue(trimmed)
        last = model.prompts("answer")[-1]
        self.assertEqual(last[0]["role"], "system")
        self.assertIn(AXIOM, last[0]["content"])
        self.assertEqual(last[-1]["content"], "four")
        self.assertLess(len(last), 1 + 2 * 3 + 1)

    def test_an_unreachable_model_has_no_window(self):
        class Broken:
            @property
            def context_window(self):
                raise wc.ModelError("down")
        self.assertIsNone(wc.context_window_of(Broken()))
        self.assertIsNone(wc.context_window_of(object()))


class ModelSwapTests(unittest.TestCase):
    def test_a_new_embedding_model_on_a_live_core_re_embeds(self):
        events = []
        core = wc.WisdomCore(AXIOM, ScriptedModel(embed_name="space-a"), path=":memory:", on_event=events.append)
        seed(core, DEBT)
        core.recall("debts", expand=False)
        new = ScriptedModel(embed_name="space-b", dims=64)
        core.model = new
        got = core.recall("which debt first", expand=False)
        self.assertEqual(got[0].name, DEBT["name"])
        self.assertTrue(new.embed_calls)
        self.assertTrue(any(e["type"] == "reindex" and e["space"] == "space-b" for e in events))

    def test_a_new_embedding_model_of_the_same_size_re_embeds_too(self):
        # Same vector size, different space: only the model's name tells the old vectors apart.
        class Reversed(ScriptedModel):
            def embed(self, texts):
                return [list(reversed(v)) for v in super().embed(texts)]

        events = []
        core = wc.WisdomCore(AXIOM, ScriptedModel(embed_name="space-a"), path=":memory:", on_event=events.append)
        seed(core, DEBT)
        core.recall("debts", expand=False)
        core.model = Reversed(embed_name="space-b")
        got = core.recall("which debt should I repay first", expand=False)
        self.assertTrue(any(e["type"] == "reindex" and e["space"] == "space-b" for e in events))
        self.assertEqual(got[0].name, DEBT["name"])
        self.assertGreater(got[0].score, 0.2)

    def test_a_new_chat_model_is_asked_what_it_accepts(self):
        core = wc.WisdomCore(AXIOM, ScriptedModel(), path=":memory:")
        self.assertEqual(core._model_kwargs(), {"messages", "json_mode", "temperature"})
        core.model, _ = streaming_function_model()
        self.assertLessEqual({"on_token", "stats", "think"}, core._model_kwargs())


class ConcurrencyTests(unittest.TestCase):
    def test_two_threads_share_one_ollama_backend(self):
        brain = ScriptedModel()
        brain.defaults["answer"] = "same answer for everyone"
        server, _ = ollama_server(brain, delay=0.001)
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "nomic-embed-text", host=server.url)
        results = []

        def run():
            got = []
            results.append((o.chat(MSGS, on_token=lambda t, k: got.append(t)), "".join(got)))
        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(results, [("same answer for everyone", "same answer for everyone")] * 4)


if __name__ == "__main__":
    unittest.main()
