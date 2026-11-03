"""The Ollama and OpenAI-compatible backends, against a fake server."""

from __future__ import annotations

import io
import os
import socket
import unittest
import urllib.error
from unittest import mock

import wisdom_core as wc
from tests.fakes import ScriptedModel, close_stores, principle
from tests.httpfake import ollama_server, openai_server

tearDownModule = close_stores          # close every database this module's tests opened


AXIOM = "Help people make sound money decisions."
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class OllamaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.brain = ollama_server()
        self.addCleanup(self.server.close)

    def ollama(self, **kw) -> wc.Ollama:
        kw.setdefault("host", self.server.url)
        return wc.Ollama(**kw)

    def test_chat_request(self):
        self.brain.queue("answer", "hello there")
        out = self.ollama(model="llama3.2", embed_model="nomic-embed-text").chat(MSGS, json_mode=True,
                                                                                 temperature=0.0)
        self.assertEqual(out, "hello there")
        body = self.server.bodies("/api/chat")[0]
        self.assertEqual(body["model"], "llama3.2")
        self.assertEqual(body["messages"], MSGS)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["options"], {"num_ctx": 8192, "temperature": 0.0})
        self.assertEqual(body["format"], "json")
        self.assertNotIn("keep_alive", body)

    def test_chat_without_hints(self):
        self.ollama(model="llama3.2", num_ctx=4096, keep_alive="10m").chat(MSGS)
        body = self.server.bodies("/api/chat")[0]
        self.assertEqual(body["options"], {"num_ctx": 4096})
        self.assertNotIn("format", body)
        self.assertEqual(body["keep_alive"], "10m")

    def test_a_core_on_ollama_writes_each_text_the_way_the_embedding_model_wants(self):
        core = wc.WisdomCore(AXIOM, self.ollama(model="llama3.2", embed_model="nomic-embed-text"), path=":memory:")
        core._apply([{"type": "add", "principle": principle("Ask when the money will be needed",
                                                             use_case="someone asks where to keep savings")}],
                    {}, wc.SynthesisReport(task="seed"), "seed")
        core.recall("Where should I keep my savings?", expand=False)
        calibration, fingerprint, *sent = [body["input"] for body in self.server.bodies("/api/embed")]
        self.assertEqual(len(calibration), 75)                          # the model is measured first...
        self.assertTrue(calibration[0].startswith("clustering: "))
        self.assertEqual(fingerprint, ["clustering: " + wc.FINGERPRINT_TEXT])   # ...and its fingerprint kept
        self.assertEqual(sent[0], ["search_document: Ask when the money will be needed. Someone asks where to keep "
                                   "savings.", "clustering: Ask when the money will be needed."])
        self.assertEqual(sent[1], ["search_query: Where should I keep my savings?"])
        core.close()

    def test_embed_in_one_batch(self):
        vecs = self.ollama(model="llama3.2", embed_model="nomic-embed-text").embed(["a b c", "d e f"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(self.server.bodies("/api/embed")[0]["input"], ["a b c", "d e f"])
        self.assertEqual(self.ollama().embed([]), [])

    def test_old_ollama_without_api_embed(self):
        server, _ = ollama_server(new_embed_api=False)
        self.addCleanup(server.close)
        o = wc.Ollama("llama3.2", "nomic-embed-text", host=server.url)
        self.assertEqual(len(o.embed(["one", "two"])), 2)
        self.assertEqual(len(o.embed(["three"])), 1)
        self.assertEqual(server.paths().count("/api/embed"), 1)       # tried once, then remembered
        self.assertEqual(server.paths().count("/api/embeddings"), 3)

    def test_a_missing_model_says_how_to_get_it(self):
        o = self.ollama(model="mistral", embed_model="bge-m3")
        with self.assertRaises(wc.ModelError) as chat_err:
            o.chat(MSGS)
        self.assertIn("ollama pull mistral", str(chat_err.exception))
        with self.assertRaises(wc.ModelError) as embed_err:
            o.embed(["x"])
        self.assertIn("ollama pull bge-m3", str(embed_err.exception))
        self.assertNotIn("/api/embeddings", self.server.paths())     # a missing model is not an old server

    def test_models_are_chosen_automatically(self):
        o = self.ollama()
        self.assertEqual(o.model, "llama3.2:latest")
        self.assertEqual(o.embed_model, "nomic-embed-text:latest")
        self.assertEqual(o.name, "ollama:llama3.2:latest")
        self.assertEqual(o.embed_name, "ollama:nomic-embed-text:latest")

    def test_models_are_chosen_without_capabilities_info(self):
        models = [{"name": "mxbai-embed-large:latest", "details": {"family": "bert", "families": ["bert"]}},
                  {"name": "qwen2.5:7b", "details": {"family": "qwen2", "families": ["qwen2"]}},
                  {"name": "all-minilm:latest", "details": {}}]
        server, _ = ollama_server(models=models, capabilities=False)
        self.addCleanup(server.close)
        o = wc.Ollama(host=server.url)
        self.assertEqual((o.model, o.embed_model), ("qwen2.5:7b", "mxbai-embed-large:latest"))

    def test_an_explicit_choice_is_kept(self):
        o = self.ollama(model="mine")
        self.assertEqual(o.embed_model, "nomic-embed-text:latest")
        self.assertEqual(o.model, "mine")

    def test_no_embedding_model(self):
        server, _ = ollama_server(models=[{"name": "llama3.2:latest", "details": {}}])
        self.addCleanup(server.close)
        with self.assertRaises(wc.ModelError) as err:
            _ = wc.Ollama(host=server.url).embed_model
        self.assertIn("ollama pull nomic-embed-text", str(err.exception))

    def test_no_chat_model(self):
        server, _ = ollama_server(models=[{"name": "nomic-embed-text:latest", "details": {}}])
        self.addCleanup(server.close)
        with self.assertRaises(wc.ModelError) as err:
            _ = wc.Ollama(host=server.url).model
        self.assertIn("no chat model", str(err.exception))

    def test_ollama_not_running(self):
        o = wc.Ollama("llama3.2", "nomic-embed-text", host=f"127.0.0.1:{closed_port()}")
        with self.assertRaises(wc.ModelError) as err:
            o.chat(MSGS)
        self.assertIn("Is Ollama running?", str(err.exception))
        self.assertIsNone(err.exception.status)

    def test_server_error(self):
        self.server.routes[("POST", "/api/chat")] = lambda body, headers: (500, {"error": "out of memory"})
        with self.assertRaises(wc.ModelError) as err:
            self.ollama(model="llama3.2").chat(MSGS)
        self.assertEqual(err.exception.status, 500)
        self.assertIn("out of memory", str(err.exception))

    def test_odd_replies(self):
        self.server.routes[("POST", "/api/chat")] = lambda body, headers: (200, "not json at all")
        with self.assertRaises(wc.ModelError):
            self.ollama(model="llama3.2").chat(MSGS)
        self.server.routes[("POST", "/api/chat")] = lambda body, headers: (200, {"message": {}})
        with self.assertRaises(wc.ModelError):
            self.ollama(model="llama3.2").chat(MSGS)
        self.server.routes[("POST", "/api/embed")] = lambda body, headers: (200, {"embeddings": [[1.0]]})
        with self.assertRaises(wc.ModelError):
            self.ollama(model="llama3.2", embed_model="nomic-embed-text").embed(["a", "b"])

    def test_ollama_host_from_the_environment(self):
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": self.server.url}):
            self.assertEqual(wc.Ollama().model, "llama3.2:latest")


class OpenAITests(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.brain = openai_server()
        self.addCleanup(self.server.close)
        self.base = self.server.url + "/v1"

    def test_chat_request_and_key(self):
        self.brain.queue("answer", "hi!")
        o = wc.OpenAICompatible("gpt-x", "embed-x", base_url=self.base, api_key="sk-test")
        self.assertEqual(o.chat(MSGS, json_mode=True, temperature=0.2), "hi!")
        method, path, headers, body = self.server.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["authorization"], "Bearer sk-test")
        self.assertEqual(body, {"model": "gpt-x", "messages": MSGS, "response_format": {"type": "json_object"},
                                "temperature": 0.2})

    def test_no_key_no_header(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            wc.OpenAICompatible("m", base_url=self.base).chat(MSGS)
        self.assertNotIn("authorization", self.server.requests[0][2])

    def test_environment_defaults(self):
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": self.base, "OPENAI_API_KEY": "sk-env"}):
            o = wc.OpenAICompatible("m")
            o.chat(MSGS)
        self.assertEqual(self.server.requests[0][2]["authorization"], "Bearer sk-env")

    def test_servers_that_reject_json_mode(self):
        real = self.server.routes[("POST", "/v1/chat/completions")]

        def picky(body, headers):
            if "response_format" in body:
                return 400, {"error": {"message": "response_format is not supported"}}
            return real(body, headers)
        self.server.routes[("POST", "/v1/chat/completions")] = picky
        o = wc.OpenAICompatible("m", base_url=self.base)
        o.chat(MSGS, json_mode=True, temperature=0.0)
        o.chat(MSGS, json_mode=True)
        bodies = self.server.bodies("/v1/chat/completions")
        self.assertEqual(len(bodies), 3)                       # one refusal, then two plain calls
        self.assertNotIn("response_format", bodies[2])
        self.assertEqual(bodies[1]["temperature"], 0.0)

    def test_models_that_reject_temperature(self):
        real = self.server.routes[("POST", "/v1/chat/completions")]

        def picky(body, headers):
            if "temperature" in body:
                return 400, {"error": {"message": "Unsupported value: 'temperature'"}}
            return real(body, headers)
        self.server.routes[("POST", "/v1/chat/completions")] = picky
        o = wc.OpenAICompatible("m", base_url=self.base)
        o.chat(MSGS, json_mode=True, temperature=0.0)
        last = self.server.bodies("/v1/chat/completions")[-1]
        self.assertNotIn("temperature", last)
        self.assertIn("response_format", last)

    def test_other_bad_requests_are_raised(self):
        self.server.routes[("POST", "/v1/chat/completions")] = lambda b, h: (400, {"error": {"message": "bad"}})
        with self.assertRaises(wc.ModelError) as err:
            wc.OpenAICompatible("m", base_url=self.base).chat(MSGS)
        self.assertEqual(err.exception.status, 400)
        self.server.routes[("POST", "/v1/chat/completions")] = lambda b, h: (401, {"error": {"message": "no key"}})
        with self.assertRaises(wc.ModelError):
            wc.OpenAICompatible("m", base_url=self.base).chat(MSGS, json_mode=True, temperature=0.1)

    def test_embeddings_come_back_in_order(self):
        o = wc.OpenAICompatible("m", "e", base_url=self.base)
        texts = ["alpha words here", "bravo other words", "charlie third"]
        self.assertEqual(o.embed(texts), self.brain.embed(texts))

    def test_embeddings_need_a_model(self):
        with self.assertRaises(wc.ModelError):
            wc.OpenAICompatible("m", base_url=self.base).embed(["x"])
        self.assertEqual(wc.OpenAICompatible("m", base_url=self.base).embed([]), [])

    def test_odd_replies(self):
        self.server.routes[("POST", "/v1/chat/completions")] = lambda b, h: (
            200, {"choices": [{"message": {"content": None, "refusal": "no"}}]})
        self.assertEqual(wc.OpenAICompatible("m", base_url=self.base).chat(MSGS), "")
        self.server.routes[("POST", "/v1/chat/completions")] = lambda b, h: (200, {"choices": []})
        with self.assertRaises(wc.ModelError):
            wc.OpenAICompatible("m", base_url=self.base).chat(MSGS)
        self.server.routes[("POST", "/v1/embeddings")] = lambda b, h: (200, {"data": [{"index": 0}]})
        with self.assertRaises(wc.ModelError):
            wc.OpenAICompatible("m", "e", base_url=self.base).embed(["x"])

    def test_model_name_required(self):
        with self.assertRaises(ValueError):
            wc.OpenAICompatible("")


class FunctionModelTests(unittest.TestCase):
    def test_hints_only_reach_functions_that_take_them(self):
        seen = {}

        def plain(messages):
            seen["plain"] = messages
            return "p"

        def named(messages, json_mode=False, temperature=None):
            seen["named"] = (json_mode, temperature)
            return "n"

        def loose(messages, **kw):
            seen["loose"] = kw
            return "l"

        def one_hint(messages, *, temperature=None):
            seen["one"] = temperature
            return None
        embed = lambda texts: [[1.0] for _ in texts]  # noqa: E731
        self.assertEqual(wc.FunctionModel(plain, embed).chat(MSGS, json_mode=True, temperature=0.1), "p")
        self.assertEqual(wc.FunctionModel(named, embed).chat(MSGS, json_mode=True, temperature=0.1), "n")
        self.assertEqual(wc.FunctionModel(loose, embed).chat(MSGS, json_mode=True), "l")
        self.assertEqual(wc.FunctionModel(one_hint, embed).chat(MSGS, temperature=0.3), "")
        self.assertEqual(seen["named"], (True, 0.1))
        self.assertEqual(seen["loose"], {"json_mode": True, "temperature": None})
        self.assertEqual(seen["one"], 0.3)
        self.assertEqual(wc.FunctionModel(plain, embed).embed(["a", "b"]), [[1.0], [1.0]])
        self.assertEqual(wc.FunctionModel(plain, embed, name="x").embed_name, "custom:x")


class EndToEndTests(unittest.TestCase):
    """A whole agent talking to a (fake) server over HTTP."""

    def run_agent(self, model: object, brain: ScriptedModel) -> wc.Agent:
        brain.queue("answer", "Pay the 24% card first.", "List them by rate; repay the costliest first.")
        brain.queue("execute", {"actions": [{"type": "add", "principle": principle(
            "Repay the debt with the highest interest rate first",
            "It saves the most money", "Debt repayment", "someone has several debts",
            ["interest", "debt"])}]})
        agent = wc.Agent(AXIOM, model, path=":memory:")
        self.addCleanup(agent.close)
        self.assertEqual(agent.ask("Car loan at 6% or card at 24%, which first?"), "Pay the 24% card first.")
        self.assertEqual(len(agent.last_report.added), 1)
        agent.ask("I also have a student loan. Which debt should I repay first?")
        self.assertEqual([p.name for p in agent.last_principles],
                         ["Repay the debt with the highest interest rate first"])
        self.assertIn("Repay the debt with the highest interest rate first", brain.prompts("answer")[1][0]["content"])
        return agent

    def test_over_ollama(self):
        server, brain = ollama_server()
        self.addCleanup(server.close)
        self.run_agent(wc.Ollama(host=server.url), brain)
        paths = server.paths()
        self.assertEqual(paths[0], "/api/tags")
        self.assertIn("/api/embed", paths)
        formats = [b.get("format") for b in server.bodies("/api/chat")]
        self.assertIn("json", formats)

    def test_over_openai_compatible(self):
        server, brain = openai_server()
        self.addCleanup(server.close)
        self.run_agent(wc.OpenAICompatible("m", "e", base_url=server.url + "/v1"), brain)
        self.assertIn("/v1/embeddings", server.paths())

    def test_learning_survives_the_server_going_away(self):
        server, brain = ollama_server()
        agent = wc.Agent(AXIOM, wc.Ollama("llama3.2", "nomic-embed-text", host=server.url), path=":memory:")
        self.addCleanup(agent.close)

        def answer_then_die(messages):
            server.routes.pop(("POST", "/api/chat"))
            return "answered"
        brain.queue("answer", answer_then_die)
        self.assertEqual(agent.ask("hello"), "answered")
        self.assertIn("404", agent.last_report.error)
        server.close()


class ErrorReplyTests(unittest.TestCase):
    """A server's error reply is read for its message and then closed, streamed or not."""

    def test_every_error_reply_is_closed(self):
        server, _ = ollama_server()
        self.addCleanup(server.close)
        server.routes[("POST", "/api/chat")] = lambda body, headers: (500, {"error": "out of memory"})
        closed: list[int] = []
        real_close = urllib.error.HTTPError.close

        def close(exc):
            closed.append(exc.code)
            real_close(exc)
        ollama = wc.Ollama("llama3.2", "nomic-embed-text", host=server.url)
        with mock.patch.object(urllib.error.HTTPError, "close", close):
            for streamed in (False, True):
                with self.subTest(streamed=streamed), self.assertRaises(wc.ModelError) as err:
                    if streamed:
                        ollama.chat(MSGS, on_token=lambda text, kind: None)
                    else:
                        ollama.chat(MSGS)
                self.assertEqual(err.exception.status, 500)
                self.assertIn("out of memory", str(err.exception))
        self.assertEqual(closed, [500, 500])

    def test_an_error_body_that_cannot_be_read_still_gives_the_status(self):
        class Broken(io.BytesIO):
            def read(self, *args):
                raise ConnectionResetError("the connection dropped")
        exc = urllib.error.HTTPError("http://x/api/chat", 502, "Bad Gateway", {}, Broken())   # type: ignore[arg-type]
        self.assertEqual(wc._error_body(exc), "")
        self.assertTrue(exc.fp.closed)


if __name__ == "__main__":
    unittest.main()
