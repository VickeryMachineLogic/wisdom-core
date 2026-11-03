"""Chat and embedding services beyond Ollama: Anthropic, OpenAI-compatible listing, and Combined."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import wisdom_core as wc
from tests.fakes import ScriptedModel, close_stores, principle
from tests.httpfake import anthropic_server, ollama_server, openai_server

tearDownModule = close_stores          # close every database this module's tests opened


AXIOM = "Help people make sound money decisions."
KEY = "sk-ant-test"
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
DEBT = principle("List debts by interest rate before suggesting which to repay first",
                 "Paying the costliest debt first saves the most money", "Debt repayment questions",
                 "someone has several debts", ["interest rate", "loans"])


class Stop(BaseException):
    """What a caller raises from on_token to stop a reply."""


class AnthropicTests(unittest.TestCase):
    def start(self, **kw):
        server, brain = anthropic_server(**kw)
        self.addCleanup(server.close)
        return server, brain, server.url

    def claude(self, url, model="claude-mid-4", **kw):
        return wc.Anthropic(model, api_key=KEY, base_url=url, **kw)

    def test_the_request_it_sends(self):
        server, brain, url = self.start()
        brain.queue("answer", "Pay the card first.")
        history = [{"role": "system", "content": "You serve the axiom."}, {"role": "user", "content": "first"},
                   {"role": "assistant", "content": "an earlier answer"}, {"role": "user", "content": "second"}]
        self.assertEqual(self.claude(url).chat(history, json_mode=True, temperature=0.2), "Pay the card first.")
        method, path, headers, body = server.requests[0]
        self.assertEqual((method, path), ("POST", "/v1/messages"))
        self.assertEqual((headers["x-api-key"], headers["anthropic-version"]), (KEY, "2023-06-01"))
        self.assertEqual(body, {"model": "claude-mid-4", "max_tokens": wc.ANTHROPIC_MAX_OUTPUT,
                                "system": "You serve the axiom.",
                                "messages": [{"role": "user", "content": "first"},
                                             {"role": "assistant", "content": "an earlier answer"},
                                             {"role": "user", "content": "second"}],
                                "temperature": 0.2})

    def test_turns_are_made_acceptable(self):
        server, brain, url = self.start()
        messy = [{"role": "assistant", "content": "a leftover reply"}, {"role": "system", "content": "one"},
                 {"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                 {"role": "assistant", "content": "  "}, {"role": "user", "content": "c"},
                 {"role": "system", "content": "two"}]
        self.claude(url).chat(messy, temperature=3.0)
        body = server.requests[0][3]
        self.assertEqual(body["system"], "one\n\ntwo")
        self.assertEqual(body["messages"], [{"role": "user", "content": "a\n\nb"},
                                            {"role": "assistant", "content": "(empty)"},
                                            {"role": "user", "content": "c"}])
        self.assertEqual(body["temperature"], 1.0)                  # clamped to Anthropic's range
        with self.assertRaises(wc.ModelError):
            self.claude(url).chat([{"role": "system", "content": "only a system prompt"}])

    def test_streaming_text_thinking_and_counts(self):
        server, brain, url = self.start()
        brain.queue("answer", "<think>weigh the rates</think>Pay the card first.")
        got, stats = [], {}
        reply = self.claude(url).chat(MSGS, on_token=lambda text, kind: got.append((kind, text)), stats=stats)
        self.assertEqual(reply, "Pay the card first.")
        self.assertEqual("".join(t for k, t in got if k == "content"), "Pay the card first.")
        self.assertEqual("".join(t for k, t in got if k == "thinking"), "weigh the rates")
        self.assertEqual(stats["thinking"], "weigh the rates")
        self.assertEqual((stats["prompt_tokens"], stats["done_reason"], stats["model"]), (21, "stop", "claude-mid-4"))
        self.assertEqual(stats["output_tokens"], len("Pay the card first.") // 4)
        self.assertIs(server.requests[0][3]["stream"], True)

    def test_a_reply_cut_off_at_the_limit_says_so(self):
        server, brain, url = self.start(stop_reason="max_tokens")
        stats = {}
        self.claude(url).chat(MSGS, stats=stats)
        self.assertEqual(stats["done_reason"], "length")
        self.claude(url).chat(MSGS, on_token=lambda t, k: None, stats=stats)
        self.assertEqual(stats["done_reason"], "length")

    def test_newer_models_that_take_no_temperature(self):
        server, brain, url = self.start(reject_temperature=True)
        claude = self.claude(url)
        self.assertEqual(claude.chat(MSGS, temperature=0.0), "OK.")
        claude.chat(MSGS, temperature=0.2)
        bodies = [r[3] for r in server.requests]
        self.assertEqual(len(bodies), 3)                            # one refusal, then two calls without it
        self.assertNotIn("temperature", bodies[1])
        self.assertNotIn("temperature", bodies[2])

    def test_the_output_limit_comes_from_the_model_list(self):
        server, brain, url = self.start()
        small = self.claude(url, "claude-small-3")
        small.list_models()
        small.chat(MSGS)
        self.assertEqual(server.requests[-1][3]["max_tokens"], 4096)
        big = self.claude(url, "claude-new-5")
        big.list_models()
        big.chat(MSGS)
        self.assertEqual(server.requests[-1][3]["max_tokens"], wc.ANTHROPIC_MAX_OUTPUT)   # capped
        self.claude(url, max_tokens=900).chat(MSGS)
        self.assertEqual(server.requests[-1][3]["max_tokens"], 900)

    def test_a_model_with_a_small_limit_is_asked_again(self):
        server, brain, url = self.start(max_output=4096)
        claude = self.claude(url, "claude-small-3")
        self.assertEqual(claude.chat(MSGS), "OK.")
        self.assertEqual([r[3]["max_tokens"] for r in server.requests], [wc.ANTHROPIC_MAX_OUTPUT, 4096])
        claude.chat(MSGS)
        self.assertEqual(server.requests[-1][3]["max_tokens"], 4096)   # remembered

    def test_the_model_list_follows_its_pages(self):
        server, brain, url = self.start(page=2)
        models = wc.Anthropic(api_key=KEY, base_url=url).list_models()
        self.assertEqual([m["id"] for m in models], ["claude-new-5", "claude-mid-4", "claude-small-3"])
        paths = [r[1] for r in server.requests]
        self.assertEqual(paths, ["/v1/models?limit=1000", "/v1/models?limit=1000&after_id=claude-mid-4"])

    def test_listing_needs_no_model_but_chatting_does(self):
        server, brain, url = self.start()
        with self.assertRaises(wc.ModelError):
            wc.Anthropic(api_key=KEY, base_url=url).chat(MSGS)
        with self.assertRaises(ValueError):
            wc.Anthropic("  ")
        self.assertEqual(server.requests, [])

    def test_errors_carry_the_services_words(self):
        server, brain, url = self.start()
        with self.assertRaises(wc.ModelError) as err:
            wc.Anthropic("claude-mid-4", api_key="wrong", base_url=url).chat(MSGS)
        self.assertEqual(err.exception.status, 401)
        self.assertIn("invalid x-api-key", str(err.exception))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with self.assertRaises(wc.ModelError):
                wc.Anthropic("claude-mid-4", base_url=url).list_models()
        self.assertNotIn("x-api-key", server.requests[-1][2])
        from tests.httpfake import Stream
        server.routes[("POST", "/v1/messages")] = lambda body, headers: (200, Stream([
            ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 3}}}),
            ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "par"}}),
            ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})],
            sse=True, done=False))
        with self.assertRaises(wc.ModelError) as err:
            self.claude(url).chat(MSGS, on_token=lambda t, k: None)
        self.assertIn("Overloaded", str(err.exception))

    def test_the_key_and_address_come_from_the_environment(self):
        server, brain, url = self.start()
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_BASE_URL": url + "/v1/"}):
            claude = wc.Anthropic("claude-mid-4")
        self.assertEqual(claude.base_url, url)                      # a trailing /v1 is tolerated
        self.assertEqual(claude.chat(MSGS), "OK.")
        self.assertEqual(server.requests[0][1], "/v1/messages")

    def test_stopping_mid_reply_closes_the_stream(self):
        server, brain, url = self.start(delay=0.002)
        brain.queue("answer", "slow words " * 200)

        def stop_soon(text, kind):
            if len(seen) > 5:
                raise Stop()
            seen.append(text)
        seen: list[str] = []
        with self.assertRaises(Stop):
            self.claude(url).chat(MSGS, on_token=stop_soon)
        import time
        deadline = time.monotonic() + 5
        while server.disconnects < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(server.disconnects, 1)

    def test_it_has_no_embeddings(self):
        with self.assertRaises(wc.ModelError) as err:
            wc.Anthropic("claude-mid-4", api_key=KEY).embed(["x"])
        self.assertIn("Combined", str(err.exception))
        self.assertEqual(wc.Anthropic("claude-mid-4").name, "anthropic:claude-mid-4")


class OpenAIListingAndUsageTests(unittest.TestCase):
    def test_the_model_list(self):
        server, _ = openai_server(models=["gpt-x", "text-embedding-3-small", "gpt-x"], key="sk-1")
        self.addCleanup(server.close)
        listing = wc.OpenAICompatible(base_url=server.url + "/v1", api_key="sk-1")
        self.assertEqual(listing.list_models(), ["gpt-x", "text-embedding-3-small"])
        with self.assertRaises(wc.ModelError) as err:
            wc.OpenAICompatible(base_url=server.url + "/v1", api_key="wrong").list_models()
        self.assertEqual(err.exception.status, 401)
        with self.assertRaises(wc.ModelError):
            listing.chat(MSGS)                                     # listing needs no model; chatting does
        server.routes[("GET", "/v1/models")] = lambda b, h: (200, [{"id": "bare-list"}, "plain-name"])
        self.assertEqual(listing.list_models(), ["bare-list", "plain-name"])
        server.routes[("GET", "/v1/models")] = lambda b, h: (200, {"models": "?"})
        with self.assertRaises(wc.ModelError):
            listing.list_models()

    def test_streamed_replies_ask_for_token_counts(self):
        server, brain = openai_server(usage_needs_option=True)
        self.addCleanup(server.close)
        stats = {}
        wc.OpenAICompatible("m", base_url=server.url + "/v1").chat(MSGS, on_token=lambda t, k: None, stats=stats)
        self.assertEqual(server.bodies("/v1/chat/completions")[0]["stream_options"], {"include_usage": True})
        self.assertEqual((stats["prompt_tokens"], stats["output_tokens"], stats["done_reason"]), (11, 7, "stop"))

    def test_servers_that_do_not_know_stream_options(self):
        server, brain = openai_server(reject_stream_options=True)
        self.addCleanup(server.close)
        o = wc.OpenAICompatible("m", base_url=server.url + "/v1")
        self.assertEqual(o.chat(MSGS, on_token=lambda t, k: None, json_mode=True, temperature=0.0), "OK.")
        o.chat(MSGS, on_token=lambda t, k: None)
        bodies = server.bodies("/v1/chat/completions")
        self.assertEqual(len(bodies), 3)                          # one refusal, then two calls without it
        self.assertNotIn("stream_options", bodies[1])
        self.assertEqual((bodies[1]["response_format"], bodies[1]["temperature"]), ({"type": "json_object"}, 0.0))


class CombinedTests(unittest.TestCase):
    def test_chat_goes_one_way_and_embeddings_the_other(self):
        chat_calls, embed_calls = [], []
        talker = wc.FunctionModel(lambda m: chat_calls.append(m) or "reply", lambda t: [[9.0]] * len(t),
                                  name="talker", embed_name="talker-space")
        searcher = wc.FunctionModel(lambda m: "never", lambda t: embed_calls.append(t) or [[1.0, 0.0]] * len(t),
                                    name="searcher", embed_name="search-space")
        both = wc.Combined(talker, searcher)
        self.assertEqual(both.chat(MSGS), "reply")
        self.assertEqual(both.embed(["a", "b"]), [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual((len(chat_calls), embed_calls), (1, [["a", "b"]]))
        self.assertEqual((both.name, both.embed_name), ("talker", "search-space"))
        self.assertIsNone(both.context_window)

    def test_the_window_is_the_chat_models(self):
        server, _ = ollama_server(context_length=32768)
        self.addCleanup(server.close)
        both = wc.Combined(wc.Ollama("llama3.2", host=server.url), wc.FunctionModel(lambda m: "", lambda t: []))
        self.assertEqual((both.context_window, wc.context_window_of(both)), (16384, 16384))

    def test_only_what_the_chat_backend_takes_is_passed(self):
        seen = {}

        def plain(messages, json_mode=False):
            seen["json_mode"] = json_mode
            return "whole reply"

        streamed = []
        both = wc.Combined(wc.FunctionModel(plain, lambda t: []), wc.FunctionModel(lambda m: "", lambda t: []))
        both.chat(MSGS, json_mode=True, temperature=0.3, on_token=lambda t, k: streamed.append((k, t)),
                  stats={}, think=False)
        self.assertEqual(seen, {"json_mode": True})
        self.assertEqual(streamed, [("content", "whole reply")])      # FunctionModel shows it whole

        class Strict:
            name = "strict"

            def chat(self, messages, *, json_mode=False, temperature=None):
                return "no streaming here"

            def embed(self, texts):
                return []
        streamed.clear()
        wc.Combined(Strict(), Strict()).chat(MSGS, on_token=lambda t, k: streamed.append(t), think=True)
        self.assertEqual(streamed, ["no streaming here"])

    def test_both_sides_must_be_backends(self):
        with self.assertRaises(TypeError):
            wc.Combined(object(), wc.FunctionModel(lambda m: "", lambda t: []))
        with self.assertRaises(TypeError):
            wc.Combined(wc.FunctionModel(lambda m: "", lambda t: []), object())

    def test_a_wisdom_core_that_thinks_with_claude_and_searches_with_ollama(self):
        claude_server, claude_brain = anthropic_server()
        self.addCleanup(claude_server.close)
        ollama, ollama_brain = ollama_server()
        self.addCleanup(ollama.close)
        claude_brain.queue("execute", {"actions": [{"type": "add", "principle": DEBT}]})
        backend = wc.Combined(wc.Anthropic("claude-mid-4", api_key=KEY, base_url=claude_server.url),
                              wc.Ollama(embed_model="nomic-embed-text", host=ollama.url))
        core = wc.WisdomCore(AXIOM, backend, path=":memory:")
        report = core.synthesize("Car loan at 6% or card at 24%?", "Pay the card first.")
        self.assertEqual([p.name for p in report.added], [DEBT["name"]])
        self.assertEqual([p.name for p in core.recall("which debt should I repay first", expand=False)],
                         [DEBT["name"]])
        self.assertEqual({r[1] for r in claude_server.requests}, {"/v1/messages"})
        self.assertNotIn("/api/chat", ollama.paths())                # Ollama only searched
        self.assertIn("/api/embed", ollama.paths())
        self.assertEqual(ollama_brain.calls, [])

    def test_swapping_the_embedding_side_re_embeds(self):
        events = []
        first = ScriptedModel(embed_name="space-a")
        both = wc.Combined(first, first)
        core = wc.WisdomCore(AXIOM, both, path=":memory:", on_event=events.append)
        from tests.fakes import seed
        seed(core, DEBT)
        core.recall("debts", expand=False)
        both.embed_backend = ScriptedModel(embed_name="space-b")
        core.recall("which debt first", expand=False)
        self.assertTrue(any(e["type"] == "reindex" and e["space"] == "space-b" for e in events))


if __name__ == "__main__":
    unittest.main()
