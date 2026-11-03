"""Reading what models write, and the small text helpers."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import wisdom_core as wc


class LenientJsonTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(wc._loads_lenient('{"actions": []}'), {"actions": []})

    def test_code_fence(self):
        self.assertEqual(wc._loads_lenient('Here you go:\n```json\n{"a": 1}\n```\nDone.'), {"a": 1})

    def test_text_around_the_object(self):
        self.assertEqual(wc._loads_lenient('Sure. {"a": [1, 2]} Hope that helps!'), {"a": [1, 2]})

    def test_trailing_commas(self):
        self.assertEqual(wc._loads_lenient('{"a": [1, 2,], "b": {"c": 3,},}'), {"a": [1, 2], "b": {"c": 3}})

    def test_smart_quotes(self):
        self.assertEqual(wc._loads_lenient("{“a”: “b”}"), {"a": "b"})

    def test_python_style_literal(self):
        self.assertEqual(wc._loads_lenient("{'a': True, 'b': None}"), {"a": True, "b": None})

    def test_thinking_block_before_json(self):
        self.assertEqual(wc._loads_lenient('<think>{"not": "this"}</think>{"a": 1}'), {"a": 1})

    def test_garbage(self):
        for text in ("", "no json here", "{broken", "[1, 2"):
            with self.subTest(text=text), self.assertRaises(wc._ParseError):
                wc._loads_lenient(text)


class ActionListTests(unittest.TestCase):
    def test_actions_key_any_case(self):
        self.assertEqual(wc._parse_actions('{"Actions": [{"type": "none"}]}'), [{"type": "none"}])

    def test_a_single_action_object(self):
        self.assertEqual(wc._parse_actions('{"type": "covered", "id": "P1"}'), [{"type": "covered", "id": "P1"}])

    def test_a_bare_list(self):
        self.assertEqual(wc._parse_actions('[{"type": "none"}, "junk", 3]'), [{"type": "none"}])

    def test_empty_object_and_null_mean_no_actions(self):
        self.assertEqual(wc._parse_actions("{}"), [])
        self.assertEqual(wc._parse_actions('{"actions": null}'), [])

    def test_a_single_action_under_actions(self):
        self.assertEqual(wc._parse_actions('{"actions": {"type": "none"}}'), [{"type": "none"}])

    def test_an_object_without_actions_is_an_error(self):
        with self.assertRaises(wc._ParseError):
            wc._parse_actions('{"thoughts": "hmm"}')
        with self.assertRaises(wc._ParseError):
            wc._parse_actions('{"actions": "add"}')

    def test_action_types_and_aliases(self):
        cases = {"add": "add", "ADD": "add", "create": "add", "New": "add", "covered": "covered",
                 "keep": "covered", "already_covered": "covered", "merge": "merge", "Combine": "merge",
                 "remove": "remove", "delete": "remove", "conflict": "remove", "none": "none",
                 "no_change": "none", "frobnicate": "frobnicate"}
        for given, expected in cases.items():
            with self.subTest(given=given):
                self.assertEqual(wc._action_type({"type": given}), expected)
        self.assertEqual(wc._action_type({"action": "merge"}), "merge")
        self.assertEqual(wc._action_type({}), "(missing)")


class PrincipleFieldTests(unittest.TestCase):
    def test_standard_fields(self):
        got = wc._principle_fields({"name": "Do X", "importance": "Because", "extent": "Here",
                                    "use_case": "Then", "related": ["a", "b"]})
        self.assertEqual(got, {"name": "Do X", "importance": "Because", "extent": "Here", "use_case": "Then",
                               "related": ["a", "b"]})

    def test_other_spellings(self):
        got = wc._principle_fields({"Principle": "Do X", "Why": "Because", "scope": "Here",
                                    "useCase": "Then", "edges": "a; b, c"})
        self.assertEqual((got["name"], got["importance"], got["extent"], got["use_case"], got["related"]),
                         ("Do X", "Because", "Here", "Then", ["a", "b", "c"]))
        got = wc._principle_fields({"title": "Do Y", "use when": "Now", "tags": ["x"]})
        self.assertEqual((got["name"], got["use_case"], got["related"]), ("Do Y", "Now", ["x"]))

    def test_a_plain_string(self):
        self.assertEqual(wc._principle_fields("Do Z")["name"], "Do Z")
        self.assertIsNone(wc._principle_fields("   "))

    def test_missing_name(self):
        self.assertIsNone(wc._principle_fields({"importance": "x"}))
        self.assertIsNone(wc._principle_fields(None))
        self.assertIsNone(wc._principle_fields(42))

    def test_nested_one_level(self):
        self.assertEqual(wc._principle_fields({"principle": {"name": "Inner"}})["name"], "Inner")

    def test_long_fields_are_clipped_and_whitespace_tidied(self):
        got = wc._principle_fields({"name": "Do   it\n now " + "x" * 1000, "importance": "y" * 2000,
                                    "related": ["z" * 200] + [f"r{i}" for i in range(20)]})
        self.assertTrue(got["name"].startswith("Do it now"))
        self.assertLessEqual(len(got["name"]), wc.NAME_CHARS + 40)
        self.assertLessEqual(len(got["importance"]), wc.FIELD_CHARS + 40)
        self.assertEqual(len(got["related"]), wc.RELATED_MAX)

    def test_duplicate_related_concepts_dropped(self):
        self.assertEqual(wc._principle_fields({"name": "n", "related": ["Tax", "tax", "VAT"]})["related"],
                         ["Tax", "VAT"])

    def test_action_principle_wherever_it_is(self):
        self.assertEqual(wc._action_principle({"type": "add", "principle": {"name": "A"}})["name"], "A")
        self.assertEqual(wc._action_principle({"type": "merge", "new_principle": {"name": "B"}})["name"], "B")
        self.assertEqual(wc._action_principle({"type": "add", "name": "C", "why": "w"})["importance"], "w")
        self.assertIsNone(wc._action_principle({"type": "add", "reason": "no principle"}))


class PlaceholderTests(unittest.TestCase):
    def test_placeholders(self):
        for text in ("", "...", "\u2026", "<reason>", "[name]", "N/A", "none", "TBD", "ok", "  . . ."):
            with self.subTest(text=text):
                self.assertTrue(wc._is_placeholder(text))

    def test_real_text(self):
        for text in ("It contradicts P2", "Be kind", "Wrong in this task", "Use the 50/30/20 rule",
                     "Противоречит оси", "与公理冲突"):
            with self.subTest(text=text):
                self.assertFalse(wc._is_placeholder(text))


class LabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p1 = wc.Principle(id="aaaa", name="First rule.")
        self.p2 = wc.Principle(id="bbbb", name="Second rule")
        self.labels = {"P1": self.p1, "P2": self.p2}

    def test_forms_that_mean_p2(self):
        for value in ("P2", "p2", " P2 ", "[P2]", "(P2)", "P 2", "P#2", "2", 2, 2.0, "Principle 2", "P2.",
                      "bbbb", "second rule", "Second rule."):
            with self.subTest(value=value):
                self.assertIs(wc._resolve_label(value, self.labels), self.p2)

    def test_unknown(self):
        for value in ("P3", 3, "P", "", None, True, "something else", 2.5, ["P1"]):
            with self.subTest(value=value):
                self.assertIsNone(wc._resolve_label(value, self.labels))

    def test_targets(self):
        self.assertEqual(wc._action_targets({"ids": ["P1", "P2"]}, "ids"), ["P1", "P2"])
        self.assertEqual(wc._action_targets({"target_ids": "P1"}, "ids", "targetids"), ["P1"])
        self.assertEqual(wc._action_targets({"id": ""}, "id"), [])


class TextHelperTests(unittest.TestCase):
    def test_strip_thinking(self):
        self.assertEqual(wc._strip_thinking("<think>x</think>\n answer"), "answer")
        self.assertEqual(wc._strip_thinking("<think>still going"), "")
        self.assertEqual(wc._strip_thinking("plain"), "plain")
        self.assertEqual(wc._strip_thinking(None), "")

    def test_clip_keeps_head_and_tail(self):
        text = "HEAD" + "m" * 1000 + "TAIL"
        out = wc._clip(text, 100)
        self.assertTrue(out.startswith("HEAD"))
        self.assertTrue(out.endswith("TAIL"))
        self.assertIn("characters cut", out)
        self.assertEqual(wc._clip("  short  ", 100), "short")

    def test_render(self):
        p = wc.Principle(id="x", name="Check totals", importance="Errors mislead", extent="Any sum",
                         use_case="When adding amounts", related=["math"])
        self.assertEqual(p.render(), "Check totals. Why: Errors mislead. Scope: Any sum. Use when: adding amounts.")
        self.assertEqual(p.embed_text(), p.render() + " Related: math.")
        self.assertEqual(wc.Principle(id="x", name="Is it right?").render(), "Is it right?")

    def test_report_lines(self):
        a = wc.Principle(id="a", name="A")
        b = wc.Principle(id="b", name="B")
        report = wc.SynthesisReport(task="t", added=[a], merged=[([a, b], b)], removed=[(a, "wrong")],
                                    covered=[b], skipped=["odd"], error="oops")
        self.assertEqual(report.lines(), ["! learning failed: oops", "+ added: A", "= merged 2 into: B",
                                          "- removed: A (wrong)", "~ already covered by: B", "? ignored: odd"])

    def test_unit_vectors(self):
        v = wc._unit([3, 4])
        self.assertAlmostEqual(v[0], 0.6, places=6)
        self.assertAlmostEqual(v[1], 0.8, places=6)
        self.assertEqual(list(wc._unit([0, 0])), [0.0, 0.0])


class OllamaHostTests(unittest.TestCase):
    def test_forms(self):
        cases = {
            None: "http://localhost:11434",
            "localhost": "http://localhost:11434",
            "192.168.1.15": "http://192.168.1.15:11434",
            "192.168.1.15:11434": "http://192.168.1.15:11434",
            "http://gpu-box:8080/": "http://gpu-box:8080",
            "https://ollama.example.com": "https://ollama.example.com:443",
            "0.0.0.0": "http://localhost:11434",
            "0.0.0.0:9999": "http://localhost:9999",
            "[::1]:11434": "http://[::1]:11434",
        }
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_HOST", None)
            for given, expected in cases.items():
                with self.subTest(given=given):
                    self.assertEqual(wc._ollama_host(given), expected)

    def test_environment_variable(self):
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "10.0.0.5"}):
            self.assertEqual(wc._ollama_host(None), "http://10.0.0.5:11434")
            self.assertEqual(wc._ollama_host("otherhost"), "http://otherhost:11434")

    def test_tag(self):
        self.assertEqual(wc._tag("nomic-embed-text"), "nomic-embed-text:latest")
        self.assertEqual(wc._tag("bge-m3:567m"), "bge-m3:567m")


if __name__ == "__main__":
    unittest.main()
