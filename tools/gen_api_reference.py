"""Generate docs/api-reference.md from wisdom_core's public names (__all__) and their docstrings.

    python tools/gen_api_reference.py > docs/api-reference.md
"""
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the checkout's wisdom_core.py
import wisdom_core as wc  # noqa: E402

ORDER = ["Agent", "WisdomCore", "Levels", "Principle", "SynthesisReport", "Event",
         "Ollama", "OpenAICompatible", "Anthropic", "Combined", "FunctionModel",
         "WisdomCoreError", "ModelError",
         "check_embeddings", "resolve_model", "fit_history", "context_window_of", "embed_prompts_for",
         "EMBED_PROMPTS", "main", "__version__"]
assert set(ORDER) == set(wc.__all__), set(wc.__all__) ^ set(ORDER)


EXTRA = {   # instance attributes set in __init__, which inspect cannot list; checked against the code by hand
    "Agent": [("core", "the `WisdomCore` underneath"), ("last_report", "the `SynthesisReport` of the last task (None until one ran)"),
              ("last_principles", "the principles injected for the last answer"),
              ("last_reasoning", "the thinking behind the last answer, if the model gave any"),
              ("messages", "the conversation so far, as role/content dicts"),
              ("instructions", "the standing instructions"), ("history", "earlier exchanges the model sees"),
              ("learn", "whether `ask` runs Wisdom Synthesis")],
    "WisdomCore": [("levels", "the `Levels` in use (kept as given, so changing it changes behaviour from the next task)"),
                   ("model", "the resolved backend"), ("path", "the database path"),
                   ("on_event", "the event handler, or None")],
    "ModelError": [("status", "the HTTP status from the server, when there was one"),
                   ("body", "the server's response body, when there was one")],
}


def args_to_list(text: str) -> str:
    """Google-style 'Args:' blocks -> a markdown bullet list."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if lines[i].strip() == "Args:":
            out.append("**Arguments**\n")
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                line = lines[i]
                m = re.match(r"^    (\w+): (.*)$", line)
                if m:
                    out.append(f"- `{m.group(1)}`: {m.group(2)}")
                elif line.startswith("        ") and out:
                    out[-1] += " " + line.strip()
                elif not line.strip():
                    if i + 1 < len(lines) and not lines[i + 1].startswith("    "):
                        break
                i += 1
            out.append("")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def doc(obj) -> str:
    d = args_to_list(inspect.getdoc(obj) or "")
    d = re.sub(r"``([^`]+)``", r"`\1`", d)
    # turn reST "::" literal blocks into fenced code
    out, lines, i = [], d.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::"):
            out.append(line.rstrip()[:-1] if not line.rstrip().endswith(" ::") else line.rstrip()[:-3])
            i += 1
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                block.append(lines[i][4:])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            while block and not block[0].strip():
                block.pop(0)
            out += ["", "```python", *block, "```", ""]
        else:
            out.append(line)
            i += 1
    return "\n".join(out).strip()


def sig(name, obj) -> str:
    try:
        text = f"{name}{inspect.signature(obj)}".replace("'", '"')
        return re.sub(r'(: |-> )"([^"]*)"', r"\1\2", text)   # annotations are strings (PEP 563): unquote them
    except (TypeError, ValueError):
        return name


print("# API reference\n")
print(f"Generated from the docstrings and signatures in `wisdom_core.py` (version {wc.__version__}) by "
      "`tools/gen_api_reference.py`. It lists only the public names in `wisdom_core.__all__` and their public "
      "methods and properties. Names starting with `_` are internal and may change without notice.\n")
print("Contents: " + " · ".join(f"[`{n}`](#{n.lower().strip('_')})" for n in ORDER) + "\n")
for name in ORDER:
    obj = getattr(wc, name)
    print(f"## {name}\n")
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            print(f"```python\nclass {name}({obj.__bases__[0].__name__})\n```\n")
        else:
            print(f"```python\n{sig(name, obj)}\n```\n")
        d = doc(obj)
        if d:
            print(d + "\n")
        if name in EXTRA:
            print("**Attributes**\n")
            for attr, what in EXTRA[name]:
                print(f"- `{attr}`: {what}")
            print()
        members = []
        for m, v in obj.__dict__.items():
            if m.startswith("_"):
                continue
            if isinstance(v, property):
                members.append((m, "property", v.fget))
            elif isinstance(v, (classmethod, staticmethod)):
                members.append((m, "method", v.__func__))
            elif inspect.isfunction(v):
                members.append((m, "method", v))
        for m, kind, fn in members:
            if kind == "property":
                print(f"### `{name}.{m}` (property)\n")
            else:
                s = sig(m, fn).replace("(self, ", "(").replace("(self)", "()").replace("(cls, ", "(")
                print(f"### `{name}.{m}`\n\n```python\n{s}\n```\n")
            d = doc(fn)
            if d:
                print(d + "\n")
    elif callable(obj):
        print(f"```python\n{sig(name, obj)}\n```\n")
        d = doc(obj)
        if d:
            print(d + "\n")
    elif name == "EMBED_PROMPTS":
        print("How an embedding model wants a search query, a stored text and a plain comparison written. "
              "The first family whose name appears anywhere in the embedding model's name wins. `{}` is the text. "
              "A backend can bring its own with an `embed_prompts` attribute.\n")
        print("| Family | Prompts |\n|---|---|")
        for fam, prompts in obj:
            shown = "; ".join(f"{k}: `{v}`".replace("\n", "\\n") for k, v in prompts.items()) or "(none)"
            print(f"| `{fam}` | {shown} |")
        print()
    else:
        print(f"`{name} = {obj!r}`\n")
