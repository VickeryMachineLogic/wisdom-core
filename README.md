<p align="center"><img src="https://raw.githubusercontent.com/VickeryMachineLogic/wisdom-core/main/.github/assets/wisdom-core-banner-1600.jpg" width="800" alt="wisdom-core by Vickery Machine Logic, LLC"></p>

# Wisdom Core

**An AI agent that learns its own principles from its work, steered by one sentence you write.**

[![PyPI](https://img.shields.io/pypi/v/wisdom-core)](https://pypi.org/project/wisdom-core/) [![Python](https://img.shields.io/pypi/pyversions/wisdom-core)](https://pypi.org/project/wisdom-core/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/LICENSE)

A language model on its own keeps nothing from one conversation to the next. Wisdom Core gives an agent a lasting memory of what its work has taught it: a short list of lessons in plain language, which it keeps adding to and cleaning up as it goes.

You write one sentence, the **Core Axiom**, saying what the agent is for. After every task, the agent works out what that task taught and writes it down as a **principle** in service of that sentence. Before every decision, the most relevant principles go into its prompt. Principles that say the same thing are combined, ones that turn out wrong are removed, and ones the user keeps pushing back on are reviewed.

It is free and open source (MIT), a single Python file with no required dependencies, and it runs on your own computer with [Ollama](https://ollama.com) by default. Everything the agent learns sits in one SQLite file that you can read, back up and edit.

```python
from wisdom_core import Agent

agent = Agent(core_axiom="Give accurate answers and say plainly when you're unsure.")
print(agent.ask("What should I check before signing a lease?"))
print(agent.last_report.summary())   # what the agent just learned
```

Wisdom Core is a framework by John Vickery.

## Why it is different

- **One sentence steers the learning.** A system prompt shapes only the answer in front of it. The Core Axiom also decides what the agent learns from every task, because every principle is written, combined or removed to fit it.
- **It keeps lessons, not transcripts.** What it stores are short, reusable principles ("Pay off the highest-interest debt first once minimum payments are covered."), each with why it matters, where it stops and when it applies.
- **It cleans up after itself.** Near-duplicates are merged, contradictions and mistakes are removed, and a principle behind answers the user keeps pushing back on comes into doubt and is reviewed. Nothing is removed on a count alone.
- **You can see everything.** Principles are plain sentences. Every change is logged with its reason, nothing is ever deleted, and the whole state exports as plain data.
- **It is yours and it is local.** One file, standard library only, Ollama on your own machine by default. No account, no hosted service and no telemetry. The only network calls go to the model server you point it at.
- **It fits the agent you already have.** Use the ready-made `Agent` chat loop, or add two calls to your own code: `core.inject(task)` before a decision, `core.synthesize(task, result)` after the task.

## Quickstart

1. Install the package, plus a chat model and an embedding model in [Ollama](https://ollama.com). The package installs in seconds; the models are a few gigabytes to download.

   ```bash
   pip install wisdom-core
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. Chat with an agent that learns:

   ```bash
   wisdom-core --axiom "Help people make sound money decisions."
   ```

   After each answer it prints what it learned, such as `+ added: ...` or `~ already held, so nothing more ran`. Type `/principles` to see what it holds.

3. Or use it from Python, as in the example at the top.

Answers come from your own model, so how fast they come and how good the principles are depends on the model you run.

**No model server yet?** [`examples/01_quickstart_offline.py`](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/examples/01_quickstart_offline.py) runs the whole learning loop with a small rule-based stand-in model, so you can see the mechanics before installing Ollama. The [Getting started guide](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/docs/getting-started.md) walks through both.

## Documentation

- [Getting started](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/docs/getting-started.md): install, first agent, other model servers, troubleshooting.
- [Concepts](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/docs/concepts.md): the Core Axiom, principles, injection, synthesis, distillation and standing, in plain words.
- [Examples](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/docs/examples.md): runnable scripts, and what each one shows.
- [FAQ](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/docs/faq.md): privacy, cost, limits, and when to use it.
- [API reference](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/docs/api-reference.md): every public class and function, generated from the code.

## The idea

- **Core Axiom.** The one sentence you write. It drives everything else, and it is close to the only thing you have to write.
- **Principles.** Short, reusable pieces of wisdom the agent writes for itself, kept in a small database that is searched by meaning.
- **Injection.** Every time the agent has to decide something, the Core Axiom and the principles most relevant to that decision (12 by default) go into its prompt exactly as written.
- **Wisdom Synthesis.** After every task the agent works out the one piece of wisdom the exchange taught. That is either what the user showed (a correction, a preference, a fact about themselves) or what its own thinking established about the subject. It then makes sure it holds a principle for it. If one of its principles already says it closely enough, nothing changes. If none does, it writes one that fits the Core Axiom.
- **Wisdom Distillation.** During synthesis, principles that say nearly the same thing are combined into one that fits the Core Axiom. A principle that conflicts with the axiom, contradicts another, or turned out to be wrong is removed. The agent has full control of its principles.
- **Principles earn their place.** The agent remembers which principles shaped each answer and how the user received it. A principle behind answers the user keeps pushing back on comes into doubt and is reviewed: it is removed, revised, or found to stand. Nothing is removed on a count alone.

Give two agents the same experiences under different axioms and they end up with different principles.

## Install

```bash
pip install wisdom-core
pip install "wisdom-core[fast]"     # optional: numpy, for faster search over large principle sets
```

Python 3.10 or newer, with no other dependencies. `wisdom_core.py` is a single file, so you can also copy it into a project.

You also need a model. By default Wisdom Core uses [Ollama](https://ollama.com) on your computer, with the first chat model and the first embedding model it finds:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Any OpenAI-compatible server, Anthropic's API, or your own functions work too (see Models, below).

## Chat from a terminal

```bash
wisdom-core --axiom "Help people make sound money decisions."
# or: python -m wisdom_core --axiom "..."
```

After every answer it prints what the Wisdom Core did, for example `+ added: ...`, `= merged 2 into: ...`, `= revised: ... -> ...`, `- removed: ...`, `~ already covered by: ...` or `~ already held, so nothing more ran`. Inside the chat, `/principles`, `/history`, `/stats`, `/axiom`, `/reset`, `/help` and `/quit` do what they say.

You need the axiom only the first time. After that it is read from the database, which is `wisdom.db` unless you pass `--db`. `wisdom-core --principles` and `wisdom-core --history` print a database without starting a model. `--model`, `--embed-model`, `--host`, `--openai-base-url`, `--api-key`, `--context` and `--no-learn` choose the rest; `wisdom-core --help` lists them.

The source package on PyPI includes `examples/demo.py`. It runs a short conversation against your own model with a temporary database, and prints what the Wisdom Core learned after each task.

## Use it inside your own agent

`Agent` is a ready-made chat loop. If you already have an agent, use `WisdomCore` directly: inject before each decision, and synthesize after each task.

```python
from wisdom_core import WisdomCore, Ollama

core = WisdomCore(core_axiom="Help people get and keep good work.",
                  model=Ollama("llama3.2"), path="career_agent.db")

task = "How should I prepare for a job interview?"
system_prompt = "You are a careful assistant.\n\n" + core.inject(task)    # before every decision
answer = my_agent(system_prompt, task)                                       # your own code
report = core.synthesize(task, answer, outcome="The user said it helped.")   # after the task
print(report.summary())
```

`synthesize` learns from whatever you can give it:

- `outcome` describes in words how the task went.
- `success=True` or `success=False` counts the result for or against the principles behind the answer.
- `reasoning` is the model's thinking before it answered.
- `conversation` is the earlier messages. Its last agent message is read as the answer the user is responding to.
- `used` is the principles injected for this task (from `core.recall`).
- `previous` is the principles behind the previous answer.

[`examples/custom_agent.py`](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/examples/custom_agent.py) shows a two-step agent that injects before each step, and [`examples/02_your_own_agent.py`](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/examples/02_your_own_agent.py) runs the same pattern offline.

## What happens in one task

1. **Recall.** Principles are ranked by meaning against what the person said. Each principle is embedded twice:
   - What it is about (its statement, the situation that calls for it, and the words a person would use), so that it is found.
   - What it says (the statement alone), to tell whether two principles teach the same thing.

   Once the agent holds more principles than it injects, the model first writes a one-sentence description of the situation. Each principle is then ranked by the better of its match to the message and to that description.
2. **Injection.** The Core Axiom and the chosen principles go into the prompt as written, most relevant first. When the backend knows its context window (Ollama does), older messages are dropped if the conversation would not leave room for a reply. The Core Axiom and the principles are never trimmed.
3. **Answer.** The model answers. A reasoning model's thinking is kept and handed to Wisdom Synthesis.
4. **First look.** One short call names the exchange's lesson without seeing the principles, and says whether the user pushed back on the previous answer. The Wisdom Core measures that lesson against every principle held. If one already says it, synthesis stops there. The first look is skipped while no principles are held, and when an outcome is given; then the whole synthesis runs. When the user pushed back, the whole synthesis always runs.
5. **Wisdom Synthesis.** On new ground, the model writes a short memo. It covers what the user showed, what the thinking established, the lesson, the principle for it, near-duplicates, conflicts, and a review of any principle in doubt. The memo reads the Core Axiom, the conversation, the answer the user was responding to, the reasoning, the answer, and up to 20 related principles (the injected ones first).
6. **Changes.** A second call turns the memo into actions:
   - `covered`: a listed principle already holds the lesson.
   - `add`: a new principle.
   - `merge`: listed principles are replaced by one that combines them. A single principle given new wording is revised.
   - `remove`: a listed principle is taken out, with a reason.
   - `stands`: a principle in doubt was reviewed and found sound.

   Each action is checked before it is applied. Only principles shown to the model can be touched, and a merge naming an unknown label changes nothing. Placeholder text is refused, and a task adds one new principle at most.

   A new principle is measured, not judged. One at least as close as `covered` to a held principle counts as already held. One at least as close as `combine` is combined with it.
7. **Wisdom Distillation.** Principles the task brought up that measure as nearly the same are combined into one, at most twice per task.

That is one call for the answer, one for the first look once principles exist, and two more on new ground. Now and then there is also a situation description, a combination or a calibration. A failed synthesis never costs the answer: it is reported in `SynthesisReport.error`.

## Semantic levels

`Levels` decides how close counts as close and how much gets written:

| Level | Default | What it does |
|---|---|---|
| `new_ground` | measured (0.85 until then) | A first-look lesson at or above this against a held principle is already held, and synthesis stops there. 0 turns the first look off. |
| `force` | `True` | On new ground the task must leave a principle; whether it is new or already held is measured. `False` lets the model decide whether a task taught anything. |
| `covered` | measured (0.95 until then) | A drafted principle at least this close to a held one counts as already held. |
| `combine` | measured (0.85 until then) | Principles at least this alike are combined into one. 0 turns it off. |
| `merge` | `0.0` | How alike principles must be for a merge the model proposes to go through. 0 leaves it to the model. |
| `recall_floor` | `0.0` | How relevant a principle must be to be injected at all. 0 means no floor. |
| `context` | `12` | How many principles are injected at each decision. |
| `max_adds` | `1` | How many new principles one task may add. |

```python
from wisdom_core import Agent, Levels

agent = Agent(core_axiom="Help people make sound money decisions.", levels=Levels(context=8, combine=0.8))
```

**Calibrated automatically.** Embedding models don't share a similarity scale: one scores two rewordings of a sentence 0.95, another 0.75. So `new_ground`, `covered` and `combine` are measured for whichever embedding model you use.

- **The first measurement.** The first time the levels are needed, one embedding request goes over a fixed set of 15 principles from different subjects. It measures how the model scores rewordings against different lessons on the same subject, and sets the levels between the two. The result is kept in the agent's database.
- **On the agent's own principles.** Once the agent holds 8 principles, it calibrates again on a sample of its own, which the chat model rewords. It does this again every 25 new principles.
- **The fingerprint.** A fixed sentence is embedded and compared at every lesson. If the same model name starts embedding it differently (new weights pulled under the same tag), every principle is embedded again and the model is measured again.
- **Levels you set.** A level you set yourself is always used as given. `core.levels = core.levels.replace(combine=None)` makes it automatic again.

`core.effective_levels()` shows the levels in force, and `core.calibration()` shows the measurement behind them.

Each text is also written the way its embedding model expects. Search queries, stored principles and plain comparisons carry the markers their model was trained with, for embeddinggemma, nomic-embed, mxbai-embed, snowflake-arctic-embed, bge (bge-m3 needs none), e5 and qwen3-embedding. A backend can bring its own markers with an `embed_prompts` attribute.

## Principles earn their place

- **Which principles shaped an answer.** A principle shapes an answer when it is among the four most relevant principles injected for it.
- **How the answer was received.** The first look reads whether the user's next message pushed back on that answer. Your application can also say outright how an answer went, with `success=True` or `success=False`.
- **In doubt.** A principle comes into doubt when it has been pushed back on at least three times since its last review, and that record would be unlikely (under 2%) for a sound principle, given how often this user pushes back on the agent's other answers.
- **Review.** The next synthesis that lists a principle in doubt reviews it: the model removes it, revises it, or says it stands. A principle that stands gets a clean start.
- **Proven.** A principle that has shaped eight answers or more, with at most a tenth pushed back on, reads as proven and isn't removed over one disagreement.

`core.standing(principle)` returns the record.

## Models

| Backend | Use |
|---|---|
| `Ollama(model=None, embed_model=None, *, host=None, num_ctx=None, timeout=None, keep_alive=None)` | Local or LAN Ollama. Models are chosen automatically if not given, and `host` falls back to `$OLLAMA_HOST`. The context window is read from the model and capped at 16,384 tokens, unless you pass `num_ctx`. |
| `OpenAICompatible(model=None, embed_model=None, *, base_url=None, api_key=None, timeout=None)` | OpenAI, LM Studio, vLLM, llama.cpp's `llama-server`, Ollama's `/v1`, OpenRouter and similar servers. `base_url` and `api_key` fall back to `$OPENAI_BASE_URL` and `$OPENAI_API_KEY`. |
| `Anthropic(model=None, *, api_key=None, base_url=None, timeout=None, max_tokens=None)` | Claude, through Anthropic's Messages API. `api_key` falls back to `$ANTHROPIC_API_KEY`. It has no embeddings, so pair it with another backend in `Combined`. |
| `Combined(chat, embed)` | Chat through one backend and search principles through another. |
| `FunctionModel(chat, embed, *, name="custom", embed_name=None)` | Your own functions: `chat(messages) -> str` and `embed(texts) -> list of vectors`. |

```python
from wisdom_core import Agent, Anthropic, Combined, Ollama

agent = Agent(core_axiom="Help people make sound money decisions.",
              model=Combined(Anthropic("claude-sonnet-4-5"), Ollama(embed_model="nomic-embed-text")))
```

- **Shorthand.** `model="llama3.2"` means `Ollama("llama3.2")`.
- **No timeouts.** Model calls have no timeout by default, so a slow local model is waited for rather than cut off.
- **Thinking.** On Ollama, the framework's own calls switch a reasoning model's thinking off, and the answer keeps the model's default.
- **Streaming.** Replies stream whenever something is watching (below).

## Watching it work

Pass `on_event` to `Agent` or `WisdomCore`, and it is called with a dict for every step:

- Recall: `recall.start`, `recall.done` (with the principles and their scores).
- Model calls: `model.start`, `model.token` (each streamed piece of the reply), `model.done`, `embed`.
- Wisdom Synthesis: `synthesis.start`, `synthesis.first_look`, `synthesis.candidates`, `synthesis.memo`, `synthesis.actions`, `synthesis.done` (with the whole report), `principles.changed`.
- Upkeep: `calibrated`, `model_changed`, `reindex`, `history.trimmed`.

An ordinary exception raised by your handler is logged and ignored. One that is not an `Exception`, such as `KeyboardInterrupt` or your own `BaseException` subclass, passes through and stops the run.

## What is stored

Each agent has one SQLite file (`wisdom.db` unless you pass `path=`).

- **Principles are never deleted.** Every principle has a status: `active`, `merged` or `removed`. When one is merged, revised or removed it is retired, not erased: it stays in the file with its status, the reason, when it ended and, for merges and revisions, the principle it became. `core.principles(include_ended=True)` lists them all, and `core.remove(principle_id, reason)` retires one by hand.
- **Every change is logged** (`core.history()`). Timestamps are stored in UTC.
- **Every Core Axiom is kept** (`core.axiom_history()`). If you change the Core Axiom, existing principles are kept, and any that conflict with the new one are removed as they come up.
- **Embedding models can change.** If you switch embedding models, stored principles are embedded again automatically.
- **Several processes can share a file.** A second process using the same file sees the changes.

`core.export_state()` returns the whole of an agent's learning as plain data: every axiom, every principle as written, the log and the calibration, without the principles' vectors. `core.import_state(state)` rebuilds it in a database that holds no principles yet (or any database, with `replace=True`), and the principles are embedded again the first time they are searched. A new database still needs a Core Axiom to open, so open it with one (for example `WisdomCore(core_axiom=state["axiom"], path="copy.db")`); the import then replaces it with the backup's axiom history.

## Private and local by default

- **Your data stays where your model runs.** With the default Ollama backend, the prompts, answers and principles stay on your computer or your own network. Point it at a hosted API (OpenAI-compatible or Anthropic) and the prompts go to that service, under its terms.
- **No telemetry.** Wisdom Core makes network calls only to the model backend you configure. It has no account, analytics or update check.
- **Plain files you control.** One SQLite file per agent, which any SQLite tool can open. `wisdom-core --db FILE --principles` and `--history` print it without starting a model.
- **Model calls have no timeout by default**, so a slow local model is waited for, not cut off. `Ollama`, `OpenAICompatible` and `Anthropic` each take a `timeout` if you want one.

## Limits

Wisdom Core is alpha software. Know these before you depend on it:

- **The interface may change before 1.0.**
- **Learning costs model calls.** Each task with learning on is one call for the answer, one for the first look once principles exist, and two more on new ground, plus occasional extra calls (see *What happens in one task*). On a small local model that takes time. `learn=False` (or `--no-learn`) answers without learning.
- **Principles are only as good as the model that writes them.** A small model can draft weak principles or replies the framework cannot read. A failed lesson never costs the answer: it shows in `SynthesisReport.error`. Read what an agent holds now and then (`/principles`, `core.principles()`, `core.history()`) and remove what you disagree with (`core.remove`).
- **What users say shapes later prompts.** Lessons learned from one conversation are injected into later ones for every user of that database. If an agent talks with people you don't trust, review its principles, give each audience its own database, or turn learning off.
- **One agent per database file.** Several processes can share a file, but don't point two different agents at one.
- **Search is simple.** Every active principle's vectors are held in memory and compared on each search. Install `wisdom-core[fast]` (numpy) to speed that up for large principle sets.
- **Anthropic's API has no embeddings,** so pair it with another backend in `Combined`.

## Reference

| | |
|---|---|
| `Agent(core_axiom=None, model=None, *, path="wisdom.db", context=None, levels=None, instructions=None, history=6, learn=True, on_event=None)` | A chat agent. `instructions` are standing instructions placed after the principles. `history` is how many earlier exchanges the model sees. |
| `agent.ask(message) -> str` | Answer one message, then learn from it. |
| `agent.respond(message) -> str`, then `agent.synthesize(outcome=None, success=None)` | The same in two steps, so the answer can go out before the agent learns, and the lesson can include how things went. |
| `agent.last_report`, `agent.last_principles`, `agent.last_reasoning` | What the last task learned, which principles were injected, and the thinking behind the answer. |
| `agent.reset()`, `agent.close()` | Forget the conversation (principles are kept), and close the database. |
| `WisdomCore(core_axiom=None, model=None, *, path="wisdom.db", context=None, levels=None, on_event=None)` | The framework on its own. `path=":memory:"` keeps everything in memory. |
| `core.inject(task, *, conversation=None, k=None, expand=True) -> str` | The Core Axiom and the relevant principles, ready for a prompt. `expand=False` skips the situation description. |
| `core.recall(task, *, conversation=None, k=None, expand=True) -> list[Principle]` | The relevant principles themselves. `core.format_context(principles)` turns them into the injected text. |
| `core.synthesize(task, result, *, outcome=None, conversation=None, used=(), reasoning=None, previous=(), success=None) -> SynthesisReport` | Wisdom Synthesis and Distillation for one finished task. |
| `core.principles(*, include_ended=False)`, `core.get(principle_id)`, `core.history(limit=50)`, `core.stats()` | Read what the agent holds and what it has done. |
| `core.core_axiom`, `core.set_core_axiom(text)`, `core.axiom_history()` | The Core Axiom, changing it, and every axiom before it. |
| `core.remove(principle_id, reason="removed by hand")` | Remove a principle by hand. Its status becomes `removed`, and it stays in the file and the log with the reason. |
| `core.levels`, `core.effective_levels()`, `core.calibration()`, `core.calibrate()` | The semantic levels, the levels in force, and the measurement behind them. |
| `core.standing(principle)` | How the answers a principle shaped were received. |
| `core.export_state()`, `core.import_state(state, *, replace=False)` | Back up an agent's learning, and rebuild it somewhere else. |
| `check_embeddings(model)` | Measure an embedding model without storing anything. |
| `SynthesisReport` | `added`, `merged`, `removed`, `covered`, `stands`, `skipped`, `first_look`, `reaction`, `doubt`, `memo`, `error`; `.changed`, `.lines()`, `.summary()`, `.to_dict()`. |
| `WisdomCoreError`, `ModelError` | The errors it raises. `ModelError` carries the server's `status` and `body`. |

Every public class, method and function, with its signature and docstring, is in the [API reference](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/docs/api-reference.md).

## Tests

The source package includes the test suite:

```bash
python -m unittest discover -s tests -t .
# or: python -m pytest -q tests
```

It uses a scripted model and fake Ollama, OpenAI-style and Anthropic servers, so it needs no model and no network.

## Help and security

- **Questions, bugs and ideas:** open an [issue](https://github.com/VickeryMachineLogic/wisdom-core/issues). For a bug, include the version (`wisdom-core --version`), your Python version, the model backend and model names, and what you expected.
- **Security problems:** please don't open a public issue. Report them privately through GitHub's [private vulnerability reporting](https://github.com/VickeryMachineLogic/wisdom-core/security/advisories/new). See [SECURITY.md](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/SECURITY.md).
- **Changes:** see the [CHANGELOG](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/CHANGELOG.md).
- **Help using it in a business:** Wisdom Core is made by [Vickery Machine Logic](https://vickerymachinelogic.com), which builds custom AI agents. [Get in touch](https://vickerymachinelogic.com/contact/).

## Contributing

Wisdom Core is here to help people build better agents, and it is free for everyone to use, fork and build on under the MIT license. Use it in your own projects, commercial ones included, and make it your own.

Wisdom Core is maintained by John Vickery at Vickery Machine Logic. Ideas and bug reports are very welcome: please [open an issue](https://github.com/VickeryMachineLogic/wisdom-core/issues). Questions are welcome too, and so is hearing what confused you. Code contributions and pull requests aren't being accepted right now; trusted maintainers may be invited in the future.

See [CONTRIBUTING.md](https://github.com/VickeryMachineLogic/wisdom-core/blob/main/CONTRIBUTING.md) for how to report a bug or share an idea.

## License

MIT. Copyright (c) 2026 Vickery Machine Logic, LLC. See the LICENSE file. The same notice sits at the top of `wisdom_core.py`, so it goes with the file if you copy it into a project.

## Status

Version 0.8.0, alpha. The interface may still change before 1.0.
