# FAQ

## What is Wisdom Core, in one paragraph?

A Python framework for AI agents built around one idea: you write one sentence, the Core Axiom, saying what the agent is for, and the agent works out the rest. After every task it writes down what that task taught as a short, reusable principle in service of that sentence, and before every decision the most relevant principles go into its prompt. Principles that repeat each other are merged, ones that conflict with the axiom or proved wrong are removed, and ones the user keeps pushing back on come into doubt and are reviewed. It works with any LLM: any OpenAI-compatible API, Anthropic, Ollama, or your own adapter.

## Is it free?

Yes. It is open source under the MIT license, so you can use it, change it and ship it in commercial products, as long as the copyright and license notice go with it. The notice is at the top of `wisdom_core.py`, so it travels with the file.

## Where does my data go?

To the model backends you configure, and nowhere else. Prompts, answers and the principles injected into them are sent to the chat and embedding models you choose, under that provider's terms. Wisdom Core itself has no telemetry, account or update check. What the agent learns is stored in a SQLite file wherever you put it.

## Where is what it learned?

In one SQLite file per agent: `wisdom.db` unless you pass `path=` (or `--db`). It holds every Core Axiom, every principle (active, merged or removed), the log of every change, and the embedding calibration. You can read it with `wisdom-core --db FILE --principles --history`, with `core.principles()` and `core.history()`, or with any SQLite tool. `core.export_state()` gives you all of it as plain data.

## Can I see why a principle exists, or undo one?

Each principle records the task it was learned from (`learned_from`), and `core.history()` logs every add, merge, revision and removal with its reason. To take one out, call `core.remove(principle_id, reason)`. Its status becomes `removed`; it is retired, not deleted, and stays in the file with your reason. There is no call in 0.8.0 that brings a retired principle back; you would add it again in your own words through a task, or edit an export and import it with `replace=True`.

## How is this different from giving the agent a long system prompt?

A system prompt is fixed, and you write all of it. Here you write one sentence, and the agent writes and maintains the rest from experience. The Core Axiom also decides what it learns, not only how it answers. Only the principles relevant to the current situation are injected (twelve by default), so the prompt stays short as the agent learns more.

## How is it different from storing chat history?

It stores lessons, not transcripts. Each task leaves at most one new principle, which is checked against what the agent already holds. Duplicates are merged rather than piled up, and wrong ones are removed.

## Which models work?

Any LLM you can reach from Python:

- **Any OpenAI-compatible API**, through `OpenAICompatible`: OpenAI itself, and servers such as LM Studio, vLLM, llama.cpp's llama-server and Ollama's `/v1`. Chat and embeddings.
- **Anthropic's Claude**, through `Anthropic` (Messages API). Chat only, so pair it with a backend that has embeddings, in `Combined`.
- **Ollama**, through `Ollama`, the default when no model is given. Chat and embeddings.
- **Anything else**, through your own `chat` and `embed` functions in `FunctionModel`, or any object with `chat()` and `embed()` methods.

You need both a chat model and an embedding model, and they can come from different backends (`Combined`).

## Which model should I use?

There is no benchmark in this project to point to. Principles are written by the chat model, so a more capable model writes better ones and fails less often to produce a reply the framework can read. `check_embeddings(model)` measures how well an embedding model tells rewordings of one lesson apart from different lessons, and tells you in words.

## Does it make answers slower or cost more?

Yes, when learning is on. Each task adds one model call for the first look once principles exist, and two more when the task is new ground, plus occasional extra calls. Each call takes time, and on a paid API it costs tokens. `learn=False` answers without learning, and `agent.respond()` followed by `agent.synthesize()` lets the answer go out before the learning runs.

## Can several users share one agent?

They can, but what one user teaches is injected into answers for the others, because principles belong to the agent, not the user. If that is not what you want, give each user or audience its own database file. If the agent talks with people you don't trust, review its principles, or run it with learning off.

## Can several processes use one database?

Yes. A second process using the same file sees the changes. Give each agent its own file, though; Wisdom Core refuses a SQLite file it did not create.

## Is it production-ready?

It is alpha (0.8.0). The interface may still change before 1.0. It has a test suite that runs with no model and no network, but it is new, and you should watch what your agents learn.

## Can Vickery Machine Logic help us use it?

Vickery Machine Logic, which makes Wisdom Core, builds custom AI agents for businesses. [Get in touch](https://vickerymachinelogic.com/contact/).

## How do I report a bug or a security problem?

Bugs, questions and ideas: [GitHub issues](https://github.com/VickeryMachineLogic/wisdom-core/issues). Security problems: please report them privately through [GitHub's private vulnerability reporting](https://github.com/VickeryMachineLogic/wisdom-core/security/advisories/new), not in a public issue. See [SECURITY.md](../SECURITY.md).

## Can I contribute?

Yes, in the way that helps most right now: ideas and bug reports are very welcome. Please [open an issue](https://github.com/VickeryMachineLogic/wisdom-core/issues), and questions are welcome there too. Wisdom Core is maintained by John Vickery at Vickery Machine Logic, and code contributions and pull requests aren't being accepted right now; trusted maintainers may be invited in the future. You are always free to fork it and build on it under the MIT license. See [CONTRIBUTING.md](../CONTRIBUTING.md).
