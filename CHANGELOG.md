# Changelog

All notable changes to this project are documented in this file.

## [0.8.0] - 2026-11-03

Initial public alpha release.

- **Core Axiom.** Write one sentence, and the agent learns its own principles in service of it.
- **Principles.** Short, reusable pieces of wisdom the agent writes for itself, kept in a small SQLite database searched by meaning.
- **Injection.** The Core Axiom and the most relevant principles (12 by default) go into the prompt at every decision.
- **Wisdom Synthesis and Distillation.** After every task the agent adds, merges, revises or removes principles to fit the Core Axiom.
- **Principles earn their place.** Principles behind answers the user keeps pushing back on come into doubt and are reviewed.
- **Semantic levels**, calibrated automatically for whichever embedding model is used.
- **Model backends:** Ollama (default), OpenAI-compatible servers, Anthropic, `Combined` and `FunctionModel`.
- **`Agent` chat loop**, the `WisdomCore` framework on its own, and a `wisdom-core` terminal chat command.
- **Event hooks** (`on_event`), change history, and `export_state()` / `import_state()`.
- Single file, Python 3.10 or newer, no required dependencies (optional `numpy` via `wisdom-core[fast]`).

[0.8.0]: https://github.com/VickeryMachineLogic/wisdom-core/releases/tag/v0.8.0
