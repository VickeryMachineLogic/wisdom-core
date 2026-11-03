# Examples

All examples live in [`examples/`](../examples/). The four numbered ones run with **no model server and no network**: they use [`offline_model.py`](../examples/offline_model.py), a small rule-based stand-in plugged in through the public `FunctionModel` hook. It is not an AI model. Its answers and "lessons" are canned, picked by keywords, but everything around them (recall, injection, synthesis, the checks, the database, the history, standing and review) is the real framework. Swap in a real backend (`OpenAICompatible(...)`, `Combined(Anthropic(...), ...)`, `Ollama()` or your own `FunctionModel`) to get real answers and real lessons.

Run them from the repository root, with the package installed (`pip install wisdom-core`) or straight from a source checkout:

```bash
python examples/01_quickstart_offline.py
```

| Script | What it shows |
|---|---|
| [`01_quickstart_offline.py`](../examples/01_quickstart_offline.py) | An `Agent` answers, learns a principle, then uses it on the next question. The second time, the first look finds the lesson already held and stops. Prints the database path so you can run `wisdom-core --db PATH --principles --history`. |
| [`02_your_own_agent.py`](../examples/02_your_own_agent.py) | `WisdomCore` inside your own agent: `recall` / `format_context` before the decision, `synthesize` after. Prints the exact text that gets injected, and `stats()`. |
| [`03_inspect_and_manage.py`](../examples/03_inspect_and_manage.py) | Reading what an agent holds: `principles()`, removing one by hand with `remove()`, seeing it retired (status `removed`) with `include_ended=True`, `history()`, `standing()`, `export_state()` / `import_state()` into a fresh database, and changing the axiom with `axiom_history()`. |
| [`04_feedback_and_events.py`](../examples/04_feedback_and_events.py) | Principles earning their place: `respond()` then `synthesize(success=...)`, the user pushing back repeatedly, a principle coming into doubt and being reviewed, and an `on_event` handler counting every step. |

These two came with the 0.8.0 release and need a real model: Ollama by default, or any OpenAI-compatible API (`demo.py --openai-base-url`; in `custom_agent.py`, change `model=`):

| Script | What it shows |
|---|---|
| [`demo.py`](../examples/demo.py) | A seven-message money conversation against a real model, with a temporary database, printing what was learned after each task. `--model`, `--embed-model`, `--host`, `--openai-base-url`, `--api-key`, `--keep`. |
| [`custom_agent.py`](../examples/custom_agent.py) | A two-step agent (plan, then answer) that injects before each step and synthesizes once at the end. Writes `career_agent.db` in the current folder. |

## What the no-model examples print

From `01_quickstart_offline.py` (the database path varies):

```
you>   I have a car loan at 6% and a credit card at 24%. Which should I pay off first?
agent> (toy answer) No principles yet, so I act from the Core Axiom alone.
       principles injected: 0
       wisdom core + added: Pay off the highest-interest debt first once minimum payments are covered.

you>   My cousin has a 19% store card and a 5% loan. What goes first?
agent> (toy answer) Following 1 principle(s), starting with: Pay off the highest-interest debt first once minimum payments are covered.
       principles injected: 1
       wisdom core ~ already covered by: Pay off the highest-interest debt first once minimum payments are covered. (1.00)
       wisdom core ~ already held, so nothing more ran (lesson: Pay off the highest-interest debt first once minimum payments are covered.; measured 1.00, new ground below 0.85)
```

From the end of `04_feedback_and_events.py`, after four pushbacks:

```
[6] you> No, that's wrong. The promo matters more than the rate.
       - removed: Pay off the highest-interest debt first once minimum payments are covered. (the user kept pushing back on answers it shaped)
       ...
       ! in doubt: Pay off the highest-interest debt first once minimum payments are covered. (the user pushed back on 4 of the 5 answers it shaped; ...)

Every principle ever held (include_ended=True):
  [removed] Pay off the highest-interest debt first once minimum payments are covered.  <- the user kept pushing back on answers it shaped
  [active] Check for promotional or 0% rates before ranking debts by their interest rate.
```

The toy model always removes a principle in doubt. A real model reads the exchange and the principle's record, and may revise it or decide it stands instead.

## Writing your own stand-in model

`FunctionModel(chat, embed)` takes any two functions: `chat(messages) -> str`, where `messages` are OpenAI-style role/content dicts, and `embed(texts) -> list of vectors`. That is how `offline_model.py` works, and how you would wrap a model API Wisdom Core doesn't support directly. If your `chat` also accepts `json_mode`, `temperature`, `on_token`, `stats` or `think` (or `**kwargs`), those are passed along. Give `embed_name=` a new value whenever you switch embedding models, so stored principles are embedded again in the new space.
