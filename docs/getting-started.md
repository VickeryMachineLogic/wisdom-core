# Getting started

This guide takes you from nothing to an agent that learns, then shows how to point it at other model servers and what to do when something goes wrong.

## 1. Install

```bash
pip install wisdom-core
```

Python 3.10 or newer. There are no other required dependencies. `pip install "wisdom-core[fast]"` adds numpy, which makes searching large principle sets faster. Because the framework is one file, `wisdom_core.py`, you can also copy it straight into a project.

Check the install:

```bash
wisdom-core --version          # wisdom-core 0.8.0
python -m wisdom_core --help   # the same command, run as a module
```

## 2. Try it with no model (optional)

The [examples](examples.md) include a small rule-based stand-in for a model, so you can watch the learning loop run before you install anything else. Get them from the repository (or the source package on PyPI), then run:

```bash
git clone https://github.com/VickeryMachineLogic/wisdom-core
cd wisdom-core
python examples/01_quickstart_offline.py
```

The answers are canned, but the principles, the reports, the database and the history are the real framework at work.

## 3. Get a model

Wisdom Core needs two kinds of model:

- a **chat model**, which answers and writes the principles, and
- an **embedding model**, which turns text into vectors so principles can be found by meaning.

The default is [Ollama](https://ollama.com) on your own computer. Install Ollama, then pull one of each:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

With no model names given, Wisdom Core uses the first chat model and the first embedding model Ollama has. The address is `http://localhost:11434` unless you set `$OLLAMA_HOST` or pass `host=`, so an Ollama server elsewhere on your network works too.

## 4. Your first agent

Write one sentence saying what the agent is for. That is the **Core Axiom**.

```python
from wisdom_core import Agent

with Agent(core_axiom="Help people make sound money decisions.", path="money.db") as agent:
    print(agent.ask("I have a car loan at 6% and a credit card at 24%. Which should I pay off first?"))
    print(agent.last_report.summary())     # what the agent learned from that task
    for p in agent.principles():
        print("-", p.name)
```

- `ask()` answers and then learns. `agent.last_report` says what changed, with lines such as `+ added: ...`, `= merged 2 into: ...`, `- removed: ...` or `~ already covered by: ...`.
- Everything the agent learns goes into `money.db`. Run the script again and the principles are still there. The Core Axiom is stored too, so after the first run `Agent(path="money.db")` is enough.
- `agent.last_principles` lists the principles that went into the prompt for the last answer.

The same thing from a terminal:

```bash
wisdom-core --axiom "Help people make sound money decisions." --db money.db
```

Inside the chat, `/principles`, `/history`, `/stats`, `/axiom`, `/reset`, `/help` and `/quit` do what they say. Outside it, `wisdom-core --db money.db --principles --history` prints the database without starting a model.

## 5. Choose models

```python
from wisdom_core import Agent, Ollama, OpenAICompatible, Anthropic, Combined, FunctionModel

Agent(core_axiom=AXIOM, model="llama3.2")                                   # Ollama, chosen chat model
Agent(core_axiom=AXIOM, model=Ollama("llama3.2", "nomic-embed-text", host="http://192.168.1.20:11434"))

# Any OpenAI-compatible server: LM Studio, vLLM, llama.cpp's llama-server, Ollama's /v1, OpenAI and others
Agent(core_axiom=AXIOM, model=OpenAICompatible("qwen2.5-7b-instruct", "text-embedding-nomic-embed-text-v1.5",
                                               base_url="http://localhost:1234/v1"))

# Claude for chat, with a local embedding model (Anthropic has no embeddings)
Agent(core_axiom=AXIOM, model=Combined(Anthropic("claude-sonnet-4-5"), Ollama(embed_model="nomic-embed-text")))

# Your own functions
Agent(core_axiom=AXIOM, model=FunctionModel(my_chat, my_embed))   # chat(messages) -> str, embed(texts) -> vectors
```

`OpenAICompatible` reads `$OPENAI_BASE_URL` and `$OPENAI_API_KEY` when you don't pass `base_url` or `api_key`; `Anthropic` reads `$ANTHROPIC_API_KEY`. From the terminal, `--openai-base-url URL --model NAME --embed-model NAME` uses an OpenAI-compatible server.

With a hosted API, your prompts (including the principles) go to that provider. With Ollama or another server on your own hardware, they stay with you.

## 6. Add it to an agent you already have

If you already have an agent, you don't need `Agent`. Use `WisdomCore` and make two calls:

```python
from wisdom_core import WisdomCore

core = WisdomCore(core_axiom="Help people get and keep good work.", model="llama3.2", path="career.db")

system_prompt = "You are a careful assistant.\n\n" + core.inject(task)   # before the decision
answer = my_agent(system_prompt, task)                                  # your own code
report = core.synthesize(task, answer, outcome="The user said it helped.")   # after the task
```

See [Concepts](concepts.md) for what each call does, and [`examples/02_your_own_agent.py`](../examples/02_your_own_agent.py) for a version that runs offline.

## Troubleshooting

| You see | What it means |
|---|---|
| `Could not reach http://localhost:11434/... Is Ollama running? Start it with: ollama serve` | No Ollama server at that address. Start it, or pass `host=` / set `$OLLAMA_HOST`. |
| `Ollama at ... has no chat model. Pull one first, for example: ollama pull llama3.2` | Ollama is running but has no chat model. |
| `Ollama at ... has no embedding model, which Wisdom Core needs ...` | Pull an embedding model, for example `ollama pull nomic-embed-text`. |
| `Ollama does not have the model 'NAME'. Pull it first: ollama pull NAME` | The model name you passed is not pulled. |
| `A Core Axiom is required the first time a database is used` | A new database needs `core_axiom=` (or `--axiom`). |
| `... is a database that Wisdom Core did not create; give each agent its own file.` | `path=` points at some other SQLite file. |
| `! learning failed: ...` in a report | The model's reply for the lesson could not be used. The answer was still given; `report.error` has the reason. Smaller models fail this more often. |

Answers seem slow? Every task with learning on makes several model calls (see [Concepts](concepts.md#what-one-task-costs)). Model calls have no timeout by default, so a slow local model is waited for. `learn=False` or `--no-learn` answers without learning.
