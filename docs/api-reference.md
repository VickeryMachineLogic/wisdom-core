# API reference

Generated from the docstrings and signatures in `wisdom_core.py` (version 0.8.0) by `tools/gen_api_reference.py`. It lists only the public names in `wisdom_core.__all__` and their public methods and properties. Names starting with `_` are internal and may change without notice.

Contents: [`Agent`](#agent) · [`WisdomCore`](#wisdomcore) · [`Levels`](#levels) · [`Principle`](#principle) · [`SynthesisReport`](#synthesisreport) · [`Event`](#event) · [`Ollama`](#ollama) · [`OpenAICompatible`](#openaicompatible) · [`Anthropic`](#anthropic) · [`Combined`](#combined) · [`FunctionModel`](#functionmodel) · [`WisdomCoreError`](#wisdomcoreerror) · [`ModelError`](#modelerror) · [`check_embeddings`](#check_embeddings) · [`resolve_model`](#resolve_model) · [`fit_history`](#fit_history) · [`context_window_of`](#context_window_of) · [`embed_prompts_for`](#embed_prompts_for) · [`EMBED_PROMPTS`](#embed_prompts) · [`main`](#main) · [`__version__`](#version)

## Agent

```python
Agent(core_axiom: str | None = None, model: Any = None, *, path: str = "wisdom.db", context: int | None = None, levels: Levels | Mapping[str, Any] | None = None, instructions: str | None = None, history: int = 6, learn: bool = True, on_event: Callable[[dict[str, Any]], None] | None = None) -> None
```

A chat agent built on the Wisdom Core.

Every call to `ask` is one task: the Core Axiom and the most relevant
principles are injected, the model answers, and then Wisdom Synthesis
runs on that task and its answer. `last_report` says what it learned.

**Arguments**

- `core_axiom`: The one sentence the agent serves (see `WisdomCore`).
- `model`: None, an Ollama model name, or a backend object.
- `path`: The SQLite file for this agent's principles.
- `context`: How many principles are injected at each decision (default 12).
- `levels`: The semantic levels (see `Levels`).
- `instructions`: Optional standing instructions (tone, format, language) placed after the Core Axiom and principles.
- `history`: How many earlier exchanges of the conversation the model sees. Older ones are also dropped when the conversation would not leave room for a reply in the model's context window.
- `learn`: Set False to answer without running Wisdom Synthesis.
- `on_event`: Watch every step (see `WisdomCore`). The answer call is reported as a model call with purpose "answer".

**Attributes**

- `core`: the `WisdomCore` underneath
- `last_report`: the `SynthesisReport` of the last task (None until one ran)
- `last_principles`: the principles injected for the last answer
- `last_reasoning`: the thinking behind the last answer, if the model gave any
- `messages`: the conversation so far, as role/content dicts
- `instructions`: the standing instructions
- `history`: earlier exchanges the model sees
- `learn`: whether `ask` runs Wisdom Synthesis

### `Agent.close`

```python
close() -> None
```

### `Agent.core_axiom` (property)

### `Agent.on_event` (property)

### `Agent.principles`

```python
principles() -> list[Principle]
```

### `Agent.reset`

```python
reset() -> None
```

Forget the conversation. Principles are kept.

### `Agent.system_prompt`

```python
system_prompt(principles: Sequence[Principle]) -> str
```

### `Agent.ask`

```python
ask(message: str) -> str
```

Answer one message, then learn from it (`respond` followed by `synthesize`).

### `Agent.respond`

```python
respond(message: str) -> str
```

Answer one message without learning from it yet.

Use this when the answer should reach the person before the agent
learns, then call `synthesize()`, optionally with an outcome.

### `Agent.synthesize`

```python
synthesize(outcome: str | None = None, success: bool | None = None) -> SynthesisReport
```

Run Wisdom Synthesis on the last answer. `outcome` is anything known about how it went, in words;
`success` says outright whether it went well, which counts for or against the principles behind it.

## WisdomCore

```python
WisdomCore(core_axiom: str | None = None, model: Any = None, *, path: str = "wisdom.db", context: int | None = None, levels: Levels | Mapping[str, Any] | None = None, on_event: Callable[[dict[str, Any]], None] | None = None) -> None
```

The framework on its own: Core Axiom, principles, injection and synthesis.

Use it inside an agent you already have:

```python
core = WisdomCore(core_axiom="...", model=Ollama("llama3.1"))
prompt = core.inject(task)          # at every decision
...
core.synthesize(task, result)       # after every task
```

`Agent` wraps it in a ready-made chat loop.

**Arguments**

- `core_axiom`: The one sentence the agent serves. Required the first time a database is used; after that it is read from the database. Passing a different sentence later changes the axiom (the old one is kept in the history, and principles that conflict with the new one are removed as they come up).
- `model`: None (Ollama, choosing models automatically), an Ollama model name, or a backend object such as Ollama, OpenAICompatible or FunctionModel.
- `path`: The SQLite file that holds this agent's principles. Use one file per agent. ":memory:" keeps everything in memory.
- `context`: How many principles are injected at each decision (default 12). A shortcut for `levels.context`.
- `levels`: The semantic levels (see `Levels`): how close counts as already covered, whether every task must write a principle, and how many are injected. The object is kept as given, so changing it later changes what this Wisdom Core does from the next task on.
- `on_event`: Optional function called with a dict for every step the Wisdom Core takes: recall, each model call (and every streamed token), each synthesis stage and its result. Use it to watch or log the framework working. An ordinary exception it raises is logged and ignored; KeyboardInterrupt-style exceptions (BaseException) pass through, which is how a caller can stop a run.

**Attributes**

- `levels`: the `Levels` in use (kept as given, so changing it changes behaviour from the next task)
- `model`: the resolved backend
- `path`: the database path
- `on_event`: the event handler, or None

### `WisdomCore.context` (property)

How many principles are injected at each decision (`levels.context`).

### `WisdomCore.close`

```python
close() -> None
```

Close the database.

### `WisdomCore.core_axiom` (property)

### `WisdomCore.set_core_axiom`

```python
set_core_axiom(text: str) -> None
```

Change the Core Axiom. The old one stays in `axiom_history()`.

### `WisdomCore.axiom_history`

```python
axiom_history() -> list[tuple[str, str]]
```

Every Core Axiom this database has had, oldest first, as (text, set_at).

### `WisdomCore.principles`

```python
principles(*, include_ended: bool = False) -> list[Principle]
```

Principles in the order they were learned. `include_ended` adds merged and removed ones.

### `WisdomCore.get`

```python
get(principle_id: str) -> Principle | None
```

### `WisdomCore.history`

```python
history(limit: int | None = 50) -> list[Event]
```

What the Wisdom Core has done, newest first.

### `WisdomCore.stats`

```python
stats() -> dict[str, int]
```

### `WisdomCore.export_state`

```python
export_state() -> dict[str, Any]
```

Everything this database holds, as plain data that `import_state` can rebuild elsewhere:
the Core Axiom and every axiom before it, every principle as written (including merged and
removed ones), and the log of what the Wisdom Core did. The vectors are left out; they are made
again from the text by whichever embedding model the new copy uses.

### `WisdomCore.import_state`

```python
import_state(state: Mapping[str, Any], *, replace: bool = False) -> dict[str, int]
```

Rebuild an exported state in this database. The principles keep their text, their history and
their ids; their vectors are made again the first time they are searched. Refuses a database that
already holds principles unless `replace` is set, which clears it first.

### `WisdomCore.remove`

```python
remove(principle_id: str, reason: str = "removed by hand") -> Principle
```

Remove a principle yourself. It stays in the log with the reason.

### `WisdomCore.embed_prompts`

```python
embed_prompts() -> dict[str, str]
```

How the embedding model in use wants each kind of text written (see `embed_prompts_for`).

### `WisdomCore.check_embeddings`

```python
check_embeddings() -> dict[str, Any]
```

Measure the embedding model in use against this core's levels (see `check_embeddings`).

### `WisdomCore.calibration`

```python
calibration(*, measure: bool = True) -> dict[str, Any] | None
```

The calibration of the embedding model in use, as kept in this database: `space`, `at`,
`levels` (the measured new_ground, covered and combine in use, or None when the model could not be
calibrated), `source` ("general" or "agent": from the agent's own principles), `general` and
`domain` (each calibration's own result), `separation`, `verdict` and the rest of
`check_embeddings`' result. When there is none for this model yet and `measure` is set, it is
measured now (one embedding request). None if it could not be measured (the model could not be
reached); that is tried again on the next call. With `measure` off it only reads, and never waits for
a lesson in progress.

### `WisdomCore.calibrate`

```python
calibrate() -> dict[str, Any] | None
```

Measure the embedding model in use now with the general check (`check_embeddings`) and keep the
result as this database's calibration, replacing any before it (a calibration on the agent's own
principles starts again from here). Returns it, or None if the model could not be reached.

### `WisdomCore.calibrate_on_principles`

```python
calibrate_on_principles() -> dict[str, Any] | None
```

Calibrate on this agent's own principles: the chat model rewords a sample of them and writes a
different lesson on the same subject for each, and the levels are set from how the embedding model
scores those pairs. What an agent learns about sits closer together than the general check's mixed
subjects, so this is what the levels should fit once it holds `DOMAIN_MIN` principles. When its
pairs overlap, the general calibration stays in use. Returns the calibration, or None when there is
none to build on or too few principles. Model problems are raised (`ModelError`).

### `WisdomCore.check_fingerprint`

```python
check_fingerprint() -> bool
```

Embed `FINGERPRINT_TEXT` and compare it with the vector kept from before. If the same model name now
embeds it differently, the model changed underneath (new weights under the same tag, a new runtime):
every principle is embedded again and the model is measured again. Returns True when it had changed.
The first check for a model only keeps the fingerprint.

### `WisdomCore.effective_levels`

```python
effective_levels() -> Levels
```

The levels in force: `levels` with each automatic one (None) filled in from the calibration of the
embedding model in use, measured now if there is none yet. Levels set by hand are used as given.

### `WisdomCore.recall`

```python
recall(task: str, *, conversation: Sequence[dict[str, Any]] | None = None, k: int | None = None, expand: bool = True) -> list[Principle]
```

The principles most relevant to `task`, most relevant first.

When there are more principles than `k`, the model first writes a
one-sentence description of the situation, and principles are ranked
by the better of their similarity to the task and to that description.
Principles describe how to act while tasks describe a subject, so the
description finds them more reliably. `expand=False` skips that call.
Principles less relevant than `levels.recall_floor` are left out.

### `WisdomCore.format_context`

```python
format_context(principles: Sequence[Principle]) -> str
```

The text injected into the agent's prompt: the Core Axiom and the principles, as written.

### `WisdomCore.inject`

```python
inject(task: str, *, conversation: Sequence[dict[str, Any]] | None = None, k: int | None = None, expand: bool = True) -> str
```

Recall the principles for `task` and return the text to put in the agent's prompt.

### `WisdomCore.synthesize`

```python
synthesize(task: str, result: str, *, outcome: str | None = None, conversation: Sequence[dict[str, Any]] | None = None, used: Iterable[Principle | str] = (), reasoning: str | None = None, previous: Iterable[Principle | str] = (), success: bool | None = None) -> SynthesisReport
```

Learn from a finished task.

The model reads the Core Axiom, what the user said (`task`), the
agent's previous answer that it responds to (the last answer in
`conversation`), the agent's `reasoning` before it answered (a
reasoning model's thinking), the agent's answer (`result`) and
`outcome` (anything known about how it went), plus the principles most
related to it, including those in `used` (the ones injected for the
task). It learns what the user showed - a reaction to the previous
answer, a preference, a fact, value or goal they stated - and what the
thinking established about the subject itself, and writes that as a
piece of wisdom. A lesson about how the answer was written counts only
if the user asked for it or reacted to it. It then keeps the principles as they are, adds one,
revises or merges principles, or removes ones the user's words or the
Core Axiom show to be wrong.

It starts with a quick first look (see `Levels.new_ground`): one
short call writes the lesson without seeing the principles, and if that
lesson measures as a principle already held, the synthesis stops there.

Principles earn their place. `previous` is the principles that were
injected for the agent's previous answer, most relevant first (the one
the user is responding to): the first look reads whether the user pushed
back on it, and the most relevant few of them are counted as questioned
or supported accordingly. `success` says outright how this task's
answer went, when the application knows; the most relevant few of
`used` are counted by it. A principle questioned far more often than
this agent's answers in general is in doubt and is listed for review in
the memo (see `standing`).

Problems with the model (unreachable, or a reply that cannot be read)
are reported in `SynthesisReport.error` rather than raised, so a
failed lesson never costs the agent its answer.

### `WisdomCore.standing`

```python
standing(principle: Principle) -> dict[str, Any]
```

A principle's standing: how the answers it shaped were received, since its last review.

`shaped` and `questioned` count answers it shaped and the ones the user pushed back on; `usual`
is the share of this user's other answers (not shaped by this principle) that are pushed back on;
`surprise` is how likely that many pushbacks would be for a sound principle, pushed back on as often
as the usual; `in_doubt` when that is below `DOUBT_ALPHA` after at least `DOUBT_MIN_QUESTIONED`
pushbacks; `proven` when it has served many times with almost no pushback; `reviews` how many
times it was reviewed and found to stand.

### `WisdomCore.usual_pushback`

```python
usual_pushback(excluding: Principle | None = None) -> float
```

The share of this agent's answers the user pushes back on, with a mild prior for a new agent; with
`excluding`, the share among the answers that principle did not shape.

## Levels

```python
Levels(new_ground: float | None = None, force: bool = True, covered: float | None = None, combine: float | None = None, merge: float = 0.0, recall_floor: float = 0.0, context: int = 12, max_adds: int = 1) -> None
```

The semantic levels: how close counts as close, and how much gets written.

`new_ground` decides whether Wisdom Synthesis runs at all. A quick first
look writes the exchange's lesson without seeing the principles, and the
Wisdom Core measures it against every principle held: at or above
`new_ground` the agent already holds that lesson and nothing more runs;
below it the exchange is new ground and the full synthesis runs. The first
look is skipped (the synthesis always runs) while no principles are held,
when an outcome is given, when the user pushed back on the previous answer,
and when `new_ground` is 0.

On new ground with `force` on (the default), the task must draft its
lesson as a principle, and `covered` decides by measurement whether the
agent already holds it: a draft that similar to a held principle is counted as covered,
anything less alike is added. With `force` off, the model decides for
itself whether a task taught anything, and `covered` only stops a draft
that is nearly identical to one already held.

Every level that compares principles (`new_ground`, `covered`,
`combine`, `merge`) measures what they say: the statement alone,
embedded as a plain comparison. `recall_floor` measures what a principle
is about (its statement, the situation that calls for it and the words a
person would use) against what the person said.

**Calibrated automatically.** Embedding models do not share a scale, so
`new_ground`, `covered` and `combine` are None by default, meaning
"measure them": the first time a Wisdom Core uses an embedding model it
runs `check_embeddings` once, keeps the result in its database, and uses
the levels measured there (see `WisdomCore.effective_levels`). A model
that cannot tell the comparisons apart keeps the fallbacks (0.85, 0.95 and
0.85). A number you set yourself is always used as given.

`combine` is enforced by the Wisdom Core itself: a drafted principle at
least that close to a held one (but below `covered`) is merged with it
instead of being added, and principles a task brings up that are that
close to each other are combined into one. 0 turns it off.

`merge` is how alike two principles must be for a merge the model
proposes to go through, and `recall_floor` how relevant a principle must
be to be injected at all; 0 for either leaves that decision where it is
today. `context` is how many principles are injected, and `max_adds`
how many new ones one task may add.

### `Levels.to_dict`

```python
to_dict() -> dict[str, Any]
```

### `Levels.from_dict`

```python
from_dict(data: Mapping[str, Any] | None) -> Levels
```

Levels from a saved dict; anything missing keeps its default, anything unknown is ignored.

### `Levels.replace`

```python
replace(**changes: Any) -> Levels
```

A copy with `changes`; None for a measured level makes it automatic again.

### `Levels.automatic`

```python
automatic() -> list[str]
```

The levels left to calibration (None).

### `Levels.resolved`

```python
resolved(measured: Mapping[str, Any] | None = None) -> Levels
```

A copy with every automatic level filled in: from `measured` (a calibration's levels) when it has
one, else the fallback. Levels set by hand are kept as they are.

## Principle

```python
Principle(id: str, name: str, importance: str = "", extent: str = "", use_case: str = "", related: list[str] = <factory>, status: str = "active", created_at: str = "", learned_from: str = "", times_injected: int = 0, shaped: int = 0, questioned: int = 0, review: dict[str, Any] | None = None, ended_at: str | None = None, end_reason: str | None = None, merged_into: str | None = None, score: float | None = None) -> None
```

One principle the agent holds (or once held).

`name` is the principle itself, one sentence telling the agent how to act.
`importance`, `extent` and `use_case` say why it matters, where it
stops, and when it applies. `related` lists related concepts, which help
the right principle come up in a search.

### `Principle.since_review`

```python
since_review() -> tuple[int, int]
```

Answers shaped and questioned since the principle was last reviewed and found to stand.

### `Principle.render`

```python
render() -> str
```

The principle as the model sees it.

### `Principle.embed_text`

```python
embed_text() -> str
```

The whole principle as one text, labels included (what earlier versions embedded).

### `Principle.recall_text`

```python
recall_text() -> str
```

What the principle is about, embedded so it is found when its situation comes up: the statement, the
situation that calls for it and the words a person would use, with no labels shared by every principle.

### `Principle.identity_text`

```python
identity_text() -> str
```

What the principle says, embedded to tell whether two principles teach the same thing: the statement
alone, so the shared shape of the other fields cannot make different lessons look alike.

### `Principle.to_dict`

```python
to_dict() -> dict[str, Any]
```

## SynthesisReport

```python
SynthesisReport(task: str, added: list[Principle] = <factory>, merged: list[tuple[list[Principle], Principle]] = <factory>, removed: list[tuple[Principle, str]] = <factory>, covered: list[Principle] = <factory>, skipped: list[str] = <factory>, memo: str = "", error: str | None = None, first_look: dict[str, Any] = <factory>, reaction: dict[str, Any] = <factory>, questioned: list[Principle] = <factory>, supported: list[Principle] = <factory>, doubt: list[Principle] = <factory>, stands: list[tuple[Principle, str]] = <factory>) -> None
```

What one round of Wisdom Synthesis did.

### `SynthesisReport.changed` (property)

True if any principle was added, merged or removed.

### `SynthesisReport.lines`

```python
lines() -> list[str]
```

### `SynthesisReport.summary`

```python
summary() -> str
```

### `SynthesisReport.to_dict`

```python
to_dict() -> dict[str, Any]
```

## Event

```python
Event(at: str, kind: str, principle_id: str | None, sources: list[str], detail: str, task: str) -> None
```

One entry in the Wisdom Core's log of what it did.

## Ollama

```python
Ollama(model: str | None = None, embed_model: str | None = None, *, host: str | None = None, num_ctx: int | None = None, timeout: float | None = None, keep_alive: str | int | None = None) -> None
```

Chat and embeddings through a local (or LAN) Ollama server.

With no arguments it uses the first chat model and the first embedding
model the server has. The address comes from `host`, then $OLLAMA_HOST,
then http://localhost:11434. The context window sent with every request
is read from the model itself and capped at 16,384 tokens, unless you pass
`num_ctx`.

### `Ollama.model` (property)

### `Ollama.embed_model` (property)

### `Ollama.name` (property)

### `Ollama.embed_name` (property)

### `Ollama.context_window` (property)

The num_ctx sent with every chat request: the model's own limit, capped at 16,384.

### `Ollama.model_limit`

```python
model_limit(name: str | None = None) -> int | None
```

The longest context the model can hold, from its own metadata; None if it does not say.

### `Ollama.list_models`

```python
list_models() -> list[dict[str, Any]]
```

### `Ollama.is_embedding_model`

```python
is_embedding_model(entry: dict[str, Any] | str) -> bool
```

True for a model that only makes embeddings (asked of the server, then guessed from the name).

### `Ollama.chat`

```python
chat(messages: Sequence[dict[str, Any]], *, json_mode: bool = False, temperature: float | None = None, on_token: Callable[[str, str], None] | None = None, stats: dict[str, Any] | None = None, think: bool | None = None) -> str
```

One reply.

`on_token(text, kind)` receives the reply as it is written, kind being
"content" or "thinking". `stats` is filled with token counts, the
window used and why generation stopped. `think=False` switches a
reasoning model's thinking off; None leaves the model's default. A model
that thinks without writing an answer is asked once more with thinking
off, and remembered.

### `Ollama.embed`

```python
embed(texts: Sequence[str]) -> list[list[float]]
```

## OpenAICompatible

```python
OpenAICompatible(model: str | None = None, embed_model: str | None = None, *, base_url: str | None = None, api_key: str | None = None, timeout: float | None = None) -> None
```

Chat and embeddings through any server that speaks the OpenAI API.

That covers OpenAI itself and local servers such as LM Studio, vLLM,
llama.cpp's llama-server and Ollama's /v1 endpoint. `base_url` defaults
to $OPENAI_BASE_URL, then https://api.openai.com/v1; `api_key` defaults
to $OPENAI_API_KEY. `model` may be left out to call `list_models()` only.

### `OpenAICompatible.name` (property)

### `OpenAICompatible.embed_name` (property)

### `OpenAICompatible.list_models`

```python
list_models() -> list[str]
```

The model names the server offers (its /models list).

### `OpenAICompatible.chat`

```python
chat(messages: Sequence[dict[str, Any]], *, json_mode: bool = False, temperature: float | None = None, on_token: Callable[[str, str], None] | None = None, stats: dict[str, Any] | None = None, think: bool | None = None) -> str
```

One reply; see `Ollama.chat`. `think` is accepted and not sent: there is no standard switch.

### `OpenAICompatible.embed`

```python
embed(texts: Sequence[str]) -> list[list[float]]
```

## Anthropic

```python
Anthropic(model: str | None = None, *, api_key: str | None = None, base_url: str | None = None, timeout: float | None = None, max_tokens: int | None = None) -> None
```

Chat through Anthropic's Messages API (the Claude models).

`api_key` defaults to $ANTHROPIC_API_KEY and `base_url` to
https://api.anthropic.com. Anthropic has no embedding model, so pair it
with a backend that has one:

```python
Combined(Anthropic("claude-sonnet-4-5"), Ollama(embed_model="embeddinggemma"))
```

`model` may be left out to call `list_models()` only. Replies are
capped at the model's own output limit (known once `list_models()` has
run), or ANTHROPIC_MAX_OUTPUT tokens, or `max_tokens` if you pass it.

### `Anthropic.name` (property)

### `Anthropic.embed_name` (property)

### `Anthropic.list_models`

```python
list_models() -> list[dict[str, Any]]
```

The models this key can use, as Anthropic lists them (newest first).

### `Anthropic.chat`

```python
chat(messages: Sequence[dict[str, Any]], *, json_mode: bool = False, temperature: float | None = None, on_token: Callable[[str, str], None] | None = None, stats: dict[str, Any] | None = None, think: bool | None = None) -> str
```

One reply; see `Ollama.chat`. `json_mode` and `think` are accepted: the prompts already ask
for JSON where it is needed, and extended thinking is left off.

### `Anthropic.embed`

```python
embed(texts: Sequence[str]) -> list[list[float]]
```

## Combined

```python
Combined(chat: Any, embed: Any) -> None
```

Chat through one backend and search principles through another.

For a chat model that has no embeddings (Anthropic), or to keep the
searching on a small local model:

```python
Combined(Anthropic("claude-sonnet-4-5"), Ollama(embed_model="embeddinggemma"))
```

`chat_backend` and `embed_backend` can be swapped at any time; the
Wisdom Core re-embeds its principles when the embedding model changes.

### `Combined.name` (property)

### `Combined.embed_name` (property)

### `Combined.context_window` (property)

### `Combined.chat`

```python
chat(messages: Sequence[dict[str, Any]], *, json_mode: bool = False, temperature: float | None = None, on_token: Callable[[str, str], None] | None = None, stats: dict[str, Any] | None = None, think: bool | None = None) -> str
```

### `Combined.embed`

```python
embed(texts: Sequence[str]) -> list[list[float]]
```

## FunctionModel

```python
FunctionModel(chat: Callable[..., str], embed: Callable[[list[str]], Sequence[Sequence[float]]], *, name: str = "custom", embed_name: str | None = None) -> None
```

Use your own functions as the model.

`chat(messages) -> str` receives OpenAI-style message dicts. If it also
accepts `json_mode`, `temperature`, `on_token`, `stats` or `think`
keywords (or **kwargs), those are passed along. `embed(texts) -> list of
vectors`. Change `embed_name` whenever you switch embedding models, so
stored principles are re-embedded in the new space.

### `FunctionModel.chat`

```python
chat(messages: Sequence[dict[str, Any]], *, json_mode: bool = False, temperature: float | None = None, on_token: Callable[[str, str], None] | None = None, stats: dict[str, Any] | None = None, think: bool | None = None) -> str
```

### `FunctionModel.embed`

```python
embed(texts: Sequence[str]) -> list[list[float]]
```

## WisdomCoreError

```python
class WisdomCoreError(Exception)
```

Base class for errors raised by wisdom_core.

## ModelError

```python
class ModelError(WisdomCoreError)
```

A model backend failed: it could not be reached, refused, or replied oddly.

**Attributes**

- `status`: the HTTP status from the server, when there was one
- `body`: the server's response body, when there was one

## check_embeddings

```python
check_embeddings(model: Any, levels: Levels | Mapping[str, Any] | None = None) -> dict[str, Any]
```

Measure an embedding model and work out the levels that fit it: its calibration.

One embedding request over a fixed set of 15 principles from different subjects: each beside the same
lesson in other words, beside a different lesson on the same subject, and beside a principle from another
subject, plus 15 questions each looked up among the 15 principles. Embedding models do not share a scale
(one scores a rewording 0.95, another 0.75), so the levels that compare principles are set from this.

- `separation` is "clear" when 90% of rewordings score above 90% of the related lessons, "partial" when
  three quarters of each are apart, and "none" otherwise.
- `calibrated` holds the measured `new_ground` and `combine` (the midpoint between related lessons
  and rewordings) and `covered` (where the closest rewordings sit), or None when there is no separation.
- `recall` says how many questions found their own principle first.
- `verdict` and `recall_verdict` say it in words.

`levels` (or the defaults) is only reported beside the result. `model` is a backend or an Ollama model
name. Nothing is stored.

## resolve_model

```python
resolve_model(model: Any = None) -> Any
```

Turn the `model` argument into a backend.

None means Ollama with automatic model choice; a string is an Ollama chat
model name; anything with `chat` and `embed` methods is used as is.

## fit_history

```python
fit_history(system: str, history: Sequence[dict[str, Any]], message: str, window: int | None) -> tuple[list[dict[str, Any]], int]
```

Drop the oldest messages until the prompt leaves room for a reply inside `window` tokens.

Only the conversation is trimmed. The system prompt (the Core Axiom and the
principles) and the new message always go in whole, so a long conversation
can never push the axiom out of the model's window. Returns the messages
kept and how many were dropped.

## context_window_of

```python
context_window_of(model: Any) -> int | None
```

The context window a backend sends with each request, when it says (Ollama does); else None.

## embed_prompts_for

```python
embed_prompts_for(model: Any) -> dict[str, str]
```

How an embedding model wants a search ("query"), a stored text ("document") and a plain comparison
("similarity") written, "{}" standing for the text: the backend's own `embed_prompts` if it has them,
else the first entry in `EMBED_PROMPTS` found in its embedding model's name. Empty: no markers needed.

## EMBED_PROMPTS

How an embedding model wants a search query, a stored text and a plain comparison written. The first family whose name appears anywhere in the embedding model's name wins. `{}` is the text. A backend can bring its own with an `embed_prompts` attribute.

| Family | Prompts |
|---|---|
| `embeddinggemma` | query: `task: search result | query: {}`; document: `title: none | text: {}`; similarity: `task: sentence similarity | query: {}` |
| `nomic-embed` | query: `search_query: {}`; document: `search_document: {}`; similarity: `clustering: {}` |
| `mxbai-embed` | query: `Represent this sentence for searching relevant passages: {}` |
| `snowflake-arctic-embed` | query: `Represent this sentence for searching relevant passages: {}` |
| `bge-m3` | (none) |
| `bge-` | query: `Represent this sentence for searching relevant passages: {}` |
| `multilingual-e5` | query: `query: {}`; document: `passage: {}`; similarity: `query: {}` |
| `e5-` | query: `query: {}`; document: `passage: {}`; similarity: `query: {}` |
| `qwen3-embedding` | query: `Instruct: Given a question, retrieve passages that answer it\nQuery: {}`; similarity: `Instruct: Retrieve text that says the same thing\nQuery: {}` |

## main

```python
main(argv: Sequence[str] | None = None) -> int
```

Chat with an agent that learns its own principles under a Core Axiom.

## __version__

`__version__ = '0.8.0'`

