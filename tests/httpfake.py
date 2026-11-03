"""A small HTTP server that plays Ollama or an OpenAI-compatible API."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from tests.fakes import ScriptedModel, bow_embed

Route = Callable[[dict[str, Any], dict[str, str]], "tuple[int, Any]"]


class Stream:
    """A streamed reply: JSON lines (Ollama) or server-sent events (OpenAI, Anthropic), written one by one.

    An item given as (event, data) is sent with an ``event:`` line first, as Anthropic does.
    ``done`` ends an SSE stream with OpenAI's [DONE] marker.
    """

    def __init__(self, items: list[Any], *, sse: bool = False, delay: float = 0.0, done: bool = True) -> None:
        self.items = items
        self.sse = sse
        self.delay = delay
        self.done = done


class FakeServer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, str], Any]] = []
        self.routes: dict[tuple[str, str], Route] = {}
        self.streamed = 0          # stream items written
        self.disconnects = 0       # streams the client walked away from
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output clean
                pass

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw.decode("utf-8", "replace")
                headers = {k.lower(): v for k, v in self.headers.items()}
                owner.requests.append((method, self.path, headers, body))
                path, _, query = self.path.partition("?")
                if query and body is None:
                    body = dict(urllib.parse.parse_qsl(query))     # a GET's query reaches the route as its body
                route = owner.routes.get((method, path))
                if route is None:
                    status, payload = 404, "404 page not found"
                else:
                    try:
                        status, payload = route(body, headers)
                    except Exception as exc:  # a scripted failure becomes a server error
                        status, payload = 500, {"error": str(exc)}
                if isinstance(payload, Stream):
                    self._stream(status, payload)
                    return
                data = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
                data = data.encode() if isinstance(data, str) else data
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, status: int, stream: Stream) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if stream.sse else "application/x-ndjson")
                self.end_headers()        # HTTP/1.0: the body runs until the connection closes
                try:
                    for item in stream.items:
                        event = None
                        if isinstance(item, tuple):
                            event, item = item
                        text = item if isinstance(item, str) else json.dumps(item)
                        line = f"data: {text}\n\n" if stream.sse else text + "\n"
                        if event:
                            line = f"event: {event}\n" + line
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                        owner.streamed += 1
                        if stream.delay:
                            time.sleep(stream.delay)
                    if stream.sse and stream.done:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    owner.disconnects += 1

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def paths(self) -> list[str]:
        return [path for _, path, _, _ in self.requests]

    def bodies(self, path: str) -> list[Any]:
        return [body for _, p, _, body in self.requests if p == path]


OLLAMA_MODELS = [
    {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert", "families": ["nomic-bert"]}},
    {"name": "llama3.2:latest", "details": {"family": "llama", "families": ["llama"]}},
]


def pieces(text: str, size: int = 7) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def ollama_server(brain: ScriptedModel | None = None, *, models: list[dict[str, Any]] | None = None,
                  capabilities: bool = True, new_embed_api: bool = True, context_length: int | None = None,
                  split_thinking: bool = False, reject_think: bool = False, starve: bool = False,
                  delay: float = 0.0) -> tuple[FakeServer, ScriptedModel]:
    """A fake Ollama whose chat replies come from a ScriptedModel.

    split_thinking: a reply's leading <think>...</think> is sent as the separate thinking field.
    reject_think:   the server answers 400 to any request carrying "think" (a model with no switch).
    starve:         unless think is false, the model thinks and writes no answer.
    """
    brain = brain or ScriptedModel()
    server = FakeServer()
    listed = OLLAMA_MODELS if models is None else models
    names = {m["name"] for m in listed} | {m["name"].split(":")[0] for m in listed}

    def tags(body, headers):
        return 200, {"models": listed}

    def show(body, headers):
        info: dict[str, Any] = {}
        if context_length:
            info["model_info"] = {"general.architecture": "gemma3", "gemma3.context_length": context_length}
        if not capabilities:
            return 200, {"details": {}, **info}
        name = body["model"]
        embed = "embed" in name or "bge" in name
        return 200, {"capabilities": ["embedding"] if embed else ["completion", "tools"], **info}

    def chat(body, headers):
        if body["model"] not in names:
            return 404, {"error": f"model \"{body['model']}\" not found, try pulling it first"}
        if reject_think and "think" in body:
            return 400, {"error": f"\"{body['model']}\" does not support thinking"}
        options = body.get("options") or {}
        reply = brain.chat(body["messages"], json_mode=body.get("format") == "json",
                           temperature=options.get("temperature"))
        thinking, content = "", reply
        if split_thinking and reply.startswith("<think>") and "</think>" in reply:
            thinking, content = reply[7:].split("</think>", 1)
        if starve and body.get("think") is not False:
            thinking, content = (thinking or "Let me think about this at length..."), ""
        final = {"model": body["model"], "done": True, "done_reason": "stop",
                 "prompt_eval_count": sum(len(m["content"]) for m in body["messages"]) // 4,
                 "eval_count": max(len(content) // 4, 1)}
        if not body.get("stream"):
            message = {"role": "assistant", "content": content}
            if thinking:
                message["thinking"] = thinking
            return 200, {**final, "message": message}
        items: list[Any] = [{"model": body["model"], "done": False,
                             "message": {"role": "assistant", "content": "", "thinking": p}}
                            for p in (pieces(thinking) if thinking else [])]
        items += [{"model": body["model"], "done": False, "message": {"role": "assistant", "content": p}}
                  for p in pieces(content)]
        items.append({**final, "message": {"role": "assistant", "content": ""}})
        return 200, Stream(items, delay=delay)

    def embed(body, headers):
        if body["model"] not in names:
            return 404, {"error": f"model \"{body['model']}\" not found, try pulling it first"}
        return 200, {"model": body["model"], "embeddings": brain.embed(body["input"])}

    def embeddings(body, headers):
        return 200, {"embedding": bow_embed([body["prompt"]], brain.dims)[0]}

    server.routes[("GET", "/api/tags")] = tags
    server.routes[("POST", "/api/show")] = show
    server.routes[("POST", "/api/chat")] = chat
    if new_embed_api:
        server.routes[("POST", "/api/embed")] = embed
    server.routes[("POST", "/api/embeddings")] = embeddings
    return server, brain


def openai_server(brain: ScriptedModel | None = None, *, delay: float = 0.0, models: list[str] | None = None,
                  key: str | None = None, usage_needs_option: bool = False,
                  reject_stream_options: bool = False) -> tuple[FakeServer, ScriptedModel]:
    """A fake OpenAI-compatible server.

    key:                   when set, requests without "Authorization: Bearer <key>" get a 401.
    usage_needs_option:    token counts are streamed only when stream_options.include_usage asks (as OpenAI).
    reject_stream_options: a 400 for any request carrying stream_options (servers that do not know it).
    """
    brain = brain or ScriptedModel()
    server = FakeServer()
    listed = ["m", "gpt-x", "text-embedding-3-small"] if models is None else models

    def refused(headers):
        if key is not None and headers.get("authorization") != f"Bearer {key}":
            return 401, {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}
        return None

    def list_models(body, headers):
        return refused(headers) or (200, {"object": "list", "data": [{"id": m, "object": "model"} for m in listed]})

    def chat(body, headers):
        no = refused(headers)
        if no:
            return no
        if reject_stream_options and "stream_options" in body:
            return 400, {"error": {"message": "Unrecognized request argument supplied: stream_options"}}
        content = brain.chat(body["messages"], json_mode="response_format" in body,
                             temperature=body.get("temperature"))
        if not body.get("stream"):
            return 200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
        thinking = ""
        if content.startswith("<think>") and "</think>" in content:
            thinking, content = content[7:].split("</think>", 1)
        items: list[Any] = [{"choices": [{"index": 0, "delta": {"reasoning_content": p}}]} for p in
                            (pieces(thinking) if thinking else [])]
        items += [{"choices": [{"index": 0, "delta": {"content": p}}]} for p in pieces(content)]
        usage = {"prompt_tokens": 11, "completion_tokens": 7}
        if usage_needs_option:
            items.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                items.append({"choices": [], "usage": usage})
        else:
            items.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
        return 200, Stream(items, sse=True, delay=delay)

    def embeddings(body, headers):
        vecs = brain.embed(body["input"])
        data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]
        return 200, {"data": list(reversed(data))}      # out of order on purpose

    server.routes[("POST", "/v1/chat/completions")] = chat
    server.routes[("POST", "/v1/embeddings")] = embeddings
    server.routes[("GET", "/v1/models")] = list_models
    return server, brain


ANTHROPIC_MODELS = [
    {"id": "claude-new-5", "display_name": "Claude New 5", "max_tokens": 64000},
    {"id": "claude-mid-4", "display_name": "Claude Mid 4", "max_tokens": 32000},
    {"id": "claude-small-3", "display_name": "Claude Small 3", "max_tokens": 4096},
]


def anthropic_server(brain: ScriptedModel | None = None, *, delay: float = 0.0, key: str = "sk-ant-test",
                     models: list[dict[str, Any]] | None = None, page: int = 1000,
                     reject_temperature: bool = False, max_output: int | None = None,
                     stop_reason: str = "end_turn") -> tuple[FakeServer, ScriptedModel]:
    """A fake of Anthropic's Messages API, strict the way the real one is.

    It refuses a wrong key or version, a missing max_tokens, empty or out-of-order turns;
    ``reject_temperature`` plays a newer model that takes no temperature, ``max_output`` a
    model with a small output limit, and ``page`` the page size of the model list.
    """
    brain = brain or ScriptedModel()
    server = FakeServer()
    listed = ANTHROPIC_MODELS if models is None else models

    def error(status: int, kind: str, message: str):
        return status, {"type": "error", "error": {"type": kind, "message": message}}

    def refused(headers):
        if headers.get("x-api-key") != key:
            return error(401, "authentication_error", "invalid x-api-key")
        if headers.get("anthropic-version") != "2023-06-01":
            return error(400, "invalid_request_error", "anthropic-version header is required")
        return None

    def list_models(body, headers):
        no = refused(headers)
        if no:
            return no
        after = (body or {}).get("after_id")
        start = next((i + 1 for i, m in enumerate(listed) if m["id"] == after), 0) if after else 0
        chunk = listed[start:start + page]
        return 200, {"data": [{"type": "model", **m} for m in chunk], "has_more": start + page < len(listed),
                     "first_id": chunk[0]["id"] if chunk else None, "last_id": chunk[-1]["id"] if chunk else None}

    def messages(body, headers):
        no = refused(headers)
        if no:
            return no
        if not isinstance(body.get("max_tokens"), int):
            return error(400, "invalid_request_error", "max_tokens: Field required")
        if max_output is not None and body["max_tokens"] > max_output:
            return error(400, "invalid_request_error",
                         f"max_tokens: {body['max_tokens']} > {max_output}, which is the maximum allowed")
        if reject_temperature and "temperature" in body:
            return error(400, "invalid_request_error", "temperature is deprecated for this model")
        turns = body.get("messages") or []
        if not turns or turns[0].get("role") != "user":
            return error(400, "invalid_request_error", "messages: first message must use the user role")
        if any(not str(t.get("content") or "").strip() for t in turns):
            return error(400, "invalid_request_error", "messages: text content blocks must be non-empty")
        if any(t.get("role") == "system" for t in turns):
            return error(400, "invalid_request_error", "messages: system is not a valid role here")
        as_openai = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + turns
        reply = brain.chat(as_openai, json_mode=False, temperature=body.get("temperature"))
        thinking = ""
        if reply.startswith("<think>") and "</think>" in reply:
            thinking, reply = reply[7:].split("</think>", 1)
        usage_in, usage_out = 21, max(1, len(reply) // 4)
        if not body.get("stream"):
            blocks = ([{"type": "thinking", "thinking": thinking, "signature": "sig"}] if thinking else [])
            blocks.append({"type": "text", "text": reply})
            return 200, {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                         "content": blocks, "stop_reason": stop_reason, "stop_sequence": None,
                         "usage": {"input_tokens": usage_in, "output_tokens": usage_out}}
        items: list[Any] = [("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": body["model"],
            "stop_reason": None, "usage": {"input_tokens": usage_in, "output_tokens": 1}}})]
        index = 0
        if thinking:
            items.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                  "content_block": {"type": "thinking", "thinking": ""}}))
            items += [("content_block_delta", {"type": "content_block_delta", "index": index,
                                               "delta": {"type": "thinking_delta", "thinking": p}})
                      for p in pieces(thinking)]
            items.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
            index += 1
        items.append(("content_block_start", {"type": "content_block_start", "index": index,
                                              "content_block": {"type": "text", "text": ""}}))
        items.append(("ping", {"type": "ping"}))
        items += [("content_block_delta", {"type": "content_block_delta", "index": index,
                                           "delta": {"type": "text_delta", "text": p}}) for p in pieces(reply)]
        items.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
        items.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason,
                                                                             "stop_sequence": None},
                                        "usage": {"output_tokens": usage_out}}))
        items.append(("message_stop", {"type": "message_stop"}))
        return 200, Stream(items, sse=True, delay=delay, done=False)

    server.routes[("GET", "/v1/models")] = list_models
    server.routes[("POST", "/v1/messages")] = messages
    return server, brain
