"""Watch the Wisdom Core learn.

Runs a short conversation against your own model and prints what the Wisdom
Core does after each task, then the principles it ended up with.

    python examples/demo.py
    python examples/demo.py --model llama3.1:8b --embed-model nomic-embed-text
    python examples/demo.py --openai-base-url http://localhost:1234/v1 --model qwen2.5-7b-instruct --embed-model text-embedding-nomic-embed-text-v1.5

It uses a fresh temporary database, so it never touches your own agents.
Add --keep to keep that database afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import wisdom_core as wc  # noqa: E402

AXIOM = "Help people make sound money decisions."

TASKS = [
    "I have a car loan at 6% and a credit card at 24%, and $200 extra each month. Which should I pay off first?",
    "Can you add up my monthly bills? Rent 1,250, car 390, phone 85, power 140.",
    "My friend says you should always pay the smallest balance first, whatever the rate. Is he right?",
    "Thanks, that's all for now!",
    "I have $5,000 saved. A coworker says to put all of it into one stock he likes. Should I?",
    "I also have a student loan at 4%. Where does it fit among my debts?",
    "Earlier you said to pay the highest rate first. Is that always the right call?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model")
    parser.add_argument("--embed-model")
    parser.add_argument("--host", help="Ollama address")
    parser.add_argument("--openai-base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--keep", action="store_true", help="keep the demo database")
    args = parser.parse_args()

    if args.openai_base_url:
        model = wc.OpenAICompatible(args.model, args.embed_model, base_url=args.openai_base_url, api_key=args.api_key)
    else:
        model = wc.Ollama(args.model, args.embed_model, host=args.host)

    folder = tempfile.mkdtemp(prefix="wisdom-demo-")
    path = os.path.join(folder, "demo.db")
    failures = 0
    with wc.Agent(AXIOM, model, path=path) as agent:
        try:
            print(f"Models: {agent.core.model.name} (chat), {wc._embed_name(agent.core.model)} (embeddings)")
        except wc.ModelError as exc:
            print(f"error: {exc}")
            return 1
        print(f"Core Axiom: {AXIOM}\n")
        for n, task in enumerate(TASKS, 1):
            start = time.monotonic()
            reply = agent.ask(task)
            took = time.monotonic() - start
            print(f"[{n}/{len(TASKS)}] you> {task}")
            print(f"       agent> {wc._clip(reply, 400)}")
            if agent.last_principles:
                print(f"       ({len(agent.last_principles)} principle(s) injected)")
            for line in agent.last_report.lines():
                print(f"       wisdom core {line}")
            print(f"       ({took:.0f}s)\n")
            failures += bool(agent.last_report.error)

        print("PRINCIPLES AT THE END")
        for i, p in enumerate(agent.core.principles(), 1):
            print(f"{i:>3}. {p.render()}")
        print("\nSTATS", ", ".join(f"{k}: {v}" for k, v in agent.core.stats().items()))
        print(f"\nLearning failed on {failures} of {len(TASKS)} tasks.")
    if args.keep:
        print(f"Database kept at {path}  (python -m wisdom_core --db {path} --history)")
    else:
        os.remove(path)
        os.rmdir(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
