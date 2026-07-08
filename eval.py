#!/usr/bin/env python3
"""Evaluate the ternary-base + LoRA-expert stack against held-out sets.

Modes:
  --adapter-id N   force one adapter for every item (single-expert probe;
                   used for the Step-3 de-risk gate and per-expert scoring)
  --adapter-id -1  base model only (all adapter scales 0)
  --route          use the external router (router.joblib) per item and also
                   report routing accuracy vs the item's `domain` field

Scoring rules:
  - answer extracted from the LAST occurrence of "answer is X"
  - numeric normalization: "26" == "26.00" == "26,00 " (comma thousands
    stripped); non-numeric answers compared case-insensitively
  - coherence: an output is degenerate if it is empty, has no "answer is"
    tail AND ends in a short repeated loop, or is one long token repetition
  - truncation is tracked via finish_reason so real scores aren't suppressed
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

CALC = ("You are a careful calculator. Work step by step, then end "
        "with exactly 'The answer is X'.")

ANSWER_RE = re.compile(r"answer is[:\s]*\**([A-Za-z0-9,.\- ]+?)\**\s*(?:[.!\n]|$)",
                       re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def normalize(s: str) -> str:
    s = s.strip().strip(".").replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.upper().replace(" ", "")


def is_degenerate(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    tail = t[-200:]
    # a short chunk (1-8 chars) repeated 10+ times consecutively at the tail
    if re.search(r"(.{1,8})\1{9,}", tail, re.DOTALL):
        return True
    # whole output is very low diversity (e.g. "53535353...")
    if len(t) > 80 and len(set(t)) <= 4:
        return True
    return False


def set_adapter(server: str, idx: int, n_adapters: int, scale: float = 1.0) -> None:
    scales = [{"id": i, "scale": scale if i == idx else 0.0}
              for i in range(n_adapters)]
    requests.post(f"{server}/lora-adapters", json=scales, timeout=30).raise_for_status()


def chat(server: str, question: str, max_tokens: int) -> tuple[str, str]:
    r = requests.post(f"{server}/v1/chat/completions", json={
        "messages": [{"role": "system", "content": CALC},
                     {"role": "user", "content": question}],
        "max_tokens": max_tokens, "temperature": 0.0}, timeout=600)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason", "?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--data", required=True, nargs="+",
                    help="jsonl file(s) with {question, answer, domain}")
    ap.add_argument("--adapter-id", type=int, default=None,
                    help="fixed adapter id; -1 = base only (all scales 0)")
    ap.add_argument("--route", action="store_true",
                    help="use router.joblib per item")
    ap.add_argument("--n-adapters", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="application scale for the active adapter")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write per-item results jsonl")
    args = ap.parse_args()

    items = []
    for path in args.data:
        with open(path, encoding="utf-8") as f:
            items += [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[:args.limit]

    router = embedder = None
    label_to_id = {"mult": 0, "roman": 1}
    if args.route:
        import joblib
        from huggingface_hub import hf_hub_download
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        router = joblib.load(hf_hub_download("UlukaDev/bitnet-moe-router",
                                             "router.joblib"))

    stats: dict[str, dict[str, int]] = {}
    route_hits = 0
    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        domain = item.get("domain", "?")
        stats.setdefault(domain, {"n": 0, "correct": 0, "degenerate": 0,
                                  "no_tail": 0, "truncated": 0})
        s = stats[domain]
        s["n"] += 1

        if args.route:
            emb = embedder.encode([item["question"]], normalize_embeddings=True)
            label = str(router.predict(emb)[0])
            idx = label_to_id.get(label, 0)
            if label == domain:
                route_hits += 1
            set_adapter(args.server, idx, args.n_adapters)
        elif args.adapter_id is not None:
            set_adapter(args.server, args.adapter_id, args.n_adapters, args.scale)
            idx = args.adapter_id

        text, finish = chat(args.server, item["question"], args.max_tokens)

        pred = extract_answer(text)
        correct = pred is not None and normalize(pred) == normalize(item["answer"])
        degen = is_degenerate(text)
        s["correct"] += correct
        s["degenerate"] += degen
        s["no_tail"] += pred is None
        s["truncated"] += finish == "length"
        results.append({**item, "prediction": pred, "correct": correct,
                        "degenerate": degen, "finish_reason": finish,
                        "adapter_id": idx, "output": text})
        done = i + 1
        if done % 20 == 0 or done == len(items):
            print(f"  [{done}/{len(items)}] "
                  + " | ".join(f"{d}: {v['correct']}/{v['n']}"
                               for d, v in stats.items()),
                  flush=True)

    dt = time.time() - t0
    print(f"\n=== results ({dt:.0f}s, {dt / max(len(items), 1):.1f}s/item) ===")
    for d, v in stats.items():
        acc = v["correct"] / v["n"]
        print(f"{d:8s} acc={acc:.3f} ({v['correct']}/{v['n']})  "
              f"degenerate={v['degenerate']}  no_answer_tail={v['no_tail']}  "
              f"truncated={v['truncated']}")
    if args.route:
        print(f"routing  acc={route_hits / len(items):.3f} "
              f"({route_hits}/{len(items)})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"per-item results -> {args.out}")


if __name__ == "__main__":
    main()
