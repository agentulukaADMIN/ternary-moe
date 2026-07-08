#!/usr/bin/env python3
"""Router-driven MoE driver: one ternary BitNet base + two hot-swapped LoRA experts.

The router (MiniLM embeddings + logistic regression, used AS-IS from
UlukaDev/bitnet-moe-router) picks an expert per query; the expert is activated
on the running llama-server via POST /lora-adapters (scale 1.0 on the chosen
adapter, 0.0 on the rest). The ternary base is never modified.

Answers stream in word by word. Ctrl+C cancels the current answer (not the
program); anything typed while the model is busy is discarded.
"""

import argparse
import json
import sys

import joblib
import requests
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

SERVER = "http://localhost:8080"
ROUTER = "UlukaDev/bitnet-moe-router"
CALC = ("You are a careful calculator. Work step by step, then end "
        "with exactly 'The answer is X'.")
# Server --lora load order: 0 = mult, 1 = roman.
# Keys match the raw labels router.joblib emits (verified live).
LABEL_TO_ID = {"mult": 0, "roman": 1}
ID_TO_NAME = {v: k for k, v in LABEL_TO_ID.items()}
SYSTEM = {0: CALC, 1: CALC}             # per-id system prompt; extend later

SERVER_DOWN_HINT = ("\nThe AI server isn't running. "
                    "Double-click START HERE.bat first, then try again.")

print("Loading router + embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
router = joblib.load(hf_hub_download(ROUTER, "router.joblib"))
DEBUG = False


def flush_typed_input() -> None:
    """Discard anything typed while the model was busy (Windows console)."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        pass


def route(q: str) -> str:
    # normalize_embeddings=True matches how the router was trained (router
    # repo moe.py); without it routing accuracy silently degrades.
    label = str(router.predict(embedder.encode([q], normalize_embeddings=True))[0])
    if DEBUG:
        print(f"[router] raw label: {label!r}")
    return label


def set_adapter(idx: int) -> None:
    scales = [{"id": i, "scale": 1.0 if i == idx else 0.0}
              for i in LABEL_TO_ID.values()]
    requests.post(f"{SERVER}/lora-adapters", json=scales, timeout=30).raise_for_status()


def answer(q: str) -> tuple[int, str]:
    """Route, activate the expert, and stream the answer to stdout."""
    print("thinking...", end="\r", flush=True)
    idx = LABEL_TO_ID.get(route(q), 0)
    set_adapter(idx)
    print(f"[expert id {idx} — {ID_TO_NAME[idx]}]")

    resp = requests.post(f"{SERVER}/v1/chat/completions", json={
        "messages": [{"role": "system", "content": SYSTEM[idx]},
                     {"role": "user",   "content": q}],
        "max_tokens": 384, "temperature": 0.0, "stream": True,
    }, stream=True, timeout=600)
    resp.raise_for_status()

    parts: list[str] = []
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload).get("choices", [{}])[0]
            delta = chunk.get("delta", {}).get("content")
            if delta:
                parts.append(delta)
                print(delta, end="", flush=True)
    finally:
        resp.close()
    print()
    return idx, "".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--debug", action="store_true",
                    help="print raw router labels")
    args = ap.parse_args()
    SERVER = args.server
    DEBUG = args.debug

    try:
        requests.get(f"{SERVER}/health", timeout=3).raise_for_status()
    except requests.exceptions.RequestException:
        print(SERVER_DOWN_HINT)
        sys.exit(1)

    import os
    os.system("")  # enables ANSI colors in the classic Windows console
    print("\n\033[92m● Ready!\033[0m  Ask a question, for example:")
    print("   What is 47 times 62?")
    print("   Convert 1994 to Roman numerals")
    print("Tip: press Ctrl+C to stop an answer early. "
          "Close the window when you are done.")

    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        try:
            answer(q)
        except KeyboardInterrupt:
            print("\n(answer cancelled — ask something else)")
        except requests.exceptions.ConnectionError:
            print(SERVER_DOWN_HINT)
            break
        finally:
            flush_typed_input()
