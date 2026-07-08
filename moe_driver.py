#!/usr/bin/env python3
"""Router-driven MoE driver: one ternary BitNet base + two hot-swapped LoRA experts.

The router (MiniLM embeddings + logistic regression, used AS-IS from
UlukaDev/bitnet-moe-router) picks an expert per query; the expert is activated
on the running llama-server via POST /lora-adapters (scale 1.0 on the chosen
adapter, 0.0 on the rest). The ternary base is never modified.
"""

import argparse

import joblib
import requests
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

SERVER = "http://localhost:8080"
ROUTER = "UlukaDev/bitnet-moe-router"
CALC = ("You are a careful calculator. Work step by step, then end "
        "with exactly 'The answer is X'.")
# Server --lora load order: 0 = mult, 1 = roman.
# NOTE: keys must match the raw labels router.joblib emits (verified live
# with --debug; fix here if the router was trained with different strings).
LABEL_TO_ID = {"mult": 0, "roman": 1}
SYSTEM = {0: CALC, 1: CALC}             # per-id system prompt; extend later

print("Loading router + embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
router = joblib.load(hf_hub_download(ROUTER, "router.joblib"))
DEBUG = False


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
    requests.post(f"{SERVER}/lora-adapters", json=scales).raise_for_status()


def answer(q: str) -> tuple[int, str]:
    idx = LABEL_TO_ID.get(route(q), 0)
    set_adapter(idx)
    r = requests.post(f"{SERVER}/v1/chat/completions", json={
        "messages": [{"role": "system", "content": SYSTEM[idx]},
                     {"role": "user",   "content": q}],
        "max_tokens": 384, "temperature": 0.0})
    r.raise_for_status()
    return idx, r.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--debug", action="store_true",
                    help="print raw router labels")
    args = ap.parse_args()
    SERVER = args.server
    DEBUG = args.debug

    while True:
        try:
            q = input("\n> ").strip()
            if not q:
                continue
            idx, txt = answer(q)
            print(f"[expert id {idx}]\n{txt}")
        except (KeyboardInterrupt, EOFError):
            break
