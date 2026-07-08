---
license: mit
base_model: microsoft/bitnet-b1.58-2B-4T
tags:
- bitnet
- ternary
- gguf
- lora
- mixture-of-experts
---

# bitnet-ternary-moe-gguf 🧮

**A 1GB mixture-of-experts AI that runs on an ordinary laptop CPU — no GPU
required.**

One ternary (1.58-bit) BitNet base model + two small LoRA expert adapters
(multiplication and Roman numerals) that are switched on **at runtime**, per
question, by an external router — the base is loaded once and never modified.
Replaces a ~5GB bf16 setup at about 1/5 the memory.

➡️ **Code, launcher, eval harness and full setup guide:**
[github.com/agentulukaADMIN/ternary-moe](https://github.com/agentulukaADMIN/ternary-moe)

## Files in this repo

| file | size | what it is |
|---|---|---|
| `bitnet-2b-tq1_0.gguf` | 1.03GB | [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) converted to ternary **TQ1_0** — all transformer matrices are ternary |
| `mult-f16.gguf` | 55MB | 2-digit multiplication expert (LoRA r=32 α=64, FFN modules only) |
| `roman-f16.gguf` | 55MB | Roman numerals expert (same shape) |
| `fabric-bitnet-fixes.patch` | 3KB | **required** engine fixes (see below) |
| `results.md` | — | full evaluation report |

## Results

Temperature 0, exact match on the required `The answer is X` line:

| task | this stack (ternary) | original bf16 | ternary base, no expert |
|---|---|---|---|
| 2-digit multiplication | **0.80** | 0.94 | 0.70 |
| Roman numerals | **0.30** | 0.24 | 0.05 |
| routing accuracy | **1.00** | — | — |

**Use TQ1_0, not TQ2_0.** Both formats store ternary weights exactly, but the
TQ2_0 runtime kernels neutralized the weaker expert (0.05, *below* the plain
base) while TQ1_0 lifted it above its own bf16 score. If your BitNet + LoRA
stack underperforms, try the other ternary format before anything else.

## Quickstart

Needs [tetherto/qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp)
built **with `fabric-bitnet-fixes.patch` applied** — the stock fork produces
repeating garbage with this model (wrong FFN activation, a double-quantization
bug, and inverted `autobitlinear` weight_scale semantics).

```bash
# serve: one base, both experts loaded but dormant
llama-server -m bitnet-2b-tq1_0.gguf \
  --lora mult-f16.gguf --lora roman-f16.gguf \
  --lora-init-without-apply -c 4096 --port 8080

# activate exactly one expert (id 0 = mult, id 1 = roman):
curl -X POST http://localhost:8080/lora-adapters \
  -d '[{"id":0,"scale":1.0},{"id":1,"scale":0.0}]'

# then use the normal OpenAI-style chat endpoint:
curl -X POST http://localhost:8080/v1/chat/completions -d '{
  "messages": [
    {"role":"system","content":"You are a careful calculator. Work step by step, then end with exactly '\''The answer is X'\''."},
    {"role":"user","content":"What is 34 times 57?"}
  ], "max_tokens": 384, "temperature": 0}'
```

For automatic routing (the router reads the question and picks the expert),
use `moe_driver.py` from the GitHub repo together with
[UlukaDev/bitnet-moe-router](https://huggingface.co/UlukaDev/bitnet-moe-router).
Note the router expects `normalize_embeddings=True` when encoding.

## ⚠️ Never merge these adapters into the base

BitNet re-quantizes its weights on every forward pass — merging a LoRA into
the base destroys it (output degenerates to repeating tokens). Always apply
the adapters as runtime deltas, as shown above. This is also what lets one
loaded base serve any number of experts.

## Credits

Base model by Microsoft Research (MIT). Engine: Tether's QVAC Fabric
llama.cpp fork (MIT). Router embedding: all-MiniLM-L6-v2. Experts trained
with PEFT LoRA on the bf16 base. Code and adapters: MIT.
