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

# bitnet-ternary-moe-gguf

A laptop-scale ternary Mixture-of-Experts stack: one 1.58-bit BitNet base
(TQ1_0 GGUF, 1.03GB) + two FP16 LoRA expert adapters hot-swapped per query at
runtime (never merged), routed by an external MiniLM + logistic-regression
router ([UlukaDev/bitnet-moe-router](https://huggingface.co/UlukaDev/bitnet-moe-router)).
Replaces a ~5GB bf16 runtime at roughly 1/5 the memory.

## Files

| file | what |
|---|---|
| `bitnet-2b-tq1_0.gguf` | microsoft/bitnet-b1.58-2B-4T converted to ternary TQ1_0 |
| `mult-f16.gguf` | 2-digit multiplication expert (LoRA r=32 α=64, FFN modules only) |
| `roman-f16.gguf` | Roman numerals expert (same shape) |
| `fabric-bitnet-fixes.patch` | 3 required fixes to tetherto/qvac-fabric-llm.cpp (see below) |
| `results.md` | full evaluation report |

## Results (temperature 0, exact-match on "The answer is X")

| | this stack (ternary) | bf16 original | ternary base alone |
|---|---|---|---|
| multiplication | **0.80** | 0.94 | 0.70 |
| roman numerals | **0.30** | 0.24 | 0.05 |
| routing accuracy | **1.00** | — | — |

TQ1_0 notably outperformed TQ2_0 here (TQ2_0 neutralized the roman adapter:
0.05 vs base 0.10). Both formats store the ternary weights exactly, so this
is a runtime kernel effect.

## Run

Requires [tetherto/qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp)
**with `fabric-bitnet-fixes.patch` applied** (wrong FFN activation, double
ternary-quantization in the converter, and inverted AutoBitLinear
weight_scale semantics — without these the Microsoft 2B-4T model degenerates
to repeating tokens).

```
llama-server -m bitnet-2b-tq1_0.gguf \
  --lora mult-f16.gguf --lora roman-f16.gguf \
  --lora-init-without-apply -c 4096 --port 8080
# activate one adapter per request:
# POST /lora-adapters  [{"id":0,"scale":1.0},{"id":1,"scale":0.0}]
```

**Never merge these adapters into the base** — BitNet re-quantizes weights
every forward pass; merging corrupts it. Runtime deltas only.

System prompt used by both experts:
`You are a careful calculator. Work step by step, then end with exactly 'The answer is X'.`
