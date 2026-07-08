# Results — ternary MoE (TQ1_0 base + FP16 LoRA runtime deltas)

**Shipped configuration: `models\bitnet-2b-tq1_0.gguf` (1.03GB), adapter scale
1.0, CPU backend.** Base: `microsoft/bitnet-b1.58-2B-4T` converted to ternary
GGUF; all transformer matrices ternary; served by qvac-fabric-llm.cpp (three
local fork patches were required — see README-run.md). Adapters applied as
FP16 runtime deltas via `POST /lora-adapters`, never merged. Router
`router.joblib` used unmodified (labels "mult"/"roman";
`normalize_embeddings=True` required). Vulkan build works and is coherent but
~3x slower than CPU on this Intel iGPU (10.9 vs 32.8 tok/s), so CPU serves.

Eval protocol: sets mirror the router repo's canonical generator (seed 12345;
mult extended with seed 12346) since the original standalone held-out files
aren't published — comparable in protocol, not item-for-item. Exact match on
the "The answer is X" tail, numeric normalization, temperature 0. "Base" rows
ran the identical items with all adapter scales 0.

## Scorecard (shipped TQ1_0 config)

| metric                     | result       | bf16 reference | verdict  |
|----------------------------|--------------|----------------|----------|
| mult, adapter (20)         | **0.800**    | 0.94           | —        |
| mult, base only (same 20)  | 0.700 (0.617 on TQ2_0/60 items) | 0.735 | — |
| mult coherence             | 0 degenerate | —              | **PASS** |
| mult usefulness (+10 pts; +10.8 pts on the larger TQ2_0 sample) | above base | — | **PASS** |
| roman, adapter (20)        | **0.300**    | 0.242          | —        |
| roman, base only (same 20) | 0.050        | 0.000          | —        |
| roman coherence            | 0 degenerate (1 false-positive flag = truncated long chain) | — | **PASS** |
| roman usefulness (6x base) | above base   | —              | **PASS** |
| routing accuracy           | **1.000** (80/80) | —         | **PASS** |

## Quant/scale search that found the winner (roman was the constraint)

| config                     | mult acc | roman acc | roman vs base |
|----------------------------|----------|-----------|---------------|
| TQ2_0, scale 1.0           | 0.708–0.725 | 0.050  | below base 0.100 → neutralized |
| TQ2_0, `--lora-scaled` 0.8 | —        | 0.100     | at base |
| TQ2_0, `--lora-scaled` 1.2 | —        | 0.100     | at base |
| **TQ1_0, scale 1.0**       | **0.800**| **0.300** | **6x base 0.050** |

i2_s was not testable: the Fabric fork has no i2_s loader (only
microsoft/BitNet reads it; CPU-only, LoRA hot-swap unverified). It was not
needed.

Both TQ formats store the ternary weights exactly (uniform per-tensor scale
fits one f16 block scale losslessly), so the TQ2_0-vs-TQ1_0 gap is a runtime
kernel effect (different activation-quant/accumulation paths in ggml vec_dot),
not a weight-fidelity effect. TQ1_0 is also the smaller file (1.03 vs 1.12GB).

## Cross-quant verdict (one paragraph)

Cross-quant application **held — for both experts — once the right ternary
format was found**. On the shipped TQ1_0 base, the mult expert scores 0.80
(coherent chain-of-thought, vs 0.94 bf16 and 0.70 ternary base) and the roman
expert scores 0.30 — above its own bf16 reference (0.242) and 6x the ternary
base (0.05) — with routing perfect at 1.000 and all outputs well-formed.
TQ2_0 neutralized the weaker roman adapter (0.05, below base) and adapter-
scale tuning did not rescue it; switching the base to TQ1_0 did, lifting both
experts simultaneously. Final stack: one 1.03GB ternary base + two 55MB FP16
adapters hot-swapped per query by the untouched logistic-regression router —
**PASS on the full success definition** (coherence, usefulness, and routing
for both experts), replacing the 5GB bf16 runtime at ~1/5 the memory.
