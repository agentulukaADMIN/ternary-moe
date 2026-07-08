# Ternary BitNet MoE 🧮

**A 1GB mixture-of-experts AI that runs on an ordinary laptop — no GPU required.**

This project takes Microsoft's [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
language model — whose weights are *ternary*, meaning every weight is just
-1, 0, or +1 — and turns it into a tiny mixture-of-experts system:

- **One compressed base model** (1.03GB, TQ1_0 GGUF) loaded once.
- **Two small "expert" adapters** (55MB each) that specialize the model — one
  for multi-digit multiplication, one for Roman numerals.
- **A router** (a 2MB text classifier) that reads each question and switches
  on the right expert *at runtime* — the base model is never modified.

The whole stack replaces a ~5GB full-precision setup at about **1/5 the
memory**, and it answers a routed question end-to-end on a mid-range laptop
CPU.

## Results

Measured at temperature 0, exact match on the required
`The answer is X` line (details in [results.md](results.md)):

| task | this stack (ternary) | original bf16 | ternary base, no expert |
|---|---|---|---|
| 2-digit multiplication | **80%** | 94% | 70% |
| Roman numerals | **30%** | 24% | 5% |
| router picks the right expert | **100%** | — | — |

Two findings worth knowing if you try something similar:

1. **The quant format matters more than you'd think.** Both TQ1_0 and TQ2_0
   store the ternary weights *exactly*, yet TQ2_0 silently neutralized the
   weaker expert (5%, below the plain base) while TQ1_0 lifted it above its
   own full-precision score. The difference is in the runtime matmul kernels,
   not the stored weights.
2. **Never merge LoRA adapters into a BitNet base.** BitNet re-quantizes
   weights on every forward pass; a merged adapter is destroyed. Adapters here
   are applied as separate FP16 deltas at runtime, which also lets one loaded
   base serve any number of experts.

## How it works

```
                     ┌───────────────────────────────────────────┐
 "What is 34 x 57?"  │  router.joblib (MiniLM embedding +        │
 ────────────────►   │  logistic regression, runs outside        │
                     │  the model)                               │
                     └──────────────┬────────────────────────────┘
                                    │ label: "mult"
                                    ▼
                     POST /lora-adapters                          
                     [{"id":0,"scale":1.0},{"id":1,"scale":0.0}]  
                                    │
                                    ▼
                     ┌───────────────────────────────────────────┐
                     │  llama-server                             │
                     │  ┌─────────────────────────────────────┐  │
                     │  │ bitnet-2b-tq1_0.gguf (frozen,       │  │
                     │  │ ternary, loaded once)               │  │
                     │  └─────────────────────────────────────┘  │
                     │    + mult-f16.gguf   (active,  FP16 delta)│
                     │    + roman-f16.gguf  (dormant, scale 0)   │
                     └──────────────┬────────────────────────────┘
                                    ▼
                     "34 x 57 = ... The answer is 1938."
```

## What's in this repo

| file | purpose |
|---|---|
| `moe_driver.py` | interactive chat: routes each question and queries the server; answers stream in live, Ctrl+C cancels the current answer without quitting |
| `eval.py` | scoring harness (per-expert accuracy, coherence, routing accuracy) |
| `convert_lora_bitnet.py` | converts PEFT LoRA adapters to GGUF for this base (fixes two converter bugs) |
| `fabric-bitnet-fixes.patch` | **required** — 3 fixes to the inference engine (see below) |
| `data/` | held-out eval sets + their generator |
| `results.md` | full evaluation report |
| `README-run.md` | terse operator notes (this file is the friendly version) |
| `START HERE.bat` | one-click launcher once everything is installed |

The model files live on Hugging Face (too big for GitHub):
**[UlukaDev/bitnet-ternary-moe-gguf](https://huggingface.co/UlukaDev/bitnet-ternary-moe-gguf)**

## Setup from scratch (Windows)

You need: [Git](https://git-scm.com/download/win),
[CMake](https://cmake.org/download/), Visual Studio 2022 (Community is fine,
with "Desktop development with C++"), and [Python 3.12](https://www.python.org/downloads/).

```powershell
# 1. Get this repo and the inference engine
git clone https://github.com/agentulukaADMIN/ternary-moe
cd ternary-moe
git clone -c core.longpaths=true https://github.com/tetherto/qvac-fabric-llm.cpp fabric

# 2. Apply the required engine fixes and build (~10 min)
git -C fabric apply ..\fabric-bitnet-fixes.patch
cmake -S fabric -B fabric\build-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build fabric\build-cpu --config Release --target llama-cli llama-server -j 8

# 3. Get the models (~1.2GB) and Python packages
pip install huggingface_hub sentence-transformers scikit-learn joblib requests
hf download UlukaDev/bitnet-ternary-moe-gguf bitnet-2b-tq1_0.gguf --local-dir models
hf download UlukaDev/bitnet-ternary-moe-gguf mult-f16.gguf  --local-dir adapters
hf download UlukaDev/bitnet-ternary-moe-gguf roman-f16.gguf --local-dir adapters

# 4. Start the server
fabric\build-cpu\bin\Release\llama-server.exe -m models\bitnet-2b-tq1_0.gguf `
  --lora adapters\mult-f16.gguf --lora adapters\roman-f16.gguf `
  --lora-init-without-apply -c 4096 --port 8080 -t 10

# 5. In a second terminal: chat!
python moe_driver.py
```

Then ask things like:

```
> What is 47 times 62?
> Convert 1994 to Roman numerals
```

After the first setup, `START HERE.bat` does steps 4–5 with a double-click.

## The three engine fixes (why the patch exists)

The Fabric fork advertises BitNet support but was only ever validated on the
older 1bitLLM checkpoints. Running Microsoft's 2B-4T through it produced
degenerate output ("the the the...") until three bugs were fixed:

1. **Wrong FFN activation** — the graph hardcoded SiLU; this model uses
   squared ReLU (`hidden_act: "relu2"`).
2. **Double quantization in the converter** — weights were ternary-quantized
   a second time after the loader had already dequantized them, collapsing
   every tensor's scale to ±mean|w|.
3. **Inverted scale semantics** — transformers has two BitNet linear classes
   with *opposite* `weight_scale` conventions; this checkpoint's
   `autobitlinear` **multiplies** by the scale, the converter divided.
   (Found by diffing per-layer activations against a transformers forward
   pass: Q/K/V were off by exactly `weight_scale²`.)

## Reproduce the evaluation

```powershell
python eval.py --data data\mult_eval.jsonl  --adapter-id 0 --out mult_results.jsonl
python eval.py --data data\roman_eval.jsonl --adapter-id 1 --out roman_results.jsonl
python eval.py --data data\mult_eval.jsonl data\roman_eval.jsonl --route   # full stack
# --adapter-id -1 measures the plain base on the same items
```

## Credits & license

- Base model: [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) (MIT)
- Inference engine: [tetherto/qvac-fabric-llm.cpp](https://github.com/tetherto/qvac-fabric-llm.cpp) (MIT), a llama.cpp fork
- Router embedding: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Experts trained with [PEFT](https://github.com/huggingface/peft) LoRA on the bf16 base

This repo's code and adapters: MIT.
