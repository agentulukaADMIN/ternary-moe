# Ternary BitNet MoE — build & run

One packed 1.58-bit ternary base (`microsoft/bitnet-b1.58-2B-4T` → **TQ1_0**
GGUF, 1.03GB — TQ1_0 outscored TQ2_0 for both experts, see results.md) served
by llama-server from the Tether QVAC Fabric fork, plus two GGUF LoRA experts
applied as FP16 runtime deltas (never merged), selected per query by an
external logistic-regression router.

## Layout

```
fabric\                     qvac-fabric-llm.cpp clone (with 3 local patches, see below)
fabric\build-cpu\           CPU build (build-vulkan\ once the Vulkan SDK is installed)
models\bitnet-2b-tq1_0.gguf ternary base (SHIPPED — all transformer mats ternary)
models\bitnet-2b-tq2_0.gguf alternate ternary base (works; weaker with the roman expert)
adapters\mult-f16.gguf      2-digit multiplication expert (LoRA r=32 a=64, FFN only)
adapters\roman-f16.gguf     roman numerals expert (same shape)
router\router.joblib        MiniLM + logistic regression router (used AS-IS)
data\mult_eval.jsonl        200 held-out items (seed 12345 mirror + seed 12346 ext)
data\roman_eval.jsonl       100 held-out items (seed 12345 mirror)
convert_lora_bitnet.py      patched PEFT->GGUF LoRA converter
moe_driver.py               interactive router-driven driver
eval.py                     eval harness (single-adapter, base-only, or routed)
```

## Local patches to the Fabric fork (REQUIRED)

The fork was validated on 1bitLLM BitNet only; three fixes were needed for
microsoft/bitnet-b1.58-2B-4T (without them the model outputs repeating
garbage):

1. `fabric\src\models\bitnet.cpp`: FFN activation `LLM_FFN_SILU` →
   `LLM_FFN_RELU_SQR` (this model uses squared ReLU). Needs rebuild.
2. `fabric\convert_hf_to_gguf.py` (`BitnetModel.modify_tensors`): skip
   `weight_quant()` for checkpoints with `quantization_config.quant_method ==
   "bitnet"` — the loader has already dequantized them; re-quantizing
   collapses every tensor to ±mean|w|.
3. `fabric\convert_hf_to_gguf.py` (`dequant_bitnet`): for
   `linear_class == "autobitlinear"` the effective weight is
   `ternary * weight_scale` (transformers' AutoBitLinear multiplies by the
   scale; only the packed BitLinear class divides).

## Build

```powershell
# CPU build (works today)
cmake -S fabric -B fabric\build-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build fabric\build-cpu --config Release --target llama-cli llama-server -j 8

# Vulkan build (after: winget install --id KhronosGroup.VulkanSDK -e  + UAC + new shell)
cmake -S fabric -B fabric\build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build fabric\build-vulkan --config Release --target llama-cli llama-server -j 8
# then add -ngl 99 to the server command line
```

## Convert (already done, for reproducibility)

```powershell
hf download microsoft/bitnet-b1.58-2B-4T --local-dir models\bitnet-2b-4t-packed
python fabric\convert_hf_to_gguf.py models\bitnet-2b-4t-packed --outtype tq2_0 --no-lazy `
    --outfile models\bitnet-2b-tq2_0.gguf
# --no-lazy is required: the lazy path crashes on the packed uint8 bit-shifts

hf download UlukaDev/bitnet-2digit-mult-expert   --local-dir adapters\mult-peft
hf download UlukaDev/bitnet-roman-numeral-expert --local-dir adapters\roman-peft
python convert_lora_bitnet.py adapters\mult-peft  --base models\bitnet-2b-4t-packed --outtype f16 --outfile adapters\mult-f16.gguf
python convert_lora_bitnet.py adapters\roman-peft --base models\bitnet-2b-4t-packed --outtype f16 --outfile adapters\roman-f16.gguf
```

## Serve (one base, two adapters, none active until selected)

```powershell
fabric\build-cpu\bin\Release\llama-server.exe -m models\bitnet-2b-tq1_0.gguf `
  --lora adapters\mult-f16.gguf --lora adapters\roman-f16.gguf `
  --lora-init-without-apply -c 4096 --port 8080 -t 10
# load order fixes ids: 0 = mult, 1 = roman
# GET  http://localhost:8080/lora-adapters          -> list, all scale 0
# POST http://localhost:8080/lora-adapters [{"id":0,"scale":1.0},{"id":1,"scale":0.0}]
```

## Drive & eval

```powershell
python moe_driver.py            # interactive; --debug prints raw router labels
python eval.py --data data\mult_eval.jsonl  --adapter-id 0  --out data\results_mult.jsonl
python eval.py --data data\roman_eval.jsonl --adapter-id 1  --out data\results_roman.jsonl
python eval.py --data data\mult_eval.jsonl data\roman_eval.jsonl --route   # full stack + routing acc
# --adapter-id -1 = base only (all adapter scales 0)
```
