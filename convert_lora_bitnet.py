#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter trained on microsoft/bitnet-b1.58-2B-4T to GGUF.

Wraps fabric's convert_lora_to_gguf.py with two fixes it needs for this base:

1. The Microsoft checkpoint declares architecture "BitNetForCausalLM"
   (capital N), which is absent from fabric's conversion.TEXT_MODEL_MAP,
   so the stock converter exits with "not supported".
2. conversion.bitnet.BitnetModel.modify_tensors() ternary-quantizes every
   attn/FFN tensor via weight_quant(). During LoRA conversion the same code
   path runs over the adapter's lora_A/lora_B tensors, which must remain
   full-precision runtime deltas — quantizing them corrupts the adapter.

Usage:
    python convert_lora_bitnet.py <peft_adapter_dir> --base <base_model_dir> \
        --outfile <out.gguf> [--outtype f16]
"""

import runpy
import sys
from pathlib import Path

FABRIC_DIR = Path(__file__).resolve().parent / "fabric"
sys.path.insert(0, str(FABRIC_DIR))
sys.path.insert(1, str(FABRIC_DIR / "gguf-py"))

import conversion                      # noqa: E402
from conversion import ModelBase       # noqa: E402
import conversion.bitnet as _bitnet    # noqa: E402

# Map the Microsoft arch name to the bitnet module so get_model_class resolves.
conversion.TEXT_MODEL_MAP["BitNetForCausalLM"] = "bitnet"


@ModelBase.register("BitNetForCausalLM")
class BitnetLoraBase(_bitnet.BitnetModel):
    model_arch = _bitnet.BitnetModel.model_arch

    # LoRA A/B tensors are FP deltas applied at runtime; they must NOT be
    # ternarized. The base model's own weights are never read during LoRA
    # conversion (only config.json is), so disabling weight_quant here is safe.
    def weight_quant(self, weight):
        return weight


# Belt and braces: the lowercase-n registration points at the same class file;
# neutralize it too in case the adapter config routes through it.
_bitnet.BitnetModel.weight_quant = lambda self, weight: weight

if __name__ == "__main__":
    runpy.run_path(str(FABRIC_DIR / "convert_lora_to_gguf.py"), run_name="__main__")
