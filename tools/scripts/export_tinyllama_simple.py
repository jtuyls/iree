#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Export TinyLlama with standard attention using iree-turbine.

Exports:
- model.mlir: Model computation graph (references external parameters)
- model.irpa: Model weights in IREE parameter archive format
- config.json: Model configuration
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StaticCache
    from iree.turbine.aot import FxProgramsBuilder, export, decompositions, externalize_module_parameters
    
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    
    # Use eager attention to decompose scaled_dot_product_attention for IREE
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=dtype, 
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    
    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create StaticCache for testing
    static_cache = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=args.max_seq_len,
        device="cpu",
        dtype=dtype,
    )
    
    print(f"StaticCache: {len(static_cache.key_cache)} layers, shape {static_cache.key_cache[0].shape}")
    
    # Test prefill
    print("\n=== Testing Prefill ===")
    test_tokens = torch.randint(0, config.vocab_size, (1, 10), dtype=torch.int64)
    static_cache.reset()
    cache_position = torch.arange(0, 10)
    
    with torch.no_grad():
        outputs = model(
            input_ids=test_tokens,
            cache_position=cache_position,
            past_key_values=static_cache,
            use_cache=True,
            return_dict=True,
        )
    print(f"Prefill logits: {outputs.logits.shape}, NaN: {torch.isnan(outputs.logits).any().item()}")
    
    # Test decode at position 600
    print("\n=== Testing Decode at position 600 ===")
    static_cache.reset()
    long_tokens = torch.randint(0, config.vocab_size, (1, 600), dtype=torch.int64)
    cache_position = torch.arange(0, 600)
    
    with torch.no_grad():
        outputs = model(
            input_ids=long_tokens,
            cache_position=cache_position,
            past_key_values=static_cache,
            use_cache=True,
            return_dict=True,
        )
    print(f"Decode at 600: logits {outputs.logits.shape}, NaN: {torch.isnan(outputs.logits).any().item()}")
    print(f"Max logit: {outputs.logits.max().item():.2f}")
    
    # Simple wrapper for export (no cache for simpler export)
    class SimpleForward(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            outputs = self.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
            return outputs.logits
    
    wrapper = SimpleForward(model)
    
    # Save weights to safetensors (which can be converted to IRPA)
    print("\n=== Saving Weights ===")
    from safetensors.torch import save_file
    
    # Save with "model." prefix to match wrapper's parameter names
    state_dict = {}
    for name, param in model.named_parameters():
        # The wrapper is SimpleForward(model), so parameters become model.<name>
        prefixed_name = f"model.{name}"
        state_dict[prefixed_name] = param.detach().cpu()
    
    safetensors_path = output_dir / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    print(f"Saved safetensors to {safetensors_path} ({safetensors_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Convert safetensors to IRPA using iree-turbine
    print("\n=== Converting to IRPA ===")
    irpa_path = output_dir / "model.irpa"
    
    try:
        import subprocess
        result = subprocess.run([
            "python", "-m", "iree.turbine.aot.params",
            "--input", str(safetensors_path),
            "--output", str(irpa_path),
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Saved IRPA to {irpa_path}")
        else:
            print(f"IRPA conversion failed: {result.stderr}")
            # Just use safetensors for now
            print("Will use safetensors directly")
    except Exception as e:
        print(f"IRPA conversion not available: {e}")
    
    # Externalize parameters for MLIR export
    print("\n=== Exporting Model to MLIR ===")
    
    # Externalize creates references to external parameters instead of embedding weights
    externalize_module_parameters(wrapper)
    
    fxb = FxProgramsBuilder(wrapper)
    
    test_input = torch.zeros(1, args.prefill_len, dtype=torch.int64)
    
    @fxb.export_program(
        name="forward",
        args=(test_input,),
        dynamic_shapes={},
    )
    def _(model, input_ids):
        return model(input_ids)
    
    print("Building MLIR...")
    output = export(fxb, import_symbolic_shape_expressions=True)
    
    mlir_path = output_dir / "model.mlir"
    output.save_mlir(str(mlir_path))
    print(f"Saved MLIR to {mlir_path}")
    
    # Save config
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    export_config = {
        "model_type": "tinyllama_simple",
        "hf_model": args.model,
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_seq_len": args.max_seq_len,
        "prefill_len": args.prefill_len,
        "dtype": args.dtype,
        "attention_type": "standard",
        "llm_assist": {
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "vocab_size": config.vocab_size,
            "context_length": args.max_seq_len,
            "model_type": "llama",
        }
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved tokenizer to {output_dir}")
    
    # Print file sizes
    print("\n=== Export Summary ===")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024*1024:
            print(f"  {f.name}: {size/1024/1024:.1f} MB")
        elif size > 1024:
            print(f"  {f.name}: {size/1024:.1f} KB")
        else:
            print(f"  {f.name}: {size} bytes")
    
    print(f"\n=== EXPORT COMPLETE ===")
    print(f"\nNext steps:")
    print(f"1. Compile: iree-compile {mlir_path} -o model.vmfb \\")
    print(f"            --iree-hal-target-device=hip --iree-hip-target=gfx950")


if __name__ == "__main__":
    main()
