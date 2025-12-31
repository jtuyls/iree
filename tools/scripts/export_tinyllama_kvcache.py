#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Export TinyLlama with KV cache support.

Due to torch.export limitations with dynamic cache operations, this exports:
1. `prefill` - Process initial tokens, return logits and KV cache
2. `forward` - Unified function for any sequence length

For incremental decode, either:
- Re-run forward with the full sequence (simple but slower)
- Use sharktank's paged attention export (full IREE KV cache support)

Both prefill and forward are in a single MLIR file.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from iree.turbine.aot import FxProgramsBuilder, export, externalize_module_parameters
    from safetensors.torch import save_file
    
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=dtype, 
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = config.hidden_size // num_heads
    
    print(f"Model: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========== Combined Module ===========
    class LLMModule(nn.Module):
        def __init__(self, model, num_layers, num_kv_heads, head_dim, max_seq_len, dtype):
            super().__init__()
            self.model = model
            self.num_layers = num_layers
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.max_seq_len = max_seq_len
            self.dtype = dtype
        
        def prefill(self, input_ids: torch.Tensor):
            """
            Process input tokens, return logits and KV cache.
            
            Args:
                input_ids: [1, seq_len]
            Returns:
                logits: [1, seq_len, vocab]
                cache_k: [num_layers, 1, num_kv_heads, seq_len, head_dim]
                cache_v: [num_layers, 1, num_kv_heads, seq_len, head_dim]
            """
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs.logits
            past_kv = outputs.past_key_values
            
            # Stack KV cache
            k_list = [past_kv[i][0] for i in range(self.num_layers)]
            v_list = [past_kv[i][1] for i in range(self.num_layers)]
            cache_k = torch.stack(k_list, dim=0)
            cache_v = torch.stack(v_list, dim=0)
            
            return logits, cache_k, cache_v
        
        def forward_no_cache(self, input_ids: torch.Tensor):
            """Simple forward without KV cache."""
            outputs = self.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
            return outputs.logits
    
    llm_module = LLMModule(model, num_layers, num_kv_heads, head_dim, args.max_seq_len, dtype)
    
    # =========== Test ===========
    print("\n=== Testing Module ===")
    
    test_tokens = torch.randint(0, config.vocab_size, (1, 10), dtype=torch.int64)
    with torch.no_grad():
        logits, cache_k, cache_v = llm_module.prefill(test_tokens)
    print(f"Prefill: logits {logits.shape}, cache_k {cache_k.shape}")
    print(f"  NaN: {torch.isnan(logits).any().item()}")
    
    # Test at position 600
    print("\nTesting prefill at 600 tokens...")
    prefill_tokens = torch.randint(0, config.vocab_size, (1, 600), dtype=torch.int64)
    with torch.no_grad():
        logits, cache_k, cache_v = llm_module.prefill(prefill_tokens)
    print(f"Prefill 600: logits {logits.shape}, NaN: {torch.isnan(logits).any().item()}")
    print(f"Max logit: {logits[:, -1, :].max().item():.2f}")
    
    # Test forward_no_cache
    with torch.no_grad():
        logits_no_cache = llm_module.forward_no_cache(test_tokens)
    print(f"\nForward (no cache): logits {logits_no_cache.shape}")
    
    # =========== Save Weights ===========
    print("\n=== Saving Weights ===")
    
    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[f"model.{name}"] = param.detach().cpu()
    
    safetensors_path = output_dir / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    print(f"Saved weights ({safetensors_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # =========== Export ===========
    print("\n=== Exporting to MLIR ===")
    
    externalize_module_parameters(llm_module)
    fxb = FxProgramsBuilder(llm_module)
    
    # Export prefill
    print("Exporting prefill...")
    prefill_tokens = torch.zeros(1, args.prefill_len, dtype=torch.int64)
    
    @fxb.export_program(
        name="prefill",
        args=(prefill_tokens,),
        dynamic_shapes={},
    )
    def _(module, input_ids):
        return module.prefill(input_ids)
    
    # Export forward (no cache) - for incremental generation via re-running
    print("Exporting forward...")
    forward_tokens = torch.zeros(1, args.prefill_len, dtype=torch.int64)
    
    @fxb.export_program(
        name="forward",
        args=(forward_tokens,),
        dynamic_shapes={},
    )
    def _(module, input_ids):
        return module.forward_no_cache(input_ids)
    
    # Build and save
    print("Building MLIR...")
    try:
        output = export(fxb, import_symbolic_shape_expressions=True)
        mlir_path = output_dir / "model.mlir"
        output.save_mlir(str(mlir_path))
        print(f"Saved MLIR to {mlir_path}")
        
        mlir_size = mlir_path.stat().st_size / 1024 / 1024
        print(f"  MLIR size: {mlir_size:.1f} MB")
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save config
    export_config = {
        "model_type": "tinyllama",
        "hf_model": args.model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_seq_len": args.max_seq_len,
        "prefill_len": args.prefill_len,
        "dtype": args.dtype,
        "functions": {
            "prefill": "prefill(input_ids[1,prefill_len]) -> (logits[1,prefill_len,vocab], cache_k, cache_v)",
            "forward": "forward(input_ids[1,prefill_len]) -> logits[1,prefill_len,vocab]",
        },
        "notes": [
            "For incremental decode, use forward with full sequence each time (simple but O(n^2))",
            "For efficient decode, use sharktank paged attention export",
        ],
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\n=== EXPORT COMPLETE ===")
    print("\nFiles:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                print(f"  {f.name}: {size/1024/1024:.1f} MB")
            else:
                print(f"  {f.name}: {size/1024:.1f} KB")
    
    print(f"\nTo compile:")
    print(f"  iree-compile {mlir_path} -o {output_dir}/model.vmfb \\")
    print(f"      --iree-hal-target-device=hip --iree-hip-target=gfx950")
    
    print(f"\nUsage pattern:")
    print(f"  # Prefill (returns KV cache for external use)")
    print(f"  logits, cache_k, cache_v = prefill(tokens)")
    print(f"  ")
    print(f"  # For generation, re-run forward with full sequence")
    print(f"  for i in range(max_new_tokens):")
    print(f"      logits = forward(full_sequence)")
    print(f"      next_token = argmax(logits[:, -1, :])")
    print(f"      full_sequence = concat(full_sequence, next_token)")


if __name__ == "__main__":
    main()
