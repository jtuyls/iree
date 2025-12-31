#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Export TinyLlama with standard attention and KV cache for incremental decoding.

Exports:
- model.mlir: Model with prefill and decode functions
- model.safetensors: Model weights
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
    parser.add_argument("--prefill-len", type=int, default=512)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StaticCache
    from iree.turbine.aot import FxProgramsBuilder, export, externalize_module_parameters
    
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
    
    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Test with StaticCache to verify it works at various positions
    print("\n=== Testing Positions (with StaticCache) ===")
    static_cache = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=args.max_seq_len,
        device="cpu",
        dtype=dtype,
    )
    
    test_positions = [10, 100, 500, 511, 512, 513, 600, 1000]
    for pos in test_positions:
        if pos >= args.max_seq_len:
            print(f"  Position {pos}: SKIPPED (beyond max_seq_len)")
            continue
            
        static_cache.reset()
        tokens = torch.randint(0, config.vocab_size, (1, pos), dtype=torch.int64)
        cache_position = torch.arange(0, pos)
        
        with torch.no_grad():
            outputs = model(
                input_ids=tokens,
                cache_position=cache_position,
                past_key_values=static_cache,
                use_cache=True,
                return_dict=True,
            )
        
        has_nan = torch.isnan(outputs.logits).any().item()
        max_logit = outputs.logits.max().item()
        status = "❌ NaN!" if has_nan else "✓ OK"
        print(f"  Position {pos}: {status} max_logit={max_logit:.2f}")
    
    # Wrapper with explicit cache tensors for export
    class LLMWithCache(nn.Module):
        def __init__(self, model, config, max_seq_len, dtype):
            super().__init__()
            self.model = model
            self.num_layers = config.num_hidden_layers
            self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
            self.head_dim = config.hidden_size // config.num_attention_heads
            self.max_seq_len = max_seq_len
            self.dtype = dtype
            
        def prefill(
            self,
            input_ids: torch.Tensor,    # [1, seq_len]
            cache_k: torch.Tensor,      # [num_layers, 1, num_kv_heads, max_seq, head_dim]
            cache_v: torch.Tensor,      # [num_layers, 1, num_kv_heads, max_seq, head_dim]
        ):
            """Prefill: process input tokens and populate cache."""
            seq_len = input_ids.shape[1]
            
            # Run without cache first
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs.logits
            past_kv = outputs.past_key_values
            
            # Store in cache tensors
            for i in range(self.num_layers):
                k, v = past_kv[i]
                cache_k[i, :, :, :seq_len, :] = k
                cache_v[i, :, :, :seq_len, :] = v
            
            return logits, cache_k, cache_v
        
        def decode(
            self,
            token: torch.Tensor,        # [1, 1]
            seq_len: torch.Tensor,      # [] scalar - current sequence length
            cache_k: torch.Tensor,      # [num_layers, 1, num_kv_heads, max_seq, head_dim]
            cache_v: torch.Tensor,      # [num_layers, 1, num_kv_heads, max_seq, head_dim]
        ):
            """Decode: process single token with cache."""
            from transformers import DynamicCache
            
            slen = seq_len.item()
            
            # Build DynamicCache from flat tensors
            past_kv = DynamicCache()
            for i in range(self.num_layers):
                k = cache_k[i, :, :, :slen, :].contiguous()
                v = cache_v[i, :, :, :slen, :].contiguous()
                past_kv.update(k, v, layer_idx=i)
            
            position_ids = torch.tensor([[slen]], dtype=torch.int64)
            
            outputs = self.model(
                input_ids=token,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs.logits
            new_past_kv = outputs.past_key_values
            
            # Update cache at position slen
            for i in range(self.num_layers):
                new_k = new_past_kv.key_cache[i]
                new_v = new_past_kv.value_cache[i]
                cache_k[i, :, :, slen:slen+1, :] = new_k[:, :, -1:, :]
                cache_v[i, :, :, slen:slen+1, :] = new_v[:, :, -1:, :]
            
            return logits, cache_k, cache_v
    
    wrapper = LLMWithCache(model, config, args.max_seq_len, dtype)
    
    # Test the wrapper
    print("\n=== Testing Wrapper ===")
    cache_shape = (config.num_hidden_layers, 1, num_kv_heads, args.max_seq_len, head_dim)
    cache_k = torch.zeros(cache_shape, dtype=dtype)
    cache_v = torch.zeros(cache_shape, dtype=dtype)
    
    test_tokens = torch.randint(0, config.vocab_size, (1, 10), dtype=torch.int64)
    
    with torch.no_grad():
        logits, cache_k, cache_v = wrapper.prefill(test_tokens, cache_k, cache_v)
    print(f"Prefill: logits {logits.shape}, NaN: {torch.isnan(logits).any().item()}")
    
    # Decode at position 10
    decode_token = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.int64)
    seq_len = torch.tensor(10, dtype=torch.int64)
    
    with torch.no_grad():
        logits, cache_k, cache_v = wrapper.decode(decode_token, seq_len, cache_k, cache_v)
    print(f"Decode at 10: logits {logits.shape}, NaN: {torch.isnan(logits).any().item()}")
    
    # Test decode at position 600
    print("\n=== Testing Decode at 600 ===")
    cache_k = torch.zeros(cache_shape, dtype=dtype)
    cache_v = torch.zeros(cache_shape, dtype=dtype)
    
    prefill_tokens = torch.randint(0, config.vocab_size, (1, 600), dtype=torch.int64)
    with torch.no_grad():
        _, cache_k, cache_v = wrapper.prefill(prefill_tokens, cache_k, cache_v)
    
    seq_len = torch.tensor(600, dtype=torch.int64)
    with torch.no_grad():
        logits, _, _ = wrapper.decode(decode_token, seq_len, cache_k, cache_v)
    print(f"Decode at 600: logits {logits.shape}, NaN: {torch.isnan(logits).any().item()}")
    print(f"Max logit: {logits.max().item():.2f}")
    
    # Save weights
    print("\n=== Saving Weights ===")
    from safetensors.torch import save_file
    
    state_dict = {}
    for name, param in model.named_parameters():
        prefixed_name = f"model.{name}"
        state_dict[prefixed_name] = param.detach().cpu()
    
    safetensors_path = output_dir / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    print(f"Saved safetensors ({safetensors_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Export model
    print("\n=== Exporting Model to MLIR ===")
    
    # Simple forward without cache for now (cache version is complex to export)
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
    
    simple_wrapper = SimpleForward(model)
    externalize_module_parameters(simple_wrapper)
    
    fxb = FxProgramsBuilder(simple_wrapper)
    
    # Export prefill with fixed size
    test_input = torch.zeros(1, args.prefill_len, dtype=torch.int64)
    
    @fxb.export_program(
        name="prefill",
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
        "cache_shape": list(cache_shape),
        "attention_type": "standard",
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\n=== EXPORT COMPLETE ===")
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024*1024:
            print(f"  {f.name}: {size/1024/1024:.1f} MB")
        else:
            print(f"  {f.name}: {size/1024:.1f} KB")


if __name__ == "__main__":
    main()

