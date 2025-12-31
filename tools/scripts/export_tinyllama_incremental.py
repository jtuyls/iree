#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Export TinyLlama with O(n) incremental decode.

Key insight: To avoid data-dependent slicing, we:
1. Pass the FULL fixed-size cache to the model
2. Use attention mask to mask out positions beyond current sequence length
3. Update cache at specific position using scatter/index_copy

This allows torch.export to trace the operations since all shapes are fixed.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prefill-len", type=int, default=512)
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
    
    # Cache shape: [num_layers, 1, num_kv_heads, max_cache_len, head_dim]
    # We use max_cache_len = prefill_len for now
    max_cache_len = args.prefill_len
    cache_shape = (num_layers, 1, num_kv_heads, max_cache_len, head_dim)
    
    # =========== Incremental LLM Module ===========
    class IncrementalLLM(nn.Module):
        """LLM with O(n) decode using full cache and attention mask."""
        
        def __init__(self, model, num_layers, num_kv_heads, head_dim, max_cache_len, dtype):
            super().__init__()
            self.model = model
            self.num_layers = num_layers
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.max_cache_len = max_cache_len
            self.dtype = dtype
        
        def prefill(
            self, 
            input_ids: torch.Tensor,   # [1, prefill_len] - padded input
            valid_len: torch.Tensor,   # [1] - actual valid length (before padding)
        ):
            """
            Process input tokens, return logits and KV cache.
            
            The input_ids are padded to prefill_len. valid_len indicates how many
            tokens are actually valid. The cache is zeroed beyond valid_len.
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
            cache_k = torch.stack(k_list, dim=0)  # [layers, 1, heads, prefill_len, dim]
            cache_v = torch.stack(v_list, dim=0)
            
            # Zero out positions beyond valid_len using mask
            # This ensures padding tokens don't pollute the cache
            seq_indices = torch.arange(self.max_cache_len, device=input_ids.device)
            valid_mask = (seq_indices < valid_len[0]).to(self.dtype).view(1, 1, 1, self.max_cache_len, 1)
            
            cache_k = cache_k * valid_mask
            cache_v = cache_v * valid_mask
            
            return logits, cache_k, cache_v
        
        def decode_step(
            self,
            token: torch.Tensor,          # [1, 1]
            seq_len: torch.Tensor,        # [1] - current valid length (new token goes at seq_len)
            attention_mask: torch.Tensor, # [1, max_cache_len + 1] - pre-computed mask
            cache_k: torch.Tensor,        # [layers, 1, heads, max_cache_len, dim]
            cache_v: torch.Tensor,        # [layers, 1, heads, max_cache_len, dim]
        ):
            """
            Decode single token at position seq_len.
            Passes FULL cache to model with attention mask for valid positions.
            
            The attention_mask should have 1s for positions 0..seq_len and 0s elsewhere.
            """
            from transformers.cache_utils import DynamicCache
            
            # Build DynamicCache from our cache tensors
            past_kv = DynamicCache()
            for i in range(self.num_layers):
                k = cache_k[i]  # [1, heads, max_cache_len, dim]
                v = cache_v[i]
                past_kv.update(k, v, layer_idx=i)
            
            # Position for the new token
            position_ids = seq_len.unsqueeze(0)  # [1, 1]
            
            outputs = self.model(
                input_ids=token,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs.logits  # [1, 1, vocab]
            new_past_kv = outputs.past_key_values
            
            # Extract the new KV entries (at position -1 in the returned cache)
            # Return them separately for the caller to update the cache
            new_k_list = []
            new_v_list = []
            for i in range(self.num_layers):
                new_k = new_past_kv.key_cache[i][:, :, -1:, :]  # [1, heads, 1, dim]
                new_v = new_past_kv.value_cache[i][:, :, -1:, :]
                new_k_list.append(new_k)
                new_v_list.append(new_v)
            
            # Stack into single tensors
            new_cache_k = torch.stack(new_k_list, dim=0)  # [layers, 1, heads, 1, dim]
            new_cache_v = torch.stack(new_v_list, dim=0)
            
            # Return: logits, new_k entries, new_v entries (caller updates cache)
            return logits, new_cache_k, new_cache_v
    
    llm = IncrementalLLM(model, num_layers, num_kv_heads, head_dim, max_cache_len, dtype)
    
    # =========== Test ===========
    print("\n=== Testing Module ===")
    
    test_tokens = torch.randint(0, config.vocab_size, (1, 10), dtype=torch.int64)
    padded = torch.zeros(1, args.prefill_len, dtype=torch.int64)
    padded[0, :10] = test_tokens[0]
    valid_len = torch.tensor([10], dtype=torch.int64)
    
    with torch.no_grad():
        logits, cache_k, cache_v = llm.prefill(padded, valid_len)
    print(f"Prefill: logits {logits.shape}, cache_k {cache_k.shape}")
    print(f"  Cache K at pos 0-9 max: {cache_k[:,:,:,:10,:].abs().max().item():.2f}")
    print(f"  Cache K at pos 10+ max: {cache_k[:,:,:,10:,:].abs().max().item():.4f}")
    
    # Helper to create attention mask
    def make_attention_mask(pos):
        """Create attention mask: 1 for positions 0..pos, 0 elsewhere."""
        mask = torch.zeros(1, max_cache_len + 1, dtype=dtype)
        mask[0, :pos + 1] = 1.0
        return mask
    
    # Test decode - now returns new_cache_k/v instead of updated cache
    decode_token = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.int64)
    seq_len_tensor = torch.tensor([10], dtype=torch.int64)
    attention_mask = make_attention_mask(10)
    
    with torch.no_grad():
        logits, new_cache_k, new_cache_v = llm.decode_step(decode_token, seq_len_tensor, attention_mask, cache_k, cache_v)
    
    # Update cache at position 10
    cache_k[:, :, :, 10:11, :] = new_cache_k
    cache_v[:, :, :, 10:11, :] = new_cache_v
    
    print(f"Decode at 10: logits {logits.shape}, NaN: {torch.isnan(logits).any().item()}")
    
    # Test generation loop
    print("\nTesting generation loop to position 30...")
    for pos in range(11, 30):
        seq_len_tensor = torch.tensor([pos], dtype=torch.int64)
        attention_mask = make_attention_mask(pos)
        next_token = torch.argmax(logits[:, -1:, :], dim=-1)
        with torch.no_grad():
            logits, new_cache_k, new_cache_v = llm.decode_step(next_token, seq_len_tensor, attention_mask, cache_k, cache_v)
        # Update cache at position pos
        cache_k[:, :, :, pos:pos+1, :] = new_cache_k
        cache_v[:, :, :, pos:pos+1, :] = new_cache_v
    print(f"  Final logits NaN: {torch.isnan(logits).any().item()}")
    
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
    
    externalize_module_parameters(llm)
    fxb = FxProgramsBuilder(llm)
    
    # Export prefill with valid_len
    print("Exporting prefill...")
    prefill_tokens = torch.zeros(1, args.prefill_len, dtype=torch.int64)
    prefill_valid_len = torch.tensor([args.prefill_len // 2], dtype=torch.int64)  # Mid-range for tracing
    
    @fxb.export_program(
        name="prefill",
        args=(prefill_tokens, prefill_valid_len),
        dynamic_shapes={},
    )
    def _(module, input_ids, valid_len):
        return module.prefill(input_ids, valid_len)
    
    # Export decode_step with attention mask as input
    print("Exporting decode_step...")
    decode_token = torch.zeros(1, 1, dtype=torch.int64)
    decode_seq_len = torch.tensor([args.prefill_len // 2], dtype=torch.int64)
    # Create attention mask for tracing position
    decode_attention_mask = torch.zeros(1, max_cache_len + 1, dtype=dtype)
    decode_attention_mask[0, :args.prefill_len // 2 + 1] = 1.0
    decode_cache_k = torch.zeros(cache_shape, dtype=dtype)
    decode_cache_v = torch.zeros(cache_shape, dtype=dtype)
    
    @fxb.export_program(
        name="decode_step",
        args=(decode_token, decode_seq_len, decode_attention_mask, decode_cache_k, decode_cache_v),
        dynamic_shapes={},
    )
    def _(module, token, seq_len, attention_mask, cache_k, cache_v):
        return module.decode_step(token, seq_len, attention_mask, cache_k, cache_v)
    
    # Build and save
    print("Building MLIR...")
    try:
        output = export(fxb, import_symbolic_shape_expressions=True)
        mlir_path = output_dir / "model.mlir"
        output.save_mlir(str(mlir_path))
        print(f"Saved MLIR to {mlir_path}")
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try exporting just prefill
        print("\n=== Trying prefill-only export ===")
        fxb2 = FxProgramsBuilder(llm)
        
        @fxb2.export_program(
            name="prefill",
            args=(prefill_tokens,),
            dynamic_shapes={},
        )
        def _(module, input_ids):
            return module.prefill(input_ids)
        
        output = export(fxb2, import_symbolic_shape_expressions=True)
        mlir_path = output_dir / "model.mlir"
        output.save_mlir(str(mlir_path))
        print(f"Saved prefill-only MLIR to {mlir_path}")
        return
    
    # Save config
    export_config = {
        "model_type": "tinyllama_incremental",
        "hf_model": args.model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_seq_len": args.max_seq_len,
        "prefill_len": args.prefill_len,
        "max_cache_len": max_cache_len,
        "dtype": args.dtype,
        "cache_shape": list(cache_shape),
        "functions": {
            "prefill": "prefill(tokens[1,prefill_len]) -> (logits, cache_k, cache_v)",
            "decode_step": "decode_step(token[1,1], seq_len[1], cache_k, cache_v) -> (logits, cache_k, cache_v)",
        },
        "notes": [
            "decode_step processes single token at given sequence position",
            "Cache is updated in-place at the given position",
            "O(1) per token decode (model only processes 1 token)",
        ],
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\n=== EXPORT COMPLETE ===")


if __name__ == "__main__":
    main()
