#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Export a simple LLM model with standard (non-paged) attention for IREE.

This creates a simpler model that:
1. Uses standard HuggingFace KV cache (no paging)
2. Has no 512 position limit
3. Works well for single-batch synchronous inference

Usage:
    python export_llm_simple.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output-dir /path/to/output \
        --max-seq-len 2048
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def create_parser():
    parser = argparse.ArgumentParser(description="Export simple LLM to IREE")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cpu")
    return parser


class SimpleLLMExport(nn.Module):
    """Wrapper for exporting HuggingFace LLM with flat cache tensors."""
    
    def __init__(self, model, config, max_seq_len: int, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.config = config
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
    
    def prefill(
        self,
        tokens: torch.Tensor,      # [batch, seq_len]
        cache_k: torch.Tensor,     # [num_layers, batch, num_kv_heads, max_seq, head_dim]
        cache_v: torch.Tensor,     # [num_layers, batch, num_kv_heads, max_seq, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prefill: process input tokens and populate cache.
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            cache_k: Updated key cache
            cache_v: Updated value cache
        """
        batch_size, seq_len = tokens.shape
        
        # Run model without cache (prefill)
        outputs = self.model(
            input_ids=tokens,
            use_cache=True,
            return_dict=True,
        )
        
        logits = outputs.logits
        past_kv = outputs.past_key_values
        
        # Store KV cache
        for i in range(self.num_layers):
            k, v = past_kv[i]
            cache_k[i, :, :, :seq_len, :] = k
            cache_v[i, :, :, :seq_len, :] = v
        
        return logits, cache_k, cache_v
    
    def decode(
        self,
        token: torch.Tensor,       # [batch, 1]
        seq_len: torch.Tensor,     # [batch] - current total sequence length
        cache_k: torch.Tensor,     # [num_layers, batch, num_kv_heads, max_seq, head_dim]
        cache_v: torch.Tensor,     # [num_layers, batch, num_kv_heads, max_seq, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode: process single token with cache.
        
        Returns:
            logits: [batch, 1, vocab_size]
            cache_k: Updated key cache
            cache_v: Updated value cache
        """
        from transformers import DynamicCache
        
        batch_size = token.shape[0]
        slen = seq_len[0].item()
        
        # Build DynamicCache from flat tensors
        past_kv = DynamicCache()
        for i in range(self.num_layers):
            k = cache_k[i, :, :, :slen, :].contiguous()
            v = cache_v[i, :, :, :slen, :].contiguous()
            past_kv.update(k, v, layer_idx=i)
        
        # Position for the new token
        position_ids = seq_len.unsqueeze(1)  # [batch, 1]
        
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
            # Take only the new token's KV (last position)
            cache_k[i, :, :, slen:slen+1, :] = new_k[:, :, -1:, :]
            cache_v[i, :, :, slen:slen+1, :] = new_v[:, :, -1:, :]
        
        return logits, cache_k, cache_v


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"Model: {config.num_hidden_layers} layers, "
          f"{config.hidden_size} hidden, {config.num_attention_heads} heads")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create wrapper
    wrapper = SimpleLLMExport(model, config, args.max_seq_len, dtype)
    
    num_kv_heads = wrapper.num_kv_heads
    head_dim = wrapper.head_dim
    
    print(f"\nModel dimensions:")
    print(f"  num_layers: {wrapper.num_layers}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  vocab_size: {wrapper.vocab_size}")
    
    # Test the wrapper
    print("\n=== Testing Prefill ===")
    batch_size = 1
    cache_shape = (wrapper.num_layers, batch_size, num_kv_heads, args.max_seq_len, head_dim)
    
    test_tokens = torch.randint(0, config.vocab_size, (batch_size, 10), dtype=torch.int64)
    test_cache_k = torch.zeros(cache_shape, dtype=dtype)
    test_cache_v = torch.zeros(cache_shape, dtype=dtype)
    
    with torch.no_grad():
        logits, cache_k, cache_v = wrapper.prefill(test_tokens, test_cache_k, test_cache_v)
    
    print(f"  Prefill logits: {logits.shape}, NaN: {torch.isnan(logits).any().item()}")
    print(f"  Cache filled to position 10")
    
    # Test decode at various positions
    print("\n=== Testing Decode at Various Positions ===")
    test_positions = [10, 100, 500, 511, 512, 513, 600, 1000]
    
    for pos in test_positions:
        if pos >= args.max_seq_len:
            print(f"  Position {pos}: SKIPPED (beyond max_seq_len)")
            continue
        
        # Fill cache up to pos
        prefill_tokens = torch.randint(0, config.vocab_size, (batch_size, pos), dtype=torch.int64)
        cache_k = torch.zeros(cache_shape, dtype=dtype)
        cache_v = torch.zeros(cache_shape, dtype=dtype)
        
        with torch.no_grad():
            _, cache_k, cache_v = wrapper.prefill(prefill_tokens, cache_k, cache_v)
        
        # Decode at pos
        decode_token = torch.randint(0, config.vocab_size, (batch_size, 1), dtype=torch.int64)
        seq_len = torch.tensor([pos], dtype=torch.int64)
        
        with torch.no_grad():
            logits, _, _ = wrapper.decode(decode_token, seq_len, cache_k, cache_v)
        
        has_nan = torch.isnan(logits).any().item()
        max_logit = logits.max().item()
        status = "❌ NaN!" if has_nan else "✓ OK"
        print(f"  Position {pos}: {status} max_logit={max_logit:.2f}")
    
    # Save config
    export_config = {
        "model_type": "simple_llm",
        "hf_model": args.model,
        "attention_type": "standard",
        "num_layers": wrapper.num_layers,
        "num_heads": wrapper.num_heads,
        "num_kv_heads": num_kv_heads,
        "hidden_size": wrapper.hidden_size,
        "head_dim": head_dim,
        "vocab_size": wrapper.vocab_size,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "cache_shape": list(cache_shape),
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    print(f"\nSaved config to {config_path}")
    
    # Save tokenizer
    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.save_pretrained(str(output_dir))
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
    
    # Export to MLIR using torch.export + iree-turbine
    print("\n=== Exporting to MLIR ===")
    
    try:
        from iree.turbine.aot import FxProgramsBuilder, export
        
        fxb = FxProgramsBuilder(wrapper)
        
        # Export prefill
        print("Exporting prefill...")
        prefill_tokens = torch.zeros(batch_size, 64, dtype=torch.int64)
        prefill_cache_k = torch.zeros(cache_shape, dtype=dtype)
        prefill_cache_v = torch.zeros(cache_shape, dtype=dtype)
        
        sl_dim = torch.export.Dim("seq_len", min=1, max=args.max_seq_len)
        
        @fxb.export_program(
            name="prefill",
            args=(prefill_tokens, prefill_cache_k, prefill_cache_v),
            dynamic_shapes={
                "tokens": {1: sl_dim},
                "cache_k": {},
                "cache_v": {},
            },
        )
        def _(model, tokens, cache_k, cache_v):
            return model.prefill(tokens, cache_k, cache_v)
        
        # Export decode
        print("Exporting decode...")
        decode_token = torch.zeros(batch_size, 1, dtype=torch.int64)
        # Trace at a mid-range position to avoid specialization
        decode_seq_len = torch.tensor([args.max_seq_len // 2], dtype=torch.int64)
        decode_cache_k = torch.zeros(cache_shape, dtype=dtype)
        decode_cache_v = torch.zeros(cache_shape, dtype=dtype)
        
        @fxb.export_program(
            name="decode",
            args=(decode_token, decode_seq_len, decode_cache_k, decode_cache_v),
            dynamic_shapes={
                "token": {},
                "seq_len": {},
                "cache_k": {},
                "cache_v": {},
            },
        )
        def _(model, token, seq_len, cache_k, cache_v):
            return model.decode(token, seq_len, cache_k, cache_v)
        
        # Build and save
        print("Building MLIR...")
        output = export(fxb, import_symbolic_shape_expressions=True)
        
        mlir_path = output_dir / "model.mlir"
        output.save_mlir(str(mlir_path))
        print(f"Saved MLIR to {mlir_path}")
        
        print("\n=== EXPORT COMPLETE ===")
        print(f"\nNext steps:")
        print(f"1. Compile: iree-compile {mlir_path} -o model.vmfb --iree-hal-target-device=hip --iree-hip-target=gfx950")
        print(f"2. Validate: python validate_llm_simple.py --config {config_path} --vmfb model.vmfb")
        
    except Exception as e:
        print(f"\nMLIR export failed: {e}")
        print("The model wrapper works correctly - export needs further debugging.")
        print("\nFor now, you can use the validate_llm_simple.py script with --use-pytorch")
        raise


if __name__ == "__main__":
    main()

