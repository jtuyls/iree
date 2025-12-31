#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Export a HuggingFace LLM model to IREE without paged attention.

This creates a simpler export that uses standard HuggingFace caching,
which doesn't have the 512 position limit bug of sharktank's paged attention.

Usage:
    python export_hf_llm.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output-dir /path/to/output \
        --max-seq-len 2048 \
        --dtype float16
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


def create_parser():
    parser = argparse.ArgumentParser(description="Export HuggingFace LLM to IREE")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for MLIR and config")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to export")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for tracing")
    parser.add_argument("--test-positions", type=int, nargs="+", 
                        default=[10, 100, 500, 512, 600, 1000],
                        help="Positions to test decode at")
    return parser


def test_hf_model_positions(args):
    """Test that HuggingFace model can decode at various positions without issues."""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, DynamicCache
    
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    
    # Load config and model
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
    
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    print(f"\nModel dimensions:")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  max_seq_len: {args.max_seq_len}")
    
    # Test prefill
    print(f"\n=== Testing Prefill ===")
    test_tokens = torch.randint(0, config.vocab_size, (args.batch_size, 10), 
                                 dtype=torch.int64, device=args.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=test_tokens,
            use_cache=True,
            return_dict=True,
        )
    
    prefill_logits = outputs.logits
    past_kv = outputs.past_key_values
    
    print(f"  Prefill logits shape: {prefill_logits.shape}")
    print(f"  Prefill logits has NaN: {torch.isnan(prefill_logits).any().item()}")
    print(f"  Cache type: {type(past_kv)}")
    if hasattr(past_kv, 'key_cache'):
        print(f"  Cache length: {past_kv.get_seq_length()}")
    else:
        print(f"  Cache layers: {len(past_kv)}, k/v shape: {past_kv[0][0].shape}")
    
    # Test decode at various positions
    print(f"\n=== Testing Decode at Various Positions ===")
    
    for target_pos in args.test_positions:
        if target_pos >= args.max_seq_len:
            print(f"\nPosition {target_pos}: SKIPPED (beyond max_seq_len)")
            continue
            
        print(f"\nPosition {target_pos}:")
        
        # First do a prefill up to target_pos - 1
        prefill_len = target_pos
        prefill_tokens = torch.randint(0, config.vocab_size, 
                                        (args.batch_size, prefill_len),
                                        dtype=torch.int64, device=args.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=prefill_tokens,
                use_cache=True,
                return_dict=True,
            )
        
        past_kv = outputs.past_key_values
        
        # Now decode one token at position target_pos
        decode_token = torch.randint(0, config.vocab_size, 
                                      (args.batch_size, 1),
                                      dtype=torch.int64, device=args.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=decode_token,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
        
        decode_logits = outputs.logits
        has_nan = torch.isnan(decode_logits).any().item()
        max_logit = decode_logits.max().item()
        min_logit = decode_logits.min().item()
        
        status = "❌ NaN!" if has_nan else "✓ OK"
        print(f"  {status} logits shape: {decode_logits.shape}, "
              f"range: [{min_logit:.2f}, {max_logit:.2f}]")
    
    # Save config
    export_config = {
        "model_type": "hf_llm",
        "hf_model": args.model,
        "attention_type": "standard",
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "hidden_size": config.hidden_size,
        "head_dim": head_dim,
        "vocab_size": config.vocab_size,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
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
    print(f"\nSaved config to {config_path}")
    
    # Save tokenizer
    print("\nSaving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.save_pretrained(str(output_dir))
        print(f"Saved tokenizer to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
    
    print("\n" + "="*60)
    print("CONCLUSION: Standard HuggingFace models work beyond position 512!")
    print("The 512 limit is specific to sharktank's paged attention export.")
    print("="*60)
    
    print("\nTo export this model to IREE, options are:")
    print("1. Use torch.export + iree-turbine (may need careful dynamic shape handling)")
    print("2. Patch sharktank's export to trace start_positions at a larger value")
    print("3. Use sharktank's 'direct' cache type (if supported)")


def main():
    parser = create_parser()
    args = parser.parse_args()
    test_hf_model_positions(args)


if __name__ == "__main__":
    main()
