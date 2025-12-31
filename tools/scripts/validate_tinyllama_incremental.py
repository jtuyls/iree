#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Validate TinyLlama incremental decode export.

Tests:
1. Prefill function
2. Decode step function with cache updates
3. Full generation loop comparing with PyTorch
4. Performance comparison between incremental and re-run approaches
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="hip")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Model: {config.get('hf_model', 'unknown')}")
    print(f"Prefill length: {config.get('prefill_len', 512)}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    # Load IREE module
    import iree.runtime as rt
    from iree.runtime import ParameterIndex
    
    vmfb_path = model_dir / "model.vmfb"
    weights_path = model_dir / "model.safetensors"
    
    print(f"\nLoading VMFB from {vmfb_path}")
    
    rt_config = rt.Config(args.device)
    param_index = ParameterIndex()
    param_index.load(str(weights_path))
    
    with open(vmfb_path, "rb") as f:
        vmfb_contents = f.read()
    
    modules = rt.load_vm_modules(
        rt.create_io_parameters_module(rt_config.vm_instance, param_index.create_provider("model")),
        rt.create_hal_module(rt_config.vm_instance, rt_config.device),
        rt.VmModule.copy_buffer(rt_config.vm_instance, vmfb_contents),
        config=rt_config,
    )
    module = modules[-1]
    
    print(f"Loaded module: {module}")
    
    # Get parameters
    prefill_len = config.get('prefill_len', 512)
    max_cache_len = config.get('max_cache_len', prefill_len)
    num_layers = config.get('num_layers', 22)
    num_kv_heads = config.get('num_kv_heads', 4)
    head_dim = config.get('head_dim', 64)
    
    cache_shape = (num_layers, 1, num_kv_heads, max_cache_len, head_dim)
    
    # Tokenize prompt
    tokens = tokenizer.encode(args.prompt, return_tensors="np").astype(np.int64)
    seq_len = tokens.shape[1]
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Tokens: {tokens.tolist()} (len={seq_len})")
    
    # =========== Test Prefill ===========
    print(f"\n=== Testing Prefill ===")
    
    # Pad tokens to prefill_len
    padded_tokens = np.zeros((1, prefill_len), dtype=np.int64)
    padded_tokens[0, :seq_len] = tokens[0, :seq_len]
    valid_len = np.array([seq_len], dtype=np.int64)
    
    start = time.perf_counter()
    result = module.prefill(padded_tokens, valid_len)
    prefill_time = time.perf_counter() - start
    
    logits = result[0].to_host()
    cache_k = result[1].to_host()
    cache_v = result[2].to_host()
    
    print(f"Prefill: logits {logits.shape}, cache_k {cache_k.shape}")
    print(f"Time: {prefill_time*1000:.1f} ms")
    print(f"NaN check: {np.isnan(logits[:, :seq_len, :]).any()}")
    
    # Get next token from last valid position
    last_logits = logits[0, seq_len - 1, :]
    next_token = int(np.argmax(last_logits))
    print(f"Predicted next token: {next_token} = '{tokenizer.decode([next_token])}'")
    
    # =========== Test Decode Step ===========
    print(f"\n=== Testing Decode Step ===")
    
    # Helper to create attention mask
    def make_attention_mask(pos, max_len):
        mask = np.zeros((1, max_len + 1), dtype=np.float16)
        mask[0, :pos + 1] = 1.0
        return mask
    
    token_input = np.array([[next_token]], dtype=np.int64)
    seq_len_input = np.array([seq_len], dtype=np.int64)
    attention_mask = make_attention_mask(seq_len, max_cache_len)
    
    start = time.perf_counter()
    result = module.decode_step(token_input, seq_len_input, attention_mask, cache_k, cache_v)
    decode_time = time.perf_counter() - start
    
    logits = result[0].to_host()
    new_cache_k = result[1].to_host()  # [layers, 1, heads, 1, dim]
    new_cache_v = result[2].to_host()
    
    # Update cache at position seq_len
    cache_k[:, :, :, seq_len:seq_len+1, :] = new_cache_k
    cache_v[:, :, :, seq_len:seq_len+1, :] = new_cache_v
    
    print(f"Decode step: logits {logits.shape}")
    print(f"Time: {decode_time*1000:.1f} ms")
    print(f"NaN check: {np.isnan(logits).any()}")
    
    # Get next token
    next_token2 = int(np.argmax(logits[0, 0, :]))
    print(f"Predicted token: {next_token2} = '{tokenizer.decode([next_token2])}'")
    
    # =========== Full Generation Loop ===========
    print(f"\n=== Full Generation (Incremental Decode) ===")
    
    # Reset - do prefill again
    result = module.prefill(padded_tokens, valid_len)
    logits = result[0].to_host()
    cache_k = result[1].to_host()
    cache_v = result[2].to_host()
    
    generated_tokens = tokens[0].tolist()
    current_pos = seq_len
    
    start = time.perf_counter()
    decode_times = []
    
    for i in range(args.max_tokens):
        # Get next token from logits
        if i == 0:
            next_token = int(np.argmax(logits[0, current_pos - 1, :]))
        else:
            next_token = int(np.argmax(logits[0, 0, :]))
        
        generated_tokens.append(next_token)
        
        if next_token == tokenizer.eos_token_id:
            break
        
        if current_pos >= max_cache_len:
            print(f"  Reached max cache length {max_cache_len}")
            break
        
        # Decode step
        token_input = np.array([[next_token]], dtype=np.int64)
        seq_len_input = np.array([current_pos], dtype=np.int64)
        attention_mask = make_attention_mask(current_pos, max_cache_len)
        
        step_start = time.perf_counter()
        result = module.decode_step(token_input, seq_len_input, attention_mask, cache_k, cache_v)
        decode_times.append(time.perf_counter() - step_start)
        
        logits = result[0].to_host()
        new_cache_k = result[1].to_host()  # [layers, 1, heads, 1, dim]
        new_cache_v = result[2].to_host()
        
        # Update cache at current_pos
        cache_k[:, :, :, current_pos:current_pos+1, :] = new_cache_k
        cache_v[:, :, :, current_pos:current_pos+1, :] = new_cache_v
        
        current_pos += 1
        
        if args.verbose:
            print(f"  Step {i+1}: pos={current_pos-1}, token={next_token} = '{tokenizer.decode([next_token])}'")
    
    total_time = time.perf_counter() - start
    
    generated_text = tokenizer.decode(generated_tokens)
    print(f"\nGenerated ({len(generated_tokens)} tokens):")
    print(f"  '{generated_text}'")
    
    avg_decode = np.mean(decode_times) * 1000 if decode_times else 0
    print(f"\nTiming:")
    print(f"  Prefill: {prefill_time*1000:.1f} ms")
    print(f"  Avg decode step: {avg_decode:.1f} ms")
    print(f"  Total: {total_time:.2f}s ({len(decode_times)/total_time:.1f} tokens/sec)")
    
    # =========== Compare with PyTorch ===========
    print(f"\n=== Comparing with PyTorch ===")
    
    import torch
    from transformers import AutoModelForCausalLM
    
    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.get('hf_model', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    hf_model.eval()
    
    input_ids = torch.tensor([tokens[0].tolist()], dtype=torch.int64)
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    pytorch_time = time.perf_counter() - start
    
    pytorch_text = tokenizer.decode(outputs[0].tolist())
    print(f"\nPyTorch generated ({outputs.shape[1]} tokens):")
    print(f"  '{pytorch_text}'")
    print(f"Time: {pytorch_time:.2f}s ({outputs.shape[1]/pytorch_time:.1f} tokens/sec)")
    
    # Check match
    if generated_text.strip() == pytorch_text.strip():
        print(f"\n✓ IREE and PyTorch outputs MATCH!")
    else:
        print(f"\n⚠ IREE and PyTorch outputs differ")
    
    print(f"\nSpeedup: {pytorch_time/total_time:.2f}x faster on IREE {args.device}")


if __name__ == "__main__":
    main()

