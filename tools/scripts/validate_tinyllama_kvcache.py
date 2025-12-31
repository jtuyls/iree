#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Validate TinyLlama export with KV cache.

Tests both the prefill function (with KV cache output) and the forward function
for text generation.
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
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Model: {config.get('hf_model', 'unknown')}")
    print(f"Vocab size: {config.get('vocab_size', 32000)}")
    print(f"Prefill length: {config.get('prefill_len', 128)}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    # Load IREE module
    import iree.runtime as rt
    
    vmfb_path = model_dir / "model.vmfb"
    weights_path = model_dir / "model.safetensors"
    
    print(f"\nLoading VMFB from {vmfb_path}")
    
    rt_config = rt.Config(args.device)
    
    # Load safetensors as parameter provider
    from iree.runtime import ParameterIndex
    param_index = ParameterIndex()
    param_index.load(str(weights_path))
    
    with open(vmfb_path, "rb") as f:
        vmfb_contents = f.read()
    
    # Create modules with parameter provider
    modules = rt.load_vm_modules(
        rt.create_io_parameters_module(rt_config.vm_instance, param_index.create_provider("model")),
        rt.create_hal_module(rt_config.vm_instance, rt_config.device),
        rt.VmModule.copy_buffer(rt_config.vm_instance, vmfb_contents),
        config=rt_config,
    )
    module = modules[-1]
    
    print(f"Loaded module: {module}")
    
    # Tokenize prompt
    tokens = tokenizer.encode(args.prompt, return_tensors="np").astype(np.int64)
    seq_len = tokens.shape[1]
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Tokens: {tokens.tolist()} (len={seq_len})")
    
    prefill_len = config.get('prefill_len', 128)
    
    # =========== Test Prefill ===========
    print(f"\n=== Testing Prefill ===")
    
    # Pad tokens to prefill_len
    padded_tokens = np.zeros((1, prefill_len), dtype=np.int64)
    padded_tokens[0, :seq_len] = tokens[0, :seq_len]
    
    start = time.perf_counter()
    result = module.prefill(padded_tokens)
    elapsed = time.perf_counter() - start
    
    if isinstance(result, tuple):
        logits = result[0].to_host()
        cache_k = result[1].to_host()
        cache_v = result[2].to_host()
        print(f"Prefill output: logits {logits.shape}, cache_k {cache_k.shape}")
    else:
        logits = result.to_host()
        cache_k = None
        print(f"Prefill output: logits {logits.shape}")
    
    print(f"Time: {elapsed*1000:.1f} ms")
    print(f"NaN check: {np.isnan(logits).any()}")
    
    # Get prediction at last valid position
    last_logits = logits[0, seq_len - 1, :]
    next_token = int(np.argmax(last_logits))
    next_word = tokenizer.decode([next_token])
    print(f"Predicted next token: {next_token} = '{next_word}'")
    
    # =========== Test Forward (for generation) ===========
    print(f"\n=== Testing Forward for Generation ===")
    
    # Generate using forward (re-running with full sequence each time)
    generated_tokens = tokens[0].tolist()
    
    print(f"Generating {args.max_tokens} tokens...")
    start = time.perf_counter()
    
    for i in range(args.max_tokens):
        # Pad to prefill_len
        current_len = len(generated_tokens)
        if current_len > prefill_len:
            print(f"\nWarning: Sequence length {current_len} > prefill_len {prefill_len}")
            break
        
        padded = np.zeros((1, prefill_len), dtype=np.int64)
        padded[0, :current_len] = generated_tokens
        
        result = module.forward(padded)
        logits = result.to_host()
        
        # Get prediction at last valid position
        last_logits = logits[0, current_len - 1, :]
        next_token = int(np.argmax(last_logits))
        generated_tokens.append(next_token)
        
        if args.verbose:
            print(f"  Step {i+1}: token {next_token} = '{tokenizer.decode([next_token])}'")
        
        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break
    
    elapsed = time.perf_counter() - start
    
    # Decode final text
    generated_text = tokenizer.decode(generated_tokens)
    print(f"\nGenerated text ({len(generated_tokens)} tokens):")
    print(f"  '{generated_text}'")
    print(f"\nGeneration time: {elapsed:.2f}s ({len(generated_tokens)/elapsed:.1f} tokens/sec)")
    
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
    
    # Generate with PyTorch
    input_ids = torch.tensor([generated_tokens[:seq_len]], dtype=torch.int64)
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    
    pytorch_text = tokenizer.decode(outputs[0].tolist())
    print(f"\nPyTorch generated ({outputs.shape[1]} tokens):")
    print(f"  '{pytorch_text}'")
    print(f"Time: {elapsed:.2f}s ({outputs.shape[1]/elapsed:.1f} tokens/sec)")
    
    # Check if outputs match
    iree_output = generated_text
    pytorch_output = pytorch_text
    
    if iree_output.strip() == pytorch_output.strip():
        print(f"\n✓ IREE and PyTorch outputs MATCH!")
    else:
        print(f"\n⚠ IREE and PyTorch outputs differ:")
        print(f"  IREE:    '{iree_output}'")
        print(f"  PyTorch: '{pytorch_output}'")


if __name__ == "__main__":
    main()

