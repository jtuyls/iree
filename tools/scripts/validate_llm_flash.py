#!/usr/bin/env python3
"""
Validate LLM with shark-ai flash attention.

This script validates the exported model with flash attention kernels.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import iree.runtime as rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory with model files")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--device", default="hip")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Model: {config['hf_model']}")
    print(f"Cache layout: {config['cache_layout']}")
    print(f"Cache shape: {config['cache_shape']}")
    
    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model_dir / config["tokenizer_path"]))
    
    # Tokenize prompt
    encoded = tokenizer.encode(args.prompt)
    prompt_tokens = encoded.ids
    prompt_len = len(prompt_tokens)
    
    print(f"\nPrompt: '{args.prompt}' ({prompt_len} tokens)")
    
    # Pad to prefill_len
    prefill_len = config["prefill_len"]
    max_cache_len = config["max_cache_len"]
    
    padded_tokens = prompt_tokens + [0] * (prefill_len - prompt_len)
    padded_tokens = np.array([padded_tokens], dtype=np.int64)
    
    # Attention mask for prefill: 0 for valid, 1 for padding
    prefill_mask = np.zeros((1, prefill_len), dtype=np.int64)
    prefill_mask[0, prompt_len:] = 1  # mask out padding
    
    # Initialize cache
    cache_shape = tuple(config["cache_shape"])
    dtype = np.float16 if config["dtype"] == "float16" else np.float32
    cache_k = np.zeros(cache_shape, dtype=dtype)
    cache_v = np.zeros(cache_shape, dtype=dtype)
    
    print(f"\n=== IREE Inference ===")
    print(f"Using weights: {config['irpa_path']}")
    
    # Load IREE modules
    vmfb_path = model_dir / config["vmfb_path"]
    weights_path = model_dir / config["irpa_path"]
    
    rt_config = rt.Config(args.device)
    device = rt_config.device
    instance = rt.VmInstance()
    
    with open(vmfb_path, "rb") as f:
        vmfb_data = f.read()
    
    params = rt.ParameterIndex()
    params.load(str(weights_path))
    
    hal_module = rt.create_hal_module(instance, device)
    params_module = rt.create_io_parameters_module(
        instance, params.create_provider("model")
    )
    vm_module = rt.VmModule.copy_buffer(instance, vmfb_data)
    
    modules = rt.load_vm_modules(params_module, hal_module, vm_module, config=rt_config)
    module = modules[-1]
    
    # Prefill
    t0 = time.perf_counter()
    result = module.prefill(padded_tokens, prefill_mask, cache_k, cache_v)
    prefill_time = time.perf_counter() - t0
    
    logits_host = result[0].to_host()
    cache_k_host = result[1].to_host()
    cache_v_host = result[2].to_host()
    
    print(f"Prefill: {prefill_time*1000:.1f}ms for {prompt_len} tokens")
    print(f"Cache K shape: {cache_k_host.shape}")
    
    # Get first token
    next_token = int(np.argmax(logits_host[0, prompt_len - 1, :]))
    generated_tokens = [next_token]
    
    # Decode loop
    decode_times = []
    pos = prompt_len
    
    for i in range(args.max_tokens - 1):
        # Token input
        token_input = np.array([[next_token]], dtype=np.int64)
        position_id = np.array([[pos]], dtype=np.int64)
        
        # Attention mask: [1, max_cache+1] - 0 for valid, 1 for masked
        attention_mask = np.ones((1, max_cache_len + 1), dtype=np.int64)
        attention_mask[0, :pos + 1] = 0  # First pos+1 positions are valid
        
        t0 = time.perf_counter()
        result = module.decode_step(token_input, position_id, attention_mask, cache_k_host, cache_v_host)
        decode_time = time.perf_counter() - t0
        decode_times.append(decode_time)
        
        logits_host = result[0].to_host()
        new_kv = result[1].to_host()  # [1, 2, layers, heads, dim]
        
        # Update cache
        # new_kv[:, 0] is new K, new_kv[:, 1] is new V
        # cache shape is [max_seq, layers, heads, dim]
        cache_k_host[pos] = new_kv[0, 0]  # [layers, heads, dim]
        cache_v_host[pos] = new_kv[0, 1]
        
        # Get next token
        next_token = int(np.argmax(logits_host[0, 0, :]))
        generated_tokens.append(next_token)
        pos += 1
        
        # Check for EOS
        if next_token == tokenizer.token_to_id("</s>") or next_token == 128001:
            break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    
    avg_decode_time = np.mean(decode_times) * 1000
    tok_per_sec = 1000.0 / avg_decode_time if avg_decode_time > 0 else 0
    
    print(f"Generated ({len(generated_tokens)} tokens): {generated_text}")
    print(f"Decode: avg {avg_decode_time:.1f}ms/token = {tok_per_sec:.1f} tok/s")
    
    print(f"\n=== Summary ===")
    print(f"Model: {config['hf_model']}")
    print(f"Prefill: {prefill_time*1000:.1f}ms ({prompt_len} tokens)")
    print(f"Decode: {avg_decode_time:.1f}ms/token = {tok_per_sec:.1f} tok/s")
    print(f"Output: {args.prompt}{generated_text}")


if __name__ == "__main__":
    main()


