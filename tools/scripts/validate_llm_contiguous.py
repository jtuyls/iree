#!/usr/bin/env python3
"""
Validate contiguous cache LLM model with IREE.
Compares IREE output with PyTorch and measures performance.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache
import iree.runtime as rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="hip")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--compare-pytorch", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Model: {config['hf_model']}")
    print(f"Cache layout: {config['cache_layout']}")
    print(f"Cache shape: {config['cache_shape']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    tokens = tokenizer.encode(args.prompt, return_tensors="pt")[0]
    prompt_len = len(tokens)
    print(f"\nPrompt: '{args.prompt}' ({prompt_len} tokens)")
    
    # Pad to prefill_len
    prefill_len = config["prefill_len"]
    padded_tokens = torch.zeros(prefill_len, dtype=torch.int64)
    padded_tokens[:prompt_len] = tokens
    
    # Create attention mask for prefill
    prefill_attention_mask = torch.zeros(1, prefill_len, dtype=torch.int64)
    prefill_attention_mask[0, :prompt_len] = 1
    
    # =========== IREE Inference ===========
    print("\n=== IREE Inference ===")
    
    vmfb_path = model_dir / "model.vmfb"
    
    # Try IRPA first, then safetensors
    irpa_path = model_dir / "model.irpa"
    safetensors_path = model_dir / "model.safetensors"
    if irpa_path.exists():
        weights_path = irpa_path
    elif safetensors_path.exists():
        weights_path = safetensors_path
    else:
        raise FileNotFoundError(f"No weights found in {model_dir}")
    
    print(f"Using weights: {weights_path.name}")
    
    rt_config = rt.Config(args.device)
    device = rt_config.device
    
    # Load module
    with open(vmfb_path, "rb") as f:
        vmfb_data = f.read()
    
    params = rt.ParameterIndex()
    params.load(str(weights_path))
    
    instance = rt.VmInstance()
    hal_module = rt.create_hal_module(instance, device)
    params_module = rt.create_io_parameters_module(
        instance, params.create_provider("model")
    )
    vm_module = rt.VmModule.copy_buffer(instance, vmfb_data)
    
    modules = rt.load_vm_modules(params_module, hal_module, vm_module, config=rt_config)
    module = modules[-1]
    
    # Prefill
    tokens_input = padded_tokens.unsqueeze(0).numpy()
    mask_input = prefill_attention_mask.numpy()
    
    t0 = time.perf_counter()
    result = module.prefill(tokens_input, mask_input)
    prefill_time = time.perf_counter() - t0
    
    logits_host = result[0].to_host()
    
    # Keep cache on device!
    cache_k_device = result[1]  # DeviceArray on device
    cache_v_device = result[2]
    
    # Get cache to host for shape info only
    cache_k_shape = cache_k_device.to_host().shape
    
    # Get first token
    next_token = int(np.argmax(logits_host[0, prompt_len - 1, :]))
    generated_tokens = [next_token]
    
    print(f"Prefill: {prefill_time*1000:.1f}ms for {prompt_len} tokens")
    print(f"Cache K shape: {cache_k_shape}")
    
    # Decode loop
    max_cache_len = config["max_cache_len"]
    attention_mask_len = config["attention_mask_len"]
    
    decode_times = []
    invoke_times = []
    to_host_times = []
    update_times = []
    pos = prompt_len  # Position of next token
    
    # Get cache on host first for updates
    cache_k_host = cache_k_device.to_host()
    cache_v_host = cache_v_device.to_host()
    
    # Try keeping cache as device arrays
    use_device_cache = True
    if use_device_cache:
        # Create device arrays from prefill output
        cache_k = cache_k_host.copy()  # Work with host copy for updates
        cache_v = cache_v_host.copy()
    else:
        cache_k = cache_k_host
        cache_v = cache_v_host
    
    for i in range(args.max_tokens - 1):
        # Prepare inputs
        token_input = np.array([[next_token]], dtype=np.int64)
        position_id = np.array([[pos]], dtype=np.int64)
        
        # Attention mask: 1s for positions 0..pos (past + current)
        attention_mask = np.zeros((1, attention_mask_len), dtype=np.int64)
        attention_mask[0, :pos + 1] = 1
        
        t0 = time.perf_counter()
        
        # Convert cache to device array for this call
        if use_device_cache:
            cache_k_dev = rt.asdevicearray(device, cache_k)
            cache_v_dev = rt.asdevicearray(device, cache_v)
            result = module.decode_step(token_input, position_id, attention_mask, cache_k_dev, cache_v_dev)
        else:
            result = module.decode_step(token_input, position_id, attention_mask, cache_k, cache_v)
        t1 = time.perf_counter()
        invoke_times.append(t1 - t0)
        
        logits = result[0].to_host()
        new_kv = result[1].to_host()
        t2 = time.perf_counter()
        to_host_times.append(t2 - t1)
        
        if i == 0:
            print(f"  new_kv shape: {new_kv.shape}")
            print(f"  cache_k[pos] shape: {cache_k[pos].shape}")
        
        # Update cache with contiguous write!
        # new_kv is [heads, 2, layers, dim] based on export
        if len(new_kv.shape) == 4:
            new_k = new_kv[:, 0, :, :].transpose(1, 0, 2)  # [layers, heads, dim]
            new_v = new_kv[:, 1, :, :].transpose(1, 0, 2)
        else:
            new_k = new_kv[0, 0]
            new_v = new_kv[0, 1]
        
        cache_k[pos] = new_k
        cache_v[pos] = new_v
        t3 = time.perf_counter()
        update_times.append(t3 - t2)
        
        decode_times.append(t3 - t0)
        
        next_token = int(np.argmax(logits[0, 0, :]))
        generated_tokens.append(next_token)
        pos += 1
        
        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break
    
    # Print timing breakdown
    print(f"\n  Timing breakdown:")
    print(f"    invoke:    {np.mean(invoke_times)*1000:.1f}ms (includes H2D for cache)")
    print(f"    to_host:   {np.mean(to_host_times)*1000:.1f}ms (logits + new_kv)")
    print(f"    update:    {np.mean(update_times)*1000:.1f}ms (host-side cache update)")
    
    avg_decode = np.mean(decode_times) * 1000
    tokens_per_sec = 1.0 / np.mean(decode_times)
    
    # Decode tokens
    generated_text = tokenizer.decode(generated_tokens)
    full_text = args.prompt + generated_text
    
    print(f"\nGenerated ({len(generated_tokens)} tokens): {generated_text}")
    print(f"Decode: avg {avg_decode:.1f}ms/token = {tokens_per_sec:.1f} tok/s")
    
    # =========== PyTorch Comparison ===========
    if args.compare_pytorch:
        print("\n=== PyTorch Comparison ===")
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            config["hf_model"],
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        hf_model.eval()
        
        # Run PyTorch generation
        input_ids = torch.tensor([tokens.tolist()], device="cuda")
        
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = hf_model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - t0
        
        pytorch_tokens = outputs[0].tolist()[prompt_len:]
        pytorch_text = tokenizer.decode(pytorch_tokens)
        
        print(f"PyTorch generated: {pytorch_text}")
        print(f"PyTorch time: {pytorch_time*1000:.1f}ms")
        
        # Compare tokens
        match = generated_tokens[:len(pytorch_tokens)] == pytorch_tokens[:len(generated_tokens)]
        print(f"\nTokens match: {match}")
        if not match:
            print(f"  IREE:    {generated_tokens[:10]}")
            print(f"  PyTorch: {pytorch_tokens[:10]}")
    
    print("\n=== Summary ===")
    print(f"Model: {config['hf_model']}")
    print(f"Prefill: {prefill_time*1000:.1f}ms ({prompt_len} tokens)")
    print(f"Decode: {avg_decode:.1f}ms/token = {tokens_per_sec:.1f} tok/s")
    print(f"Output: {full_text}")


if __name__ == "__main__":
    main()

