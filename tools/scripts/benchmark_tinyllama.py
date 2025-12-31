#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark TinyLlama at various sequence lengths.
Tests both numerics (NaN checks) and performance.
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
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--tokens-to-generate", type=int, default=20)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    prefill_len = config.get('prefill_len', 128)
    print(f"Model: {config.get('hf_model', 'unknown')}")
    print(f"Export prefill_len: {prefill_len}")
    print(f"Testing sequence lengths: {args.seq_lengths}")
    print(f"Tokens to generate: {args.tokens_to_generate}")
    
    # Load IREE module
    import iree.runtime as rt
    from iree.runtime import ParameterIndex
    
    vmfb_path = model_dir / "model.vmfb"
    weights_path = model_dir / "model.safetensors"
    
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
    
    print(f"\nLoaded IREE module from {vmfb_path}")
    
    # =========== Benchmark at different sequence lengths ===========
    print(f"\n{'='*70}")
    print(f"{'Seq Len':>8} | {'Prefill (ms)':>12} | {'Tok/s (gen)':>12} | {'Gen Time':>10} | {'NaN':>5}")
    print(f"{'='*70}")
    
    results = []
    
    for seq_len in args.seq_lengths:
        if seq_len > prefill_len:
            print(f"{seq_len:>8} | {'SKIP (>prefill_len)':>40}")
            continue
        
        # Create random input tokens
        tokens = np.random.randint(0, config.get('vocab_size', 32000), (1, seq_len), dtype=np.int64)
        
        # Pad to prefill_len
        padded = np.zeros((1, prefill_len), dtype=np.int64)
        padded[0, :seq_len] = tokens[0]
        
        # ===== Prefill =====
        # Warmup
        _ = module.prefill(padded)
        
        # Timed run
        start = time.perf_counter()
        result = module.prefill(padded)
        prefill_time = time.perf_counter() - start
        
        if isinstance(result, tuple):
            logits = result[0].to_host()
        else:
            logits = result.to_host()
        
        prefill_nan = np.isnan(logits[:, :seq_len, :]).any()
        
        # ===== Generation (re-run approach) =====
        generated = list(tokens[0])
        gen_times = []
        gen_nan = False
        
        for i in range(args.tokens_to_generate):
            current_len = len(generated)
            if current_len > prefill_len:
                break
            
            padded = np.zeros((1, prefill_len), dtype=np.int64)
            padded[0, :current_len] = generated
            
            start = time.perf_counter()
            result = module.forward(padded)
            gen_times.append(time.perf_counter() - start)
            
            logits = result.to_host()
            
            if np.isnan(logits[:, current_len-1, :]).any():
                gen_nan = True
            
            next_token = int(np.argmax(logits[0, current_len-1, :]))
            generated.append(next_token)
        
        total_gen_time = sum(gen_times)
        tokens_per_sec = len(gen_times) / total_gen_time if total_gen_time > 0 else 0
        
        nan_status = "❌" if (prefill_nan or gen_nan) else "✓"
        
        print(f"{seq_len:>8} | {prefill_time*1000:>12.2f} | {tokens_per_sec:>12.1f} | {total_gen_time:>10.3f}s | {nan_status:>5}")
        
        results.append({
            'seq_len': seq_len,
            'prefill_ms': prefill_time * 1000,
            'tokens_per_sec': tokens_per_sec,
            'gen_time': total_gen_time,
            'nan': prefill_nan or gen_nan,
        })
    
    print(f"{'='*70}")
    
    # =========== Analysis ===========
    print(f"\n=== Performance Analysis ===")
    
    if len(results) >= 2:
        # Check for O(n) vs O(n^2) scaling
        first = results[0]
        last = results[-1]
        
        seq_ratio = last['seq_len'] / first['seq_len']
        time_ratio = last['gen_time'] / first['gen_time'] if first['gen_time'] > 0 else 0
        
        print(f"Sequence length ratio: {seq_ratio:.1f}x")
        print(f"Generation time ratio: {time_ratio:.1f}x")
        
        if time_ratio > seq_ratio * 1.5:
            print(f"⚠ Scaling is worse than O(n) - likely O(n²) due to re-running full sequence")
        else:
            print(f"✓ Scaling is approximately O(n)")
    
    # =========== Numerics check at high position ===========
    print(f"\n=== Numerics Check at High Positions ===")
    
    # Test positions beyond 512 (the paged attention limit)
    test_positions = [100, 256, 512, 768, 1024]
    
    for pos in test_positions:
        if pos > prefill_len:
            print(f"Position {pos}: SKIP (>prefill_len {prefill_len})")
            continue
        
        # Create input with `pos` tokens
        tokens = np.random.randint(1, 1000, (1, pos), dtype=np.int64)
        padded = np.zeros((1, prefill_len), dtype=np.int64)
        padded[0, :pos] = tokens[0]
        
        result = module.forward(padded)
        logits = result.to_host()
        
        # Check logits at last valid position
        last_logits = logits[0, pos-1, :]
        has_nan = np.isnan(last_logits).any()
        max_logit = np.max(last_logits)
        min_logit = np.min(last_logits)
        
        status = "❌ NaN" if has_nan else "✓ OK"
        print(f"Position {pos:>4}: {status} | logits range: [{min_logit:.2f}, {max_logit:.2f}]")


if __name__ == "__main__":
    main()

