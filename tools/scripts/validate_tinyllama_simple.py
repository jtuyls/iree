#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Validate TinyLlama simple export with IREE.
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
    args = parser.parse_args()
    
    import iree.runtime as rt
    from transformers import AutoTokenizer
    from safetensors import safe_open
    
    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    print(f"Model: {config['num_layers']} layers, {config['hidden_size']} hidden")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    
    # Load VMFB
    vmfb_path = model_dir / "model_gfx950.vmfb"
    if not vmfb_path.exists():
        vmfb_path = model_dir / "model.vmfb"
    
    print(f"Loading VMFB: {vmfb_path}")
    instance = rt.VmInstance()
    device = rt.get_device(args.device)
    rt_config = rt.Config(device=device)
    
    with open(vmfb_path, "rb") as f:
        vm_module = rt.VmModule.copy_buffer(instance, f.read())
    
    # Load parameters from safetensors
    safetensors_path = model_dir / "model.safetensors"
    print(f"Loading weights: {safetensors_path}")
    
    params = rt.ParameterIndex()
    params.load(str(safetensors_path))
    
    # Create modules
    hal_module = rt.create_hal_module(instance, device)
    params_module = rt.create_io_parameters_module(
        instance, params.create_provider("model")
    )
    
    modules = rt.load_vm_modules(
        params_module,
        hal_module,
        vm_module,
        config=rt_config,
    )
    module = modules[-1]
    
    print("Model loaded!")
    
    # Encode prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="np")
    print(f"\nPrompt: {args.prompt}")
    print(f"Input tokens: {input_ids.shape}")
    
    # Need to pad to prefill_len
    prefill_len = config.get("prefill_len", 128)
    if input_ids.shape[1] < prefill_len:
        padded = np.zeros((1, prefill_len), dtype=np.int64)
        padded[0, :input_ids.shape[1]] = input_ids[0]
        input_ids = padded
    
    # Run forward
    print("\nRunning forward pass...")
    start_time = time.time()
    result = module.forward(input_ids)
    forward_time = time.time() - start_time
    
    logits = result.to_host()
    print(f"Forward time: {forward_time*1000:.1f}ms")
    print(f"Output logits: {logits.shape}")
    print(f"Has NaN: {np.isnan(logits).any()}")
    
    # Get predicted token for original input length
    orig_len = len(tokenizer.encode(args.prompt))
    next_token_logits = logits[0, orig_len - 1, :]
    next_token = int(np.argmax(next_token_logits))
    print(f"\nPredicted next token: {next_token} = '{tokenizer.decode([next_token])}'")
    
    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()

