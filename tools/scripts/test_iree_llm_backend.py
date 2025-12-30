#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test script for the IREE LLM backend.

This script tests loading and running a compiled LLM VMFB with parameters.
It's useful for verifying the model export and understanding the function signatures.
"""

import argparse
import json
import numpy as np
import sys

# Try to import iree runtime
try:
    import iree.runtime as ireert
except ImportError:
    print("Error: iree.runtime not found. Make sure to install iree-runtime.")
    sys.exit(1)


def load_model(vmfb_path: str, irpa_path: str, device: str = "local-task"):
    """Load the VMFB module with parameters."""
    # Create configuration with parameters
    config = ireert.Config(device)

    # Load parameters
    params = ireert.ParameterIndex()
    params.load(irpa_path)

    # Create the module with parameters
    ctx = ireert.SystemContext(config=config)
    ctx.set_parameter_index(params)

    # Load the module
    with open(vmfb_path, "rb") as f:
        vmfb_data = f.read()
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_data)
    ctx.add_vm_module(vm_module)

    return ctx


def print_function_info(ctx):
    """Print information about exported functions."""
    module = ctx.modules.module
    print("\n=== Exported Functions ===")

    # Try to access functions
    for name in ["prefill_bs1", "decode_bs1", "prefill_bs1$async", "decode_bs1$async"]:
        try:
            fn = getattr(module, name.replace("$", "_"))
            print(f"  - {name}: Found")
        except AttributeError:
            print(f"  - {name}: Not found")


def test_prefill(ctx, tokens):
    """Test the prefill function with sample tokens."""
    print("\n=== Testing Prefill ===")
    module = ctx.modules.module

    # Get the device
    hal_device = ctx.system_bundle.device

    # Convert tokens to numpy array
    tokens_np = np.array(tokens, dtype=np.int64).reshape(1, -1)  # [batch=1, seq_len]
    seq_len = np.array([len(tokens)], dtype=np.int64)

    print(f"Input tokens shape: {tokens_np.shape}")
    print(f"Input tokens: {tokens_np}")
    print(f"Sequence length: {seq_len}")

    # TODO: Set up KV-cache and page IDs based on model config
    # This is model-specific and needs the cache allocation

    print("\nNote: Full inference requires KV-cache setup, which is model-specific.")
    print("The C++ IREEBackend handles this via KVCacheManager.")


def main():
    parser = argparse.ArgumentParser(description="Test IREE LLM Backend")
    parser.add_argument("--vmfb", required=True, help="Path to VMFB file")
    parser.add_argument("--irpa", required=True, help="Path to IRPA file")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--device", default="local-task", help="IREE device")
    parser.add_argument("--test-tokens", type=str, default="1,2,3,4,5",
                        help="Comma-separated test token IDs")
    args = parser.parse_args()

    print("=== IREE LLM Backend Test ===")
    print(f"VMFB: {args.vmfb}")
    print(f"IRPA: {args.irpa}")
    print(f"Device: {args.device}")

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        print(f"\nModel config: {json.dumps(config.get('llm_assist', {}), indent=2)}")

    # Load model
    print("\nLoading model...")
    try:
        ctx = load_model(args.vmfb, args.irpa, args.device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1

    # Print function info
    print_function_info(ctx)

    # Test prefill
    tokens = [int(x) for x in args.test_tokens.split(",")]
    test_prefill(ctx, tokens)

    print("\n=== Test Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())


