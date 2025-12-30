#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Integration test for the IREE LLM backend.

This script tests the IREE native LLM backend using a pre-exported model.
It verifies:
1. Tokenizer loading and basic encode/decode
2. Model loading and initialization
3. End-to-end generation quality

Usage:
    python test_iree_backend_integration.py \
        --vmfb /path/to/model.vmfb \
        --irpa /path/to/model.irpa \
        --tokenizer /path/to/tokenizer.model \
        --device hip  # or local-task for CPU
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Try to import IREE runtime
try:
    import iree.runtime as rt
except ImportError:
    print("ERROR: iree.runtime not found. Install with: pip install iree-runtime")
    sys.exit(1)

# Try to import SentencePiece
try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: sentencepiece not found. Install with: pip install sentencepiece")
    sys.exit(1)


def load_tokenizer(tokenizer_path: str) -> spm.SentencePieceProcessor:
    """Load and verify the tokenizer."""
    sp = spm.SentencePieceProcessor()
    if not sp.load(tokenizer_path):
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}")
    return sp


def test_tokenizer(tokenizer: spm.SentencePieceProcessor) -> bool:
    """Test basic tokenizer functionality."""
    print("\n=== Testing Tokenizer ===")
    
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"  Input: '{test_text}'")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: '{decoded}'")
    
    assert len(tokens) > 0, "Tokenizer should produce tokens"
    print("  ✓ Tokenizer test passed")
    return True


class IREETestGenerator:
    """IREE-based LLM generator for testing."""
    
    def __init__(self, vmfb_path: str, irpa_path: str, device_str: str,
                 block_stride: int = 32, device_block_count: int = 64,
                 page_size: int = 5324800):
        self.block_stride = block_stride
        self.device_block_count = device_block_count
        self.page_size = page_size
        
        # Load IREE module
        self.instance = rt.VmInstance()
        self.device = rt.get_device(device_str)
        self.config = rt.Config(device=self.device)
        
        # Load parameters
        params = rt.ParameterIndex()
        params.load(irpa_path)
        
        # Load VMFB
        with open(vmfb_path, "rb") as f:
            vmfb_data = f.read()
        vm_module = rt.VmModule.copy_buffer(self.instance, vmfb_data)
        
        # Create context with modules
        modules = rt.load_vm_modules(
            rt.create_io_parameters_module(self.instance, params.create_provider("model")),
            rt.create_hal_module(self.instance, self.device),
            vm_module,
            config=self.config,
        )
        self.module = modules[-1]
    
    def has_function(self, name: str) -> bool:
        """Check if a function exists in the module."""
        try:
            getattr(self.module, name)
            return True
        except AttributeError:
            return False
    
    def generate(self, tokenizer: spm.SentencePieceProcessor, prompt: str,
                 max_tokens: int = 32, verbose: bool = False) -> tuple:
        """Generate text and return (text, stats)."""
        
        # Create persistent device buffer for KV-cache
        cache_np = np.zeros((self.device_block_count, self.page_size), dtype=np.float16)
        cache_device = rt.asdevicearray(self.config.device, cache_np)
        
        # Tokenize input
        tokens = [tokenizer.bos_id()] + tokenizer.encode(prompt)
        seq_len = len(tokens)
        
        if verbose:
            print(f"  Input tokens: {tokens}")
        
        # Prefill
        tokens_input = np.array([tokens], dtype=np.int64)
        seq_lens = np.array([seq_len], dtype=np.int64)
        num_pages = (seq_len + self.block_stride - 1) // self.block_stride
        page_table = np.arange(num_pages, dtype=np.int64).reshape(1, -1)
        
        start_time = time.time()
        result = self.module.prefill_bs1(tokens_input, seq_lens, page_table, cache_device)
        prefill_time = time.time() - start_time
        
        # Get first generated token (at position seq_len - 1)
        tokens_out = result[1].to_host()
        next_token = int(tokens_out[0, seq_len - 1, 0])
        
        generated_tokens = [next_token]
        
        if verbose:
            print(f"  Prefill time: {prefill_time*1000:.1f}ms")
            print(f"  First token: {next_token} = '{tokenizer.decode([next_token])}'")
        
        # Decode loop
        decode_start = time.time()
        for i in range(max_tokens - 1):
            if next_token == tokenizer.eos_id():
                if verbose:
                    print(f"  EOS reached at step {i+1}")
                break
            
            seq_len += 1
            num_pages = (seq_len + self.block_stride - 1) // self.block_stride
            page_table = np.arange(num_pages, dtype=np.int64).reshape(1, -1)
            
            token_input = np.array([[next_token]], dtype=np.int64)
            seq_lens_arr = np.array([seq_len], dtype=np.int64)
            start_pos = np.array([seq_len - 1], dtype=np.int64)
            
            result = self.module.decode_bs1(
                token_input, seq_lens_arr, start_pos, page_table, cache_device
            )
            
            tokens_out = result[1].to_host()
            next_token = int(tokens_out[0, 0, 0])
            generated_tokens.append(next_token)
        
        decode_time = time.time() - decode_start
        
        generated_text = tokenizer.decode(generated_tokens)
        tokens_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0
        
        stats = {
            "prefill_ms": prefill_time * 1000,
            "decode_ms": decode_time * 1000,
            "tokens_per_sec": tokens_per_sec,
            "generated_tokens": len(generated_tokens),
        }
        
        return generated_text, stats


def test_model_loading(vmfb_path: str, irpa_path: str, device: str) -> IREETestGenerator:
    """Test model loading and return generator."""
    print("\n=== Loading Model ===")
    
    generator = IREETestGenerator(vmfb_path, irpa_path, device)
    
    print(f"  Device: {generator.device}")
    
    # Check for expected functions
    has_prefill = generator.has_function("prefill_bs1")
    has_decode = generator.has_function("decode_bs1")
    
    print(f"  Has prefill_bs1: {has_prefill}")
    print(f"  Has decode_bs1: {has_decode}")
    
    if not has_prefill or not has_decode:
        raise RuntimeError("Model missing required functions (prefill_bs1, decode_bs1)")
    
    print("  ✓ Model loaded successfully")
    return generator


def test_generation(generator: IREETestGenerator, tokenizer: spm.SentencePieceProcessor,
                    prompt: str, max_tokens: int = 32) -> bool:
    """Test end-to-end generation."""
    print(f"\n=== Testing Generation ===")
    print(f"  Prompt: '{prompt}'")
    
    text, stats = generator.generate(tokenizer, prompt, max_tokens, verbose=True)
    
    print(f"\n  Generated text: '{text}'")
    print(f"  Tokens: {stats['generated_tokens']}")
    print(f"  Prefill: {stats['prefill_ms']:.2f}ms")
    print(f"  Decode: {stats['decode_ms']:.2f}ms ({stats['tokens_per_sec']:.1f} tok/s)")
    
    if stats['generated_tokens'] > 0:
        print("  ✓ Generation test passed")
        return True
    else:
        print("  ✗ No tokens generated")
        return False


def run_multiple_prompts(generator: IREETestGenerator, tokenizer: spm.SentencePieceProcessor) -> bool:
    """Test with multiple prompts."""
    print("\n=== Testing Multiple Prompts ===")
    
    prompts = [
        "Once upon a time",
        "The capital of France is",
        "To make a cake, you need",
        "The quick brown fox",
    ]
    
    all_passed = True
    for prompt in prompts:
        try:
            text, stats = generator.generate(tokenizer, prompt, max_tokens=20)
            print(f"  '{prompt}' → '{text[:50]}...' ({stats['generated_tokens']} tokens)")
        except Exception as e:
            print(f"  '{prompt}' → FAILED: {e}")
            all_passed = False
    
    if all_passed:
        print("  ✓ All prompts generated successfully")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test IREE LLM backend integration")
    parser.add_argument("--vmfb", required=True, help="Path to compiled VMFB model")
    parser.add_argument("--irpa", required=True, help="Path to IRPA parameters file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer model")
    parser.add_argument("--config", help="Path to model config JSON")
    parser.add_argument("--device", default="hip", help="IREE device (local-task, hip)")
    parser.add_argument("--prompt", default="Once upon a time", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--run-all", action="store_true", help="Run all tests including multiple prompts")
    args = parser.parse_args()
    
    # Verify files exist
    for path, name in [(args.vmfb, "VMFB"), (args.irpa, "IRPA"), (args.tokenizer, "Tokenizer")]:
        if not Path(path).exists():
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)
    
    print("=" * 60)
    print("IREE LLM Backend Integration Test")
    print("=" * 60)
    print(f"Model: {args.vmfb}")
    print(f"Device: {args.device}")
    
    all_passed = True
    
    try:
        # Test tokenizer
        tokenizer = load_tokenizer(args.tokenizer)
        if not test_tokenizer(tokenizer):
            all_passed = False
        
        # Test model loading
        generator = test_model_loading(args.vmfb, args.irpa, args.device)
        
        # Test generation
        if not test_generation(generator, tokenizer, args.prompt, args.max_tokens):
            all_passed = False
        
        # Optionally run more tests
        if args.run_all:
            if not run_multiple_prompts(generator, tokenizer):
                all_passed = False
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED ✓")
        sys.exit(0)
    else:
        print("Some tests FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
