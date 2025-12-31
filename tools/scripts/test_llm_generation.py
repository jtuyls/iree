#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test script for IREE LLM generation with paged attention.

This script demonstrates end-to-end text generation using an IREE-compiled
LLM model with paged attention KV-cache.

Key learnings:
1. Use rt.asdevicearray() for persistent device buffers (KV-cache must survive across calls)
2. Output shape is [1, block_stride, 1] - only first seq_len positions are valid
3. After prefill, next token is at position (seq_len - 1) in output
4. Use .to_host() to read device arrays back to CPU

Usage:
    python test_llm_generation.py \
        --vmfb /path/to/model.vmfb \
        --irpa /path/to/model.irpa \
        --tokenizer /path/to/tokenizer.model \
        --device hip \
        --prompt "The meaning of life is"
"""

import argparse
import numpy as np
import sys
import time

try:
    import iree.runtime as rt
except ImportError:
    print("Error: iree.runtime not found. Install with: pip install iree-runtime")
    sys.exit(1)

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

if not HAS_SENTENCEPIECE and not HAS_TRANSFORMERS:
    print("Error: Need either sentencepiece or transformers. Install with:")
    print("  pip install sentencepiece  # for .model files")
    print("  pip install transformers   # for .json files")
    sys.exit(1)


class TokenizerWrapper:
    """Wrapper to provide unified interface for different tokenizer types."""
    
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        
        if tokenizer_path.endswith('.json'):
            if not HAS_TRANSFORMERS:
                raise RuntimeError("transformers required for .json tokenizers")
            # Load from directory containing tokenizer.json
            import os
            tok_dir = os.path.dirname(tokenizer_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
            self.tokenizer_type = "bpe"
            self.bos_id = self.tokenizer.bos_token_id or 128000
            self.eos_id = self.tokenizer.eos_token_id or 128001
        else:
            if not HAS_SENTENCEPIECE:
                raise RuntimeError("sentencepiece required for .model tokenizers")
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(tokenizer_path)
            self.tokenizer_type = "sentencepiece"
            self.bos_id = self.tokenizer.bos_id()
            self.eos_id = self.tokenizer.eos_id()
    
    def encode(self, text: str) -> list:
        if self.tokenizer_type == "bpe":
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.tokenizer.encode(text)
    
    def decode(self, tokens: list) -> str:
        if self.tokenizer_type == "bpe":
            return self.tokenizer.decode(tokens)
        else:
            return self.tokenizer.decode(tokens)


class IREELLMGenerator:
    """IREE-based LLM text generator with paged attention."""
    
    def __init__(self, vmfb_path: str, irpa_path: str, tokenizer_path: str,
                 device: str = "hip", block_stride: int = 32,
                 device_block_count: int = 64, page_size: int = 5324800):
        self.block_stride = block_stride
        self.device_block_count = device_block_count
        self.page_size = page_size
        
        # Load tokenizer (auto-detect type)
        self.tokenizer = TokenizerWrapper(tokenizer_path)
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        
        # Load IREE module
        self.instance = rt.VmInstance()
        self.device = rt.get_device(device)
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
    
    def generate(self, prompt: str, max_tokens: int = 50,
                 stop_on_eos: bool = True, verbose: bool = False,
                 budget_aware: bool = False) -> str:
        """Generate text completion for the given prompt.
        
        Args:
            prompt: Input text to complete
            max_tokens: Maximum tokens to generate
            stop_on_eos: Stop generation on end-of-sequence token
            verbose: Print timing information
            budget_aware: Prepend instruction telling LLM to wrap up within token budget
        """
        
        # Optionally wrap prompt with budget instructions
        if budget_aware:
            prompt = (
                f"[Complete the following in approximately {max_tokens} words or less. "
                f"Wrap up with a complete sentence before reaching the limit.]\n\n"
                f"{prompt}"
            )
        
        # Create persistent device buffer for KV-cache
        cache_np = np.zeros((self.device_block_count, self.page_size), dtype=np.float16)
        cache_device = rt.asdevicearray(self.config.device, cache_np)
        
        # Tokenize input
        tokens = [self.bos_id] + self.tokenizer.encode(prompt)
        seq_len = len(tokens)
        
        if verbose:
            print(f"Input tokens: {tokens}")
        
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
            print(f"Prefill time: {prefill_time*1000:.1f}ms")
            print(f"First token: {next_token} = '{self.tokenizer.decode([next_token])}'")
        
        # Decode loop
        decode_start = time.time()
        for i in range(max_tokens - 1):
            if stop_on_eos and next_token == self.eos_id:
                if verbose:
                    print(f"EOS reached at step {i+1}")
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
        
        if verbose:
            tokens_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0
            print(f"Decode time: {decode_time*1000:.1f}ms ({tokens_per_sec:.1f} tok/s)")
        
        return self.tokenizer.decode(generated_tokens)


def main():
    parser = argparse.ArgumentParser(description="Test IREE LLM Generation")
    parser.add_argument("--vmfb", required=True, help="Path to compiled VMFB file")
    parser.add_argument("--irpa", required=True, help="Path to IRPA weights file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.model")
    parser.add_argument("--device", default="hip", help="IREE device (hip, local-task)")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Print timing info")
    parser.add_argument("--test-prompts", action="store_true",
                        help="Run multiple test prompts")
    parser.add_argument("--budget-aware", action="store_true",
                        help="Instruct LLM to wrap up within token budget")
    parser.add_argument("--block-stride", type=int, default=32,
                        help="Block sequence stride for paged attention")
    parser.add_argument("--device-block-count", type=int, default=64,
                        help="Number of blocks for KV cache on device")
    parser.add_argument("--page-size", type=int, default=None,
                        help="Page size in elements (auto-calculated if not specified)")
    args = parser.parse_args()
    
    print("Loading model...")
    generator = IREELLMGenerator(
        vmfb_path=args.vmfb,
        irpa_path=args.irpa,
        tokenizer_path=args.tokenizer,
        device=args.device,
        block_stride=args.block_stride,
        device_block_count=args.device_block_count,
        page_size=args.page_size if args.page_size else 5324800,  # Default from open_llama_3b
    )
    print("Model loaded!\n")
    
    if args.test_prompts:
        prompts = [
            "Once upon a time",
            "The capital of France is",
            "Python is a programming language that",
            "To make a cake, you need",
            "The quick brown fox",
        ]
        
        for prompt in prompts:
            output = generator.generate(
                prompt, max_tokens=30, verbose=args.verbose,
                budget_aware=args.budget_aware
            )
            print(f"Prompt: {prompt}")
            print(f"Output: {prompt}{output}")
            print()
    else:
        output = generator.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            budget_aware=args.budget_aware
        )
        print(f"Prompt: {args.prompt}")
        print(f"Output: {args.prompt}{output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

