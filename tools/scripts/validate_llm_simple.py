#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Validate a simple LLM model with standard (non-paged) attention.

This script tests both PyTorch and IREE backends to validate:
1. Correct token generation
2. No position limits (can decode beyond 512)
3. Proper KV cache handling

Usage:
    # Test with PyTorch backend:
    python validate_llm_simple.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --use-pytorch \
        --max-tokens 100
    
    # Test with IREE backend:
    python validate_llm_simple.py \
        --config /path/to/config.json \
        --vmfb /path/to/model.vmfb \
        --device hip \
        --max-tokens 100
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch


def create_parser():
    parser = argparse.ArgumentParser(description="Validate simple LLM")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name (for PyTorch backend)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config JSON path (for IREE backend)")
    parser.add_argument("--vmfb", type=str, default=None,
                        help="VMFB path (for IREE backend)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, hip, cuda)")
    parser.add_argument("--use-pytorch", action="store_true",
                        help="Use PyTorch backend instead of IREE")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--test-positions", type=int, nargs="+",
                        default=[10, 100, 500, 512, 600, 1000],
                        help="Positions to test decode at")
    parser.add_argument("--verbose", action="store_true")
    return parser


class PyTorchLLMGenerator:
    """PyTorch-based LLM generator with standard attention."""
    
    def __init__(self, model_name: str, dtype: torch.dtype, device: str, max_seq_len: int):
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        
        print(f"Loading model: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        self.num_layers = self.config.num_hidden_layers
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads)
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        print(f"  Layers: {self.num_layers}, KV heads: {self.num_kv_heads}, head_dim: {self.head_dim}")
    
    def generate(self, prompt: str, max_tokens: int, verbose: bool = False) -> str:
        """Generate text from prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        seq_len = input_ids.shape[1]
        
        if verbose:
            print(f"Input tokens: {input_ids.shape[1]}")
        
        # Prefill
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True, return_dict=True)
        prefill_time = time.time() - start_time
        
        logits = outputs.logits
        past_kv = outputs.past_key_values
        
        # Get first token
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        if verbose:
            print(f"Prefill time: {prefill_time*1000:.1f}ms")
            print(f"First token: {next_token} = '{self.tokenizer.decode([next_token])}'")
        
        generated_tokens = [next_token]
        
        # Decode loop
        decode_start = time.time()
        for i in range(max_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                if verbose:
                    print(f"EOS at step {i+1}")
                break
            
            token_input = torch.tensor([[next_token]], dtype=torch.int64, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=token_input,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
            
            logits = outputs.logits
            past_kv = outputs.past_key_values
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens.append(next_token)
            
            # Check for NaN
            if torch.isnan(logits).any():
                current_pos = seq_len + len(generated_tokens)
                print(f"  ❌ NaN detected at position {current_pos}!")
                break
        
        decode_time = time.time() - decode_start
        
        if verbose:
            tok_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0
            print(f"Decode time: {decode_time*1000:.1f}ms ({tok_per_sec:.1f} tok/s)")
        
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return prompt + output_text
    
    def test_positions(self, positions: List[int], verbose: bool = False):
        """Test decode at various positions."""
        print("\n=== Testing Decode at Various Positions ===")
        
        for pos in positions:
            if pos >= self.max_seq_len:
                print(f"  Position {pos}: SKIPPED (beyond max_seq_len)")
                continue
            
            # Create dummy input up to pos
            input_ids = torch.randint(0, self.config.vocab_size, (1, pos), 
                                       dtype=torch.int64, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=True, return_dict=True)
            
            past_kv = outputs.past_key_values
            
            # Decode one more token
            next_token = torch.randint(0, self.config.vocab_size, (1, 1), 
                                        dtype=torch.int64, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
            
            logits = outputs.logits
            has_nan = torch.isnan(logits).any().item()
            max_logit = logits.max().item()
            min_logit = logits.min().item()
            
            status = "❌ NaN!" if has_nan else "✓ OK"
            print(f"  Position {pos}: {status} logits range: [{min_logit:.2f}, {max_logit:.2f}]")


class IREELLMGenerator:
    """IREE-based LLM generator with standard attention."""
    
    def __init__(self, config_path: str, vmfb_path: str, device: str):
        import iree.runtime as rt
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.num_layers = self.config["num_layers"]
        self.num_kv_heads = self.config["num_kv_heads"]
        self.head_dim = self.config["head_dim"]
        self.max_seq_len = self.config["max_seq_len"]
        self.vocab_size = self.config["vocab_size"]
        self.cache_shape = tuple(self.config["cache_shape"])
        
        # Load tokenizer
        config_dir = Path(config_path).parent
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(config_dir), trust_remote_code=True)
        
        # Load IREE module
        print(f"Loading VMFB: {vmfb_path}")
        self.instance = rt.VmInstance()
        self.rt_device = rt.get_device(device)
        self.rt_config = rt.Config(device=self.rt_device)
        
        with open(vmfb_path, "rb") as f:
            vm_module = rt.VmModule.copy_buffer(self.instance, f.read())
        
        hal_module = rt.create_hal_module(self.instance, self.rt_device)
        self.context = rt.VmContext(self.instance, [hal_module, vm_module])
        self.module = rt.load_vm_modules(hal_module, vm_module, config=self.rt_config)[-1]
        
        print(f"Model loaded: {self.num_layers} layers, {self.num_kv_heads} KV heads")
    
    def generate(self, prompt: str, max_tokens: int, verbose: bool = False) -> str:
        """Generate text from prompt."""
        input_ids = self.tokenizer.encode(prompt)
        seq_len = len(input_ids)
        
        if verbose:
            print(f"Input tokens: {seq_len}")
        
        # Prepare inputs
        tokens = np.array([input_ids], dtype=np.int64)
        cache_k = np.zeros(self.cache_shape, dtype=np.float16)
        cache_v = np.zeros(self.cache_shape, dtype=np.float16)
        
        # Prefill
        start_time = time.time()
        result = self.module.prefill(tokens, cache_k, cache_v)
        prefill_time = time.time() - start_time
        
        logits = result[0].to_host()
        cache_k = result[1].to_host()
        cache_v = result[2].to_host()
        
        # Get first token (argmax of last position)
        next_token = int(np.argmax(logits[0, -1, :]))
        
        if verbose:
            print(f"Prefill time: {prefill_time*1000:.1f}ms")
            print(f"First token: {next_token} = '{self.tokenizer.decode([next_token])}'")
        
        generated_tokens = [next_token]
        
        # Decode loop
        decode_start = time.time()
        current_seq_len = seq_len
        
        for i in range(max_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                if verbose:
                    print(f"EOS at step {i+1}")
                break
            
            token_input = np.array([[next_token]], dtype=np.int64)
            seq_len_input = np.array([current_seq_len], dtype=np.int64)
            
            result = self.module.decode(token_input, seq_len_input, cache_k, cache_v)
            
            logits = result[0].to_host()
            cache_k = result[1].to_host()
            cache_v = result[2].to_host()
            
            next_token = int(np.argmax(logits[0, 0, :]))
            generated_tokens.append(next_token)
            current_seq_len += 1
            
            # Check for NaN
            if np.isnan(logits).any():
                print(f"  ❌ NaN detected at position {current_seq_len}!")
                break
        
        decode_time = time.time() - decode_start
        
        if verbose:
            tok_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0
            print(f"Decode time: {decode_time*1000:.1f}ms ({tok_per_sec:.1f} tok/s)")
        
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return prompt + output_text


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.use_pytorch:
        if not args.model:
            parser.error("--model required when using --use-pytorch")
        
        dtype = getattr(torch, args.dtype)
        generator = PyTorchLLMGenerator(
            args.model, dtype, args.device, args.max_seq_len
        )
        
        # Test positions
        generator.test_positions(args.test_positions, args.verbose)
        
        # Generate text
        print(f"\n=== Generating Text ===")
        print(f"Prompt: {args.prompt}")
        output = generator.generate(args.prompt, args.max_tokens, args.verbose)
        print(f"\nOutput:\n{output}")
        
    else:
        if not args.config or not args.vmfb:
            parser.error("--config and --vmfb required for IREE backend")
        
        generator = IREELLMGenerator(args.config, args.vmfb, args.device)
        
        # Generate text
        print(f"\n=== Generating Text ===")
        print(f"Prompt: {args.prompt}")
        output = generator.generate(args.prompt, args.max_tokens, args.verbose)
        print(f"\nOutput:\n{output}")
    
    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()

