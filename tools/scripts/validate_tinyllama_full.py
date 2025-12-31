#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Comprehensive validation of TinyLlama with:
1. Position tests (including beyond 512)
2. KV cache incremental decoding  
3. Full text generation loop

Supports both PyTorch and IREE backends.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model-dir", default=None, help="For IREE: directory with vmfb and weights")
    parser.add_argument("--use-iree", action="store_true", help="Use IREE backend")
    parser.add_argument("--device", default="cpu", help="Device (cpu, hip, cuda)")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--test-positions", type=int, nargs="+", 
                        default=[10, 100, 500, 511, 512, 513, 600, 1000, 1500])
    parser.add_argument("--verbose", action="store_true")
    return parser


class PyTorchGenerator:
    """PyTorch-based generator with KV cache."""
    
    def __init__(self, model_name: str, dtype: torch.dtype, device: str, max_seq_len: int):
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StaticCache
        
        print(f"Loading PyTorch model: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.eval()
        self.model.to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        # Cache setup
        self.static_cache = StaticCache(
            config=self.config,
            max_batch_size=1,
            max_cache_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  KV heads: {getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads)}")
    
    def test_positions(self, positions: list):
        """Test decode at various positions."""
        print("\n=== Testing Positions ===")
        results = {}
        
        for pos in positions:
            if pos >= self.max_seq_len:
                print(f"  Position {pos}: SKIPPED (beyond max_seq_len)")
                continue
            
            self.static_cache.reset()
            tokens = torch.randint(0, self.config.vocab_size, (1, pos), 
                                    dtype=torch.int64, device=self.device)
            cache_position = torch.arange(0, pos, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=tokens,
                    cache_position=cache_position,
                    past_key_values=self.static_cache,
                    use_cache=True,
                    return_dict=True,
                )
            
            has_nan = torch.isnan(outputs.logits).any().item()
            max_logit = outputs.logits.max().item()
            min_logit = outputs.logits.min().item()
            
            results[pos] = {"nan": has_nan, "max": max_logit, "min": min_logit}
            status = "❌ NaN!" if has_nan else "✓ OK"
            print(f"  Position {pos}: {status} logits range: [{min_logit:.2f}, {max_logit:.2f}]")
        
        return results
    
    def generate(self, prompt: str, max_tokens: int, verbose: bool = False):
        """Generate text with KV cache."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        seq_len = input_ids.shape[1]
        
        if verbose:
            print(f"Input: {seq_len} tokens")
        
        # Reset cache
        self.static_cache.reset()
        
        # Prefill
        start_time = time.time()
        cache_position = torch.arange(0, seq_len, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                cache_position=cache_position,
                past_key_values=self.static_cache,
                use_cache=True,
                return_dict=True,
            )
        prefill_time = time.time() - start_time
        
        # Get first token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        if verbose:
            print(f"Prefill: {prefill_time*1000:.1f}ms")
            print(f"First token: {next_token} = '{self.tokenizer.decode([next_token])}'")
        
        generated = [next_token]
        
        # Decode loop
        decode_start = time.time()
        current_pos = seq_len
        
        for i in range(max_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                if verbose:
                    print(f"EOS at step {i+1}")
                break
            
            if current_pos >= self.max_seq_len - 1:
                if verbose:
                    print(f"Max sequence length reached at {current_pos}")
                break
            
            token_input = torch.tensor([[next_token]], dtype=torch.int64, device=self.device)
            cache_position = torch.tensor([current_pos], device=self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=token_input,
                    cache_position=cache_position,
                    past_key_values=self.static_cache,
                    use_cache=True,
                    return_dict=True,
                )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Check for NaN
            if torch.isnan(next_token_logits).any():
                print(f"  ❌ NaN at position {current_pos}!")
                break
            
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated.append(next_token)
            current_pos += 1
        
        decode_time = time.time() - decode_start
        
        if verbose:
            tok_per_sec = len(generated) / decode_time if decode_time > 0 else 0
            print(f"Decode: {decode_time*1000:.1f}ms ({tok_per_sec:.1f} tok/s)")
            print(f"Generated {len(generated)} tokens, final position: {current_pos}")
        
        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return prompt + output_text


class IREEGenerator:
    """IREE-based generator (prefill only for now)."""
    
    def __init__(self, model_dir: str, device: str):
        import iree.runtime as rt
        from transformers import AutoTokenizer
        
        model_dir = Path(model_dir)
        
        with open(model_dir / "config.json") as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        
        vmfb_path = model_dir / "model_gfx950.vmfb"
        if not vmfb_path.exists():
            vmfb_path = model_dir / "model.vmfb"
        
        print(f"Loading IREE model: {vmfb_path}")
        
        instance = rt.VmInstance()
        self.rt_device = rt.get_device(device)
        rt_config = rt.Config(device=self.rt_device)
        
        with open(vmfb_path, "rb") as f:
            vm_module = rt.VmModule.copy_buffer(instance, f.read())
        
        # Load parameters
        safetensors_path = model_dir / "model.safetensors"
        params = rt.ParameterIndex()
        params.load(str(safetensors_path))
        
        hal_module = rt.create_hal_module(instance, self.rt_device)
        params_module = rt.create_io_parameters_module(
            instance, params.create_provider("model")
        )
        
        modules = rt.load_vm_modules(params_module, hal_module, vm_module, config=rt_config)
        self.module = modules[-1]
        
        print(f"  Layers: {self.config['num_layers']}")
        print(f"  Prefill length: {self.config['prefill_len']}")
    
    def generate(self, prompt: str, max_tokens: int, verbose: bool = False):
        """Generate using IREE with autoregressive loop.
        
        Since we only have a fixed-length forward pass, we re-run the model
        for each new token, prepending the generated tokens to the input.
        """
        input_ids = self.tokenizer.encode(prompt)
        orig_len = len(input_ids)
        prefill_len = self.config["prefill_len"]
        
        if verbose:
            print(f"Input: {orig_len} tokens")
        
        generated = []
        total_time = 0
        
        for step in range(max_tokens):
            # Build current sequence: prompt + generated tokens
            current_seq = input_ids + generated
            current_len = len(current_seq)
            
            if current_len >= prefill_len:
                if verbose:
                    print(f"Max prefill length reached at {current_len}")
                break
            
            # Pad to prefill_len
            padded = [0] * prefill_len
            padded[:current_len] = current_seq
            input_array = np.array([padded], dtype=np.int64)
            
            # Run forward
            start_time = time.time()
            result = self.module.forward(input_array)
            step_time = time.time() - start_time
            total_time += step_time
            
            logits = result.to_host()
            
            # Get next token from last valid position
            next_logits = logits[0, current_len - 1, :]
            
            if np.isnan(next_logits).any():
                print(f"  ❌ NaN at position {current_len - 1}!")
                break
            
            next_token = int(np.argmax(next_logits))
            generated.append(next_token)
            
            if verbose and step == 0:
                print(f"First token: {next_token} = '{self.tokenizer.decode([next_token])}'")
            
            if next_token == self.tokenizer.eos_token_id:
                if verbose:
                    print(f"EOS at step {step + 1}")
                break
        
        if verbose:
            tok_per_sec = len(generated) / total_time if total_time > 0 else 0
            print(f"Generated {len(generated)} tokens in {total_time*1000:.1f}ms ({tok_per_sec:.1f} tok/s)")
        
        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return prompt + output_text


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.use_iree:
        if not args.model_dir:
            parser.error("--model-dir required for IREE backend")
        generator = IREEGenerator(args.model_dir, args.device)
    else:
        dtype = getattr(torch, args.dtype)
        generator = PyTorchGenerator(args.model, dtype, args.device, args.max_seq_len)
    
    # Test positions (PyTorch only)
    if hasattr(generator, 'test_positions'):
        results = generator.test_positions(args.test_positions)
        
        # Summary
        passed = sum(1 for r in results.values() if not r["nan"])
        total = len(results)
        print(f"\n✓ Position tests: {passed}/{total} passed")
        
        if passed == total:
            print("  All positions work - no 512 limit!")
    
    # Generate text
    print(f"\n=== Text Generation ===")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    
    output = generator.generate(args.prompt, args.max_tokens, args.verbose)
    
    print(f"\n--- Output ---")
    print(output)
    print("--- End ---")
    
    # Test long context generation (PyTorch only)
    if hasattr(generator, 'static_cache') and args.max_seq_len >= 700:
        print("\n=== Long Context Test (start at 600) ===")
        # Create a long prompt
        long_prompt = "A " * 600 + "The answer is"
        print(f"Prompt: {len(long_prompt.split())} words (starts at ~600 tokens)")
        
        output = generator.generate(long_prompt, 20, verbose=True)
        print(f"Output (truncated): ...{output[-100:]}")
    
    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()

