#!/usr/bin/env python3
"""
Validate LLM generation using HuggingFace transformers directly.
This script tests that the model works correctly with standard attention
(no paged attention) to verify the 512 position issue is paged-attention specific.
"""

import argparse
import time
import torch


def main():
    parser = argparse.ArgumentParser(description="Validate LLM with HuggingFace")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace model name or local path")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--test-long-context", action="store_true",
                        help="Test with long context (>512 tokens)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}, dtype: {args.dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.hidden_size} hidden")
    
    # Test 1: Normal generation
    print("\n=== Test 1: Normal Generation ===")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    print(f"Input tokens: {inputs.input_ids.shape[1]}")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    
    print(f"Generated {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
    print(f"Output: {generated_text}")
    
    # Test 2: Manual prefill + decode with KV cache
    print("\n=== Test 2: Manual Prefill + Decode ===")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    
    # Prefill
    start_time = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            use_cache=True,
            return_dict=True,
        )
    prefill_time = time.time() - start_time
    
    logits = outputs.logits
    kv_cache = outputs.past_key_values
    next_token = logits[0, -1].argmax().item()
    
    print(f"Prefill time: {prefill_time*1000:.1f}ms")
    print(f"Logits shape: {logits.shape}")
    print(f"KV cache: {len(kv_cache)} layers, k/v shape: {kv_cache[0][0].shape}")
    print(f"First token: {next_token} = '{tokenizer.decode([next_token])}'")
    
    # Decode loop
    generated_tokens = [next_token]
    decode_start = time.time()
    
    for i in range(args.max_tokens - 1):
        input_ids = torch.tensor([[next_token]], device=args.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                use_cache=True,
                return_dict=True,
            )
        
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        next_token = logits[0, -1].argmax().item()
        generated_tokens.append(next_token)
        
        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            print(f"Hit EOS at step {i+1}")
            break
    
    decode_time = time.time() - decode_start
    
    print(f"Decode time: {decode_time*1000:.1f}ms for {len(generated_tokens)} tokens")
    print(f"Generated: {tokenizer.decode(generated_tokens)}")
    
    # Test 3: Long context (if requested)
    if args.test_long_context:
        print("\n=== Test 3: Long Context (>512 tokens) ===")
        
        # Create a long prompt
        long_prompt = "Hello world. " * 100  # About 300 tokens
        inputs = tokenizer(long_prompt, return_tensors="pt").to(args.device)
        seq_len = inputs.input_ids.shape[1]
        print(f"Long prompt tokens: {seq_len}")
        
        # Prefill
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                use_cache=True,
                return_dict=True,
            )
        
        logits = outputs.logits
        kv_cache = outputs.past_key_values
        print(f"Prefill logits shape: {logits.shape}")
        print(f"KV cache seq_len: {kv_cache[0][0].shape[2]}")
        
        # Generate tokens past position 512
        next_token = logits[0, -1].argmax().item()
        generated_tokens = [next_token]
        
        target_positions = [510, 511, 512, 513, 520, 600]
        current_pos = seq_len
        
        for target_pos in target_positions:
            # Generate tokens until we reach target position
            while current_pos < target_pos:
                input_ids = torch.tensor([[next_token]], device=args.device)
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        past_key_values=kv_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                
                logits = outputs.logits
                kv_cache = outputs.past_key_values
                next_token = logits[0, -1].argmax().item()
                generated_tokens.append(next_token)
                current_pos += 1
            
            # Check logits at this position
            logit_val = logits[0, 0, next_token].item()
            is_nan = torch.isnan(logits[0, 0]).any().item()
            cache_seq_len = kv_cache[0][0].shape[2]
            
            print(f"Position {current_pos}: token={next_token}, "
                  f"logit={logit_val:.4f}, NaN={is_nan}, cache_len={cache_seq_len}")
        
        print(f"\nGenerated {len(generated_tokens)} tokens beyond position 512 successfully!")
        print(f"Last 10 tokens: {tokenizer.decode(generated_tokens[-10:])}")

if __name__ == "__main__":
    main()

