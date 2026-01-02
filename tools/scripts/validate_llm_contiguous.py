#!/usr/bin/env python3
"""
Validate contiguous cache LLM model with in-place updates.
Compares against HuggingFace reference and measures performance.
Supports both PyTorch and IREE runtime validation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def run_iree_inference(model_dir, config, tokenizer, prompt, max_new_tokens=20, 
                       device="hip", num_warmup=3, num_runs=10, vmfb_path=None):
    """Run inference and profiling using IREE runtime."""
    import iree.runtime as rt
    
    vmfb_path = Path(vmfb_path) if vmfb_path else model_dir / "model.vmfb"
    irpa_path = model_dir / "model.irpa"
    
    if not vmfb_path.exists():
        print(f"ERROR: VMFB not found at {vmfb_path}")
        print("Run: iree-compile model.mlir -o model.vmfb --iree-hal-target-backends=rocm --iree-hip-target=gfx950")
        return None
    
    print(f"\n=== IREE Runtime Inference ({device}) ===")
    print(f"VMFB: {vmfb_path}")
    print(f"Weights: {irpa_path}")
    
    # Setup IREE runtime
    rt_config = rt.Config(device)
    rt_device = rt_config.device
    
    with open(vmfb_path, "rb") as f:
        vmfb_data = f.read()
    
    params = rt.ParameterIndex()
    params.load(str(irpa_path))
    
    instance = rt.VmInstance()
    hal_module = rt.create_hal_module(instance, rt_device)
    params_module = rt.create_io_parameters_module(instance, params.create_provider("model"))
    vm_module = rt.VmModule.copy_buffer(instance, vmfb_data)
    
    modules = rt.load_vm_modules(params_module, hal_module, vm_module, config=rt_config)
    module = modules[-1]
    
    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = tokens.shape[1]
    cache_len = config['cache_len']
    
    # Prepare prefill inputs
    padded_tokens = np.zeros((1, cache_len), dtype=np.int64)
    padded_tokens[0, :prompt_len] = tokens[0].numpy()
    attention_mask = np.zeros((1, cache_len), dtype=np.int64)
    attention_mask[0, :prompt_len] = 1
    
    print(f"Prompt: '{prompt}' -> {prompt_len} tokens")
    
    # Warmup prefill
    print(f"\nWarming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        result = module.prefill(padded_tokens, attention_mask)
        # Force sync
        _ = result[0].to_host()
    
    # Profile prefill
    print(f"Profiling prefill ({num_runs} runs)...")
    prefill_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = module.prefill(padded_tokens, attention_mask)
        logits_host = result[0].to_host()
        prefill_times.append(time.perf_counter() - t0)
    
    avg_prefill = sum(prefill_times) / len(prefill_times) * 1000
    print(f"Prefill: {avg_prefill:.2f} ms ({prompt_len} tokens)")
    
    # Get cache from last prefill
    cache = result[1]  # Keep on device
    cache_host = cache.to_host()
    
    # Get first generated token
    next_token = int(np.argmax(logits_host[0, prompt_len - 1, :]))
    print(f"First token: {next_token} = '{tokenizer.decode([next_token])}'")
    
    # Profile decode
    print(f"\nProfiling decode ({num_runs} runs)...")
    decode_times = []
    
    for _ in range(num_runs):
        token_input = np.array([[next_token]], dtype=np.int64)
        position_id = np.array([[prompt_len]], dtype=np.int64)
        valid_len = np.array([[prompt_len + 1]], dtype=np.int64)
        
        # Use fresh cache each time
        cache_dev = rt.asdevicearray(rt_device, cache_host.copy())
        
        t0 = time.perf_counter()
        result = module.decode_step(token_input, position_id, cache_dev, valid_len)
        logits_host = result[0].to_host()
        decode_times.append(time.perf_counter() - t0)
    
    avg_decode = sum(decode_times) / len(decode_times) * 1000
    tokens_per_sec = 1000.0 / avg_decode
    print(f"Decode: {avg_decode:.2f} ms = {tokens_per_sec:.1f} tok/s")
    
    # Full generation
    print(f"\nFull generation ({max_new_tokens} tokens)...")
    
    # Fresh prefill
    t_start = time.perf_counter()
    result = module.prefill(padded_tokens, attention_mask)
    logits_host = result[0].to_host()
    cache_host = result[1].to_host()
    
    next_token = int(np.argmax(logits_host[0, prompt_len - 1, :]))
    generated_tokens = [next_token]
    pos = prompt_len
    
    for step in range(max_new_tokens - 1):
        token_input = np.array([[generated_tokens[-1]]], dtype=np.int64)
        position_id = np.array([[pos]], dtype=np.int64)
        valid_len = np.array([[pos + 1]], dtype=np.int64)
        
        cache_dev = rt.asdevicearray(rt_device, cache_host)
        result = module.decode_step(token_input, position_id, cache_dev, valid_len)
        
        logits_host = result[0].to_host()
        cache_host = result[1].to_host()
        
        next_token = int(np.argmax(logits_host[0, 0, :]))
        generated_tokens.append(next_token)
        pos += 1
        
        if next_token == tokenizer.eos_token_id:
            break
        
        if pos >= cache_len:
            print(f"  Cache full at position {pos}")
            break
    
    total_time = time.perf_counter() - t_start
    
    generated_text = tokenizer.decode(generated_tokens)
    full_text = prompt + generated_text
    
    print(f"\nGenerated: {full_text}")
    print(f"Total time: {total_time*1000:.2f} ms for {len(generated_tokens)+prompt_len} tokens")
    print(f"Throughput: {(len(generated_tokens)+prompt_len) / total_time:.1f} tok/s")
    
    return {
        'prefill_ms': avg_prefill,
        'decode_ms': avg_decode, 
        'tokens_per_sec': tokens_per_sec,
        'total_time_ms': total_time * 1000,
        'generated_text': generated_text,
    }


def profile_pytorch(llm, tokens, cache_len, tokenizer, num_warmup=3, num_runs=10, max_new_tokens=20):
    """Profile prefill and decode performance."""
    prompt_len = tokens.shape[1]
    dtype = llm.dtype
    
    # Prepare inputs
    padded_tokens = torch.zeros(1, cache_len, dtype=torch.int64)
    padded_tokens[0, :prompt_len] = tokens[0]
    attention_mask = torch.zeros(1, cache_len, dtype=torch.int64)
    attention_mask[0, :prompt_len] = 1
    
    print(f"\n=== Profiling PyTorch ({num_runs} runs, {num_warmup} warmup) ===")
    
    # Warmup prefill
    for _ in range(num_warmup):
        with torch.no_grad():
            _, cache = llm.prefill(padded_tokens, attention_mask)
    
    # Profile prefill
    prefill_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, cache = llm.prefill(padded_tokens, attention_mask)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        prefill_times.append(time.perf_counter() - t0)
    
    avg_prefill = sum(prefill_times) / len(prefill_times) * 1000
    print(f"Prefill ({prompt_len} tokens): {avg_prefill:.2f} ms")
    
    # Get first token
    next_token = torch.argmax(logits[0, prompt_len - 1]).item()
    
    # Warmup decode
    for _ in range(num_warmup):
        token = torch.tensor([[next_token]], dtype=torch.int64)
        position_id = torch.tensor([[prompt_len]], dtype=torch.int64)
        valid_len = torch.tensor([[prompt_len + 1]], dtype=torch.int64)
        with torch.no_grad():
            _, _ = llm.decode_step(token, position_id, cache.clone(), valid_len)
    
    # Profile decode (single step)
    decode_times = []
    for _ in range(num_runs):
        token = torch.tensor([[next_token]], dtype=torch.int64)
        position_id = torch.tensor([[prompt_len]], dtype=torch.int64)
        valid_len = torch.tensor([[prompt_len + 1]], dtype=torch.int64)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            _, _ = llm.decode_step(token, position_id, cache.clone(), valid_len)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        decode_times.append(time.perf_counter() - t0)
    
    avg_decode = sum(decode_times) / len(decode_times) * 1000
    tokens_per_sec = 1000.0 / avg_decode
    print(f"Decode (1 token):  {avg_decode:.2f} ms = {tokens_per_sec:.1f} tok/s")
    
    # Profile full generation
    gen_times = []
    for run in range(min(3, num_runs)):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        
        with torch.no_grad():
            logits, current_cache = llm.prefill(padded_tokens, attention_mask)
        
        next_tok = torch.argmax(logits[0, prompt_len - 1]).item()
        generated = [next_tok]
        pos = prompt_len
        
        for _ in range(max_new_tokens - 1):
            token = torch.tensor([[generated[-1]]], dtype=torch.int64)
            position_id = torch.tensor([[pos]], dtype=torch.int64)
            valid_len = torch.tensor([[pos + 1]], dtype=torch.int64)
            
            with torch.no_grad():
                logits, current_cache = llm.decode_step(token, position_id, current_cache, valid_len)
            
            next_tok = torch.argmax(logits[0, 0]).item()
            generated.append(next_tok)
            pos += 1
            
            if next_tok == tokenizer.eos_token_id:
                break
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        gen_times.append(time.perf_counter() - t0)
    
    avg_gen = sum(gen_times) / len(gen_times) * 1000
    total_tokens = prompt_len + len(generated)
    print(f"Full generation ({total_tokens} tokens): {avg_gen:.2f} ms = {total_tokens / (avg_gen/1000):.1f} tok/s")
    
    return {
        'prefill_ms': avg_prefill,
        'decode_ms': avg_decode,
        'tokens_per_sec': tokens_per_sec,
        'full_gen_ms': avg_gen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--compare-hf", action="store_true", help="Compare against HuggingFace")
    parser.add_argument("--profile", action="store_true", help="Run profiling benchmarks")
    parser.add_argument("--profile-runs", type=int, default=10, help="Number of profiling runs")
    parser.add_argument("--iree", action="store_true", default=True, help="Run with IREE runtime (default: True)")
    parser.add_argument("--no-iree", action="store_true", help="Disable IREE runtime")
    parser.add_argument("--iree-device", type=str, default="hip", help="IREE device (hip, cuda, local-task)")
    parser.add_argument("--vmfb", type=str, default=None, help="Path to VMFB file (default: model.vmfb in model-dir)")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--iree-only", action="store_true", default=True, help="Run only IREE validation (default: True)")
    parser.add_argument("--pytorch", action="store_true", help="Also run PyTorch validation")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Model: {config['hf_model']}")
    print(f"Cache length: {config['cache_len']}")
    print(f"Cache shape: {config['cache_shape']}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    tokens = tokenizer.encode(args.prompt, return_tensors="pt")
    prompt_len = tokens.shape[1]
    print(f"\nPrompt: '{args.prompt}' -> {prompt_len} tokens")
    
    cache_len = config['cache_len']
    if prompt_len > cache_len:
        print(f"ERROR: Prompt too long ({prompt_len} > {cache_len})")
        return 1
    
    # Handle flags
    if args.no_iree:
        args.iree = False
        args.iree_only = False
    if args.pytorch:
        args.iree_only = False
    
    # =========== IREE-only mode ===========
    if args.iree_only:
        print("\n=== IREE-only mode (skipping PyTorch) ===")
        
        iree_results = run_iree_inference(
            model_dir, config, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.iree_device,
            num_warmup=3,
            num_runs=args.profile_runs,
            vmfb_path=args.vmfb
        )
        
        if iree_results:
            print("\n=== IREE Profiling Summary ===")
            print(f"  Prefill:    {iree_results['prefill_ms']:.2f} ms")
            print(f"  Decode:     {iree_results['decode_ms']:.2f} ms/token")
            print(f"  Throughput: {iree_results['tokens_per_sec']:.1f} tok/s")
        
        print("\n=== VALIDATION COMPLETE ===")
        return 0
    
    # =========== Test with PyTorch module ===========
    print("\n=== Testing PyTorch Module ===")
    
    # Import and recreate module
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Use the InPlaceCacheLLM from the export script
    from transformers import AutoModelForCausalLM, AutoConfig
    
    dtype = getattr(torch, config['dtype'])
    hf_config = AutoConfig.from_pretrained(config['hf_model'], trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        config['hf_model'],
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    hf_model.eval()
    
    # Create wrapper inline (same as in export script)
    import math
    
    class RotaryEmbedding(torch.nn.Module):
        def __init__(self, dim, max_position_embeddings=2048, base=500000.0):
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            
            t = torch.arange(max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos(), persistent=False)
            self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
        def forward(self, position_ids):
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
            return cos, sin
    
    def apply_rotary_pos_emb(q, k, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
    def repeat_kv(hidden_states, n_rep):
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    
    class InPlaceCacheLLM(torch.nn.Module):
        def __init__(self, hf_model, config, cache_len, dtype):
            super().__init__()
            self.hf_model = hf_model
            self.cache_len = cache_len
            self.dtype = dtype
            
            self.num_layers = config.num_hidden_layers
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
            self.head_dim = config.hidden_size // self.num_heads
            self.hidden_size = config.hidden_size
            self.num_kv_groups = self.num_heads // self.num_kv_heads
            
            rope_theta = getattr(config, 'rope_theta', 500000.0)
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=cache_len * 4,
                base=rope_theta
            )
        
        def _get_layer_components(self, layer_idx):
            layer = self.hf_model.model.layers[layer_idx]
            return {
                'input_layernorm': layer.input_layernorm,
                'q_proj': layer.self_attn.q_proj,
                'k_proj': layer.self_attn.k_proj,
                'v_proj': layer.self_attn.v_proj,
                'o_proj': layer.self_attn.o_proj,
                'post_attention_layernorm': layer.post_attention_layernorm,
                'mlp': layer.mlp,
            }
        
        def prefill(self, input_ids, attention_mask):
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )
            
            logits = outputs.logits
            past_kv = outputs.past_key_values
            
            cache_k_list = []
            cache_v_list = []
            for layer_kv in past_kv:
                cache_k_list.append(layer_kv[0])
                cache_v_list.append(layer_kv[1])
            
            cache_k_model = torch.stack(cache_k_list, dim=0)
            cache_v_model = torch.stack(cache_v_list, dim=0)
            
            cache_k = cache_k_model.squeeze(1).permute(2, 0, 1, 3)
            cache_v = cache_v_model.squeeze(1).permute(2, 0, 1, 3)
            cache = torch.stack([cache_k, cache_v], dim=2)
            
            valid_mask = attention_mask.squeeze(0).view(-1, 1, 1, 1, 1).to(self.dtype)
            cache = cache * valid_mask
            
            return logits, cache
        
        def decode_step(self, token, position_id, cache, valid_len):
            batch_size = 1
            seq_len = 1
            
            write_pos = position_id.squeeze() % self.cache_len
            pos_mask = F.one_hot(write_pos.long(), num_classes=self.cache_len).to(self.dtype)
            pos_mask = pos_mask.view(-1, 1, 1)
            
            hidden_states = self.hf_model.model.embed_tokens(token)
            
            cos, sin = self.rotary_emb(position_id)
            cos = cos.to(self.dtype)
            sin = sin.to(self.dtype)
            
            positions = torch.arange(self.cache_len, device=token.device)
            valid_positions = positions < valid_len.squeeze()
            attn_mask = torch.zeros(self.cache_len, dtype=self.dtype, device=token.device)
            attn_mask = attn_mask.masked_fill(~valid_positions, float('-inf'))
            attn_mask = attn_mask.view(1, 1, 1, self.cache_len)
            
            layer_caches_k = []
            layer_caches_v = []
            
            for layer_idx in range(self.num_layers):
                components = self._get_layer_components(layer_idx)
                
                normed = components['input_layernorm'](hidden_states)
                
                q = components['q_proj'](normed)
                k = components['k_proj'](normed)
                v = components['v_proj'](normed)
                
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                
                new_k = k.squeeze(0).squeeze(1)
                new_v = v.squeeze(0).squeeze(1)
                
                layer_cache_k = cache[:, layer_idx, 0, :, :]
                layer_cache_v = cache[:, layer_idx, 1, :, :]
                
                layer_cache_k = layer_cache_k * (1 - pos_mask) + new_k.unsqueeze(0) * pos_mask
                layer_cache_v = layer_cache_v * (1 - pos_mask) + new_v.unsqueeze(0) * pos_mask
                
                layer_caches_k.append(layer_cache_k)
                layer_caches_v.append(layer_cache_v)
                
                k_full = layer_cache_k.unsqueeze(0).transpose(1, 2)
                v_full = layer_cache_v.unsqueeze(0).transpose(1, 2)
                
                k_expanded = repeat_kv(k_full, self.num_kv_groups)
                v_expanded = repeat_kv(v_full, self.num_kv_groups)
                
                attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_weights = attn_weights + attn_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(self.dtype)
                attn_output = torch.matmul(attn_weights, v_expanded)
                
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
                attn_output = components['o_proj'](attn_output)
                
                hidden_states = hidden_states + attn_output
                
                normed = components['post_attention_layernorm'](hidden_states)
                hidden_states = hidden_states + components['mlp'](normed)
            
            cache_k_stacked = torch.stack(layer_caches_k, dim=1)
            cache_v_stacked = torch.stack(layer_caches_v, dim=1)
            cache = torch.stack([cache_k_stacked, cache_v_stacked], dim=2)
            
            hidden_states = self.hf_model.model.norm(hidden_states)
            logits = self.hf_model.lm_head(hidden_states)
            
            return logits, cache
    
    llm = InPlaceCacheLLM(hf_model, hf_config, cache_len, dtype)
    
    # Prepare prefill input
    padded_tokens = torch.zeros(1, cache_len, dtype=torch.int64)
    padded_tokens[0, :prompt_len] = tokens[0]
    attention_mask = torch.zeros(1, cache_len, dtype=torch.int64)
    attention_mask[0, :prompt_len] = 1
    
    # Run prefill
    with torch.no_grad():
        prefill_logits, cache = llm.prefill(padded_tokens, attention_mask)
    
    print(f"Prefill logits: {prefill_logits.shape}")
    print(f"Cache: {cache.shape}")
    
    # Get first generated token
    last_logits = prefill_logits[0, prompt_len - 1]
    next_token = torch.argmax(last_logits).item()
    print(f"Next token after prefill: {next_token} = '{tokenizer.decode([next_token])}'")
    
    # =========== Multi-step generation ===========
    print("\n=== Multi-step Generation ===")
    
    generated_tokens = [next_token]
    current_cache = cache.clone()
    current_pos = prompt_len
    
    for step in range(args.max_new_tokens - 1):
        token = torch.tensor([[generated_tokens[-1]]], dtype=torch.int64)
        position_id = torch.tensor([[current_pos]], dtype=torch.int64)
        valid_len = torch.tensor([[current_pos + 1]], dtype=torch.int64)
        
        with torch.no_grad():
            decode_logits, current_cache = llm.decode_step(
                token, position_id, current_cache, valid_len
            )
        
        next_token = torch.argmax(decode_logits[0, 0]).item()
        generated_tokens.append(next_token)
        current_pos += 1
        
        if next_token == tokenizer.eos_token_id:
            break
    
        if current_pos >= cache_len:
            print(f"  [Step {step+1}] Cache full at position {current_pos}")
            break
    
    generated_text = tokenizer.decode(generated_tokens)
    full_text = args.prompt + generated_text
    print(f"\nGenerated: {full_text}")
    
    # =========== Compare with HuggingFace ===========
    if args.compare_hf:
        print("\n=== Comparing with HuggingFace ===")
        
        # HF prefill
        with torch.no_grad():
            hf_outputs = hf_model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=True,
            )
        
        hf_prefill_logits = hf_outputs.logits
        print(f"HF prefill logits: {hf_prefill_logits.shape}")
        
        # Compare prefill logits (only for valid positions)
        our_logits = prefill_logits[0, :prompt_len]
        hf_logits = hf_prefill_logits[0]
        
        max_diff = (our_logits - hf_logits).abs().max().item()
        mean_diff = (our_logits - hf_logits).abs().mean().item()
        
        print(f"Prefill logits comparison:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff < args.atol or torch.allclose(our_logits, hf_logits, atol=args.atol, rtol=args.rtol):
            print("  ✓ PASS: Prefill matches HF")
        else:
            print("  ✗ FAIL: Prefill differs from HF")
        
        # Compare decode step
        print("\nDecode step comparison:")
        
        token = torch.tensor([[generated_tokens[0]]], dtype=torch.int64)
        position_id = torch.tensor([[prompt_len]], dtype=torch.int64)
        valid_len = torch.tensor([[prompt_len + 1]], dtype=torch.int64)
        
        with torch.no_grad():
            our_decode_logits, _ = llm.decode_step(token, position_id, cache, valid_len)
        
        hf_past = hf_outputs.past_key_values
        with torch.no_grad():
            hf_decode_outputs = hf_model(
                input_ids=token,
                attention_mask=torch.ones(1, prompt_len + 1, dtype=torch.int64),
                past_key_values=hf_past,
                use_cache=True,
            )
        
        hf_decode_logits = hf_decode_outputs.logits
        
        max_diff = (our_decode_logits - hf_decode_logits).abs().max().item()
        mean_diff = (our_decode_logits - hf_decode_logits).abs().mean().item()
        
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff < args.atol or torch.allclose(our_decode_logits, hf_decode_logits, atol=args.atol, rtol=args.rtol):
            print("  ✓ PASS: Decode matches HF")
        else:
            print("  ✗ FAIL: Decode differs from HF")
        
        our_next = torch.argmax(our_decode_logits[0, 0]).item()
        hf_next = torch.argmax(hf_decode_logits[0, 0]).item()
        
        print(f"\nNext token prediction:")
        print(f"  Ours: {our_next} = '{tokenizer.decode([our_next])}'")
        print(f"  HF:   {hf_next} = '{tokenizer.decode([hf_next])}'")
        
        if our_next == hf_next:
            print("  ✓ PASS: Same token predicted")
        else:
            print("  ✗ FAIL: Different tokens predicted")
    
    # =========== Verify dimensions ===========
    print("\n=== Verifying Dimensions ===")
    
    is_power_of_2 = (cache_len & (cache_len - 1)) == 0
    
    print(f"Cache length: {cache_len}")
    print(f"Power of 2: {is_power_of_2}")
    print(f"Prefill attention: [1, {config['num_heads']}, {cache_len}, {cache_len}]")
    print(f"Decode attention:  [1, {config['num_heads']}, 1, {cache_len}]")
    
    if is_power_of_2:
        print("\n✓ All attention dimensions are powers of 2!")
    else:
        print(f"\n✗ Cache length {cache_len} is not a power of 2")
    
    # =========== PyTorch Profiling ===========
    if args.profile:
        profile_results = profile_pytorch(
            llm, tokens, cache_len, tokenizer,
            num_warmup=3, num_runs=args.profile_runs,
            max_new_tokens=args.max_new_tokens
        )
        
        print("\n=== PyTorch Profiling Summary ===")
        print(f"  Prefill:    {profile_results['prefill_ms']:.2f} ms")
        print(f"  Decode:     {profile_results['decode_ms']:.2f} ms/token")
        print(f"  Throughput: {profile_results['tokens_per_sec']:.1f} tok/s")
    
    # =========== IREE Runtime ===========
    if args.iree:
        iree_results = run_iree_inference(
            model_dir, config, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.iree_device,
            num_warmup=3,
            num_runs=args.profile_runs,
            vmfb_path=args.vmfb
        )
        
        if iree_results:
            print("\n=== IREE Profiling Summary ===")
            print(f"  Prefill:    {iree_results['prefill_ms']:.2f} ms")
            print(f"  Decode:     {iree_results['decode_ms']:.2f} ms/token")
            print(f"  Throughput: {iree_results['tokens_per_sec']:.1f} tok/s")
    
    print("\n=== VALIDATION COMPLETE ===")
    return 0


if __name__ == "__main__":
    exit(main())
