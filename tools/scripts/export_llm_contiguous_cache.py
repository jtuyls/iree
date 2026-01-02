#!/usr/bin/env python3
"""
Export LLM with contiguous cache layout and in-place updates.

Key insights:
1. Cache layout: [seq, layers, 2, heads, dim] for contiguous memory access
2. In-place update: Write K/V to cache BEFORE attention, giving power-of-2 dimensions
3. Decode attention is [1, heads, 1, cache_len] instead of [1, heads, 1, cache_len+1]

This gives O(1) cache update inside the kernel with power-of-2 attention dimensions.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to Q and K."""
    cos = cos.unsqueeze(1)  # [batch, 1, seq, dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    """Repeat KV heads to match query heads for GQA."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """Precompute rotary embeddings for all positions."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin for all positions
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, position_ids):
        # position_ids: [batch, seq_len]
        cos = self.cos_cached[position_ids]  # [batch, seq_len, dim]
        sin = self.sin_cached[position_ids]
        return cos, sin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-len", type=int, default=512,
                        help="Cache length (power of 2 recommended)")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from iree.turbine.aot import FxProgramsBuilder, export, externalize_module_parameters, save_module_parameters
    
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=dtype, 
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use SDPA for fused attention
    )
    model.eval()
    
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = config.hidden_size // num_heads
    hidden_size = config.hidden_size
    num_kv_groups = num_heads // num_kv_heads
    
    cache_len = args.cache_len
    
    print(f"Model: {num_layers} layers, {num_heads} Q heads, {num_kv_heads} KV heads, {head_dim} head_dim")
    print(f"Cache length: {cache_len} (power of 2: {cache_len & (cache_len - 1) == 0})")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache layout: [seq, layers, 2, heads, dim] where 2 = K and V stacked
    cache_shape = (cache_len, num_layers, 2, num_kv_heads, head_dim)
    
    print(f"Cache shape: {cache_shape}")
    cache_size_bytes = math.prod(cache_shape) * 2  # f16
    print(f"Cache size: {cache_size_bytes / 1024 / 1024:.1f} MB")
    
    class InPlaceCacheLLM(nn.Module):
        """LLM with in-place cache update for power-of-2 attention dimensions."""
        
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
            
            # RoPE
            rope_theta = getattr(config, 'rope_theta', 500000.0)
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=cache_len * 4,
                base=rope_theta
            )
        
        def _get_layer_components(self, layer_idx):
            """Get components from HF model layer."""
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
            """
            Prefill using HF model, return cache in contiguous layout.
            
            Args:
                input_ids: [1, cache_len]
                attention_mask: [1, cache_len]
            
            Returns:
                logits: [1, cache_len, vocab_size]
                cache: [cache_len, layers, 2, kv_heads, head_dim]
            """
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            
            # Use HF model directly for prefill
            with torch.no_grad():
                outputs = self.hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )
            
            logits = outputs.logits
            past_kv = outputs.past_key_values
            
            # Extract and stack cache
            cache_k_list = []
            cache_v_list = []
            for layer_kv in past_kv:
                cache_k_list.append(layer_kv[0])  # [1, kv_heads, seq, dim]
                cache_v_list.append(layer_kv[1])
            
            cache_k_model = torch.stack(cache_k_list, dim=0)  # [layers, 1, kv_heads, seq, dim]
            cache_v_model = torch.stack(cache_v_list, dim=0)
            
            # Convert to contiguous layout: [seq, layers, kv_heads, dim]
            cache_k = cache_k_model.squeeze(1).permute(2, 0, 1, 3)
            cache_v = cache_v_model.squeeze(1).permute(2, 0, 1, 3)
            
            # Stack K and V: [seq, layers, 2, kv_heads, dim]
            cache = torch.stack([cache_k, cache_v], dim=2)
            
            # Zero out padding positions
            valid_mask = attention_mask.squeeze(0).view(-1, 1, 1, 1, 1).to(self.dtype)
            cache = cache * valid_mask
            
            return logits, cache
        
        def decode_step(self, token, position_id, cache, valid_len):
            """
            Decode with in-place cache update using mask-based writes.
            
            Args:
                token: [1, 1]
                position_id: [1, 1]
                cache: [cache_len, layers, 2, kv_heads, head_dim]
                valid_len: [1, 1] - number of valid positions (for attention mask)
            
            Returns:
                logits: [1, 1, vocab_size]
                cache: [cache_len, layers, 2, kv_heads, head_dim] - updated
            """
            batch_size = 1
            seq_len = 1
            
            # Get write position (circular buffer)
            write_pos = position_id.squeeze() % self.cache_len
            
            # Create position mask for cache update (one-hot)
            pos_mask = F.one_hot(write_pos.long(), num_classes=self.cache_len).to(self.dtype)
            pos_mask = pos_mask.view(-1, 1, 1)  # [cache_len, 1, 1]
            
            # Embeddings
            hidden_states = self.hf_model.model.embed_tokens(token)
            
            # RoPE for current position
            cos, sin = self.rotary_emb(position_id)
            cos = cos.to(self.dtype)
            sin = sin.to(self.dtype)
            
            # Create attention mask for decode
            # Only attend to positions [0, valid_len)
            positions = torch.arange(self.cache_len, device=token.device)
            valid_positions = positions < valid_len.squeeze()
            attn_mask = torch.zeros(self.cache_len, dtype=self.dtype, device=token.device)
            attn_mask = attn_mask.masked_fill(~valid_positions, float('-inf'))
            attn_mask = attn_mask.view(1, 1, 1, self.cache_len)
            
            # Collect updated layer caches
            layer_caches_k = []
            layer_caches_v = []
            
            for layer_idx in range(self.num_layers):
                components = self._get_layer_components(layer_idx)
                
                # Pre-attention norm
                normed = components['input_layernorm'](hidden_states)
                
                # Squeeze to [batch, hidden] for better codegen on decode (avoids [1,1,hidden] matmuls)
                normed_2d = normed.squeeze(1)  # [1, 4096]
                
                # Q, K, V projections with 2D input
                q = components['q_proj'](normed_2d)  # [1, 4096]
                k = components['k_proj'](normed_2d)  # [1, 1024]
                v = components['v_proj'](normed_2d)  # [1, 1024]
                
                # Reshape to attention format
                q = q.view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)  # [1, 32, 1, 128]
                k = k.view(batch_size, self.num_kv_heads, self.head_dim).unsqueeze(2)  # [1, 8, 1, 128]
                v = v.view(batch_size, self.num_kv_heads, self.head_dim).unsqueeze(2)  # [1, 8, 1, 128]
                
                # Apply RoPE
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                
                # Get new K, V
                new_k = k.squeeze(0).squeeze(1)  # [kv_heads, dim]
                new_v = v.squeeze(0).squeeze(1)
                
                # Get current layer's cache
                layer_cache_k = cache[:, layer_idx, 0, :, :]  # [cache_len, kv_heads, dim]
                layer_cache_v = cache[:, layer_idx, 1, :, :]
                
                # Update cache using mask (traceable!)
                layer_cache_k = layer_cache_k * (1 - pos_mask) + new_k.unsqueeze(0) * pos_mask
                layer_cache_v = layer_cache_v * (1 - pos_mask) + new_v.unsqueeze(0) * pos_mask
                
                # Store for rebuilding cache
                layer_caches_k.append(layer_cache_k)
                layer_caches_v.append(layer_cache_v)
                
                # Prepare for attention
                k_full = layer_cache_k.unsqueeze(0).transpose(1, 2)  # [1, kv_heads, cache_len, dim]
                v_full = layer_cache_v.unsqueeze(0).transpose(1, 2)
                
                # Expand for GQA
                k_expanded = repeat_kv(k_full, self.num_kv_groups)
                v_expanded = repeat_kv(v_full, self.num_kv_groups)
                
                # Attention: [1, heads, 1, cache_len] - power of 2!
                # Use SDPA for fused attention kernel
                attn_output = F.scaled_dot_product_attention(
                    q, k_expanded, v_expanded,
                    attn_mask=attn_mask,
                    scale=1.0 / math.sqrt(self.head_dim),
                )
                
                # Output projection - squeeze to 2D for better codegen
                attn_output = attn_output.squeeze(2)  # [1, 32, 128]
                attn_output = attn_output.view(batch_size, self.hidden_size)  # [1, 4096]
                attn_output = components['o_proj'](attn_output)  # [1, 4096]
                
                # Residual (squeeze hidden_states to 2D)
                hidden_states = hidden_states.squeeze(1) + attn_output  # [1, 4096]
                
                # FFN with 2D input
                normed = components['post_attention_layernorm'](hidden_states.unsqueeze(1)).squeeze(1)
                hidden_states = hidden_states + components['mlp'](normed.unsqueeze(1)).squeeze(1)
            
            # Rebuild cache
            cache_k_stacked = torch.stack(layer_caches_k, dim=1)  # [cache_len, layers, kv_heads, dim]
            cache_v_stacked = torch.stack(layer_caches_v, dim=1)
            cache = torch.stack([cache_k_stacked, cache_v_stacked], dim=2)  # [cache_len, layers, 2, kv_heads, dim]
            
            # Final norm and LM head (restore to 3D for compatibility)
            hidden_states = hidden_states.unsqueeze(1)  # [1, 1, 4096]
            hidden_states = self.hf_model.model.norm(hidden_states)
            logits = self.hf_model.lm_head(hidden_states)
            
            return logits, cache
    
    llm = InPlaceCacheLLM(model, config, cache_len, dtype)
    
    # =========== Test ===========
    print("\n=== Testing Module ===")
    
    test_len = 10
    test_tokens = torch.randint(0, config.vocab_size, (1, test_len), dtype=torch.int64)
    padded = torch.zeros(1, cache_len, dtype=torch.int64)
    padded[0, :test_len] = test_tokens[0]
    attention_mask = torch.zeros(1, cache_len, dtype=torch.long)
    attention_mask[0, :test_len] = 1
    
    with torch.no_grad():
        logits, cache = llm.prefill(padded, attention_mask)
    
    print(f"Prefill: logits {logits.shape}, cache {cache.shape}")
    print(f"  Cache at pos 0-9 max: {cache[:10].abs().max().item():.2f}")
    print(f"  Cache at pos 10+ max: {cache[10:].abs().max().item():.4f}")
    
    # Test decode
    decode_token = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.int64)
    position_id = torch.tensor([[test_len]], dtype=torch.int64)
    valid_len = torch.tensor([[test_len + 1]], dtype=torch.int64)
    
    with torch.no_grad():
        logits, cache_updated = llm.decode_step(decode_token, position_id, cache, valid_len)
    
    print(f"Decode: logits {logits.shape}")
    print(f"  Decode attention: [1, {num_heads}, 1, {cache_len}] - POWER OF 2!")
    print(f"  Cache at pos 10 updated: {cache_updated[10].abs().max().item():.2f}")
    
    # =========== Export to MLIR ===========
    print("\n=== Exporting to MLIR ===")
    
    externalize_module_parameters(llm)
    
    fxb = FxProgramsBuilder(llm)
    
    # Prefill inputs
    prefill_input_ids = torch.zeros((1, cache_len), dtype=torch.int64)
    prefill_attention_mask = torch.ones((1, cache_len), dtype=torch.int64)
    
    @fxb.export_program(
        name="prefill",
        args=(prefill_input_ids, prefill_attention_mask),
    )
    def prefill_export(module, input_ids, attention_mask):
        return module.prefill(input_ids, attention_mask)
    
    # Decode inputs
    decode_token = torch.zeros((1, 1), dtype=torch.int64)
    decode_position_id = torch.zeros((1, 1), dtype=torch.int64)
    decode_cache = torch.zeros(cache_shape, dtype=dtype)
    decode_valid_len = torch.ones((1, 1), dtype=torch.int64)
    
    @fxb.export_program(
        name="decode_step",
        args=(decode_token, decode_position_id, decode_cache, decode_valid_len),
    )
    def decode_export(module, token, position_id, cache, valid_len):
        return module.decode_step(token, position_id, cache, valid_len)
    
    print("Tracing programs...")
    output = export(fxb, import_symbolic_shape_expressions=True)
    
    mlir_path = output_dir / "model.mlir"
    output.save_mlir(str(mlir_path))
    print(f"Saved MLIR to {mlir_path}")
    
    # =========== Save Weights ===========
    print("\n=== Saving Weights ===")
    
    irpa_path = output_dir / "model.irpa"
    save_module_parameters(str(irpa_path), llm)
    print(f"Saved IRPA ({irpa_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # =========== Save Config ===========
    config_dict = {
        "model_type": "llama_inplace_cache",
        "hf_model": args.model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "vocab_size": config.vocab_size,
        "cache_len": cache_len,
        "dtype": args.dtype,
        "cache_layout": "inplace_contiguous",
        "cache_shape": list(cache_shape),
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved tokenizer")
    
    print("\n=== EXPORT COMPLETE ===")
    print(f"\nDimension summary:")
    print(f"  Prefill attention: [{cache_len}, {cache_len}]")
    print(f"  Decode attention:  [1, {cache_len}] - POWER OF 2!")
    print(f"  Cache: {cache_shape}")
    print(f"\nNext: iree-compile {mlir_path} -o {output_dir}/model.vmfb --iree-hal-target-backends=rocm")


if __name__ == "__main__":
    main()
