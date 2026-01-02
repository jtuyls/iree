#!/usr/bin/env python3
"""
Export LLM with shark-ai's flash attention kernels.

Key advantages:
- Dynamic sequence dimensions (no hardcoded 513)
- Direct iree_linalg_ext.attention emission
- Better optimized attention kernels
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add shark-ai to path
sys.path.insert(0, '/home/jornt/workspace/shark-ai/sharktank')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prefill-len", type=int, default=512)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from iree.turbine.aot import FxProgramsBuilder, export
    from iree.turbine.aot import externalize_module_parameters, save_module_parameters
    
    # Import shark-ai flash attention
    from sharktank.kernels.attention import masked_flash_attention
    
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=dtype, 
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = config.hidden_size // num_heads
    
    print(f"Model: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache shape: [seq, layers, heads, dim] for contiguous updates
    max_cache_len = args.prefill_len
    cache_shape_contiguous = (max_cache_len, num_layers, num_kv_heads, head_dim)
    
    print(f"Cache shape (contiguous): {cache_shape_contiguous}")
    
    class FlashAttentionLLM(nn.Module):
        """LLM wrapper using shark-ai's flash attention with dynamic dimensions."""
        
        def __init__(self, model, num_layers, num_heads, num_kv_heads, head_dim, max_cache_len, dtype):
            super().__init__()
            self.model = model
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.max_cache_len = max_cache_len
            self.dtype = dtype
            self.scale = torch.tensor(1.0 / math.sqrt(head_dim), dtype=torch.float32)
            
            # GQA: repeat KV heads to match query heads
            self.kv_repeat = num_heads // num_kv_heads
            
            # Rotary embedding is at model.model.rotary_emb
            self.rotary_emb = model.model.rotary_emb
        
        def _reshape_cache_to_model(self, cache_k, cache_v, seq_len):
            """Convert from [seq, layers, heads, dim] to per-layer [batch, heads, seq, dim]."""
            # cache_k/v: [max_seq, layers, heads, dim]
            # Need: list of [batch=1, heads, seq, dim] per layer
            cache_k_valid = cache_k[:seq_len]  # [seq, layers, heads, dim]
            cache_v_valid = cache_v[:seq_len]
            
            # Transpose to [layers, seq, heads, dim]
            cache_k_t = cache_k_valid.permute(1, 0, 2, 3)  # [layers, seq, heads, dim]
            cache_v_t = cache_v_valid.permute(1, 0, 2, 3)
            
            # Add batch dim and transpose to [layers, batch, heads, seq, dim]
            cache_k_t = cache_k_t.unsqueeze(1).permute(0, 1, 3, 2, 4)  # [layers, 1, heads, seq, dim]
            cache_v_t = cache_v_t.unsqueeze(1).permute(0, 1, 3, 2, 4)
            
            return cache_k_t, cache_v_t
        
        def _flash_attention(self, q, k, v, mask=None):
            """Apply flash attention using shark-ai kernel.
            
            Args:
                q: [batch, num_heads, seq_q, head_dim]
                k: [batch, num_kv_heads, seq_kv, head_dim]  
                v: [batch, num_kv_heads, seq_kv, head_dim]
                mask: [seq_q, seq_kv] or None
            
            Returns:
                [batch, num_heads, seq_q, head_dim]
            """
            batch, num_heads, seq_q, head_dim = q.shape
            seq_kv = k.shape[2]
            
            # Expand KV heads for GQA
            if self.kv_repeat > 1:
                k = k.repeat_interleave(self.kv_repeat, dim=1)  # [batch, num_heads, seq_kv, head_dim]
                v = v.repeat_interleave(self.kv_repeat, dim=1)
            
            # Create causal mask if not provided
            if mask is None:
                mask = torch.triu(
                    torch.full((seq_q, seq_kv), float("-inf"), dtype=q.dtype, device=q.device),
                    diagonal=seq_kv - seq_q + 1
                )
            
            # shark-ai flash attention expects f16 inputs
            q_f16 = q.to(torch.float16) if q.dtype != torch.float16 else q
            k_f16 = k.to(torch.float16) if k.dtype != torch.float16 else k
            v_f16 = v.to(torch.float16) if v.dtype != torch.float16 else v
            mask_f32 = mask.to(torch.float32) if mask.dtype != torch.float32 else mask
            
            # Call shark-ai masked flash attention
            # Result is f32, convert back to input dtype
            result = masked_flash_attention(q_f16, k_f16, v_f16, mask_f32, self.scale)
            return result.to(q.dtype)
        
        def prefill(self, tokens, attention_mask, cache_k, cache_v):
            """Prefill: process prompt and populate cache.
            
            Args:
                tokens: [1, seq_len]
                attention_mask: [1, seq_len] (unused, we use causal)
                cache_k: [max_seq, layers, heads, dim]
                cache_v: [max_seq, layers, heads, dim]
            
            Returns:
                logits: [1, seq_len, vocab]
                new_cache_k: [max_seq, layers, heads, dim]
                new_cache_v: [max_seq, layers, heads, dim]
            """
            seq_len = tokens.shape[1]
            
            # Get embeddings
            hidden = self.model.model.embed_tokens(tokens)
            
            # Process through layers
            new_keys = []
            new_values = []
            
            for layer_idx, layer in enumerate(self.model.model.layers):
                # Pre-attention norm
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                
                # Self attention
                bsz, q_len, _ = hidden.shape
                
                # QKV projections
                q = layer.self_attn.q_proj(hidden)
                k = layer.self_attn.k_proj(hidden)
                v = layer.self_attn.v_proj(hidden)
                
                # Reshape
                q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                
                # Apply rotary embeddings
                position_ids = torch.arange(q_len, device=tokens.device).unsqueeze(0)
                cos, sin = self.rotary_emb(v, position_ids)
                q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
                
                # Store KV for cache
                new_keys.append(k)  # [1, heads, seq, dim]
                new_values.append(v)
                
                # Flash attention (causal)
                attn_out = self._flash_attention(q, k, v)
                
                # Output projection
                attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                attn_out = layer.self_attn.o_proj(attn_out)
                
                hidden = residual + attn_out
                
                # MLP
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden
            
            # Final norm and LM head
            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)
            
            # Build new cache in contiguous format
            # Stack keys/values: [layers, 1, heads, seq, dim] -> [seq, layers, heads, dim]
            stacked_k = torch.stack(new_keys, dim=0)  # [layers, 1, heads, seq, dim]
            stacked_v = torch.stack(new_values, dim=0)
            stacked_k = stacked_k.squeeze(1).permute(2, 0, 1, 3)  # [seq, layers, heads, dim]
            stacked_v = stacked_v.squeeze(1).permute(2, 0, 1, 3)
            
            # Copy to cache
            new_cache_k = cache_k.clone()
            new_cache_v = cache_v.clone()
            new_cache_k[:seq_len] = stacked_k
            new_cache_v[:seq_len] = stacked_v
            
            return logits, new_cache_k, new_cache_v
        
        def decode_step(self, token, position_id, attention_mask, cache_k, cache_v):
            """Decode: generate one token using cached KV.
            
            Uses full cache and attention mask to avoid dynamic slicing.
            
            Args:
                token: [1, 1]
                position_id: [1, 1]
                attention_mask: [1, max_cache+1] - mask with 0s for valid, -inf for invalid
                cache_k: [max_seq, layers, heads, dim]
                cache_v: [max_seq, layers, heads, dim]
            
            Returns:
                logits: [1, 1, vocab]
                new_kv: [1, 2, layers, heads, dim] (new K and V to append)
            """
            # Get embedding
            hidden = self.model.model.embed_tokens(token)
            
            new_keys = []
            new_values = []
            
            for layer_idx, layer in enumerate(self.model.model.layers):
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                
                bsz, q_len, _ = hidden.shape  # q_len = 1
                
                # QKV projections for new token
                q = layer.self_attn.q_proj(hidden)
                k = layer.self_attn.k_proj(hidden)
                v = layer.self_attn.v_proj(hidden)
                
                q = q.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = v.view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                
                # Rotary embeddings for current position
                cos, sin = self.rotary_emb(v, position_id)
                q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
                
                # Store new KV
                new_keys.append(k.squeeze(2))  # [1, heads, dim]
                new_values.append(v.squeeze(2))
                
                # Get full cached KV for this layer
                # cache_k is [max_seq, layers, heads, dim]
                cached_k = cache_k[:, layer_idx, :, :].unsqueeze(0).permute(0, 2, 1, 3)  # [1, heads, max_seq, dim]
                cached_v = cache_v[:, layer_idx, :, :].unsqueeze(0).permute(0, 2, 1, 3)
                
                # Concat with new KV: [1, heads, max_seq+1, dim]
                full_k = torch.cat([cached_k, k], dim=2)
                full_v = torch.cat([cached_v, v], dim=2)
                
                # Use attention mask to mask out invalid positions
                # attention_mask is [1, max_cache+1] with 0 for valid, will use as mask
                # For flash attention, convert to [1, max_cache+1] float mask
                mask = attention_mask.float()  # [1, max_cache+1]
                # Set invalid positions to -inf
                mask = mask.masked_fill(mask == 1, float("-inf"))
                
                attn_out = self._flash_attention(q, full_k, full_v, mask)
                
                attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, 1, -1)
                attn_out = layer.self_attn.o_proj(attn_out)
                
                hidden = residual + attn_out
                
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden
            
            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)
            
            # Pack new KV: [1, 2, layers, heads, dim]
            stacked_k = torch.stack(new_keys, dim=0)  # [layers, 1, heads, dim]
            stacked_v = torch.stack(new_values, dim=0)
            new_kv = torch.stack([stacked_k.squeeze(1), stacked_v.squeeze(1)], dim=0)  # [2, layers, heads, dim]
            new_kv = new_kv.unsqueeze(0)  # [1, 2, layers, heads, dim]
            
            return logits, new_kv
        
        def _apply_rotary_pos_emb(self, q, k, cos, sin):
            """Apply rotary position embeddings."""
            cos = cos.unsqueeze(1)  # [1, 1, seq, dim]
            sin = sin.unsqueeze(1)
            
            q_embed = (q * cos) + (self._rotate_half(q) * sin)
            k_embed = (k * cos) + (self._rotate_half(k) * sin)
            return q_embed, k_embed
        
        def _rotate_half(self, x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
    
    # Create wrapper
    llm = FlashAttentionLLM(
        model, num_layers, num_heads, num_kv_heads, head_dim, max_cache_len, dtype
    )
    
    # Test before export
    print("\n=== Testing model before export ===")
    test_tokens = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.int64)
    test_cache_k = torch.zeros(cache_shape_contiguous, dtype=dtype)
    test_cache_v = torch.zeros(cache_shape_contiguous, dtype=dtype)
    test_mask = torch.zeros(1, 8, dtype=torch.long)
    
    with torch.no_grad():
        logits, cache_k_out, cache_v_out = llm.prefill(test_tokens, test_mask, test_cache_k, test_cache_v)
        print(f"Prefill logits shape: {logits.shape}")
        print(f"Cache K out shape: {cache_k_out.shape}")
        
        # Test decode
        test_token = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.int64)
        test_pos = torch.tensor([[8]], dtype=torch.int64)
        # attention_mask: [1, max_cache+1] - 0 for valid, 1 for masked
        test_decode_mask = torch.ones(1, max_cache_len + 1, dtype=torch.long)
        test_decode_mask[0, :9] = 0  # First 9 positions are valid (8 cached + 1 new)
        
        logits_d, new_kv = llm.decode_step(test_token, test_pos, test_decode_mask, cache_k_out, cache_v_out)
        print(f"Decode logits shape: {logits_d.shape}")
        print(f"New KV shape: {new_kv.shape}")
    
    print("\n=== Exporting to MLIR ===")
    
    # Externalize parameters
    externalize_module_parameters(llm, external_scope="model")
    
    # Export
    fxb = FxProgramsBuilder(llm)
    
    seq_dim = torch.export.Dim("seq", max=args.prefill_len)
    mask_dim = torch.export.Dim("mask_len", max=max_cache_len + 1)
    
    @fxb.export_program(
        name="prefill",
        args=(
            torch.randint(0, config.vocab_size, (1, args.prefill_len), dtype=torch.int64),
            torch.zeros(1, args.prefill_len, dtype=torch.long),
            torch.zeros(cache_shape_contiguous, dtype=dtype),
            torch.zeros(cache_shape_contiguous, dtype=dtype),
        ),
        dynamic_shapes={
            "tokens": {1: seq_dim},
            "attention_mask": {1: seq_dim},
            "cache_k": None,
            "cache_v": None,
        },
    )
    def prefill_export(module, tokens, attention_mask, cache_k, cache_v):
        return module.prefill(tokens, attention_mask, cache_k, cache_v)
    
    @fxb.export_program(
        name="decode_step",
        args=(
            torch.randint(0, config.vocab_size, (1, 1), dtype=torch.int64),
            torch.tensor([[args.prefill_len]], dtype=torch.int64),
            torch.zeros(1, max_cache_len + 1, dtype=torch.long),
            torch.zeros(cache_shape_contiguous, dtype=dtype),
            torch.zeros(cache_shape_contiguous, dtype=dtype),
        ),
        dynamic_shapes={
            "token": None,
            "position_id": None,
            "attention_mask": {1: mask_dim},
            "cache_k": None,
            "cache_v": None,
        },
    )
    def decode_step_export(module, token, position_id, attention_mask, cache_k, cache_v):
        return module.decode_step(token, position_id, attention_mask, cache_k, cache_v)
    
    output = export(fxb)
    
    mlir_path = output_dir / "model.mlir"
    output.save_mlir(str(mlir_path))
    print(f"Saved MLIR to {mlir_path}")
    
    # Save parameters
    irpa_path = output_dir / "model.irpa"
    save_module_parameters(str(irpa_path), llm)
    print(f"Saved IRPA ({irpa_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    
    # Save config
    config_out = {
        "model_type": "llama_flash",
        "hf_model": args.model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.vocab_size,
        "max_seq_len": args.max_seq_len,
        "prefill_len": args.prefill_len,
        "max_cache_len": max_cache_len,
        "attention_mask_len": max_cache_len + 1,
        "dtype": args.dtype,
        "cache_layout": "contiguous",
        "cache_shape": list(cache_shape_contiguous),
        "new_kv_shape": [1, 2, num_layers, num_kv_heads, head_dim],
        "vmfb_path": "model.vmfb",
        "irpa_path": "model.irpa",
        "tokenizer_path": "tokenizer.json",
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    
    print(f"\nExport complete! Files in {output_dir}")
    print(f"  MLIR: {mlir_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

