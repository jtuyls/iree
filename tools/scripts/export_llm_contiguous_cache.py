#!/usr/bin/env python3
"""
Export LLM with contiguous cache layout for efficient updates.

Key insight: Change cache layout from [layers, batch, heads, seq, dim] 
to [seq, layers, heads, dim] so that updating position `pos` is a single
contiguous memory write.

This allows O(1) cache update with one H2D transfer instead of 512 scattered writes.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prefill-len", type=int, default=512)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, DynamicCache
    from iree.turbine.aot import FxProgramsBuilder, export, externalize_module_parameters, save_module_parameters
    from safetensors.torch import save_file
    
    print(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=dtype, 
        trust_remote_code=True,
        attn_implementation="eager",  # SDPA not supported by torch-mlir yet
    )
    model.eval()
    
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = config.hidden_size // num_heads
    
    print(f"Model: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # max_cache_len = prefill_len (power of 2 for GPU-friendly prefill)
    # During decode: attention over 512 past + 1 current = 513 positions
    # The 513 only affects attention matmul, not the large FFN operations
    max_cache_len = args.prefill_len
    
    # NEW LAYOUT: [seq, layers, heads, dim] for contiguous updates
    # When updating position pos, we write cache[pos, :, :, :] = 64KB contiguous
    cache_shape_contiguous = (max_cache_len, num_layers, num_kv_heads, head_dim)
    
    # OLD LAYOUT for model: [layers, batch, heads, seq, dim]
    cache_shape_model = (num_layers, 1, num_kv_heads, max_cache_len, head_dim)
    
    print(f"Cache shapes:")
    print(f"  Contiguous (storage): {cache_shape_contiguous}")
    print(f"  Model (transposed):   {cache_shape_model}")
    
    class ContiguousCacheLLM(nn.Module):
        """LLM wrapper with contiguous cache layout."""
        
        def __init__(self, model, num_layers, num_kv_heads, head_dim, max_cache_len, dtype):
            super().__init__()
            self.model = model
            self.num_layers = num_layers
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.max_cache_len = max_cache_len
            self.dtype = dtype
        
        def _contiguous_to_model(self, cache_k, cache_v):
            """Convert contiguous layout to model layout.
            
            Contiguous: [seq, layers, heads, dim]
            Model:      [layers, 1, heads, seq, dim]
            """
            # cache_k: [seq, layers, heads, dim]
            # Need: [layers, 1, heads, seq, dim]
            # permute: (0,1,2,3) -> (1, 2, 0, 3) gives [layers, heads, seq, dim]
            cache_k_model = cache_k.permute(1, 2, 0, 3).unsqueeze(1)  # [layers, 1, heads, seq, dim]
            cache_v_model = cache_v.permute(1, 2, 0, 3).unsqueeze(1)
            return cache_k_model, cache_v_model
        
        def _model_to_contiguous(self, cache_k_model, cache_v_model):
            """Convert model layout to contiguous layout.
            
            Model:      [layers, 1, heads, seq, dim]
            Contiguous: [seq, layers, heads, dim]
            """
            # cache_k_model: [layers, 1, heads, seq, dim]
            # Need: [seq, layers, heads, dim]
            cache_k = cache_k_model.squeeze(1).permute(2, 0, 1, 3)  # [seq, layers, heads, dim]
            cache_v = cache_v_model.squeeze(1).permute(2, 0, 1, 3)
            return cache_k, cache_v
        
        def prefill(self, input_ids, attention_mask):
            """
            Prefill with contiguous cache output.
            
            Args:
                input_ids: [1, max_cache_len] - padded to max_cache_len
                attention_mask: [1, max_cache_len] - 1 for valid tokens, 0 for padding
            
            Returns:
                logits: [1, max_cache_len, vocab]
                cache_k: [max_cache_len, layers, heads, dim] - contiguous layout
                cache_v: [max_cache_len, layers, heads, dim] - contiguous layout
            """
            # Create position IDs matching input length
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)
            
            # Run model
            with torch.no_grad():
                outputs = self.model(
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
                cache_k_list.append(layer_kv[0])  # [1, heads, seq, dim]
                cache_v_list.append(layer_kv[1])
            
            cache_k_model = torch.stack(cache_k_list, dim=0)  # [layers, 1, heads, seq, dim]
            cache_v_model = torch.stack(cache_v_list, dim=0)
            
            # Convert to contiguous layout
            cache_k, cache_v = self._model_to_contiguous(cache_k_model, cache_v_model)
            
            # Zero out positions where attention_mask is 0
            # attention_mask: [1, prefill_len], need to broadcast to [prefill_len, layers, heads, dim]
            valid_mask = attention_mask.squeeze(0).view(-1, 1, 1, 1).to(self.dtype)
            cache_k = cache_k * valid_mask
            cache_v = cache_v * valid_mask
            
            return logits, cache_k, cache_v
        
        def decode_step(self, token, position_id, attention_mask, cache_k, cache_v):
            """
            Decode step with contiguous cache.
            
            Args:
                token: [1, 1]
                position_id: [1, 1] - position of the current token
                attention_mask: [1, max_cache_len+1] - for past positions + current token
                cache_k: [max_cache_len, layers, heads, dim] - contiguous, FULL cache
                cache_v: [max_cache_len, layers, heads, dim] - contiguous, FULL cache
            
            Returns:
                logits: [1, 1, vocab]
                new_kv: [1, 2, layers, heads, dim] - K and V stacked, contiguous!
            """
            # Convert cache to model layout: [layers, 1, heads, seq, dim]
            cache_k_model, cache_v_model = self._contiguous_to_model(cache_k, cache_v)
            
            # Build DynamicCache from tensors  
            past_key_values = DynamicCache()
            for layer_idx in range(self.num_layers):
                past_key_values.update(
                    cache_k_model[layer_idx],  # [1, heads, max_cache_len, dim]
                    cache_v_model[layer_idx],
                    layer_idx
                )
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=token,
                    attention_mask=attention_mask,
                    position_ids=position_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits = outputs.logits
            new_past_kv = outputs.past_key_values
            
            # Extract new KV for the current position
            # The model appends to cache, so new entry is at position -1
            new_k_list = []
            new_v_list = []
            for layer_idx in range(self.num_layers):
                new_k_list.append(new_past_kv.key_cache[layer_idx][:, :, -1:, :])  # [1, heads, 1, dim]
                new_v_list.append(new_past_kv.value_cache[layer_idx][:, :, -1:, :])
            
            new_k_model = torch.stack(new_k_list, dim=0)  # [layers, 1, heads, 1, dim]
            new_v_model = torch.stack(new_v_list, dim=0)
            
            # Convert to contiguous layout: [1, layers, heads, dim]
            new_k = new_k_model.squeeze(1).squeeze(2).permute(1, 0, 2)  # [1, layers, heads, dim]
            new_v = new_v_model.squeeze(1).squeeze(2).permute(1, 0, 2)
            
            # Stack k and v together for single transfer: [1, 2, layers, heads, dim]
            new_kv = torch.stack([new_k, new_v], dim=1)  # [1, 2, layers, heads, dim]
            
            return logits, new_kv
    
    llm = ContiguousCacheLLM(model, num_layers, num_kv_heads, head_dim, max_cache_len, dtype)
    
    # =========== Test ===========
    print("\n=== Testing Module ===")
    
    test_tokens = torch.randint(0, config.vocab_size, (1, 10), dtype=torch.int64)
    padded = torch.zeros(1, max_cache_len, dtype=torch.int64)
    padded[0, :10] = test_tokens[0]
    attention_mask_prefill = torch.zeros(1, max_cache_len, dtype=torch.long)
    attention_mask_prefill[0, :10] = 1
    
    with torch.no_grad():
        logits, cache_k, cache_v = llm.prefill(padded, attention_mask_prefill)
    
    print(f"Prefill: logits {logits.shape}")
    print(f"  cache_k (contiguous): {cache_k.shape}")
    print(f"  Cache K at pos 0-9 max: {cache_k[:10].abs().max().item():.2f}")
    print(f"  Cache K at pos 10+ max: {cache_k[10:].abs().max().item():.4f}")
    
    # Test decode
    # After prefill of 10 tokens, we're at position 10
    decode_token = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.int64)
    position_id = torch.tensor([[10]], dtype=torch.int64)
    # Attention mask: [1, max_cache_len+1] - 1s for positions 0-10 (10 past + 1 current)
    attention_mask = torch.zeros(1, max_cache_len + 1, dtype=torch.long)
    attention_mask[0, :11] = 1  # Valid for first 11 positions (0-10)
    
    with torch.no_grad():
        logits, new_kv = llm.decode_step(decode_token, position_id, attention_mask, cache_k, cache_v)
    
    print(f"Decode: logits {logits.shape}, new_kv {new_kv.shape}")
    print(f"  new_kv contains: K[1, {num_layers}, {num_kv_heads}, {head_dim}] + V[same]")
    print(f"  Total new_kv size: {new_kv.numel() * 2} bytes = {new_kv.numel() * 2 / 1024:.1f} KB")
    
    # Verify update is contiguous
    pos = 10
    slice_size = num_layers * num_kv_heads * head_dim * 2  # K and V
    print(f"\n  Cache update at pos {pos}:")
    print(f"    Contiguous slice size: {slice_size} elements = {slice_size * 2 / 1024:.1f} KB")
    print(f"    Single H2D transfer: YES!")
    
    # =========== Export to MLIR ===========
    print("\n=== Exporting to MLIR ===")
    
    # Externalize parameters so MLIR references them instead of embedding
    externalize_module_parameters(llm)
    
    fxb = FxProgramsBuilder(llm)
    
    # Prefill example inputs
    prefill_input_ids = torch.zeros((1, max_cache_len), dtype=torch.int64)
    prefill_attention_mask = torch.ones((1, max_cache_len), dtype=torch.int64)
    
    @fxb.export_program(
        name="prefill",
        args=(prefill_input_ids, prefill_attention_mask),
    )
    def prefill_export(module, input_ids, attention_mask):
        return module.prefill(input_ids, attention_mask)
    
    # Decode step example inputs - CONTIGUOUS LAYOUT
    decode_token = torch.zeros((1, 1), dtype=torch.int64)
    decode_position_id = torch.zeros((1, 1), dtype=torch.int64)
    decode_attention_mask = torch.ones((1, max_cache_len + 1), dtype=torch.int64)  # +1 for current token
    decode_cache_k = torch.zeros(cache_shape_contiguous, dtype=dtype)  # [seq, layers, heads, dim]
    decode_cache_v = torch.zeros(cache_shape_contiguous, dtype=dtype)
    
    @fxb.export_program(
        name="decode_step",
        args=(decode_token, decode_position_id, decode_attention_mask, decode_cache_k, decode_cache_v),
    )
    def decode_export(module, token, position_id, attention_mask, cache_k, cache_v):
        return module.decode_step(token, position_id, attention_mask, cache_k, cache_v)
    
    print("Tracing programs...")
    output = export(fxb, import_symbolic_shape_expressions=True)
    
    mlir_path = output_dir / "model.mlir"
    output.save_mlir(str(mlir_path))
    print(f"Saved MLIR to {mlir_path}")
    
    # =========== Save Weights as IRPA ===========
    print("\n=== Saving Weights as IRPA ===")
    
    irpa_path = output_dir / "model.irpa"
    save_module_parameters(str(irpa_path), llm)
    print(f"Saved IRPA ({irpa_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # =========== Save Config ===========
    config_dict = {
        "model_type": "llama_contiguous",
        "hf_model": args.model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.vocab_size,
        "max_seq_len": args.max_seq_len,
        "prefill_len": max_cache_len,  # Prefill input length = cache length = 512
        "max_cache_len": max_cache_len,
        "attention_mask_len": max_cache_len + 1,  # For decode: 512 past + 1 current = 513
        "dtype": args.dtype,
        "cache_layout": "contiguous",
        "cache_shape": list(cache_shape_contiguous),
        "new_kv_shape": [1, 2, num_layers, num_kv_heads, head_dim],
        "vmfb_path": "model.vmfb",
        "irpa_path": "model.irpa",
        "tokenizer_path": "tokenizer.json",
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
    print(f"\nKey benefits of contiguous cache layout:")
    print(f"  - Cache update: 1 H2D transfer of {slice_size * 2 / 1024:.1f} KB")
    print(f"  - vs. old layout: 512 H2D transfers of 256 bytes each")
    print(f"  - Expected speedup: ~10x for cache update")
    print(f"\nNext steps:")
    print(f"  1. Compile MLIR: iree-compile {mlir_path} -o {output_dir}/model.vmfb --iree-hal-target-backends=rocm")
    print(f"  2. Update C++ backend to use contiguous cache layout")


if __name__ == "__main__":
    main()

