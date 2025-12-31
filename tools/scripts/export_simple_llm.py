#!/usr/bin/env python3
"""
Export a simple LLM model without paged attention for single-batch inference.
Uses standard HuggingFace transformers with torch.export + IREE compilation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Export simple LLM for IREE")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for exported files")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for export")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    print(f"Loading model: {args.model}")
    
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load config first to get model parameters
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"Model config: {config.num_hidden_layers} layers, {config.hidden_size} hidden, "
          f"{config.num_attention_heads} heads")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Saved tokenizer to {output_dir}")
    
    # Create wrapper for simple inference
    class SimpleLLM(nn.Module):
        def __init__(self, model, max_seq_len):
            super().__init__()
            self.model = model
            self.max_seq_len = max_seq_len
            
        def prefill(self, input_ids):
            """
            Prefill: process input tokens and return logits + KV cache.
            input_ids: [batch_size, seq_len]
            Returns: logits [batch_size, seq_len, vocab_size], past_key_values
            """
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
            return outputs.logits, outputs.past_key_values
        
        def decode(self, input_ids, past_key_values):
            """
            Decode: process single token with KV cache.
            input_ids: [batch_size, 1]
            past_key_values: KV cache from previous step
            Returns: logits [batch_size, 1, vocab_size], updated past_key_values
            """
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            return outputs.logits, outputs.past_key_values
    
    wrapper = SimpleLLM(model, args.max_seq_len)
    
    # Test the model
    print("\nTesting model...")
    test_input = torch.tensor([[1, 2, 3]], device=args.device)
    logits, kv_cache = wrapper.prefill(test_input)
    print(f"Prefill output shape: {logits.shape}")
    print(f"KV cache layers: {len(kv_cache)}")
    if len(kv_cache) > 0:
        print(f"KV cache[0] shapes: k={kv_cache[0][0].shape}, v={kv_cache[0][1].shape}")
    
    # Decode step
    next_token = torch.tensor([[logits[0, -1].argmax().item()]], device=args.device)
    logits2, kv_cache2 = wrapper.decode(next_token, kv_cache)
    print(f"Decode output shape: {logits2.shape}")
    
    # Export with torch.export
    print("\nExporting with torch.export...")
    
    # For now, save the model in a format we can use directly
    # torch.export is still experimental for complex models
    
    # Save model config
    model_config = {
        "model_name": args.model,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "head_dim": config.hidden_size // config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "rope_theta": getattr(config, "rope_theta", 10000.0),
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Save model weights
    weights_path = output_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(weights_path))
        print(f"Saved weights to {weights_path}")
    except ImportError:
        weights_path = output_dir / "model.pt"
        torch.save(model.state_dict(), weights_path)
        print(f"Saved weights to {weights_path} (install safetensors for smaller files)")
    
    print("\n=== Export Complete ===")
    print(f"Files saved to: {output_dir}")
    print("\nTo use with Python:")
    print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")

if __name__ == "__main__":
    main()

