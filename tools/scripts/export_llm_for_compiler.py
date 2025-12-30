#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Export an LLM for use with IREE LLMAssist.

This script uses shark-ai/sharktank to export a HuggingFace transformer model
to MLIR + IRPA weights format.

Usage:
    # Activate the shark-ai venv first:
    source ../.shark_ai/bin/activate
    
    # Export using a predefined dataset:
    python export_llm_for_compiler.py \
        --hf-dataset open_llama_3b_v2_f16_gguf \
        --output-dir ../llm-assist-files \
        --bs 1

    # Or with a GGUF file (e.g., Qwen):
    python export_llm_for_compiler.py \
        --gguf-file /path/to/qwen2.5-coder-1.5b.gguf \
        --output-dir ../llm-assist-files \
        --bs 1

The output directory will contain:
    - model.mlir: Exported MLIR module
    - config.json: Model and service configuration
    - tokenizer.model: SentencePiece tokenizer (if available)

Requirements:
    - shark-ai venv with sharktank installed
    - pip install huggingface_hub sentencepiece

Supported Models (via GGUF):
    - Llama family (Llama, Llama2, Llama3)
    - OpenLlama
    - Mistral, Mixtral
    - Qwen/Qwen2 (Llama-compatible architecture)
    - Any other Llama-architecture model in GGUF format
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def setup_sharktank_path():
    """Add sharktank to path if not already installed."""
    # Try to import sharktank
    try:
        import sharktank
        return True
    except ImportError:
        pass
    
    # Try to find sharktank in parent directories
    script_dir = Path(__file__).parent.resolve()
    for parent in [script_dir.parent.parent.parent.parent.parent.parent,  # From LLMAssist/scripts
                   Path.cwd()]:
        sharktank_path = parent / "shark-ai" / "sharktank"
        if sharktank_path.exists():
            sys.path.insert(0, str(sharktank_path))
            try:
                import sharktank
                print(f"Found sharktank at: {sharktank_path}")
                return True
            except ImportError:
                pass
    
    return False


def export_model(args):
    """Export the model using sharktank."""
    
    if not setup_sharktank_path():
        print("ERROR: Could not find sharktank. Make sure you're in the .shark-ai conda environment")
        print("       or that shark-ai is available at ../shark-ai")
        return 1
    
    # Import sharktank components
    from sharktank.utils import cli
    from sharktank.examples.export_paged_llm_v1 import export_llm_v1
    from sharktank.layers.configs import LlamaModelConfig
    from sharktank.models.llm.config import ExportConfig
    from sharktank.types.pipelining import pipeline_parallelize_llm_theta
    from sharktank.layers.configs import ParallelismConfig
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build arguments for sharktank CLI parser
    sharktank_args = []
    
    # Add input source
    if args.hf_dataset:
        sharktank_args.extend(["--hf-dataset", args.hf_dataset])
    elif args.gguf_file:
        sharktank_args.extend(["--gguf-file", str(args.gguf_file)])
    elif args.irpa_file:
        sharktank_args.extend(["--irpa-file", str(args.irpa_file)])
    else:
        print("ERROR: Must specify --hf-dataset, --gguf-file, or --irpa-file")
        return 1
    
    # Add export options
    sharktank_args.extend([
        "--bs-prefill", str(args.bs),
        "--bs-decode", str(args.bs),
        "--output-mlir", str(output_dir / "model.mlir"),
        "--output-config", str(output_dir / "config.json"),
        "--block-seq-stride", str(args.block_seq_stride),
        "--device-block-count", str(args.device_block_count),
        "--attention-kernel", args.attention_kernel,
    ])
    
    if args.activation_dtype:
        sharktank_args.extend(["--activation-dtype", args.activation_dtype])
    
    # Parse with sharktank CLI
    parser = cli.create_parser(
        description="Export LLM for IREE LLMAssist"
    )
    cli.add_input_dataset_options(parser)
    cli.add_model_options(parser)
    cli.add_export_artifacts(parser)
    cli.add_quantization_options(parser)
    
    parsed = cli.parse(parser, args=sharktank_args)
    
    print(f"Loading dataset...")
    dataset = cli.get_input_dataset(parsed)
    
    # Configure export
    export_config = ExportConfig(
        top_k=1,  # Argmax for deterministic output
        device_block_count=parsed.device_block_count,
        logits_normalization=parsed.logits_normalization if hasattr(parsed, 'logits_normalization') else None,
        prefill_final_logits=getattr(parsed, 'prefill_final_logits', False),
        bs_prefill=parsed.bs_prefill,
        bs_decode=parsed.bs_decode,
    )
    
    # Configure model
    dtype_flags = cli.get_dtype_flags(parsed)
    llama_config = LlamaModelConfig.from_dataset(
        dataset=dataset,
        attention_kernel=parsed.attention_kernel,
        block_seq_stride=parsed.block_seq_stride,
        **dtype_flags,
    )
    
    hp = llama_config.hp
    parallelism_config = ParallelismConfig.default_config(
        block_count=hp.block_count,
        tp=1,  # No tensor parallelism for compiler use
        pp=1,  # No pipeline parallelism
    )
    llama_config.parallelism_config = parallelism_config
    
    pipeline_parallelize_llm_theta(dataset.root_theta, llama_config.parallelism_config)
    
    print(f"Exporting model with config:")
    print(f"  - Batch size: {args.bs}")
    print(f"  - Block stride: {args.block_seq_stride}")
    print(f"  - Device blocks: {args.device_block_count}")
    print(f"  - Attention kernel: {args.attention_kernel}")
    
    # Export
    output_export, service_config = export_llm_v1(
        llama_config=llama_config,
        theta=dataset.root_theta,
        export_config=export_config,
        strict=False,
    )
    
    # Save MLIR
    mlir_path = output_dir / "model.mlir"
    print(f"Saving MLIR to: {mlir_path}")
    output_export.save_mlir(str(mlir_path))
    
    # Save weights as IRPA
    irpa_path = output_dir / "model.irpa"
    print(f"Saving weights to: {irpa_path}")
    dataset.save(str(irpa_path))
    
    # Save service config
    import dataclasses
    config_path = output_dir / "config.json"
    print(f"Saving config to: {config_path}")
    config_dict = dataclasses.asdict(service_config)
    
    # Add extra info for LLMAssist
    config_dict["llm_assist"] = {
        "num_layers": hp.block_count,
        "num_heads": hp.attention_head_count,
        "num_kv_heads": hp.attention_head_count_kv,
        "head_dim": hp.attn_head_dim,
        "vocab_size": hp.vocab_size,
        "block_seq_stride": args.block_seq_stride,
        "model_type": "llama",
        "context_length": hp.context_length,
    }
    
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Try to find and copy tokenizer
    try_copy_tokenizer(args, output_dir)
    
    print(f"\nExport complete! Files saved to: {output_dir}")
    print(f"Contents:")
    for item in sorted(output_dir.iterdir()):
        size = item.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  - {item.name}: {size_str}")
    
    return 0


def try_copy_tokenizer(args, output_dir: Path):
    """Try to find and copy the tokenizer model."""
    tokenizer_paths = []
    
    if args.hf_dataset:
        # Try to download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # Map dataset name to repo if needed
            repo_id = args.hf_dataset
            
            # Check for common tokenizer files
            try:
                files = list_repo_files(repo_id)
                for fname in ["tokenizer.model", "tokenizer.json", "spiece.model"]:
                    if fname in files:
                        local_path = hf_hub_download(repo_id=repo_id, filename=fname)
                        tokenizer_paths.append(Path(local_path))
                        break
            except Exception as e:
                print(f"Note: Could not list repo files: {e}")
        except ImportError:
            pass
    
    elif args.gguf_file:
        # Look for tokenizer next to GGUF file
        gguf_path = Path(args.gguf_file)
        for name in ["tokenizer.model", "tokenizer.json", "spiece.model"]:
            candidate = gguf_path.parent / name
            if candidate.exists():
                tokenizer_paths.append(candidate)
                break
    
    # Copy found tokenizer
    for tok_path in tokenizer_paths:
        dest = output_dir / tok_path.name
        shutil.copy(tok_path, dest)
        print(f"Copied tokenizer: {tok_path.name}")
        return
    
    print("Note: No tokenizer model found. You may need to provide one manually.")


def main():
    parser = argparse.ArgumentParser(
        description="Export an LLM for use with IREE LLMAssist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export using a known HuggingFace dataset
    python export_llm_for_compiler.py \\
        --hf-dataset open-llama-3b-v2-f16-gguf \\
        --output-dir ../llm-assist-files

    # Export from a GGUF file
    python export_llm_for_compiler.py \\
        --gguf-file /path/to/model.gguf \\
        --output-dir ../llm-assist-files

Available HF datasets (from sharktank):
    - open-llama-3b-v2-f16-gguf
    - llama3_8b_f16
    - llama3_8b_fp8
    - llama3_70b_f16
    - llama3_405b_fp8
    (and more - check sharktank/utils/hf_datasets.py)
"""
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--hf-dataset",
        type=str,
        help="HuggingFace dataset name (from sharktank's known datasets)",
    )
    input_group.add_argument(
        "--gguf-file",
        type=Path,
        help="Path to a GGUF model file",
    )
    input_group.add_argument(
        "--irpa-file",
        type=Path,
        help="Path to an IRPA weights file",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for exported files",
    )
    
    # Model configuration
    parser.add_argument(
        "--bs",
        type=int,
        default=1,
        help="Batch size for prefill and decode (default: 1)",
    )
    parser.add_argument(
        "--block-seq-stride",
        type=int,
        default=32,
        help="Block sequence stride for paged KV cache (default: 32)",
    )
    parser.add_argument(
        "--device-block-count",
        type=int,
        default=256,
        help="Number of cache blocks per device (default: 256)",
    )
    parser.add_argument(
        "--attention-kernel",
        type=str,
        default="decomposed",
        choices=["decomposed", "torch", "sharktank"],
        help="Attention kernel implementation (default: decomposed)",
    )
    parser.add_argument(
        "--activation-dtype",
        type=str,
        default="float16",
        help="Activation dtype (default: float16)",
    )
    
    args = parser.parse_args()
    
    return export_model(args)


if __name__ == "__main__":
    sys.exit(main())
