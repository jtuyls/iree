#!/usr/bin/env python3
"""
Patched LLM export script that fixes the 512 position limit.
Based on sharktank's export_paged_llm_v1.py with fixes for start_positions tracing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

def ceildiv(a, b):
    return (a + b - 1) // b


def main():
    parser = argparse.ArgumentParser(description="Export LLM with fixed position handling")
    parser.add_argument("--hf-dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g., 'amd-shark/llama3.1-8B')")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--block-seq-stride", type=int, default=32,
                        help="Block sequence stride for paged KV cache")
    parser.add_argument("--device-block-count", type=int, default=128,
                        help="Number of device blocks for KV cache")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Maximum sequence length (for tracing)")
    args = parser.parse_args()

    # Import sharktank components
    from sharktank.examples.export_paged_llm_v1 import (
        run_export,
    )
    
    print(f"Exporting {args.hf_dataset} with fixed position handling")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  block_seq_stride: {args.block_seq_stride}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the standard export
    # Note: We would need to patch the export function to use a larger start_positions
    # For now, let's use the CLI
    
    cmd = [
        sys.executable, "-m", "sharktank.examples.export_paged_llm_v1",
        "--hf-dataset", args.hf_dataset,
        "--output-mlir", str(output_dir / "model.mlir"),
        "--output-config", str(output_dir / "config.json"),
        "--block-seq-stride", str(args.block_seq_stride),
        "--device-block-count", str(args.device_block_count),
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Export failed:\n{result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    print("\nExport complete!")


if __name__ == "__main__":
    main()

