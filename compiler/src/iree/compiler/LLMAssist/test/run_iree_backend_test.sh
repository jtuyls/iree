#!/bin/bash
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script to run IREE backend integration test with actual model files.
#
# Usage:
#   ./run_iree_backend_test.sh [OPTIONS]
#
# Required environment variables or options:
#   --vmfb PATH      Path to compiled VMFB model
#   --irpa PATH      Path to IRPA weights file
#   --tokenizer PATH Path to tokenizer model
#   --device DEVICE  IREE device (default: hip)
#
# Example:
#   ./run_iree_backend_test.sh \
#       --vmfb /path/to/model.vmfb \
#       --irpa /path/to/model.irpa \
#       --tokenizer /path/to/tokenizer.model \
#       --device hip

set -e

# Default values
DEVICE="${IREE_DEVICE:-hip}"
IREE_OPT="${IREE_OPT:-iree-opt}"
VMFB=""
IRPA=""
TOKENIZER=""
TASK="Optimize the multiply by 2 using a left shift"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vmfb)
            VMFB="$2"
            shift 2
            ;;
        --irpa)
            IRPA="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --iree-opt)
            IREE_OPT="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VMFB" || -z "$IRPA" || -z "$TOKENIZER" ]]; then
    echo "Error: --vmfb, --irpa, and --tokenizer are required"
    echo "Usage: $0 --vmfb PATH --irpa PATH --tokenizer PATH [--device DEVICE]"
    exit 1
fi

# Check files exist
for FILE in "$VMFB" "$IRPA" "$TOKENIZER"; do
    if [[ ! -f "$FILE" ]]; then
        echo "Error: File not found: $FILE"
        exit 1
    fi
done

# Create test MLIR
TEST_MLIR=$(mktemp /tmp/llm_test_XXXXXX.mlir)
cat > "$TEST_MLIR" << 'EOF'
module @test {
  func.func @multiply_by_two(%arg0: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %0 = arith.muli %arg0, %c2 : i32
    return %0 : i32
  }
}
EOF

echo "=============================================="
echo "IREE LLM Backend Integration Test"
echo "=============================================="
echo "VMFB: $VMFB"
echo "IRPA: $IRPA"
echo "Tokenizer: $TOKENIZER"
echo "Device: $DEVICE"
echo "Task: $TASK"
echo "=============================================="
echo ""

# Run the pass
echo "Running iree-opt with IREE backend..."
echo ""

"$IREE_OPT" "$TEST_MLIR" \
    --pass-pipeline="builtin.module(iree-llm-assisted-transform{ \
        backend=iree \
        iree-vmfb=$VMFB \
        iree-irpa=$IRPA \
        iree-tokenizer=$TOKENIZER \
        iree-device=$DEVICE \
        task=\"$TASK\" \
        verbose=true \
    })" 2>&1

RESULT=$?

# Cleanup
rm -f "$TEST_MLIR"

echo ""
echo "=============================================="
if [[ $RESULT -eq 0 ]]; then
    echo "Test PASSED"
else
    echo "Test FAILED (exit code: $RESULT)"
fi
echo "=============================================="

exit $RESULT

