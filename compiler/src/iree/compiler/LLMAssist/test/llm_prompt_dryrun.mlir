// RUN: iree-opt --pass-pipeline='builtin.module(iree-llm-assisted-transform{dry-run=true task="Add documentation comments"})' %s 2>&1 | FileCheck %s

// This test verifies the prompt building in dry-run mode.
// Dry-run mode does not require Ollama to be running.

// CHECK: DRY RUN
// CHECK: expert MLIR compiler engineer
// CHECK: Task:
// CHECK: Add documentation comments
// CHECK: Input MLIR
// CHECK: module @dryrun_test
// CHECK: func.func @example

module @dryrun_test {
  func.func @example(%arg0: f32) -> f32 {
    %cst = arith.constant 2.0 : f32
    %0 = arith.mulf %arg0, %cst : f32
    return %0 : f32
  }
}
