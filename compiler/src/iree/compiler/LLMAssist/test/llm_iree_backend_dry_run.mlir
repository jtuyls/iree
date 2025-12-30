// RUN: iree-opt %s --pass-pipeline='builtin.module(iree-llm-assisted-transform{backend=iree iree-vmfb=/nonexistent/model.vmfb iree-tokenizer=/nonexistent/tokenizer.model task="test task" verbose=true})' 2>&1 | FileCheck %s

// Test that the IREE backend gracefully handles missing files.
// This verifies:
// 1. The IREE backend option is recognized
// 2. The pass handles missing files gracefully (doesn't crash)
// 3. Appropriate error message is produced

// CHECK: IREEBackend::initialize: Starting
// CHECK: LLMAssistedTransformPass: Failed to create IREE backend
// CHECK: LLMAssistedTransformPass: Starting
// CHECK: Backend: iree
// CHECK: VMFB: /nonexistent/model.vmfb
// CHECK: Tokenizer: /nonexistent/tokenizer.model
// CHECK: Device: local-task
// CHECK: Task: test task

module @test {
  func.func @simple(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}

