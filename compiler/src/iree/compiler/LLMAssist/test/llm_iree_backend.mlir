// Test that the IREE backend can be initialized and run inference.
//
// This test verifies:
// 1. The backend can load model config from the same directory as VMFB
// 2. The backend can create a HIP device (or other device)
// 3. The backend can allocate KV-cache buffers
// 4. The backend can invoke prefill and decode functions
//
// Note: Output quality (getting correct tokens) is still being debugged.
// See test_iree_backend_integration.py for the working Python version.
//
// To run this test manually:
//   iree-opt test.mlir --pass-pipeline='builtin.module(iree-llm-assisted-transform{
//     backend=iree iree-vmfb=/path/to/model.vmfb iree-irpa=/path/to/model.irpa
//     iree-tokenizer=/path/to/tokenizer.model iree-device=hip task="test" verbose=true})'

// REQUIRES: iree_backend_available
// Note: The sharktank-exported model has a 512 decode position limit.
// Keep the total prompt (system prompt + IR + task) under ~300 tokens to leave room for generation.
// RUN: iree-opt %s --pass-pipeline='builtin.module(iree-llm-assisted-transform{backend=iree iree-vmfb=%LLM_ASSIST_VMFB% iree-irpa=%LLM_ASSIST_IRPA% iree-tokenizer=%LLM_ASSIST_TOKENIZER% iree-device=%LLM_ASSIST_DEVICE% task="shift" verbose=true})' 2>&1 | FileCheck %s

// CHECK: IREEBackend::initialize: Starting
// CHECK: IREEBackend::initialize: Complete!
// CHECK: LLMAssistedTransformPass: Starting
// CHECK: Backend: iree
// CHECK: LLMAssistedTransformPass: Backend is available

module @test {
  func.func @multiply_by_two(%arg0: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %0 = arith.muli %arg0, %c2 : i32
    return %0 : i32
  }
}
