// RUN: iree-opt --pass-pipeline='builtin.module(iree-llm-assisted-transform{verbose=true endpoint="http://localhost:99999"})' %s 2>&1 | FileCheck %s --check-prefix=CHECK-UNAVAILABLE
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llm-assisted-transform{verbose=true})' %s 2>&1 | FileCheck %s --check-prefix=CHECK-PASS

// This test verifies that the LLMAssistedTransformPass can be loaded and runs
// without crashing. It tests both scenarios:
// 1. When the backend is unavailable (graceful degradation)
// 2. When the backend is available (pass loads correctly)

// Test 1: Backend unavailable - should gracefully degrade
// CHECK-UNAVAILABLE: LLMAssistedTransformPass: Starting
// CHECK-UNAVAILABLE: Backend: ollama
// CHECK-UNAVAILABLE: module @test_module

// Test 2: Pass runs with available backend (if Ollama is running)
// CHECK-PASS: LLMAssistedTransformPass: Starting
// CHECK-PASS: Backend: ollama
// CHECK-PASS: module @test_module

module @test_module {
  func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}

