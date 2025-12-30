// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_PASSES
#define IREE_COMPILER_LLMASSIST_PASSES

include "mlir/Pass/PassBase.td"

def LLMAssistedTransformPass :
    Pass<"iree-llm-assisted-transform", "ModuleOp"> {
  let summary = "Applies LLM-assisted transformations to IR";
  let description = [{
    This pass uses a Large Language Model (LLM) backend to suggest and apply
    transformations to the IR. The LLM is queried with serialized IR fragments
    and task descriptions, and the suggested transformations are validated
    before being applied.
    
    This is an experimental pass and requires an LLM backend (e.g., Ollama)
    to be available.
  }];
  let options = [
    Option<"backendType", "backend", "std::string", "\"ollama\"",
      "LLM backend type: 'ollama' or 'iree'">,
    Option<"endpoint", "endpoint", "std::string", "\"http://localhost:11434\"",
      "LLM backend endpoint URL (for ollama)">,
    Option<"model", "model", "std::string", "\"qwen2.5-coder:7b\"",
      "Model name to use for generation (for ollama)">,
    Option<"taskDescription", "task", "std::string", "\"\"",
      "Task description for the LLM (what transformation to apply)">,
    Option<"dryRun", "dry-run", "bool", "false",
      "If true, only print the LLM suggestion without applying it">,
    Option<"verbose", "verbose", "bool", "false",
      "Enable verbose output for debugging">,
    // IREE backend options
    Option<"ireeVmfbPath", "iree-vmfb", "std::string", "\"\"",
      "Path to IREE compiled model (.vmfb) for IREE backend">,
    Option<"ireeIrpaPath", "iree-irpa", "std::string", "\"\"",
      "Path to IREE parameter archive (.irpa) for IREE backend">,
    Option<"ireeTokenizerPath", "iree-tokenizer", "std::string", "\"\"",
      "Path to tokenizer model for IREE backend">,
    Option<"ireeDevice", "iree-device", "std::string", "\"local-task\"",
      "IREE device URI (e.g., 'local-task', 'hip')">,
  ];
}

#endif // IREE_COMPILER_LLMASSIST_PASSES

