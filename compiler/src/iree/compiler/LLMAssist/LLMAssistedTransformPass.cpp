// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Backend/LLMBackend.h"
#include "iree/compiler/LLMAssist/Backend/OllamaBackend.h"
#include "iree/compiler/LLMAssist/Backend/IREEBackend.h"
#include "iree/compiler/LLMAssist/IR/IRParser.h"
#include "iree/compiler/LLMAssist/IR/IRSerializer.h"
#include "iree/compiler/LLMAssist/Passes.h"
#include "iree/compiler/LLMAssist/Prompt/PromptBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::LLMAssist {

#define GEN_PASS_DEF_LLMASSISTEDTRANSFORMPASS
#include "iree/compiler/LLMAssist/Passes.h.inc"

namespace {

class LLMAssistedTransformPass
    : public impl::LLMAssistedTransformPassBase<LLMAssistedTransformPass> {
public:
  using impl::LLMAssistedTransformPassBase<
      LLMAssistedTransformPass>::LLMAssistedTransformPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Starting\n";
      llvm::outs() << "  Backend: " << backendType << "\n";
      if (backendType == "ollama") {
        llvm::outs() << "  Endpoint: " << endpoint << "\n";
        llvm::outs() << "  Model: " << model << "\n";
      } else if (backendType == "iree") {
        llvm::outs() << "  VMFB: " << ireeVmfbPath << "\n";
        llvm::outs() << "  IRPA: " << ireeIrpaPath << "\n";
        llvm::outs() << "  Tokenizer: " << ireeTokenizerPath << "\n";
        llvm::outs() << "  Device: " << ireeDevice << "\n";
      }
      llvm::outs() << "  Task: "
                   << (taskDescription.empty() ? "(none)"
                                               : taskDescription.c_str())
                   << "\n";
      llvm::outs() << "  Dry run: " << (dryRun ? "yes" : "no") << "\n";
    }

    // Create the LLM backend.
    std::unique_ptr<LLMBackend> backend;
    if (backendType == "ollama") {
      backend = createOllamaBackend(endpoint);
    } else if (backendType == "iree") {
      // Validate required IREE backend options.
      if (ireeVmfbPath.empty() || ireeTokenizerPath.empty()) {
        emitError(module.getLoc())
            << "IREE backend requires --iree-vmfb and --iree-tokenizer options";
        return signalPassFailure();
      }
      IREEBackendConfig ireeConfig;
      ireeConfig.vmfbPath = ireeVmfbPath.getValue();
      ireeConfig.irpaPath = ireeIrpaPath.getValue();
      ireeConfig.tokenizerPath = ireeTokenizerPath.getValue();
      ireeConfig.deviceUri = ireeDevice.getValue();
      
      auto backendOrErr = IREEBackend::create(ireeConfig);
      if (!backendOrErr) {
        llvm::errs() << "LLMAssistedTransformPass: Failed to create IREE backend: "
                     << llvm::toString(backendOrErr.takeError()) << "\n";
        return;
      }
      backend = std::move(*backendOrErr);
    } else {
      emitError(module.getLoc())
          << "Unknown LLM backend type: " << backendType
          << " (supported: 'ollama', 'iree')";
      return signalPassFailure();
    }

    // Check if backend is available.
    if (!backend->isAvailable()) {
      // Graceful degradation: log warning but don't fail the pass.
      llvm::errs()
          << "LLMAssistedTransformPass: LLM backend '" << backend->getName()
          << "' is not available. Pass will have no effect.\n";
      return;
    }

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Backend is available\n";
    }

    // If no task is specified, do nothing.
    if (taskDescription.empty()) {
      if (verbose) {
        llvm::outs() << "LLMAssistedTransformPass: No task specified, "
                        "skipping LLM query\n";
      }
      return;
    }

    // Serialize the IR to text.
    SerializationOptions serOpts;
    serOpts.useGenericForm = false; // Pretty print is more readable for LLM
    serOpts.includeLocations = false;
    std::string irText = IRSerializer::serializeModule(module, serOpts);

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Serialized IR ("
                   << irText.size() << " chars)\n";
    }

    // Build the prompt.
    PromptBuilder promptBuilder;
    PromptConfig promptConfig;
    promptConfig.taskDescription = taskDescription.getValue();
    promptConfig.includeFewShot = true;
    promptConfig.maxFewShotExamples = 1;
    promptConfig.requestExplanation = false;

    std::string prompt = promptBuilder.buildTransformPrompt(irText, promptConfig);

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Built prompt ("
                   << prompt.size() << " chars)\n";
    }

    if (dryRun) {
      llvm::outs() << "=== DRY RUN: Prompt ===\n" << prompt << "\n";
      return;
    }

    // Configure generation.
    GenerationConfig config;
    config.model = model.getValue();
    config.temperature = 0.0f; // Deterministic
    config.maxTokens = 4096;

    // Query the LLM.
    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Querying LLM...\n";
    }

    auto result = backend->generate(prompt, config);
    if (!result) {
      llvm::errs() << "LLMAssistedTransformPass: LLM generation failed: "
                   << llvm::toString(result.takeError()) << "\n";
      return;
    }

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: LLM response received\n";
      llvm::outs() << "  Prompt tokens: " << result->promptTokens << "\n";
      llvm::outs() << "  Completion tokens: " << result->completionTokens
                   << "\n";
      llvm::outs() << "  Latency: " << result->latencyMs << " ms\n";
    }

    // Extract MLIR from the response.
    std::string extractedMLIR =
        IRParser::extractMLIRFromResponse(result->content);

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Extracted MLIR ("
                   << extractedMLIR.size() << " chars)\n";
    }

    // Parse the extracted MLIR.
    auto parseResult =
        IRParser::parseModule(extractedMLIR, module.getContext());

    if (!parseResult) {
      llvm::errs() << "LLMAssistedTransformPass: Failed to parse LLM response "
                      "as valid MLIR\n";
      for (const auto &diag : parseResult.diagnostics) {
        llvm::errs() << "  " << diag << "\n";
      }
      llvm::errs() << "=== Raw LLM Response ===\n" << result->content << "\n";
      llvm::errs() << "=== Extracted MLIR ===\n" << extractedMLIR << "\n";
      // Don't fail the pass, just skip the transformation.
      return;
    }

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Successfully parsed "
                      "transformed MLIR\n";
    }

    // Validate compatibility.
    if (!IRParser::validateCompatibility(module, parseResult.module.get())) {
      llvm::errs() << "LLMAssistedTransformPass: Transformed IR is not "
                      "compatible with original (signature mismatch)\n";
      return;
    }

    if (verbose) {
      llvm::outs()
          << "LLMAssistedTransformPass: Compatibility validation passed\n";
    }

    // Apply the transformation by replacing the module body.
    // We need to be careful here - we'll replace the contents of each function.
    applyTransformation(module, parseResult.module.get());

    if (verbose) {
      llvm::outs() << "LLMAssistedTransformPass: Transformation applied\n";
    }
  }

private:
  /// Apply the transformation from the LLM-generated module to the original.
  void applyTransformation(ModuleOp original, ModuleOp transformed) {
    // Build a map from function name to transformed function.
    llvm::StringMap<mlir::func::FuncOp> transMap;
    for (auto func : transformed.getOps<mlir::func::FuncOp>()) {
      transMap[func.getName()] = func;
    }

    // For each original function, replace its body with the transformed one.
    for (auto origFunc :
         llvm::make_early_inc_range(original.getOps<mlir::func::FuncOp>())) {
      auto it = transMap.find(origFunc.getName());
      if (it == transMap.end()) {
        if (verbose) {
          llvm::outs() << "  Skipping function " << origFunc.getName()
                       << " (not in transformed module)\n";
        }
        continue;
      }

      mlir::func::FuncOp transFunc = it->second;

      // Clear the original function body.
      origFunc.getBody().getBlocks().clear();

      // Clone the transformed function's blocks into the original.
      IRMapping mapping;
      for (auto &block : transFunc.getBody()) {
        Block *newBlock = new Block();
        origFunc.getBody().push_back(newBlock);

        // Map block arguments.
        for (auto arg : block.getArguments()) {
          Value newArg = newBlock->addArgument(arg.getType(), arg.getLoc());
          mapping.map(arg, newArg);
        }

        // Clone operations.
        for (auto &op : block) {
          newBlock->push_back(op.clone(mapping));
        }
      }

      if (verbose) {
        llvm::outs() << "  Replaced function body: " << origFunc.getName()
                     << "\n";
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::LLMAssist
