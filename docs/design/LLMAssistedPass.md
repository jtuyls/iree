# LLM-Assisted MLIR Pass for IREE

## Executive Summary

This document outlines a comprehensive plan to integrate Large Language Models (LLMs) into IREE's MLIR compiler infrastructure, enabling AI-assisted IR transformations. The system will allow passes to query an LLM for suggested code transformations, validate the suggestions, and apply them to the IR.

### High-Level Goals

1. **Enable AI-assisted compiler transformations** - Use LLMs to suggest optimizations, refactorings, and transformations that are difficult to express as traditional pattern-based rewrites
2. **Progressive implementation** - Start with a simple HTTP-based prototype (Ollama), then migrate to native IREE execution for performance
3. **Full self-hosting** - Eventually run the entire LLM inference stack (model + tokenizer) using IREE itself

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM-Assisted Pass Architecture                        │
│                                                                              │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐   │
│  │   Input    │     │   Prompt   │     │    LLM     │     │  Validate  │   │
│  │    IR      │────▶│  Builder   │────▶│  Backend   │────▶│  & Apply   │   │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘   │
│                                              │                              │
│                            ┌─────────────────┼─────────────────┐            │
│                            ▼                 ▼                 ▼            │
│                     ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│                     │  Ollama  │      │   IREE   │      │   IREE   │       │
│                     │  (HTTP)  │      │ + SPM    │      │ + IREE   │       │
│                     │ Phase 1  │      │ Phase 2  │      │ Phase 3  │       │
│                     └──────────┘      └──────────┘      └──────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

| Phase | Description | Duration | Key Deliverable |
|-------|-------------|----------|-----------------|
| **Phase 1** | Ollama HTTP prototype | 3-4 weeks | Working end-to-end with external LLM |
| **Phase 2** | IREE native + SentencePiece | 4-5 weeks | In-process inference, no HTTP |
| **Phase 3** | IREE-compiled tokenizer | 3-4 weeks | Fully self-contained system |
| **Phase 4** | Experimentation | Ongoing | Model tuning, cache algorithms |

### Key Design Decisions

1. **Tokenizer Strategy**: Use HuggingFace `tokenizers` or SentencePiece (proven, fast) before attempting IREE-compiled tokenization
2. **KV-Cache**: Explicit cache state as function arguments (following shark-ai patterns)
3. **Validation**: Always validate LLM output before applying transformations
4. **Graceful Degradation**: Pass should never fail compilation; fall back to no-op if LLM unavailable

---

## Table of Contents

- [Phase 1: Ollama Prototype](#phase-1-ollama-prototype-3-4-weeks)
  - [1.0 Prerequisites & Environment Setup](#10-prerequisites--environment-setup)
  - [1.1 Project Setup & Infrastructure](#11-project-setup--infrastructure-week-1)
  - [1.2 Ollama HTTP Backend](#12-ollama-http-backend-week-1-2)
  - [1.3 IR Serialization & Parsing](#13-ir-serialization--parsing-week-2)
  - [1.4 Prompt Engineering System](#14-prompt-engineering-system-week-2-3)
  - [1.5 LLMAssistedPass Implementation](#15-llmassistedpass-implementation-week-3)
  - [1.6 Testing & Validation](#16-testing--validation-week-3-4)
- [Phase 2: IREE Native Integration](#phase-2-iree-native-integration-4-5-weeks)
  - [2.1 SentencePiece Integration](#21-sentencepiece-integration-week-5)
  - [2.2 LLM Model Export](#22-llm-model-export-week-5-6)
  - [2.3 KV-Cache Management Layer](#23-kv-cache-management-layer-week-6-7)
  - [2.4 IREE LLM Backend](#24-iree-llm-backend-week-7-8)
  - [2.5 Backend Switching & Testing](#25-backend-switching--testing-week-8-9)
- [Phase 3: IREE-Compiled Tokenizer](#phase-3-iree-compiled-tokenizer-3-4-weeks)
  - [3.1 Tokenizer Model Analysis](#31-tokenizer-model-analysis-week-10)
  - [3.2 Tokenizer Model Implementation](#32-tokenizer-model-implementation-week-10-11)
  - [3.3 C++ IREE Tokenizer Integration](#33-c-iree-tokenizer-integration-week-11-12)
  - [3.4 Validation & Optimization](#34-validation--optimization-week-12-13)
- [Phase 4: Advanced Experimentation](#phase-4-advanced-experimentation-ongoing)
  - [4.1 Model Experimentation](#41-model-experimentation)
  - [4.2 KV-Cache Algorithm Experiments](#42-kv-cache-algorithm-experiments)
  - [4.3 Fine-Tuning for MLIR](#43-fine-tuning-for-mlir)
  - [4.4 Evaluation Framework](#44-evaluation-framework)
- [Summary Timeline](#summary-timeline)

---

## Phase 1: Ollama Prototype (3-4 weeks)

**Goal**: Validate the concept with a working end-to-end prototype using Ollama as the LLM backend.

### 1.0 Prerequisites & Environment Setup

Before developing or testing the LLM-assisted pass, you need to set up an Ollama server with a code-capable model.

#### 1.0.1 Installing Ollama

Ollama is available for Linux, macOS, and Windows. Choose the appropriate installation method:

**Linux (recommended for development):**
```bash
# One-line install script
curl -fsSL https://ollama.com/install.sh | sh

# Or using package manager (Ubuntu/Debian)
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

**macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.com/download
```

**Docker (for isolated environments):**
```bash
# Pull and run Ollama in Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# For GPU support (NVIDIA)
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### 1.0.2 Pulling a Code Model

For MLIR/compiler work, we recommend code-specialized models. Pull one of these:

```bash
# Recommended: Qwen2.5-Coder (best for code tasks, multiple sizes)
ollama pull qwen2.5-coder:7b      # 7B parameters, ~4GB, good balance
ollama pull qwen2.5-coder:14b     # 14B parameters, ~8GB, better quality
ollama pull qwen2.5-coder:32b     # 32B parameters, ~18GB, best quality

# Alternative: DeepSeek-Coder
ollama pull deepseek-coder:6.7b   # 6.7B parameters
ollama pull deepseek-coder:33b    # 33B parameters

# Alternative: CodeLlama
ollama pull codellama:7b          # 7B parameters
ollama pull codellama:13b         # 13B parameters

# List downloaded models
ollama list
```

**Model Selection Guidelines:**
| Model | Size | VRAM Required | Use Case |
|-------|------|---------------|----------|
| qwen2.5-coder:7b | ~4GB | 8GB+ | Development/testing |
| qwen2.5-coder:14b | ~8GB | 16GB+ | Better quality, still fast |
| qwen2.5-coder:32b | ~18GB | 32GB+ | Best quality for complex IR |
| deepseek-coder:6.7b | ~4GB | 8GB+ | Alternative, good at code |

#### 1.0.3 Starting the Ollama Server

```bash
# Start the Ollama server (runs on http://localhost:11434 by default)
ollama serve

# Or run in background
ollama serve &

# Check if server is running
curl http://localhost:11434/api/tags
```

**Custom Configuration:**
```bash
# Run on a different port
OLLAMA_HOST=0.0.0.0:8080 ollama serve

# Limit GPU memory usage
OLLAMA_GPU_MEMORY=8GB ollama serve

# Enable debug logging
OLLAMA_DEBUG=1 ollama serve
```

#### 1.0.4 Verifying the Setup

Test that Ollama is working correctly:

```bash
# Quick test via CLI
ollama run qwen2.5-coder:7b "Write a hello world in Python"

# Test via API (same as what our pass will use)
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:7b",
  "prompt": "Write a function that adds two numbers",
  "stream": false
}'

# Check available models
curl http://localhost:11434/api/tags
```

#### 1.0.5 Testing with IREE (Once Implemented)

After building IREE with `-DIREE_ENABLE_LLM_ASSIST=ON`:

```bash
# Basic test - should show backend availability
echo 'module { func.func @test() { return } }' | \
  iree-opt --pass-pipeline='builtin.module(iree-llm-assisted-transform{verbose=true})'

# Test with a task description
echo 'module { func.func @add(%a: i32, %b: i32) -> i32 { 
  %c = arith.addi %a, %b : i32 
  return %c : i32 
} }' | \
  iree-opt --pass-pipeline='builtin.module(iree-llm-assisted-transform{
    verbose=true 
    task="Add documentation comments to this function"
  })'
```

#### 1.0.6 Troubleshooting

**Common Issues:**

1. **"Connection refused" error:**
   ```bash
   # Check if Ollama is running
   pgrep -f ollama
   
   # Restart the server
   pkill ollama && ollama serve
   ```

2. **"Model not found" error:**
   ```bash
   # List available models
   ollama list
   
   # Pull the missing model
   ollama pull qwen2.5-coder:7b
   ```

3. **Out of memory (OOM):**
   ```bash
   # Use a smaller model
   ollama pull qwen2.5-coder:3b
   
   # Or limit context size in API call
   # (handled in pass options)
   ```

4. **Slow inference:**
   ```bash
   # Check GPU is being used
   nvidia-smi  # Should show ollama process
   
   # Ensure CUDA is available
   ollama run qwen2.5-coder:7b --verbose
   ```

#### 1.0.7 Environment Variables

The LLM-assisted pass respects these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IREE_LLM_ENDPOINT` | `http://localhost:11434` | Ollama server URL |
| `IREE_LLM_MODEL` | `qwen2.5-coder:7b` | Default model name |
| `IREE_LLM_TIMEOUT` | `300` | Request timeout (seconds) |
| `IREE_LLM_DISABLED` | `0` | Set to `1` to disable LLM calls |

---

### 1.1 Project Setup & Infrastructure (Week 1)

#### 1.1.1 Create Module Structure

```
compiler/src/iree/compiler/LLMAssist/
├── CMakeLists.txt
├── BUILD.bazel
├── LLMAssist.h                    # Public API header
├── Passes.td                      # TableGen pass definitions
├── Passes.h                       # Pass registration
├── Passes.cpp                     # Pass implementations
├── Backend/
│   ├── LLMBackend.h              # Abstract backend interface
│   ├── LLMBackend.cpp
│   ├── OllamaBackend.h           # Ollama HTTP implementation
│   └── OllamaBackend.cpp
├── IR/
│   ├── IRSerializer.h            # IR to text conversion
│   ├── IRSerializer.cpp
│   ├── IRParser.h                # Text to IR parsing
│   └── IRParser.cpp
├── Prompt/
│   ├── PromptBuilder.h           # Prompt template system
│   ├── PromptBuilder.cpp
│   └── Templates/                # Prompt template files
│       ├── transform.txt
│       └── few_shot_examples.txt
├── Cache/
│   ├── ResponseCache.h           # Caching layer
│   └── ResponseCache.cpp
└── test/
    ├── ollama_backend_test.cpp
    ├── ir_serializer_test.cpp
    └── llm_assisted_pass_test.mlir
```

#### 1.1.2 CMake Integration

```cmake
# compiler/src/iree/compiler/LLMAssist/CMakeLists.txt

iree_cc_library(
  NAME
    LLMAssist
  HDRS
    "LLMAssist.h"
    "Passes.h"
  SRCS
    "Passes.cpp"
    "Backend/LLMBackend.cpp"
    "Backend/OllamaBackend.cpp"
    "IR/IRSerializer.cpp"
    "IR/IRParser.cpp"
    "Prompt/PromptBuilder.cpp"
    "Cache/ResponseCache.cpp"
  DEPS
    LLVMSupport
    MLIRParser
    MLIRPass
    iree::compiler::Dialect::Util::IR
    # HTTP client library (cpp-httplib or similar)
  PUBLIC
)

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)
```

#### 1.1.3 Deliverables - Week 1

- [x] Module skeleton with CMake/Bazel build
- [x] Abstract `LLMBackend` interface defined
- [x] Basic test infrastructure
- [x] CMake flag `IREE_ENABLE_LLM_ASSIST` (off by default)
- [x] Stub `OllamaBackend` implementation
- [x] Pass registered and accessible via `--iree-llm-assisted-transform`

---

### 1.2 Ollama HTTP Backend (Week 1-2)

#### 1.2.1 LLM Backend Interface

```cpp
// Backend/LLMBackend.h

namespace mlir::iree_compiler::LLMAssist {

struct GenerationConfig {
  float temperature = 0.0f;  // Deterministic by default
  int maxTokens = 4096;
  std::optional<int> seed;
  std::string model = "qwen2.5-coder:7b";
};

struct GenerationResult {
  std::string content;
  int promptTokens;
  int completionTokens;
  float latencyMs;
};

class LLMBackend {
public:
  virtual ~LLMBackend() = default;
  
  // Check if backend is available
  virtual bool isAvailable() const = 0;
  
  // Synchronous generation
  virtual FailureOr<GenerationResult> generate(
      StringRef prompt,
      const GenerationConfig &config) = 0;
  
  // Get backend name for logging
  virtual StringRef getName() const = 0;
};

// Factory function
std::unique_ptr<LLMBackend> createOllamaBackend(StringRef endpoint);

} // namespace
```

#### 1.2.2 Ollama Implementation

```cpp
// Backend/OllamaBackend.cpp

#include "httplib.h"  // cpp-httplib or similar
#include "llvm/Support/JSON.h"

class OllamaBackend : public LLMBackend {
  httplib::Client client_;
  std::string endpoint_;
  
public:
  OllamaBackend(StringRef endpoint) 
      : endpoint_(endpoint.str()),
        client_(endpoint.str()) {
    client_.set_read_timeout(300);  // 5 min timeout for long generations
  }
  
  bool isAvailable() const override {
    // Check /api/tags endpoint
    auto res = client_.Get("/api/tags");
    return res && res->status == 200;
  }
  
  FailureOr<GenerationResult> generate(
      StringRef prompt, 
      const GenerationConfig &config) override {
    
    // Build request JSON
    llvm::json::Object request;
    request["model"] = config.model;
    request["prompt"] = prompt.str();
    request["stream"] = false;
    
    llvm::json::Object options;
    options["temperature"] = config.temperature;
    options["num_predict"] = config.maxTokens;
    if (config.seed)
      options["seed"] = *config.seed;
    request["options"] = std::move(options);
    
    std::string body;
    llvm::raw_string_ostream os(body);
    os << llvm::json::Value(std::move(request));
    
    auto start = std::chrono::steady_clock::now();
    auto res = client_.Post("/api/generate", body, "application/json");
    auto end = std::chrono::steady_clock::now();
    
    if (!res || res->status != 200) {
      return failure();
    }
    
    // Parse response
    auto parsed = llvm::json::parse(res->body);
    if (!parsed)
      return failure();
    
    auto *obj = parsed->getAsObject();
    GenerationResult result;
    result.content = obj->getString("response")->str();
    result.latencyMs = std::chrono::duration<float, std::milli>(end - start).count();
    
    if (auto tokens = obj->getInteger("prompt_eval_count"))
      result.promptTokens = *tokens;
    if (auto tokens = obj->getInteger("eval_count"))
      result.completionTokens = *tokens;
    
    return result;
  }
  
  StringRef getName() const override { return "Ollama"; }
};
```

#### 1.2.3 Deliverables - Week 1-2

- [x] Ollama HTTP client implementation (POSIX sockets, no external dependencies)
- [x] JSON request/response handling (using LLVM's JSON library)
- [x] Error handling and timeouts
- [x] Basic prompt building with IR serialization
- [x] End-to-end testing with live Ollama server

---

### 1.3 IR Serialization & Parsing (Week 2)

#### 1.3.1 IR Serializer

```cpp
// IR/IRSerializer.h

namespace mlir::iree_compiler::LLMAssist {

struct SerializationOptions {
  bool useGenericForm = true;      // More regular for LLM
  bool includeLocations = false;   // Usually noise for LLM
  bool prettyPrint = true;
  int contextLines = 5;            // Lines before/after for snippets
};

class IRSerializer {
public:
  // Serialize full operation
  static std::string serialize(Operation *op, const SerializationOptions &opts);
  
  // Serialize region (e.g., function body)
  static std::string serializeRegion(Region &region, const SerializationOptions &opts);
  
  // Serialize with context (surrounding ops)
  static std::string serializeWithContext(
      Operation *op, 
      Operation *contextRoot,
      const SerializationOptions &opts);
  
  // Extract just the body of a function (no signature)
  static std::string serializeFunctionBody(func::FuncOp funcOp, 
                                           const SerializationOptions &opts);
};

} // namespace
```

#### 1.3.2 IR Parser with Validation

```cpp
// IR/IRParser.h

namespace mlir::iree_compiler::LLMAssist {

struct ParseResult {
  OwningOpRef<Operation *> op;
  std::vector<std::string> diagnostics;
  bool hadErrors = false;
};

class IRParser {
public:
  // Parse a complete module
  static ParseResult parseModule(StringRef irText, MLIRContext *ctx);
  
  // Parse operations to insert into existing function
  static ParseResult parseOperations(StringRef irText, 
                                     MLIRContext *ctx,
                                     Block *insertionBlock);
  
  // Validate that parsed IR matches expected types
  static LogicalResult validateTypes(Operation *original, 
                                     Operation *transformed);
  
  // Validate SSA value replacement is valid
  static LogicalResult validateReplacement(Value original, Value replacement);
};

} // namespace
```

#### 1.3.3 Deliverables - Week 2

- [x] IR serialization with generic form option (IRSerializer class)
- [x] IR parsing with diagnostic collection (IRParser class)
- [x] Type validation between original and transformed IR
- [x] MLIR extraction from markdown code blocks

---

### 1.4 Prompt Engineering System (Week 2-3)

#### 1.4.1 Prompt Template System

```cpp
// Prompt/PromptBuilder.h

namespace mlir::iree_compiler::LLMAssist {

struct PromptConfig {
  StringRef taskType;              // "optimize", "vectorize", "fuse", etc.
  bool includeFewShot = true;
  int maxFewShotExamples = 3;
  bool includeDialectDocs = false;
  StringRef customInstructions;
};

class PromptBuilder {
  llvm::StringMap<std::string> templates_;
  llvm::StringMap<std::vector<std::string>> fewShotExamples_;
  
public:
  // Load templates from directory
  LogicalResult loadTemplates(StringRef templateDir);
  
  // Build prompt for IR transformation
  std::string buildTransformPrompt(
      StringRef irText,
      const PromptConfig &config);
  
  // Add few-shot example
  void addFewShotExample(StringRef taskType, 
                         StringRef input, 
                         StringRef output);
};

} // namespace
```

#### 1.4.2 Base Prompt Template

```text
# Prompt/Templates/transform.txt

You are an expert MLIR compiler engineer. Your task is to transform 
the given IREE MLIR code according to the instructions.

## Rules:
1. Output ONLY valid MLIR code, no explanations
2. Preserve the function signature exactly
3. Maintain SSA form
4. Keep type annotations correct
5. Do not introduce undefined values

## IREE Dialect Notes:
- `iree_linalg_ext.` operations are IREE-specific extensions
- `flow.dispatch` creates execution boundaries
- Use `tensor.empty` for allocation, not `bufferization.alloc_tensor`

{{#if few_shot_examples}}
## Examples:
{{#each few_shot_examples}}
### Example {{@index}}:
Input:
```mlir
{{this.input}}
```
Output:
```mlir
{{this.output}}
```
{{/each}}
{{/if}}

## Task: {{task_description}}

## Input IR:
```mlir
{{input_ir}}
```

## Output the transformed MLIR:
```mlir
```

#### 1.4.3 Response Extraction

```cpp
// Prompt/PromptBuilder.cpp

std::string extractMLIRFromResponse(StringRef response) {
  // Find ```mlir ... ``` blocks
  static const llvm::Regex mlirBlock("```mlir\\n([\\s\\S]*?)```");
  
  SmallVector<StringRef, 4> matches;
  if (mlirBlock.match(response, &matches) && matches.size() > 1) {
    return matches[1].str();
  }
  
  // Fallback: try to find module { ... } or func.func
  // ...
  
  return response.str();
}
```

#### 1.4.4 Deliverables - Week 2-3

- [x] Template loading and rendering (PromptBuilder class)
- [x] Few-shot example management
- [x] Response extraction (MLIR from markdown)
- [x] Task-specific prompt variants

---

### 1.5 LLMAssistedPass Implementation (Week 3)

#### 1.5.1 Pass Definition

```tablegen
// Passes.td

def LLMAssistedTransformPass : Pass<"iree-llm-assisted-transform", "func::FuncOp"> {
  let summary = "Apply LLM-suggested transformations to functions";
  let description = [{
    This pass sends function IR to an LLM and applies the suggested
    transformations. It validates the transformed IR before applying.
    
    The pass is designed for experimentation and should be used with
    care in production pipelines.
  }];
  
  let options = [
    Option<"ollamaEndpoint", "ollama-endpoint",
           "std::string", /*default=*/"\"http://localhost:11434\"",
           "Ollama API endpoint">,
    Option<"model", "model",
           "std::string", /*default=*/"\"qwen2.5-coder:7b\"",
           "LLM model to use">,
    Option<"taskDescription", "task",
           "std::string", /*default=*/"\"Optimize this code\"",
           "Task description for the LLM">,
    Option<"dryRun", "dry-run",
           "bool", /*default=*/"false",
           "Print transformations without applying">,
    Option<"cacheDir", "cache-dir",
           "std::string", /*default=*/"\"\"",
           "Directory for caching LLM responses">,
  ];
  
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect",
    // ... other dialects
  ];
}
```

#### 1.5.2 Pass Implementation

```cpp
// Passes.cpp

struct LLMAssistedTransformPass 
    : public impl::LLMAssistedTransformPassBase<LLMAssistedTransformPass> {
  
  std::unique_ptr<LLMBackend> backend_;
  std::unique_ptr<PromptBuilder> promptBuilder_;
  std::unique_ptr<ResponseCache> cache_;
  
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    // Skip small functions
    if (funcOp.getBody().front().getOperations().size() < 3)
      return;
    
    // Initialize backend lazily
    if (!backend_) {
      backend_ = createOllamaBackend(ollamaEndpoint);
      if (!backend_->isAvailable()) {
        funcOp.emitWarning() << "LLM backend not available, skipping";
        return;
      }
      
      promptBuilder_ = std::make_unique<PromptBuilder>();
      if (!cacheDir.empty())
        cache_ = std::make_unique<ResponseCache>(cacheDir);
    }
    
    // Serialize IR
    SerializationOptions serOpts;
    serOpts.useGenericForm = true;
    std::string irText = IRSerializer::serialize(funcOp, serOpts);
    
    // Check cache
    std::string cacheKey = computeHash(irText + taskDescription);
    if (cache_) {
      if (auto cached = cache_->lookup(cacheKey)) {
        applyTransformation(funcOp, *cached);
        return;
      }
    }
    
    // Build prompt
    PromptConfig promptConfig;
    promptConfig.taskType = "optimize";
    promptConfig.customInstructions = taskDescription;
    std::string prompt = promptBuilder_->buildTransformPrompt(irText, promptConfig);
    
    // Query LLM
    GenerationConfig genConfig;
    genConfig.model = model;
    genConfig.temperature = 0.0f;  // Deterministic
    
    auto result = backend_->generate(prompt, genConfig);
    if (failed(result)) {
      funcOp.emitWarning() << "LLM generation failed";
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "LLM response (" << result->latencyMs 
                            << "ms, " << result->completionTokens << " tokens):\n"
                            << result->content << "\n");
    
    // Extract MLIR from response
    std::string transformedIR = extractMLIRFromResponse(result->content);
    
    // Cache result
    if (cache_)
      cache_->store(cacheKey, transformedIR);
    
    // Apply transformation
    if (dryRun) {
      llvm::outs() << "=== LLM Suggested Transformation ===\n"
                   << transformedIR << "\n";
      return;
    }
    
    applyTransformation(funcOp, transformedIR);
  }
  
  void applyTransformation(func::FuncOp original, StringRef newIR) {
    // Parse new IR
    auto parsed = IRParser::parseModule(newIR, &getContext());
    if (parsed.hadErrors) {
      original.emitWarning() << "Failed to parse LLM output: "
                             << llvm::join(parsed.diagnostics, "; ");
      return;
    }
    
    // Find the function in parsed module
    auto *parsedModule = parsed.op.get();
    func::FuncOp newFunc;
    parsedModule->walk([&](func::FuncOp f) {
      if (f.getName() == original.getName())
        newFunc = f;
    });
    
    if (!newFunc) {
      original.emitWarning() << "LLM output missing function " << original.getName();
      return;
    }
    
    // Validate types match
    if (failed(IRParser::validateTypes(original, newFunc))) {
      original.emitWarning() << "Type mismatch in LLM transformation";
      return;
    }
    
    // Replace function body
    original.getBody().takeBody(newFunc.getBody());
    
    // Run verifier
    if (failed(verify(original))) {
      original.emitError() << "LLM transformation produced invalid IR";
      signalPassFailure();
    }
  }
};
```

#### 1.5.3 Deliverables - Week 3

- [x] Pass TableGen definition
- [x] Full pass implementation with transformation application
- [x] Validation and error handling (graceful degradation)
- [x] Integration tests with Ollama (end-to-end working)

---

### 1.6 Testing & Validation (Week 3-4) ✅

> **Status**: IMPLEMENTED - Unit tests and lit tests are in place.

#### 1.6.1 Test Infrastructure

The testing infrastructure is organized into two categories:

**Unit Tests** (in `compiler/src/iree/compiler/LLMAssist/unittests/`):
- `IRSerializerTest.cpp` - Tests for IR serialization to text
- `IRParserTest.cpp` - Tests for parsing MLIR from LLM responses
- `PromptBuilderTest.cpp` - Tests for prompt construction

Run unit tests with:
```bash
cd iree-build
ninja iree_compiler_LLMAssist_unittests_IRSerializerTest
./tools/IRSerializerTest
```

**Lit Tests** (in `compiler/src/iree/compiler/LLMAssist/test/`):
- `llm_assisted_transform.mlir` - Basic pass behavior (works with/without Ollama)
- `llm_prompt_dryrun.mlir` - Verifies prompt generation without LLM
- `llm_transform_e2e.mlir` - End-to-end transformation (requires Ollama)

Run lit tests with:
```bash
cd iree-build
ninja compiler/src/iree/compiler/LLMAssist/test/test
```

#### 1.6.2 Example Unit Test Pattern

```cpp
// unittests/IRSerializerTest.cpp

class IRSerializerTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
  }

  MLIRContext ctx;
};

TEST_F(IRSerializerTest, SerializeSimpleModule) {
  StringRef input = R"mlir(
    module {
      func.func @add(%arg0: i32, %arg1: i32) -> i32 {
        %result = arith.addi %arg0, %arg1 : i32
        return %result : i32
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(input, &ctx);
  ASSERT_TRUE(module);

  std::string output = IRSerializer::serializeModule(*module);
  EXPECT_THAT(output, HasSubstr("arith.addi"));
}
```

#### 1.6.3 Example Lit Tests

#### 1.6.2 Lit Tests

```mlir
// test/llm_assisted_pass_test.mlir

// RUN: iree-opt %s --iree-llm-assisted-transform="task='Vectorize the loop'" \
// RUN:   --dry-run 2>&1 | FileCheck %s

// CHECK: LLM Suggested Transformation

func.func @simple_loop(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.0 : f32
  %result = scf.for %i = %c0 to %c128 step %c1 
      iter_args(%acc = %arg0) -> tensor<128xf32> {
    %elem = tensor.extract %acc[%i] : tensor<128xf32>
    %scaled = arith.mulf %elem, %cst : f32
    %updated = tensor.insert %scaled into %acc[%i] : tensor<128xf32>
    scf.yield %updated : tensor<128xf32>
  }
  return %result : tensor<128xf32>
}
```

#### 1.6.3 Deliverables - Week 3-4

- [x] Unit tests for each component (`IRSerializerTest`, `IRParserTest`, `PromptBuilderTest`)
- [x] Integration tests with Ollama (`llm_transform_e2e.mlir`)
- [x] Lit tests for pass behavior (`llm_assisted_transform.mlir`, `llm_prompt_dryrun.mlir`)
- [x] Documentation and examples (this document)

### Phase 1 Exit Criteria

- [x] Working prototype that can transform simple IR via Ollama
- [ ] Caching to avoid repeated LLM calls (future enhancement)
- [x] Clear error messages for invalid LLM output
- [ ] At least 80% success rate on simple transformations (needs more testing)
- [ ] Latency metrics collection (future enhancement)

---

## Phase 2: IREE Native Integration (4-5 weeks)

**Goal**: Replace Ollama with IREE-compiled LLM + SentencePiece tokenizer for in-process inference.

> **Status**: INFRASTRUCTURE COMPLETE - Core components implemented, pending model export/integration.

### Implementation Summary

The following components have been implemented:

1. **Tokenizer Interface** (`Tokenizer/Tokenizer.h`)
   - Abstract `Tokenizer` class with encode/decode/special token APIs
   - SentencePiece implementation (conditional, enabled with `IREE_LLM_ASSIST_ENABLE_SENTENCEPIECE`)

2. **KV-Cache Manager** (`KVCache/KVCacheManager.h/.cpp`)
   - Simple linear page allocation for single-request inference
   - IREE buffer allocation and management
   - Page ID tracking for cache access

3. **IREE Backend** (`Backend/IREEBackend.h/.cpp`)
   - Full `LLMBackend` implementation
   - Model loading from VMFB
   - Config loading from JSON
   - Token generation loop structure

4. **Export Script** (`scripts/export_llm_for_compiler.py`)
   - Model preparation helper for HuggingFace models
   - Configuration extraction

### Enabling the IREE Backend

To use the IREE backend, you need:

1. Build with SentencePiece:
   ```bash
   cmake -DIREE_ENABLE_LLM_ASSIST=ON -DIREE_LLM_ASSIST_ENABLE_SENTENCEPIECE=ON ...
   ```

2. Export an LLM model using the provided script (requires shark-ai venv):
   ```bash
   # Activate shark-ai environment
   source /path/to/.shark_ai/bin/activate
   
   # Export model (e.g., open_llama_3b)
   python iree/tools/scripts/export_llm_for_compiler.py \
       --hf-dataset "open_llama_3b_v2_f16_gguf" \
       --output-dir ./llm-assist-files/open_llama_3b \
       --bs 1
   
   # Compile to VMFB
   iree-compile \
       --iree-hal-target-backends=llvm-cpu \
       --iree-llvmcpu-target-cpu=host \
       --iree-llvmcpu-stack-allocation-limit=1048576 \
       ./llm-assist-files/open_llama_3b/model.mlir \
       -o ./llm-assist-files/open_llama_3b/model.vmfb
   ```

3. Configure the backend with paths to VMFB, tokenizer, and config

### Exported Model Files

After export, the output directory contains:
- `model.mlir` - MLIR graph with external weight references
- `model.irpa` - Model weights in IREE parameter archive format
- `model.vmfb` - Compiled IREE module (after iree-compile)
- `tokenizer.model` - SentencePiece tokenizer
- `config.json` - Model configuration

### Current Limitations

- **Weight Loading**: IRPA weights must be loaded at runtime (not embedded in VMFB)
- **Prefill/Decode**: Currently placeholder implementations; need to implement actual VM function invocation
- **No GPU Support**: Currently only `local-task` device tested
- **Qwen Models**: Not yet supported by shark-ai (Llama architecture only)

---

### 2.1 SentencePiece Integration (Week 5)

#### 2.1.1 Add SentencePiece Dependency

```cmake
# compiler/src/iree/compiler/LLMAssist/CMakeLists.txt

find_package(SentencePiece QUIET)
if(NOT SentencePiece_FOUND)
  # Fetch and build SentencePiece
  include(FetchContent)
  FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.0
  )
  FetchContent_MakeAvailable(sentencepiece)
endif()

iree_cc_library(
  NAME
    Tokenizer
  HDRS
    "Tokenizer/Tokenizer.h"
  SRCS
    "Tokenizer/SentencePieceTokenizer.cpp"
  DEPS
    sentencepiece-static
    LLVMSupport
)
```

#### 2.1.2 Tokenizer Interface

```cpp
// Tokenizer/Tokenizer.h

namespace mlir::iree_compiler::LLMAssist {

class Tokenizer {
public:
  virtual ~Tokenizer() = default;
  
  // Encode text to token IDs
  virtual std::vector<int64_t> encode(StringRef text) = 0;
  
  // Decode token IDs to text
  virtual std::string decode(ArrayRef<int64_t> ids) = 0;
  
  // Get vocabulary size
  virtual size_t vocabSize() const = 0;
  
  // Special token IDs
  virtual int64_t bosId() const = 0;
  virtual int64_t eosId() const = 0;
  virtual int64_t padId() const = 0;
};

// Factory functions
std::unique_ptr<Tokenizer> createSentencePieceTokenizer(StringRef modelPath);

} // namespace
```

#### 2.1.3 SentencePiece Implementation

```cpp
// Tokenizer/SentencePieceTokenizer.cpp

#include <sentencepiece_processor.h>

class SentencePieceTokenizer : public Tokenizer {
  sentencepiece::SentencePieceProcessor processor_;
  
public:
  SentencePieceTokenizer(StringRef modelPath) {
    auto status = processor_.Load(modelPath.str());
    if (!status.ok()) {
      llvm::report_fatal_error("Failed to load tokenizer: " + 
                               status.error_message());
    }
  }
  
  std::vector<int64_t> encode(StringRef text) override {
    std::vector<int> ids;
    processor_.Encode(text.str(), &ids);
    return std::vector<int64_t>(ids.begin(), ids.end());
  }
  
  std::string decode(ArrayRef<int64_t> ids) override {
    std::vector<int> intIds(ids.begin(), ids.end());
    std::string text;
    processor_.Decode(intIds, &text);
    return text;
  }
  
  size_t vocabSize() const override { 
    return processor_.GetPieceSize(); 
  }
  
  int64_t bosId() const override { return processor_.bos_id(); }
  int64_t eosId() const override { return processor_.eos_id(); }
  int64_t padId() const override { return processor_.pad_id(); }
};
```

#### 2.1.4 Deliverables - Week 5

- [x] SentencePiece build integration (CMake with FetchContent)
- [x] Tokenizer interface and implementation (`Tokenizer.h`, `SentencePieceTokenizer.cpp`)
- [ ] Tests with various tokenizer models (requires SentencePiece enabled)

---

### 2.2 LLM Model Export (Week 5-6)

#### 2.2.1 Export Script

```python
# scripts/export_llm_for_compiler.py

"""
Export an LLM for use in the IREE compiler.
Uses shark-ai patterns for paged attention.
"""

import torch
from sharktank.models.llm import PagedLlmModelV1
from sharktank.models.llm.export import ServicePagedLlmModelV1
from sharktank.utils import load_llm
from iree.turbine.aot import FxProgramsBuilder, export

def export_llm_for_compiler(
    model_name: str,
    output_dir: str,
    max_seq_len: int = 8192,
    batch_size: int = 1,
    target_backend: str = "rocm",  # or "cuda", "llvm-cpu"
):
    """
    Export LLM with simplified interface for compiler use.
    
    Single batch size, simple KV-cache management.
    """
    # Load model
    model, config = load_llm(model_name)
    
    # Wrap for export
    service_model = ServicePagedLlmModelV1(
        model=model,
        config=ExportConfig(
            bs_prefill=[batch_size],
            bs_decode=[batch_size],
            device_block_count=max_seq_len // config.block_seq_stride,
            top_k=1,  # Always argmax for compiler use
        )
    )
    
    # Export using FxProgramsBuilder
    fxb = FxProgramsBuilder(service_model)
    
    # ... (export prefill and decode functions)
    
    # Compile
    output = export(fxb)
    output.compile(target_backends=[target_backend])
    output.save(f"{output_dir}/llm_model.vmfb")
    
    # Save config
    config.save(f"{output_dir}/config.json")
    
    # Copy tokenizer
    shutil.copy(f"{model_name}/tokenizer.model", 
                f"{output_dir}/tokenizer.model")
    
    print(f"Exported to {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="rocm")
    args = parser.parse_args()
    
    export_llm_for_compiler(args.model, args.output, target_backend=args.target)
```

#### 2.2.2 Deliverables - Week 5-6

- [x] Export script for compiler-friendly LLM (`scripts/export_llm_for_compiler.py`)
- [ ] Pre-exported model artifacts (VMFB + tokenizer) - requires shark-ai setup
- [x] Model configuration schema (JSON loading in `LLMModelConfig`)

---

### 2.3 KV-Cache Management Layer (Week 6-7)

#### 2.3.1 Simple KV-Cache Manager

```cpp
// KVCache/KVCacheManager.h

namespace mlir::iree_compiler::LLMAssist {

struct KVCacheConfig {
  int numLayers;
  int numHeads;
  int headDim;
  int maxSeqLen;
  int blockSeqStride;
  iree_hal_element_type_t dtype;
};

class KVCacheManager {
  KVCacheConfig config_;
  iree_hal_device_t *device_;
  iree_hal_allocator_t *allocator_;
  
  // Page table storage
  iree_hal_buffer_t *pageTable_;
  int numPages_;
  int currentSeqLen_;
  
  // Page allocation tracking (simple linear allocation)
  std::vector<int> allocatedPages_;
  
public:
  KVCacheManager(const KVCacheConfig &config, 
                 iree_hal_device_t *device);
  ~KVCacheManager();
  
  // Allocate pages for a new sequence
  LogicalResult allocateSequence(int seqLen);
  
  // Get page IDs for current sequence
  std::vector<int64_t> getPageIds() const;
  
  // Get the page table buffer view for passing to model
  iree_hal_buffer_view_t *getPageTableView();
  
  // Reset for new generation
  void reset();
  
  // Extend allocation for new tokens
  LogicalResult extendAllocation(int newTokens);
};

} // namespace
```

#### 2.3.2 Implementation

```cpp
// KVCache/KVCacheManager.cpp

KVCacheManager::KVCacheManager(const KVCacheConfig &config,
                               iree_hal_device_t *device)
    : config_(config), device_(device) {
  
  allocator_ = iree_hal_device_allocator(device);
  
  // Calculate page size
  // Shape: [num_pages, layers * 2 * heads * block_stride * head_dim]
  int pageElements = config.numLayers * 2 * config.numHeads * 
                     config.blockSeqStride * config.headDim;
  numPages_ = config.maxSeqLen / config.blockSeqStride;
  
  // Allocate page table
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  
  size_t byteSize = numPages_ * pageElements * 
                    iree_hal_element_dense_byte_count(config.dtype);
  
  iree_hal_buffer_allocate_result_t alloc;
  iree_hal_allocator_allocate_buffer(
      allocator_, params, byteSize, &alloc);
  pageTable_ = alloc.buffer;
  
  currentSeqLen_ = 0;
}

LogicalResult KVCacheManager::allocateSequence(int seqLen) {
  int pagesNeeded = (seqLen + config_.blockSeqStride - 1) / config_.blockSeqStride;
  
  if (pagesNeeded > numPages_) {
    return failure();
  }
  
  allocatedPages_.clear();
  for (int i = 0; i < pagesNeeded; ++i) {
    allocatedPages_.push_back(i);  // Simple linear allocation
  }
  
  currentSeqLen_ = seqLen;
  return success();
}

std::vector<int64_t> KVCacheManager::getPageIds() const {
  return std::vector<int64_t>(allocatedPages_.begin(), allocatedPages_.end());
}
```

#### 2.3.3 Deliverables - Week 6-7

- [x] KV-cache manager implementation (`KVCache/KVCacheManager.h/.cpp`)
- [x] Page allocation/deallocation (linear allocation strategy)
- [x] Integration with IREE runtime (buffer allocation, buffer views)

---

### 2.4 IREE LLM Backend (Week 7-8)

#### 2.4.1 Backend Implementation

```cpp
// Backend/IREEBackend.h

namespace mlir::iree_compiler::LLMAssist {

struct IREEBackendConfig {
  std::string vmfbPath;
  std::string tokenizerPath;
  std::string configPath;
  std::string deviceUri = "local-task";  // or "rocm://0", "cuda://0"
};

class IREEBackend : public LLMBackend {
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<KVCacheManager> kvCache_;
  
  iree_vm_instance_t *instance_ = nullptr;
  iree_hal_device_t *device_ = nullptr;
  iree_vm_context_t *context_ = nullptr;
  iree_vm_function_t prefillFn_;
  iree_vm_function_t decodeFn_;
  
  LLMModelConfig modelConfig_;
  
public:
  static std::unique_ptr<IREEBackend> create(const IREEBackendConfig &config);
  
  ~IREEBackend();
  
  bool isAvailable() const override;
  
  FailureOr<GenerationResult> generate(
      StringRef prompt,
      const GenerationConfig &config) override;
  
  StringRef getName() const override { return "IREE"; }
  
private:
  // Internal generation loop
  FailureOr<std::vector<int64_t>> generateTokens(
      ArrayRef<int64_t> promptTokens,
      int maxNewTokens);
  
  // Single forward pass
  FailureOr<int64_t> prefill(ArrayRef<int64_t> tokens);
  FailureOr<int64_t> decode(int64_t token, int position);
};

} // namespace
```

#### 2.4.2 Generation Loop

```cpp
// Backend/IREEBackend.cpp

FailureOr<GenerationResult> IREEBackend::generate(
    StringRef prompt,
    const GenerationConfig &config) {
  
  auto start = std::chrono::steady_clock::now();
  
  // Tokenize input
  std::vector<int64_t> inputTokens = tokenizer_->encode(prompt);
  inputTokens.insert(inputTokens.begin(), tokenizer_->bosId());
  
  // Reset KV cache
  kvCache_->reset();
  if (failed(kvCache_->allocateSequence(inputTokens.size() + config.maxTokens))) {
    return failure();
  }
  
  // Prefill
  auto firstToken = prefill(inputTokens);
  if (failed(firstToken))
    return failure();
  
  // Decode loop
  std::vector<int64_t> outputTokens = {*firstToken};
  int position = inputTokens.size();
  
  for (int i = 1; i < config.maxTokens; ++i) {
    auto nextToken = decode(outputTokens.back(), position++);
    if (failed(nextToken))
      return failure();
    
    outputTokens.push_back(*nextToken);
    
    // Check for EOS
    if (*nextToken == tokenizer_->eosId())
      break;
    
    // Check for stop sequences (``` for MLIR output)
    std::string decoded = tokenizer_->decode(outputTokens);
    if (decoded.find("```\n") != std::string::npos &&
        decoded.rfind("```") > decoded.find("```mlir"))
      break;
  }
  
  auto end = std::chrono::steady_clock::now();
  
  GenerationResult result;
  result.content = tokenizer_->decode(outputTokens);
  result.promptTokens = inputTokens.size();
  result.completionTokens = outputTokens.size();
  result.latencyMs = std::chrono::duration<float, std::milli>(end - start).count();
  
  return result;
}
```

#### 2.4.3 Deliverables - Week 7-8

- [x] IREE backend implementation (`Backend/IREEBackend.h/.cpp`)
- [x] IRPA weight loading via `io_parameters` module
- [x] Prefill and decode function invocation
- [x] Token generation loop with stopping criteria
- [x] Model export script (`tools/scripts/export_llm_for_compiler.py`)
- [x] Test VMFB compiled (open_llama_3b for gfx950)
- [x] End-to-end Python integration test (`tools/scripts/test_iree_backend_integration.py`)
- [x] KVCacheManager C++ unit tests
- [x] IREEBackend C++ unit tests
- [x] Python test with multiple prompts showing coherent generation

---

### 2.5 Backend Switching & Testing (Week 8-9)

#### 2.5.1 Backend Factory

```cpp
// Backend/BackendFactory.h

namespace mlir::iree_compiler::LLMAssist {

enum class BackendType {
  Ollama,
  IREE,
  Auto  // Try IREE first, fall back to Ollama
};

struct BackendFactoryConfig {
  BackendType type = BackendType::Auto;
  
  // Ollama config
  std::string ollamaEndpoint = "http://localhost:11434";
  std::string ollamaModel = "qwen2.5-coder:7b";
  
  // IREE config  
  std::string ireeVmfbPath;
  std::string ireeTokenizerPath;
  std::string ireeDeviceUri = "local-task";
};

std::unique_ptr<LLMBackend> createBackend(const BackendFactoryConfig &config);

} // namespace
```

#### 2.5.2 Pass Update

```tablegen
// Updated Passes.td

def LLMAssistedTransformPass : Pass<"iree-llm-assisted-transform", "func::FuncOp"> {
  // ... existing options ...
  
  let options = [
    Option<"backendType", "backend",
           "std::string", /*default=*/"\"auto\"",
           "Backend type: 'ollama', 'iree', or 'auto'">,
    
    // Ollama options
    Option<"ollamaEndpoint", "ollama-endpoint", ...>,
    Option<"ollamaModel", "ollama-model", ...>,
    
    // IREE options
    Option<"ireeVmfbPath", "iree-vmfb", ...>,
    Option<"ireeTokenizerPath", "iree-tokenizer", ...>,
    Option<"ireeDevice", "iree-device", ...>,
  ];
}
```

#### 2.5.3 Benchmark Comparison

```cpp
// test/benchmark_backends.cpp

void benchmarkBackends() {
  std::vector<std::string> testPrompts = loadTestPrompts();
  
  auto ollamaBackend = createOllamaBackend("http://localhost:11434");
  auto ireeBackend = IREEBackend::create({
      .vmfbPath = "llm_model.vmfb",
      .tokenizerPath = "tokenizer.model",
  });
  
  for (const auto &prompt : testPrompts) {
    // Ollama timing
    auto t0 = now();
    auto ollamaResult = ollamaBackend->generate(prompt, config);
    auto ollamaTime = elapsed(t0);
    
    // IREE timing  
    auto t1 = now();
    auto ireeResult = ireeBackend->generate(prompt, config);
    auto ireeTime = elapsed(t1);
    
    std::cout << "Prompt length: " << prompt.size() 
              << " Ollama: " << ollamaTime << "ms"
              << " IREE: " << ireeTime << "ms"
              << " Speedup: " << ollamaTime / ireeTime << "x\n";
  }
}
```

#### 2.5.4 Deliverables - Week 8-9

- [ ] Backend factory with auto-selection
- [ ] Updated pass with backend options
- [ ] Benchmark suite comparing backends
- [ ] Documentation for using either backend

### Phase 2 Exit Criteria

- [ ] IREE backend produces identical results to Ollama
- [ ] IREE backend is faster for typical prompts (>2x expected)
- [ ] Works on at least CPU and one GPU backend
- [ ] Memory usage is reasonable (<16GB for 7B model)
- [ ] All Phase 1 tests pass with IREE backend

---

## Phase 3: IREE-Compiled Tokenizer (3-4 weeks)

**Goal**: Replace SentencePiece with an IREE-compiled tokenizer for full self-containment.

### 3.1 Tokenizer Model Analysis (Week 10)

#### 3.1.1 Extract Tokenizer Data

```python
# scripts/extract_tokenizer_tensors.py

"""
Extract BPE tokenizer as tensor representations for IREE compilation.
"""

import json
import torch
from sentencepiece import SentencePieceProcessor

def extract_bpe_tensors(sp_model_path: str, output_dir: str):
    sp = SentencePieceProcessor(model_file=sp_model_path)
    
    # 1. Vocabulary table: token_id -> bytes
    vocab_size = sp.GetPieceSize()
    max_token_len = max(len(sp.IdToPiece(i).encode('utf-8')) 
                        for i in range(vocab_size))
    
    vocab_table = torch.zeros(vocab_size, max_token_len, dtype=torch.uint8)
    vocab_lens = torch.zeros(vocab_size, dtype=torch.int32)
    
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        piece_bytes = piece.encode('utf-8')
        vocab_lens[i] = len(piece_bytes)
        for j, b in enumerate(piece_bytes):
            vocab_table[i, j] = b
    
    # 2. Byte-to-initial-token mapping
    byte_to_token = torch.zeros(256, dtype=torch.int32)
    for byte_val in range(256):
        byte_char = bytes([byte_val]).decode('latin-1', errors='replace')
        token_id = sp.PieceToId(byte_char)
        byte_to_token[byte_val] = token_id if token_id != sp.unk_id() else sp.unk_id()
    
    # Save tensors
    torch.save({
        'vocab_table': vocab_table,
        'vocab_lens': vocab_lens,
        'byte_to_token': byte_to_token,
        'bos_id': sp.bos_id(),
        'eos_id': sp.eos_id(),
        'pad_id': sp.pad_id(),
        'unk_id': sp.unk_id(),
    }, f"{output_dir}/tokenizer_tensors.pt")
    
    print(f"Extracted tokenizer with vocab_size={vocab_size}")
```

#### 3.1.2 Deliverables - Week 10

- [ ] Tokenizer tensor extraction script
- [ ] Understanding of target tokenizer format
- [ ] Test data for validation

---

### 3.2 Tokenizer Model Implementation (Week 10-11)

#### 3.2.1 PyTorch Tokenizer for Export

```python
# scripts/tokenizer_model.py

import torch
import torch.nn as nn

class IREETokenizerEncode(nn.Module):
    """
    BPE encoding as tensor operations, exportable to IREE.
    
    Simplified version: uses vocabulary lookup with longest match.
    """
    
    def __init__(self, tokenizer_data: dict, max_seq_len: int = 8192):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Register buffers (will be embedded in VMFB)
        self.register_buffer('vocab_table', tokenizer_data['vocab_table'])
        self.register_buffer('vocab_lens', tokenizer_data['vocab_lens'])
        self.register_buffer('byte_to_token', tokenizer_data['byte_to_token'])
        
        self.bos_id = tokenizer_data['bos_id']
        self.eos_id = tokenizer_data['eos_id']
    
    def forward(
        self,
        byte_input: torch.Tensor,   # [max_len] uint8
        input_length: torch.Tensor,  # [] int32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode bytes to token IDs.
        
        Uses greedy longest-match against vocabulary.
        """
        # Simple byte-level tokenization (baseline)
        # Each byte becomes a token via lookup
        tokens = self.byte_to_token[byte_input.long()]
        
        # Add BOS
        result = torch.zeros(self.max_seq_len, dtype=torch.int64)
        result[0] = self.bos_id
        result[1:input_length + 1] = tokens[:input_length]
        
        output_length = input_length + 1
        
        return result, output_length


class IREETokenizerDecode(nn.Module):
    """Decode token IDs back to bytes."""
    
    def __init__(self, tokenizer_data: dict, max_seq_len: int = 8192):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.register_buffer('vocab_table', tokenizer_data['vocab_table'])
        self.register_buffer('vocab_lens', tokenizer_data['vocab_lens'])
    
    def forward(
        self,
        token_ids: torch.Tensor,    # [seq_len] int64
        token_length: torch.Tensor,  # [] int32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode tokens to bytes."""
        
        max_bytes = self.max_seq_len * self.vocab_table.shape[1]
        output = torch.zeros(max_bytes, dtype=torch.uint8)
        
        output_pos = 0
        for i in range(token_length):
            token_id = token_ids[i]
            token_len = self.vocab_lens[token_id]
            token_bytes = self.vocab_table[token_id, :token_len]
            output[output_pos:output_pos + token_len] = token_bytes
            output_pos += token_len
        
        return output, torch.tensor(output_pos, dtype=torch.int32)
```

#### 3.2.2 Export to IREE

```python
# scripts/export_tokenizer.py

import iree.turbine.aot as aot

def export_tokenizer(tokenizer_data_path: str, output_path: str):
    data = torch.load(tokenizer_data_path)
    
    encoder = IREETokenizerEncode(data)
    decoder = IREETokenizerDecode(data)
    
    class ExportedTokenizer(aot.CompiledModule):
        encoder_params = aot.export_parameters(encoder)
        decoder_params = aot.export_parameters(decoder)
        
        def encode(
            self,
            byte_input=aot.AbstractTensor(8192, dtype=torch.uint8),
            input_length=aot.AbstractTensor(None, dtype=torch.int32),
        ):
            return aot.jittable(encoder.forward)(byte_input, input_length)
        
        def decode(
            self,
            token_ids=aot.AbstractTensor(None, dtype=torch.int64),
            token_length=aot.AbstractTensor(None, dtype=torch.int32),
        ):
            return aot.jittable(decoder.forward)(token_ids, token_length)
    
    exported = aot.export(ExportedTokenizer)
    compiled = exported.compile(target_backends=["llvm-cpu"])
    compiled.save(output_path)
    print(f"Exported tokenizer to {output_path}")
```

#### 3.2.3 Deliverables - Week 10-11

- [ ] PyTorch tokenizer model (encode/decode)
- [ ] Export script for VMFB
- [ ] Basic accuracy testing

---

### 3.3 C++ IREE Tokenizer Integration (Week 11-12)

#### 3.3.1 IREE Tokenizer Implementation

```cpp
// Tokenizer/IREETokenizer.h

namespace mlir::iree_compiler::LLMAssist {

class IREETokenizer : public Tokenizer {
  iree_vm_instance_t *instance_ = nullptr;
  iree_hal_device_t *device_ = nullptr;
  iree_vm_context_t *context_ = nullptr;
  
  iree_vm_function_t encodeFn_;
  iree_vm_function_t decodeFn_;
  
  int64_t bosId_, eosId_, padId_;
  size_t vocabSize_;
  
public:
  static std::unique_ptr<IREETokenizer> create(
      StringRef vmfbPath,
      StringRef configPath,
      StringRef deviceUri = "local-task");
  
  ~IREETokenizer();
  
  std::vector<int64_t> encode(StringRef text) override;
  std::string decode(ArrayRef<int64_t> ids) override;
  
  size_t vocabSize() const override { return vocabSize_; }
  int64_t bosId() const override { return bosId_; }
  int64_t eosId() const override { return eosId_; }
  int64_t padId() const override { return padId_; }
};

} // namespace
```

#### 3.3.2 Deliverables - Week 11-12

- [ ] IREE tokenizer C++ wrapper
- [ ] Integration with existing Tokenizer interface
- [ ] Accuracy validation vs SentencePiece

---

### 3.4 Validation & Optimization (Week 12-13)

#### 3.4.1 Accuracy Tests

```cpp
// test/iree_tokenizer_test.cpp

TEST(IREETokenizerTest, MatchesSentencePiece) {
  auto spTokenizer = createSentencePieceTokenizer("tokenizer.model");
  auto ireeTokenizer = IREETokenizer::create("tokenizer.vmfb", "config.json");
  
  std::vector<std::string> testStrings = {
    "Hello, world!",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "tensor<4x4xf32>",
    "// This is a comment with special chars: @#$%",
    // ... many more test cases
  };
  
  for (const auto &s : testStrings) {
    auto spTokens = spTokenizer->encode(s);
    auto ireeTokens = ireeTokenizer->encode(s);
    
    EXPECT_EQ(spTokens, ireeTokens) << "Mismatch for: " << s;
    
    auto spDecoded = spTokenizer->decode(spTokens);
    auto ireeDecoded = ireeTokenizer->decode(ireeTokens);
    
    EXPECT_EQ(spDecoded, ireeDecoded);
  }
}
```

#### 3.4.2 Performance Comparison

```cpp
// test/tokenizer_benchmark.cpp

void benchmarkTokenizers() {
  auto spTokenizer = createSentencePieceTokenizer("tokenizer.model");
  auto ireeTokenizer = IREETokenizer::create("tokenizer.vmfb", "config.json");
  
  std::vector<std::string> prompts = loadTestPrompts();
  
  // Warm up
  for (int i = 0; i < 10; ++i) {
    spTokenizer->encode(prompts[0]);
    ireeTokenizer->encode(prompts[0]);
  }
  
  // Benchmark
  double spTotal = 0, ireeTotal = 0;
  for (const auto &prompt : prompts) {
    auto t0 = now();
    spTokenizer->encode(prompt);
    spTotal += elapsed(t0);
    
    auto t1 = now();
    ireeTokenizer->encode(prompt);
    ireeTotal += elapsed(t1);
  }
  
  std::cout << "SentencePiece: " << spTotal << "ms total\n";
  std::cout << "IREE: " << ireeTotal << "ms total\n";
  std::cout << "Ratio: " << spTotal / ireeTotal << "x\n";
}
```

#### 3.4.3 Deliverables - Week 12-13

- [ ] Comprehensive accuracy test suite
- [ ] Performance benchmarks
- [ ] Bug fixes and optimizations
- [ ] Documentation

### Phase 3 Exit Criteria

- [ ] IREE tokenizer matches SentencePiece output exactly
- [ ] Performance is within 2x of SentencePiece (acceptable for prototype)
- [ ] Works on CPU backend
- [ ] Fully self-contained (no external tokenizer dependency needed)

---

## Phase 4: Advanced Experimentation (Ongoing)

**Goal**: Explore different models, fine-tuning, and optimizations.

### 4.1 Model Experimentation

#### 4.1.1 Model Variants

| Model | Size | Best For |
|-------|------|----------|
| Qwen2.5-Coder-1.5B | 1.5B | Fast iteration, simple transforms |
| Qwen2.5-Coder-7B | 7B | Balanced quality/speed |
| Qwen2.5-Coder-32B | 32B | Complex transformations |
| DeepSeek-Coder-V2-Lite | 16B | Strong code understanding |
| CodeLlama-34B | 34B | Specialized for code |

#### 4.1.2 Model Switching Infrastructure

```cpp
// Config file approach
struct LLMAssistConfig {
  std::string modelName;
  std::string vmfbPath;
  std::string tokenizerPath;
  
  // Model-specific parameters
  int maxSeqLen;
  int numLayers;
  int numHeads;
  // ...
  
  static LLMAssistConfig loadFromJson(StringRef path);
};
```

### 4.2 KV-Cache Algorithm Experiments

#### 4.2.1 Algorithm Variants

**Linear Allocation (Current)**
```
Simple: allocate pages sequentially
Pros: Simple, predictable
Cons: No sharing, no prefix caching
```

**Prefix Caching (Trie-based)**
```
Share pages for common prefixes (e.g., system prompts)
Pros: Memory efficient for similar prompts
Cons: More complex, overhead for unique prompts
```

**Sliding Window**
```
Keep only recent tokens in cache
Pros: Bounded memory
Cons: Loses long-range context
```

#### 4.2.2 Pluggable Cache Interface

```cpp
// KVCache/CacheStrategy.h

class CacheStrategy {
public:
  virtual ~CacheStrategy() = default;
  
  virtual LogicalResult allocate(int seqLen) = 0;
  virtual void extend(int newTokens) = 0;
  virtual void reset() = 0;
  
  virtual std::vector<int64_t> getPageIds() const = 0;
  virtual iree_hal_buffer_view_t *getPageTable() = 0;
};

std::unique_ptr<CacheStrategy> createLinearCache(const Config &);
std::unique_ptr<CacheStrategy> createTrieCache(const Config &);
std::unique_ptr<CacheStrategy> createSlidingWindowCache(const Config &);
```

### 4.3 Fine-Tuning for MLIR

#### 4.3.1 Data Collection

```python
# Collect training data from IREE test suite

def collect_mlir_pairs():
    """
    Collect (input_ir, transformed_ir) pairs from IREE tests.
    """
    pairs = []
    
    # Parse lit tests
    for test_file in glob("**/*test*.mlir"):
        input_ir, expected_ir = parse_run_and_check(test_file)
        if input_ir and expected_ir:
            pairs.append({
                "instruction": extract_pass_description(test_file),
                "input": input_ir,
                "output": expected_ir,
            })
    
    return pairs
```

#### 4.3.2 Fine-Tuning Script

```python
# Fine-tune using LoRA

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

def finetune_for_mlir(
    base_model: str,
    train_data: list,
    output_dir: str,
):
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Training loop
    # ...
    
    model.save_pretrained(output_dir)
```

### 4.4 Evaluation Framework

#### 4.4.1 Metrics

```cpp
struct TransformationMetrics {
  int totalAttempts;
  int parseSuccesses;
  int validationSuccesses;
  int semanticCorrect;  // Verified by execution
  
  float avgLatencyMs;
  float avgTokensGenerated;
  
  void report() const {
    float parseRate = 100.0f * parseSuccesses / totalAttempts;
    float validRate = 100.0f * validationSuccesses / totalAttempts;
    float correctRate = 100.0f * semanticCorrect / totalAttempts;
    
    std::cout << "Parse success: " << parseRate << "%\n";
    std::cout << "Validation success: " << validRate << "%\n";
    std::cout << "Semantic correctness: " << correctRate << "%\n";
    std::cout << "Avg latency: " << avgLatencyMs << "ms\n";
  }
};
```

#### 4.4.2 Semantic Correctness Testing

```cpp
// Compare execution results

bool verifySemanticEquivalence(
    func::FuncOp original,
    func::FuncOp transformed,
    ArrayRef<TypedValue> testInputs) {
  
  // Compile and run both versions
  auto originalResult = compileAndRun(original, testInputs);
  auto transformedResult = compileAndRun(transformed, testInputs);
  
  // Compare outputs
  return resultsMatch(originalResult, transformedResult);
}
```

---

## Summary Timeline

```
Week 1-2:   Phase 1.1-1.2  Project setup, Ollama backend
Week 2-3:   Phase 1.3-1.4  IR serialization, prompt engineering
Week 3-4:   Phase 1.5-1.6  LLMAssistedPass, testing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 5:     Phase 2.1      SentencePiece integration
Week 5-6:   Phase 2.2      LLM model export
Week 6-7:   Phase 2.3      KV-cache management
Week 7-8:   Phase 2.4      IREE backend implementation
Week 8-9:   Phase 2.5      Backend switching, testing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 10:    Phase 3.1      Tokenizer analysis
Week 10-11: Phase 3.2      Tokenizer model
Week 11-12: Phase 3.3      IREE tokenizer integration
Week 12-13: Phase 3.4      Validation, optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 14+:   Phase 4        Experimentation (ongoing)
```

**Total estimated time**: ~13 weeks for core implementation, then ongoing experimentation.

---

## Appendix A: Quick Start Guide

### Running with Ollama (Phase 1)

```bash
# 1. Start Ollama
ollama serve

# 2. Pull model
ollama pull qwen2.5-coder:7b

# 3. Run pass
iree-opt input.mlir \
  --iree-llm-assisted-transform="task='Optimize memory access patterns'" \
  --iree-llm-assisted-transform-ollama-endpoint="http://localhost:11434" \
  --iree-llm-assisted-transform-model="qwen2.5-coder:7b"
```

### Running with IREE Native (Phase 2+)

```bash
# 1. Export model (one-time)
python scripts/export_llm_for_compiler.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output ./llm_artifacts \
  --target llvm-cpu

# 2. Run pass
iree-opt input.mlir \
  --iree-llm-assisted-transform="task='Vectorize this loop'" \
  --iree-llm-assisted-transform-backend="iree" \
  --iree-llm-assisted-transform-iree-vmfb="./llm_artifacts/llm_model.vmfb" \
  --iree-llm-assisted-transform-iree-tokenizer="./llm_artifacts/tokenizer.model"
```

---

## Appendix B: Related Work

- **shark-ai/shortfin**: IREE-based LLM serving infrastructure
- **llama.cpp**: Reference for efficient LLM inference
- **vLLM**: Paged attention and prefix caching
- **MLIR Transform Dialect**: Structured IR transformation framework

