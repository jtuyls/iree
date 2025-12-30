// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/IR/IRParser.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::iree_compiler::LLMAssist;
using namespace testing;

class IRParserTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
  }

  MLIRContext ctx;
};

TEST_F(IRParserTest, ParseValidModule) {
  StringRef input = R"mlir(
    module {
      func.func @add(%arg0: i32, %arg1: i32) -> i32 {
        %result = arith.addi %arg0, %arg1 : i32
        return %result : i32
      }
    }
  )mlir";

  auto result = IRParser::parseModule(input, &ctx);

  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_FALSE(result.hadErrors);
  EXPECT_TRUE(result.module);
}

TEST_F(IRParserTest, ParseInvalidModule) {
  StringRef input = R"mlir(
    module {
      func.func @broken(
        // Missing closing paren and body
    }
  )mlir";

  auto result = IRParser::parseModule(input, &ctx);

  EXPECT_FALSE(static_cast<bool>(result));
  EXPECT_TRUE(result.hadErrors);
}

TEST_F(IRParserTest, ExtractMLIRFromMarkdownBlock) {
  StringRef response = R"(
Here's the optimized code:

```mlir
module {
  func.func @optimized() -> i32 {
    %c3 = arith.constant 3 : i32
    return %c3 : i32
  }
}
```

This folds the constants at compile time.
)";

  std::string extracted = IRParser::extractMLIRFromResponse(response);

  EXPECT_THAT(extracted, HasSubstr("module"));
  EXPECT_THAT(extracted, HasSubstr("@optimized"));
  EXPECT_THAT(extracted, HasSubstr("arith.constant 3"));
}

TEST_F(IRParserTest, ExtractMLIRFromPlainBlock) {
  StringRef response = R"(
Here's the optimized code:

```
module {
  func.func @plain() -> i32 {
    %c5 = arith.constant 5 : i32
    return %c5 : i32
  }
}
```
)";

  std::string extracted = IRParser::extractMLIRFromResponse(response);

  EXPECT_THAT(extracted, HasSubstr("module"));
  EXPECT_THAT(extracted, HasSubstr("@plain"));
}

TEST_F(IRParserTest, ExtractMLIRWithNoCodeBlock) {
  // When there's no code block, it should return the input as-is
  // or try to find MLIR-like content
  StringRef response = R"(
module {
  func.func @raw() -> i32 {
    %c1 = arith.constant 1 : i32
    return %c1 : i32
  }
}
)";

  std::string extracted = IRParser::extractMLIRFromResponse(response);

  // Should still find the module
  EXPECT_THAT(extracted, HasSubstr("module"));
}

TEST_F(IRParserTest, ValidateCompatibleModules) {
  StringRef original = R"mlir(
    module {
      func.func @compute(%arg0: i32) -> i32 {
        %c2 = arith.constant 2 : i32
        %result = arith.muli %arg0, %c2 : i32
        return %result : i32
      }
    }
  )mlir";

  StringRef transformed = R"mlir(
    module {
      func.func @compute(%arg0: i32) -> i32 {
        %c1 = arith.constant 1 : i32
        %result = arith.shli %arg0, %c1 : i32
        return %result : i32
      }
    }
  )mlir";

  auto origModule = parseSourceString<ModuleOp>(original, &ctx);
  auto transModule = parseSourceString<ModuleOp>(transformed, &ctx);
  ASSERT_TRUE(origModule);
  ASSERT_TRUE(transModule);

  // Same function name and signature = compatible
  bool compatible =
      IRParser::validateCompatibility(*origModule, *transModule);
  EXPECT_TRUE(compatible);
}

TEST_F(IRParserTest, ValidateIncompatibleModulesDifferentFunctionName) {
  StringRef original = R"mlir(
    module {
      func.func @original(%arg0: i32) -> i32 {
        return %arg0 : i32
      }
    }
  )mlir";

  StringRef transformed = R"mlir(
    module {
      func.func @different_name(%arg0: i32) -> i32 {
        return %arg0 : i32
      }
    }
  )mlir";

  auto origModule = parseSourceString<ModuleOp>(original, &ctx);
  auto transModule = parseSourceString<ModuleOp>(transformed, &ctx);
  ASSERT_TRUE(origModule);
  ASSERT_TRUE(transModule);

  bool compatible =
      IRParser::validateCompatibility(*origModule, *transModule);
  EXPECT_FALSE(compatible);
}

TEST_F(IRParserTest, ValidateIncompatibleModulesDifferentSignature) {
  StringRef original = R"mlir(
    module {
      func.func @compute(%arg0: i32) -> i32 {
        return %arg0 : i32
      }
    }
  )mlir";

  StringRef transformed = R"mlir(
    module {
      func.func @compute(%arg0: i64) -> i64 {
        return %arg0 : i64
      }
    }
  )mlir";

  auto origModule = parseSourceString<ModuleOp>(original, &ctx);
  auto transModule = parseSourceString<ModuleOp>(transformed, &ctx);
  ASSERT_TRUE(origModule);
  ASSERT_TRUE(transModule);

  bool compatible =
      IRParser::validateCompatibility(*origModule, *transModule);
  EXPECT_FALSE(compatible);
}

TEST_F(IRParserTest, ParseAndValidateRoundTrip) {
  StringRef input = R"mlir(
    module {
      func.func @roundtrip(%arg0: f32, %arg1: f32) -> f32 {
        %result = arith.addf %arg0, %arg1 : f32
        return %result : f32
      }
    }
  )mlir";

  auto result = IRParser::parseModule(input, &ctx);
  ASSERT_TRUE(static_cast<bool>(result));

  // Parse the same input again
  auto result2 = IRParser::parseModule(input, &ctx);
  ASSERT_TRUE(static_cast<bool>(result2));

  // They should be compatible with each other
  bool compatible =
      IRParser::validateCompatibility(*result.module, *result2.module);
  EXPECT_TRUE(compatible);
}

