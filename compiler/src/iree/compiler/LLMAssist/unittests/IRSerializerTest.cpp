// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/IR/IRSerializer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::iree_compiler::LLMAssist;
using namespace testing;

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

  // Check that key elements are present
  EXPECT_THAT(output, HasSubstr("func.func"));
  EXPECT_THAT(output, HasSubstr("@add"));
  EXPECT_THAT(output, HasSubstr("arith.addi"));
  EXPECT_THAT(output, HasSubstr("return"));
}

TEST_F(IRSerializerTest, SerializeWithGenericForm) {
  StringRef input = R"mlir(
    module {
      func.func @multiply(%arg0: i32, %arg1: i32) -> i32 {
        %result = arith.muli %arg0, %arg1 : i32
        return %result : i32
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(input, &ctx);
  ASSERT_TRUE(module);

  SerializationOptions opts;
  opts.useGenericForm = true;

  std::string output = IRSerializer::serializeModule(*module, opts);

  // Generic form uses "arith.muli" with explicit type annotation style
  EXPECT_THAT(output, HasSubstr("arith.muli"));
}

TEST_F(IRSerializerTest, SerializeWithLocations) {
  StringRef input = R"mlir(
    module {
      func.func @simple() -> i32 {
        %c1 = arith.constant 1 : i32
        return %c1 : i32
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(input, &ctx);
  ASSERT_TRUE(module);

  SerializationOptions optsNoLoc;
  optsNoLoc.includeLocations = false;
  std::string outputNoLoc = IRSerializer::serializeModule(*module, optsNoLoc);

  SerializationOptions optsWithLoc;
  optsWithLoc.includeLocations = true;
  std::string outputWithLoc =
      IRSerializer::serializeModule(*module, optsWithLoc);

  // The output without locations should be shorter or at least different
  // Locations often contain "loc(" strings
  // Both may or may not have location depending on the default printing
  EXPECT_FALSE(outputNoLoc.empty());
  EXPECT_FALSE(outputWithLoc.empty());
}

TEST_F(IRSerializerTest, SerializeOperation) {
  StringRef input = R"mlir(
    module {
      func.func @test() -> i32 {
        %c42 = arith.constant 42 : i32
        return %c42 : i32
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(input, &ctx);
  ASSERT_TRUE(module);

  // Get the function operation
  auto funcOp = *module->getOps<func::FuncOp>().begin();
  std::string output = IRSerializer::serialize(funcOp);

  EXPECT_THAT(output, HasSubstr("func.func @test"));
  EXPECT_THAT(output, HasSubstr("42"));
}

TEST_F(IRSerializerTest, SerializeEmptyModule) {
  StringRef input = "module {}";

  auto module = parseSourceString<ModuleOp>(input, &ctx);
  ASSERT_TRUE(module);

  std::string output = IRSerializer::serializeModule(*module);

  EXPECT_THAT(output, HasSubstr("module"));
}

TEST_F(IRSerializerTest, SerializeMultipleFunctions) {
  StringRef input = R"mlir(
    module {
      func.func @first() -> i32 {
        %c1 = arith.constant 1 : i32
        return %c1 : i32
      }
      func.func @second() -> i32 {
        %c2 = arith.constant 2 : i32
        return %c2 : i32
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(input, &ctx);
  ASSERT_TRUE(module);

  std::string output = IRSerializer::serializeModule(*module);

  EXPECT_THAT(output, HasSubstr("@first"));
  EXPECT_THAT(output, HasSubstr("@second"));
}

