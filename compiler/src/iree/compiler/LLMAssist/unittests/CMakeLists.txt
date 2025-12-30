# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# LLMAssist unit tests - only built when IREE_ENABLE_LLM_ASSIST is ON

if(NOT IREE_ENABLE_LLM_ASSIST)
  return()
endif()

iree_cc_test(
  NAME
    IRSerializerTest
  SRCS
    "IRSerializerTest.cpp"
  DEPS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    gmock
    gtest
    iree::compiler::LLMAssist::IR
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    IRParserTest
  SRCS
    "IRParserTest.cpp"
  DEPS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    gmock
    gtest
    iree::compiler::LLMAssist::IR
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    PromptBuilderTest
  SRCS
    "PromptBuilderTest.cpp"
  DEPS
    gmock
    gtest
    iree::compiler::LLMAssist::Prompt
    iree::testing::gtest_main
)

