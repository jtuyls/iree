# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# LLMAssist tests - only built when IREE_ENABLE_LLM_ASSIST is ON

if(NOT IREE_ENABLE_LLM_ASSIST)
  return()
endif()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "llm_assisted_transform.mlir"
    "llm_prompt_dryrun.mlir"
    "llm_transform_e2e.mlir"
  TOOLS
    FileCheck
    iree-opt
  TIMEOUT
    120
)

