# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# LLMAssist module - experimental LLM-assisted compiler passes
# This module is only built when IREE_ENABLE_LLM_ASSIST is ON

if(NOT IREE_ENABLE_LLM_ASSIST)
  return()
endif()

message(STATUS "Building LLMAssist module (experimental)")

iree_add_all_subdirs()

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    PassHeaders
  HDRS
    "Passes.h"
    "Passes.h.inc"
  DEPS
    ::PassesIncGen
    MLIRPass
    MLIRTransforms
  PUBLIC
)

iree_cc_library(
  NAME
    Backend
  HDRS
    "Backend/LLMBackend.h"
    "Backend/OllamaBackend.h"
  SRCS
    "Backend/LLMBackend.cpp"
    "Backend/OllamaBackend.cpp"
  DEPS
    LLVMSupport
  PUBLIC
)

iree_cc_library(
  NAME
    IR
  HDRS
    "IR/IRSerializer.h"
    "IR/IRParser.h"
  SRCS
    "IR/IRSerializer.cpp"
    "IR/IRParser.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRFuncDialect
    MLIRParser
  PUBLIC
)

iree_cc_library(
  NAME
    Prompt
  HDRS
    "Prompt/PromptBuilder.h"
  SRCS
    "Prompt/PromptBuilder.cpp"
  DEPS
    LLVMSupport
  PUBLIC
)

iree_cc_library(
  NAME
    LLMAssist
  HDRS
    "Passes.h"
  SRCS
    "Passes.cpp"
    "LLMAssistedTransformPass.cpp"
  DEPS
    ::Backend
    ::IR
    ::PassHeaders
    ::PassesIncGen
    ::Prompt
    LLVMSupport
    MLIRIR
    MLIRFuncDialect
    MLIRPass
    MLIRParser
    MLIRSupport
    iree::compiler::Dialect::Util::IR
  DEFINES
    "IREE_HAVE_LLM_ASSIST"
  PUBLIC
)

iree_tablegen_doc(
  NAME
    LLMAssistPassesDocGen
  CATEGORY
    "Passes"
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-doc LLMAssistPasses.md
)

