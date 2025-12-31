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

#-------------------------------------------------------------------------------
# Optional SentencePiece integration for native IREE backend
#-------------------------------------------------------------------------------
option(IREE_LLM_ASSIST_ENABLE_SENTENCEPIECE "Enable SentencePiece tokenizer" OFF)

if(IREE_LLM_ASSIST_ENABLE_SENTENCEPIECE)
  find_package(SentencePiece QUIET)
  if(NOT SentencePiece_FOUND)
    message(STATUS "SentencePiece not found, fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
      sentencepiece
      GIT_REPOSITORY https://github.com/google/sentencepiece.git
      GIT_TAG v0.2.0
      GIT_SHALLOW TRUE
    )
    # Disable building tools/tests for sentencepiece
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_NO_TCMALLOC ON CACHE BOOL "" FORCE)
    set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(sentencepiece)
    # Set include directory for fetched sentencepiece
    set(SENTENCEPIECE_INCLUDE_DIR "${sentencepiece_SOURCE_DIR}/src")
  else()
    # Use system SentencePiece
    get_target_property(SENTENCEPIECE_INCLUDE_DIR sentencepiece-static INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  add_compile_definitions(IREE_LLM_ASSIST_HAVE_SENTENCEPIECE)
  message(STATUS "LLMAssist: SentencePiece tokenizer ENABLED")
  message(STATUS "LLMAssist: SentencePiece include: ${SENTENCEPIECE_INCLUDE_DIR}")
endif()

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

#-------------------------------------------------------------------------------
# Tokenizer library
#-------------------------------------------------------------------------------
set(TOKENIZER_DEPS LLVMSupport)
set(TOKENIZER_SRCS 
  "Tokenizer/Tokenizer.cpp"
  "Tokenizer/BPETokenizer.cpp"
)
if(IREE_LLM_ASSIST_ENABLE_SENTENCEPIECE)
  list(APPEND TOKENIZER_SRCS "Tokenizer/SentencePieceTokenizer.cpp")
  list(APPEND TOKENIZER_DEPS sentencepiece-static)
endif()

iree_cc_library(
  NAME
    Tokenizer
  HDRS
    "Tokenizer/Tokenizer.h"
    "Tokenizer/BPETokenizer.h"
  SRCS
    ${TOKENIZER_SRCS}
  DEPS
    ${TOKENIZER_DEPS}
  PUBLIC
)

# Add include directory for SentencePiece if enabled
if(IREE_LLM_ASSIST_ENABLE_SENTENCEPIECE AND SENTENCEPIECE_INCLUDE_DIR)
  target_include_directories(iree_compiler_LLMAssist_Tokenizer.objects
    PRIVATE "${SENTENCEPIECE_INCLUDE_DIR}")
endif()

#-------------------------------------------------------------------------------
# KV-Cache management
#-------------------------------------------------------------------------------
iree_cc_library(
  NAME
    KVCache
  HDRS
    "KVCache/KVCacheManager.h"
  SRCS
    "KVCache/KVCacheManager.cpp"
  DEPS
    LLVMSupport
    iree::base
    iree::hal
  PUBLIC
)

#-------------------------------------------------------------------------------
# IREE Native LLM Backend
#-------------------------------------------------------------------------------
set(IREE_BACKEND_HAL_DEPS
  iree::hal::drivers::local_task::registration
)

# Add HIP driver if available
if(IREE_HAL_DRIVER_HIP)
  list(APPEND IREE_BACKEND_HAL_DEPS iree::hal::drivers::hip::registration)
  set(IREE_BACKEND_DEFINES "IREE_HAVE_HAL_HIP_DRIVER")
endif()

iree_cc_library(
  NAME
    IREEBackend
  HDRS
    "Backend/IREEBackend.h"
  SRCS
    "Backend/IREEBackend.cpp"
  DEPS
    ::Backend
    ::Tokenizer
    LLVMSupport
    iree::base
    iree::hal
    ${IREE_BACKEND_HAL_DEPS}
    iree::io::file_handle
    iree::io::formats::parser_registry
    iree::io::parameter_index
    iree::io::parameter_index_provider
    iree::modules::hal
    iree::modules::io::parameters
    iree::vm
    iree::vm::bytecode::module
  DEFINES
    ${IREE_BACKEND_DEFINES}
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
    ::IREEBackend
    ::KVCache
    ::PassHeaders
    ::PassesIncGen
    ::Prompt
    ::Tokenizer
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

