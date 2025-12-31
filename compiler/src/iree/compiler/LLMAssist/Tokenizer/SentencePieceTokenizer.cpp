// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Tokenizer/Tokenizer.h"

#ifdef IREE_LLM_ASSIST_HAVE_SENTENCEPIECE

#include "sentencepiece_processor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::LLMAssist {

/// SentencePiece-based tokenizer implementation.
class SentencePieceTokenizer : public Tokenizer {
public:
  explicit SentencePieceTokenizer(
      std::unique_ptr<sentencepiece::SentencePieceProcessor> processor)
      : processor_(std::move(processor)) {}

  std::vector<int64_t> encode(llvm::StringRef text) const override {
    std::vector<int> ids;
    auto status = processor_->Encode(text.str(), &ids);
    if (!status.ok()) {
      llvm::errs() << "SentencePiece encode error: " << status.ToString()
                   << "\n";
      return {};
    }
    return std::vector<int64_t>(ids.begin(), ids.end());
  }

  std::string decode(llvm::ArrayRef<int64_t> ids) const override {
    std::vector<int> intIds(ids.begin(), ids.end());
    std::string text;
    auto status = processor_->Decode(intIds, &text);
    if (!status.ok()) {
      llvm::errs() << "SentencePiece decode error: " << status.ToString()
                   << "\n";
      return "";
    }
    return text;
  }

  std::string decodeToken(int64_t id) const override {
    return processor_->IdToPiece(static_cast<int>(id));
  }

  size_t vocabSize() const override {
    return static_cast<size_t>(processor_->GetPieceSize());
  }

  int64_t bosId() const override { return processor_->bos_id(); }
  int64_t eosId() const override { return processor_->eos_id(); }
  int64_t padId() const override { return processor_->pad_id(); }

  llvm::StringRef getName() const override { return "SentencePiece"; }

private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
};

bool isSentencePieceAvailable() { return true; }

llvm::Expected<std::unique_ptr<Tokenizer>>
createSentencePieceTokenizer(llvm::StringRef modelPath) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(modelPath.str());
  if (!status.ok()) {
    return llvm::createStringError(
        std::errc::io_error,
        "Failed to load SentencePiece model '%s': %s",
        modelPath.str().c_str(), status.ToString().c_str());
  }

  return std::make_unique<SentencePieceTokenizer>(std::move(processor));
}

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_LLM_ASSIST_HAVE_SENTENCEPIECE

