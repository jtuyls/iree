// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_KVCACHE_KVCACHEMANAGER_H_
#define IREE_COMPILER_LLMASSIST_KVCACHE_KVCACHEMANAGER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlir::iree_compiler::LLMAssist {

/// Configuration for the KV-cache.
struct KVCacheConfig {
  int numLayers = 32;       // Number of transformer layers
  int numKVHeads = 8;       // Number of key/value attention heads
  int headDim = 128;        // Dimension per attention head
  int maxSeqLen = 8192;     // Maximum sequence length
  int blockSeqStride = 16;  // Tokens per cache block (page size)
  iree_hal_element_type_t dtype = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
};

/// Manages KV-cache allocation for LLM inference.
///
/// This implements a simple linear page allocation strategy suitable for
/// single-request inference within the compiler. For more advanced use cases
/// (multiple concurrent requests, cache eviction), see shark-ai's paged
/// attention implementation.
class KVCacheManager {
public:
  /// Create a KV-cache manager.
  static llvm::Expected<std::unique_ptr<KVCacheManager>>
  create(const KVCacheConfig &config, iree_hal_device_t *device,
         iree_allocator_t hostAllocator);

  ~KVCacheManager();

  // Non-copyable, non-movable
  KVCacheManager(const KVCacheManager &) = delete;
  KVCacheManager &operator=(const KVCacheManager &) = delete;

  /// Reset the cache for a new generation.
  void reset();

  /// Allocate cache space for a sequence of given length.
  /// Returns failure if the sequence is too long.
  llvm::Error allocateForSequence(int seqLen);

  /// Extend the allocation for additional tokens during decoding.
  llvm::Error extendAllocation(int newTokens);

  /// Get the current sequence length.
  int getCurrentSeqLen() const { return currentSeqLen_; }

  /// Get the allocated page indices for the current sequence.
  const std::vector<int64_t> &getPageIds() const { return allocatedPages_; }

  /// Get the cache buffer for passing to the model.
  /// This is the actual KV-cache storage.
  iree_hal_buffer_t *getCacheBuffer() const { return cacheBuffer_; }

  /// Get page indices as a buffer view for passing to model functions.
  /// Caller takes ownership of the returned buffer view.
  llvm::Expected<iree_hal_buffer_view_t *> createPageIdsBufferView();

  /// Get the cache buffer views for all layers (for paged attention).
  /// Returns a vector of buffer views, one per layer.
  /// The caller does NOT own these views; they are managed by the cache.
  llvm::Expected<std::vector<iree_hal_buffer_view_t *>> getCacheBufferViews();

  /// Get page IDs as a buffer view. The caller does NOT own this view.
  llvm::Expected<iree_hal_buffer_view_t *> getPageIdsBufferView();

  /// Get the config.
  const KVCacheConfig &getConfig() const { return config_; }

  /// Get the total number of available pages.
  int getTotalPages() const { return numPages_; }

  /// Get the number of pages currently allocated.
  int getAllocatedPageCount() const {
    return static_cast<int>(allocatedPages_.size());
  }

private:
  KVCacheManager(const KVCacheConfig &config, iree_hal_device_t *device,
                 iree_allocator_t hostAllocator);

  llvm::Error initialize();

  KVCacheConfig config_;
  iree_hal_device_t *device_;
  iree_hal_allocator_t *deviceAllocator_;
  iree_allocator_t hostAllocator_;

  // Cache storage
  iree_hal_buffer_t *cacheBuffer_ = nullptr;
  int numPages_ = 0;
  int pageElements_ = 0;

  // Current allocation state
  int currentSeqLen_ = 0;
  std::vector<int64_t> allocatedPages_;

  // Cached buffer views
  std::vector<iree_hal_buffer_view_t *> cacheBufferViews_;
  iree_hal_buffer_view_t *pageIdsBufferView_ = nullptr;
};

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_KVCACHE_KVCACHEMANAGER_H_

