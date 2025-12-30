// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/KVCache/KVCacheManager.h"
#include "llvm/Support/Error.h"

namespace mlir::iree_compiler::LLMAssist {

namespace {

/// Helper to format an IREE status to a string.
std::string formatStatus(iree_status_t status) {
  char buffer[256];
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(buffer), buffer, &length);
  return std::string(buffer, std::min(length, sizeof(buffer) - 1));
}

} // namespace

llvm::Expected<std::unique_ptr<KVCacheManager>>
KVCacheManager::create(const KVCacheConfig &config, iree_hal_device_t *device,
                       iree_allocator_t hostAllocator) {
  auto manager = std::unique_ptr<KVCacheManager>(
      new KVCacheManager(config, device, hostAllocator));

  if (auto err = manager->initialize()) {
    return std::move(err);
  }

  return manager;
}

KVCacheManager::KVCacheManager(const KVCacheConfig &config,
                               iree_hal_device_t *device,
                               iree_allocator_t hostAllocator)
    : config_(config), device_(device), hostAllocator_(hostAllocator) {
  iree_hal_device_retain(device_);
  deviceAllocator_ = iree_hal_device_allocator(device_);
}

KVCacheManager::~KVCacheManager() {
  // Release cached buffer views
  for (auto *view : cacheBufferViews_) {
    if (view) {
      iree_hal_buffer_view_release(view);
    }
  }
  cacheBufferViews_.clear();

  if (pageIdsBufferView_) {
    iree_hal_buffer_view_release(pageIdsBufferView_);
    pageIdsBufferView_ = nullptr;
  }

  if (cacheBuffer_) {
    iree_hal_buffer_release(cacheBuffer_);
  }
  if (device_) {
    iree_hal_device_release(device_);
  }
}

llvm::Error KVCacheManager::initialize() {
  // Calculate cache dimensions
  // Each page stores: [layers, 2 (K+V), heads, blockStride, headDim]
  // We flatten this to [numPages, pageElements]
  pageElements_ = config_.numLayers * 2 * config_.numKVHeads *
                  config_.blockSeqStride * config_.headDim;
  numPages_ =
      (config_.maxSeqLen + config_.blockSeqStride - 1) / config_.blockSeqStride;

  // Calculate byte size
  size_t elementSize = iree_hal_element_dense_byte_count(config_.dtype);
  size_t totalBytes =
      static_cast<size_t>(numPages_) * static_cast<size_t>(pageElements_) *
      elementSize;

  // Allocate the cache buffer
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_status_t status = iree_hal_allocator_allocate_buffer(
      deviceAllocator_, params, totalBytes, &cacheBuffer_);

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Failed to allocate KV-cache buffer: %s",
                                   msg.c_str());
  }

  return llvm::Error::success();
}

void KVCacheManager::reset() {
  currentSeqLen_ = 0;
  allocatedPages_.clear();

  // Release cached page IDs buffer view since allocation changed
  if (pageIdsBufferView_) {
    iree_hal_buffer_view_release(pageIdsBufferView_);
    pageIdsBufferView_ = nullptr;
  }
}

llvm::Error KVCacheManager::allocateForSequence(int seqLen) {
  if (seqLen > config_.maxSeqLen) {
    return llvm::createStringError(std::errc::value_too_large,
                                   "Sequence length %d exceeds maximum %d",
                                   seqLen, config_.maxSeqLen);
  }

  // Calculate pages needed
  int pagesNeeded =
      (seqLen + config_.blockSeqStride - 1) / config_.blockSeqStride;

  if (pagesNeeded > numPages_) {
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Need %d pages but only %d available",
                                   pagesNeeded, numPages_);
  }

  // Simple linear allocation
  allocatedPages_.clear();
  allocatedPages_.reserve(pagesNeeded);
  for (int i = 0; i < pagesNeeded; ++i) {
    allocatedPages_.push_back(static_cast<int64_t>(i));
  }

  currentSeqLen_ = seqLen;
  return llvm::Error::success();
}

llvm::Error KVCacheManager::extendAllocation(int newTokens) {
  int newSeqLen = currentSeqLen_ + newTokens;

  if (newSeqLen > config_.maxSeqLen) {
    return llvm::createStringError(
        std::errc::value_too_large,
        "Extended sequence length %d exceeds maximum %d", newSeqLen,
        config_.maxSeqLen);
  }

  int pagesNeeded =
      (newSeqLen + config_.blockSeqStride - 1) / config_.blockSeqStride;

  // Add new pages if needed
  while (static_cast<int>(allocatedPages_.size()) < pagesNeeded) {
    int nextPage = static_cast<int>(allocatedPages_.size());
    if (nextPage >= numPages_) {
      return llvm::createStringError(std::errc::not_enough_memory,
                                     "No more pages available");
    }
    allocatedPages_.push_back(static_cast<int64_t>(nextPage));
  }

  currentSeqLen_ = newSeqLen;
  return llvm::Error::success();
}

llvm::Expected<iree_hal_buffer_view_t *>
KVCacheManager::createPageIdsBufferView() {
  if (allocatedPages_.empty()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "No pages allocated");
  }

  // Create a host buffer with the page IDs
  size_t numPages = allocatedPages_.size();
  size_t byteSize = numPages * sizeof(int64_t);

  iree_hal_buffer_params_t params = {};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  iree_hal_buffer_t *buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      deviceAllocator_, params, byteSize, &buffer);

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Failed to allocate page IDs buffer: %s",
                                   msg.c_str());
  }

  // Map and copy data
  iree_hal_buffer_mapping_t mapping;
  status = iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                     IREE_HAL_MEMORY_ACCESS_WRITE, 0, byteSize,
                                     &mapping);

  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to map page IDs buffer: %s",
                                   msg.c_str());
  }

  memcpy(mapping.contents.data, allocatedPages_.data(), byteSize);
  iree_hal_buffer_unmap_range(&mapping);

  // Create buffer view with shape [numPages]
  iree_hal_dim_t shape[] = {static_cast<iree_hal_dim_t>(numPages)};
  iree_hal_buffer_view_t *bufferView = nullptr;
  status = iree_hal_buffer_view_create(
      buffer, /*shape_rank=*/1, shape, IREE_HAL_ELEMENT_TYPE_INT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, hostAllocator_, &bufferView);

  iree_hal_buffer_release(buffer); // View holds a reference

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create buffer view: %s",
                                   msg.c_str());
  }

  return bufferView;
}

llvm::Expected<std::vector<iree_hal_buffer_view_t *>>
KVCacheManager::getCacheBufferViews() {
  // For paged attention, the cache is typically structured as:
  // [num_pages, layers, 2 (K+V), heads, block_stride, head_dim]
  // We create a single buffer view over the whole cache
  // and the model internally slices it.

  if (!cacheBufferViews_.empty()) {
    return cacheBufferViews_;
  }

  if (!cacheBuffer_) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Cache buffer not initialized");
  }

  // Create a buffer view for the entire cache
  // Shape: [num_pages, num_layers * 2 * num_kv_heads * block_stride * head_dim]
  // (flattened for simplicity)
  iree_hal_dim_t shape[] = {
      static_cast<iree_hal_dim_t>(numPages_),
      static_cast<iree_hal_dim_t>(pageElements_)};

  iree_hal_buffer_view_t *bufferView = nullptr;
  iree_status_t status = iree_hal_buffer_view_create(
      cacheBuffer_, /*shape_rank=*/2, shape, config_.dtype,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, hostAllocator_, &bufferView);

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create cache buffer view: %s",
                                   msg.c_str());
  }

  cacheBufferViews_.push_back(bufferView);
  return cacheBufferViews_;
}

llvm::Expected<iree_hal_buffer_view_t *> KVCacheManager::getPageIdsBufferView() {
  if (pageIdsBufferView_) {
    return pageIdsBufferView_;
  }

  auto viewOrErr = createPageIdsBufferView();
  if (!viewOrErr) {
    return viewOrErr.takeError();
  }

  pageIdsBufferView_ = *viewOrErr;
  return pageIdsBufferView_;
}

} // namespace mlir::iree_compiler::LLMAssist
