// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/KVCache/KVCacheManager.h"

#include "gtest/gtest.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/vm/api.h"

namespace mlir::iree_compiler::LLMAssist {
namespace {

class KVCacheManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    hostAllocator_ = iree_allocator_system();

    // Register the local-task driver (ignore if already registered)
    iree_status_t status = 
        iree_hal_local_task_driver_module_register(iree_hal_driver_registry_default());
    if (!iree_status_is_ok(status) && 
        iree_status_code(status) != IREE_STATUS_ALREADY_EXISTS) {
      iree_status_free(status);
      GTEST_SKIP() << "Failed to register local-task driver";
      return;
    }
    iree_status_ignore(status);

    // Create a device
    iree_hal_driver_t *driver = nullptr;
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), hostAllocator_, &driver);
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "Failed to create local-task driver";
      return;
    }

    status = iree_hal_driver_create_default_device(driver, hostAllocator_, &device_);
    iree_hal_driver_release(driver);
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "Failed to create device";
      return;
    }
  }

  void TearDown() override {
    if (device_) {
      iree_hal_device_release(device_);
    }
  }

  iree_allocator_t hostAllocator_;
  iree_hal_device_t *device_ = nullptr;
};

TEST_F(KVCacheManagerTest, ConfigDefaults) {
  KVCacheConfig config;
  EXPECT_EQ(config.numLayers, 32);
  EXPECT_EQ(config.numKVHeads, 8);
  EXPECT_EQ(config.headDim, 128);
  EXPECT_EQ(config.blockSeqStride, 16);
  EXPECT_EQ(config.maxSeqLen, 8192);
  EXPECT_EQ(config.dtype, IREE_HAL_ELEMENT_TYPE_FLOAT_16);
}

TEST_F(KVCacheManagerTest, CreateWithValidConfig) {
  ASSERT_NE(device_, nullptr) << "Device not available";

  KVCacheConfig config;
  config.numLayers = 4;
  config.numKVHeads = 4;
  config.headDim = 64;
  config.blockSeqStride = 8;
  config.maxSeqLen = 512;

  auto managerOrErr = KVCacheManager::create(config, device_, hostAllocator_);
  ASSERT_TRUE(static_cast<bool>(managerOrErr))
      << llvm::toString(managerOrErr.takeError());

  auto &manager = *managerOrErr;
  EXPECT_EQ(manager->getCurrentSeqLen(), 0);
}

TEST_F(KVCacheManagerTest, AllocateForSequence) {
  ASSERT_NE(device_, nullptr) << "Device not available";

  KVCacheConfig config;
  config.numLayers = 4;
  config.numKVHeads = 4;
  config.headDim = 64;
  config.blockSeqStride = 8;
  config.maxSeqLen = 512;

  auto managerOrErr = KVCacheManager::create(config, device_, hostAllocator_);
  ASSERT_TRUE(static_cast<bool>(managerOrErr));

  auto &manager = *managerOrErr;

  // Allocate for a short sequence
  auto err = manager->allocateForSequence(32);
  ASSERT_FALSE(static_cast<bool>(err)) << llvm::toString(std::move(err));
  EXPECT_EQ(manager->getCurrentSeqLen(), 32);

  // Check that pages were allocated correctly (32 tokens / 8 block_stride = 4 pages)
  const auto &pages = manager->getPageIds();
  EXPECT_EQ(pages.size(), 4u);
  EXPECT_EQ(pages[0], 0);
  EXPECT_EQ(pages[1], 1);
  EXPECT_EQ(pages[2], 2);
  EXPECT_EQ(pages[3], 3);
}

TEST_F(KVCacheManagerTest, ExtendAllocation) {
  ASSERT_NE(device_, nullptr) << "Device not available";

  KVCacheConfig config;
  config.numLayers = 4;
  config.numKVHeads = 4;
  config.headDim = 64;
  config.blockSeqStride = 8;
  config.maxSeqLen = 512;

  auto managerOrErr = KVCacheManager::create(config, device_, hostAllocator_);
  ASSERT_TRUE(static_cast<bool>(managerOrErr));

  auto &manager = *managerOrErr;

  // Allocate for initial sequence
  auto err = manager->allocateForSequence(16);
  ASSERT_FALSE(static_cast<bool>(err));
  EXPECT_EQ(manager->getPageIds().size(), 2u);

  // Extend allocation
  err = manager->extendAllocation(10);
  ASSERT_FALSE(static_cast<bool>(err));
  EXPECT_EQ(manager->getCurrentSeqLen(), 26);
  EXPECT_EQ(manager->getPageIds().size(), 4u); // (26 + 7) / 8 = 4 pages
}

TEST_F(KVCacheManagerTest, AllocateExceedingMaxFails) {
  ASSERT_NE(device_, nullptr) << "Device not available";

  KVCacheConfig config;
  config.numLayers = 4;
  config.numKVHeads = 4;
  config.headDim = 64;
  config.blockSeqStride = 8;
  config.maxSeqLen = 64;

  auto managerOrErr = KVCacheManager::create(config, device_, hostAllocator_);
  ASSERT_TRUE(static_cast<bool>(managerOrErr));

  auto &manager = *managerOrErr;

  // Try to allocate more than max
  auto err = manager->allocateForSequence(100);
  EXPECT_TRUE(static_cast<bool>(err));
  llvm::consumeError(std::move(err));
}

TEST_F(KVCacheManagerTest, Reset) {
  ASSERT_NE(device_, nullptr) << "Device not available";

  KVCacheConfig config;
  config.numLayers = 4;
  config.numKVHeads = 4;
  config.headDim = 64;
  config.blockSeqStride = 8;
  config.maxSeqLen = 512;

  auto managerOrErr = KVCacheManager::create(config, device_, hostAllocator_);
  ASSERT_TRUE(static_cast<bool>(managerOrErr));

  auto &manager = *managerOrErr;

  auto err = manager->allocateForSequence(32);
  ASSERT_FALSE(static_cast<bool>(err));
  EXPECT_GT(manager->getCurrentSeqLen(), 0);

  manager->reset();
  EXPECT_EQ(manager->getCurrentSeqLen(), 0);
  EXPECT_TRUE(manager->getPageIds().empty());
}

TEST_F(KVCacheManagerTest, GetCacheBufferViews) {
  ASSERT_NE(device_, nullptr) << "Device not available";

  KVCacheConfig config;
  config.numLayers = 4;
  config.numKVHeads = 4;
  config.headDim = 64;
  config.blockSeqStride = 8;
  config.maxSeqLen = 512;

  auto managerOrErr = KVCacheManager::create(config, device_, hostAllocator_);
  ASSERT_TRUE(static_cast<bool>(managerOrErr));

  auto &manager = *managerOrErr;

  auto viewsOrErr = manager->getCacheBufferViews();
  ASSERT_TRUE(static_cast<bool>(viewsOrErr))
      << llvm::toString(viewsOrErr.takeError());

  auto &views = *viewsOrErr;
  EXPECT_FALSE(views.empty());
}

TEST_F(KVCacheManagerTest, CreatePageIdsBufferView) {
  ASSERT_NE(device_, nullptr) << "Device not available";

  KVCacheConfig config;
  config.numLayers = 4;
  config.numKVHeads = 4;
  config.headDim = 64;
  config.blockSeqStride = 8;
  config.maxSeqLen = 512;

  auto managerOrErr = KVCacheManager::create(config, device_, hostAllocator_);
  ASSERT_TRUE(static_cast<bool>(managerOrErr));

  auto &manager = *managerOrErr;

  // First allocate for a sequence
  auto err = manager->allocateForSequence(16);
  ASSERT_FALSE(static_cast<bool>(err));

  // Now get page IDs
  auto viewOrErr = manager->getPageIdsBufferView();
  ASSERT_TRUE(static_cast<bool>(viewOrErr))
      << llvm::toString(viewOrErr.takeError());

  auto *view = *viewOrErr;
  ASSERT_NE(view, nullptr);

  // Check shape - should be [num_pages]
  EXPECT_EQ(iree_hal_buffer_view_shape_rank(view), 1u);
  EXPECT_EQ(iree_hal_buffer_view_shape_dim(view, 0), 2u); // 16 tokens / 8 = 2 pages
}

} // namespace
} // namespace mlir::iree_compiler::LLMAssist

