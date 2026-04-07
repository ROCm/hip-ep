/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeTestFixture.h"

//===----------------------------------------------------------------------===//
// State cleanup
//===----------------------------------------------------------------------===//

TEST(RuntimeStateTest, CleanupNullStateReturnsZero) {
  EXPECT_EQ(hipdnn_ep_state_cleanup(nullptr), 0);
}

//===----------------------------------------------------------------------===//
// State accessors
//===----------------------------------------------------------------------===//

class StateAccessorTest : public RuntimeTestFixture {};

TEST_F(StateAccessorTest, GetStreamReturnsNonNull) {
  void *stream = hipdnn_ep_state_get_stream(state);
  EXPECT_NE(stream, nullptr);
}

TEST_F(StateAccessorTest, GetStreamNullStateReturnsNull) {
  void *stream = hipdnn_ep_state_get_stream(nullptr);
  EXPECT_EQ(stream, nullptr);
}

TEST_F(StateAccessorTest, GetMiopenHandleReturnsNonNull) {
  void *handle = hipdnn_ep_state_get_miopen_handle(state);
  EXPECT_NE(handle, nullptr);
}

TEST_F(StateAccessorTest, GetMiopenHandleNullStateReturnsNull) {
  void *handle = hipdnn_ep_state_get_miopen_handle(nullptr);
  EXPECT_EQ(handle, nullptr);
}

TEST_F(StateAccessorTest, GetHipblasHandleReturnsNonNull) {
  void *handle = hipdnn_ep_state_get_hipblas_handle(state);
  EXPECT_NE(handle, nullptr);
}

TEST_F(StateAccessorTest, GetHipblasHandleNullStateReturnsNull) {
  void *handle = hipdnn_ep_state_get_hipblas_handle(nullptr);
  EXPECT_EQ(handle, nullptr);
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

TEST_F(StateAccessorTest, ConstantGetNullStateReturnsNull) {
  void *ptr = hipdnn_ep_constant_get(nullptr, 0);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(StateAccessorTest, ConstantGetOutOfBoundsReturnsNull) {
  // state has num_constants=0, so any index is out of bounds
  void *ptr = hipdnn_ep_constant_get(state, 0);
  EXPECT_EQ(ptr, nullptr);
}

//===----------------------------------------------------------------------===//
// Memory pool
//===----------------------------------------------------------------------===//

class PoolTest : public RuntimeTestFixture {};

TEST_F(PoolTest, GetPoolBaseWithoutInitReturnsNull) {
  void *base = hipdnn_ep_get_pool_base(state);
  EXPECT_EQ(base, nullptr);
}

TEST_F(PoolTest, InitPoolAndGetBase) {
  size_t offsets[] = {0, 1024, 2048};
  int rc = hipdnn_ep_pool_init(state, 4096, offsets, 3);
  EXPECT_EQ(rc, 0);

  void *base = hipdnn_ep_get_pool_base(state);
  EXPECT_NE(base, nullptr);
}

TEST_F(PoolTest, GetBufferFromPool) {
  size_t offsets[] = {0, 1024, 2048};
  int rc = hipdnn_ep_pool_init(state, 4096, offsets, 3);
  EXPECT_EQ(rc, 0);

  void *buf0 = hipdnn_ep_get_buffer_from_pool(state, 0);
  void *buf1 = hipdnn_ep_get_buffer_from_pool(state, 1);
  void *buf2 = hipdnn_ep_get_buffer_from_pool(state, 2);
  EXPECT_NE(buf0, nullptr);
  EXPECT_NE(buf1, nullptr);
  EXPECT_NE(buf2, nullptr);

  // Buffers should be at different offsets from base
  char *base = (char *)hipdnn_ep_get_pool_base(state);
  EXPECT_EQ(buf0, base + 0);
  EXPECT_EQ(buf1, base + 1024);
  EXPECT_EQ(buf2, base + 2048);
}

TEST_F(PoolTest, GetBufferOutOfBoundsReturnsNull) {
  size_t offsets[] = {0};
  hipdnn_ep_pool_init(state, 1024, offsets, 1);

  void *buf = hipdnn_ep_get_buffer_from_pool(state, 99);
  EXPECT_EQ(buf, nullptr);
}

//===----------------------------------------------------------------------===//
// Workspace
//===----------------------------------------------------------------------===//

class WorkspaceTest : public RuntimeTestFixture {};

TEST_F(WorkspaceTest, InitialWorkspaceIsNullAndSizeZero) {
  EXPECT_EQ(hipdnn_ep_state_get_workspace(state), nullptr);
  EXPECT_EQ(hipdnn_ep_state_get_workspace_size(state), 0u);
}

TEST_F(WorkspaceTest, EnsureWorkspaceAllocates) {
  int rc = hipdnn_ep_state_ensure_workspace(state, 4096);
  EXPECT_EQ(rc, 0);
  EXPECT_NE(hipdnn_ep_state_get_workspace(state), nullptr);
  EXPECT_GE(hipdnn_ep_state_get_workspace_size(state), 4096u);
}

TEST_F(WorkspaceTest, EnsureWorkspaceGrows) {
  hipdnn_ep_state_ensure_workspace(state, 1024);
  void *ws1 = hipdnn_ep_state_get_workspace(state);

  hipdnn_ep_state_ensure_workspace(state, 8192);
  EXPECT_GE(hipdnn_ep_state_get_workspace_size(state), 8192u);
  // Workspace pointer may change after growth (realloc)
  EXPECT_NE(hipdnn_ep_state_get_workspace(state), nullptr);
}

TEST_F(WorkspaceTest, EnsureWorkspaceNoShrink) {
  hipdnn_ep_state_ensure_workspace(state, 8192);
  size_t bigSize = hipdnn_ep_state_get_workspace_size(state);

  // Requesting smaller size should not shrink
  hipdnn_ep_state_ensure_workspace(state, 1024);
  EXPECT_GE(hipdnn_ep_state_get_workspace_size(state), bigSize);
}
