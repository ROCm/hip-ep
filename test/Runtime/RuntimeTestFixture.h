/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef RUNTIME_TEST_FIXTURE_H
#define RUNTIME_TEST_FIXTURE_H

#include <gtest/gtest.h>

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdlib>
#include <cstring>
#include <vector>

/// Base fixture for runtime operator unit tests.
/// Creates a mock RuntimeState with fake GPU handles in SetUp()
/// and cleans up via hipdnn_ep_state_cleanup() in TearDown().
class RuntimeTestFixture : public ::testing::Test {
protected:
  RuntimeState *state = nullptr;

  void SetUp() override {
    state = (RuntimeState *)calloc(1, sizeof(RuntimeState));
    ASSERT_NE(state, nullptr);

    // Create mock handles (in mock mode these are just malloc'd void*)
    hipStreamCreate(&state->stream);
    miopenCreate(&state->miopen_handle);
    miopenSetStream(state->miopen_handle, state->stream);
    hipblasLtCreate(&state->hipblas_handle);
  }

  void TearDown() override {
    if (state) {
      hipdnn_ep_state_cleanup(state);
      state = nullptr;
    }
    // Free any tracked buffers
    for (void *buf : trackedBuffers_) {
      free(buf);
    }
    trackedBuffers_.clear();
  }

  /// Allocate a buffer and track it for automatic cleanup.
  /// Use this for dummy GPU/host pointers in tests.
  void *allocBuffer(size_t bytes) {
    void *buf = malloc(bytes);
    EXPECT_NE(buf, nullptr);
    memset(buf, 0, bytes);
    trackedBuffers_.push_back(buf);
    return buf;
  }

  /// Allocate a buffer filled with a specific byte value.
  void *allocBufferFilled(size_t bytes, int fillValue) {
    void *buf = malloc(bytes);
    EXPECT_NE(buf, nullptr);
    memset(buf, fillValue, bytes);
    trackedBuffers_.push_back(buf);
    return buf;
  }

private:
  std::vector<void *> trackedBuffers_;
};

#endif // RUNTIME_TEST_FIXTURE_H
