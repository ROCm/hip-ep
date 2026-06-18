/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- op_state_buffers.h - Grow-on-demand buffers for op states --------===//
//
// Small device/pinned-host buffer helpers used inside per-op state structs
// (see docs/design/op-state-slots-design.md). They mirror the grow-on-demand
// policy the legacy RuntimeState scratch fields used: monotonic 1.5x growth,
// stream-sync before freeing the old buffer, never shrink, and free themselves
// in their destructor (run by the OpState deletor at session cleanup).
//
// Real-runtime only (depends on HIP). Mock op states never own device memory,
// so this header is not included by the mock build.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_REAL_OP_STATE_BUFFERS_H
#define HIPDNN_EP_REAL_OP_STATE_BUFFERS_H

#include "runtime_types.h" // HIP types + hipMalloc/hipFree/hipHostMalloc/...

#include <cstddef>

// Grow-on-demand device buffer (hipMalloc/hipFree).
struct GrowableDeviceBuffer {
  void *ptr = nullptr;
  size_t size = 0;

  void *ensure(size_t needed, hipStream_t stream) {
    if (needed == 0)
      return ptr;
    if (size >= needed)
      return ptr;
    size_t alloc = needed;
    if (size > 0) {
      size_t grown = size + size / 2;
      if (grown > alloc)
        alloc = grown;
    }
    if (ptr) {
      if (stream)
        hipStreamSynchronize(stream);
      hipFree(ptr);
      ptr = nullptr;
      size = 0;
    }
    if (hipMalloc(&ptr, alloc) != hipSuccess) {
      ptr = nullptr;
      size = 0;
      return nullptr;
    }
    size = alloc;
    return ptr;
  }

  ~GrowableDeviceBuffer() {
    if (ptr)
      hipFree(ptr);
  }
};

// Grow-on-demand pinned (host-mapped) buffer for small D2H readbacks
// (hipHostMalloc/hipHostFree).
struct GrowablePinnedBuffer {
  void *ptr = nullptr;
  size_t size = 0;

  void *ensure(size_t needed, hipStream_t stream) {
    if (needed == 0)
      return ptr;
    if (size >= needed)
      return ptr;
    size_t alloc = needed;
    if (size > 0) {
      size_t grown = size + size / 2;
      if (grown > alloc)
        alloc = grown;
    }
    if (ptr) {
      if (stream)
        hipStreamSynchronize(stream);
      hipHostFree(ptr);
      ptr = nullptr;
      size = 0;
    }
    if (hipHostMalloc(&ptr, alloc, hipHostMallocDefault) != hipSuccess) {
      ptr = nullptr;
      size = 0;
      return nullptr;
    }
    size = alloc;
    return ptr;
  }

  ~GrowablePinnedBuffer() {
    if (ptr)
      hipHostFree(ptr);
  }
};

#endif // HIPDNN_EP_REAL_OP_STATE_BUFFERS_H
