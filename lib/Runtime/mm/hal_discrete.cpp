/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hal_discrete.cpp - Discrete GPU HalAllocator implementation --------===//
//
// Discrete GPU (PCIe / XGMI) backend. GPU and CPU have separate physical
// memory; data movement requires explicit DMA.
//
// Key design choices:
//
// 1. Both backing stores (VRAM + pinned host) are allocated at alloc() time.
//    This avoids hipHostMalloc latency at eviction time, which can stall if
//    the host is under memory pressure — a latency-sensitive operation we need
//    to keep out of the decode hot path.
//
// 2. Eviction (GPU → CPU) uses hipMemcpyAsync D2H on the provided stream.
//    VRAM is NOT freed — it is retained as a "clean slot" so restore_to_gpu
//    can issue an H2D memcpy into the same allocation without a new hipMalloc.
//
// 3. restore_to_gpu() issues hipMemcpyAsync H2D. The prefetch path calls this
//    before the next compute launch on the same stream, so it overlaps with
//    prior GPU work at no extra sync cost.
//
// Under HIPDNN_EP_MM_MOCK_HAL, all HIP calls are replaced with
// malloc/free/memcpy for GPU-free unit testing.
//
//===----------------------------------------------------------------------===//

#include "hal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef HIPDNN_EP_MM_MOCK_HAL
#include "runtime_types.h"
#else
// Mock stubs for unit tests
static inline int hipMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  return *ptr ? 0 : -1;
}
static inline int hipFree(void *ptr) {
  ::free(ptr);
  return 0;
}
static inline int hipHostMalloc(void **ptr, size_t size, unsigned int /*f*/) {
  *ptr = malloc(size);
  return *ptr ? 0 : -1;
}
static inline int hipHostFree(void *ptr) {
  ::free(ptr);
  return 0;
}
static inline int hipMemcpyAsync(void *dst, const void *src, size_t n,
                                  int /*kind*/, void * /*stream*/) {
  memcpy(dst, src, n);
  return 0;
}
#define hipHostMallocDefault 0
#define hipMemcpyDeviceToHost 1
#define hipMemcpyHostToDevice 0
#define hipSuccess 0
#endif // HIPDNN_EP_MM_MOCK_HAL

HalBlock DiscreteHalAllocator::alloc(size_t bytes, MemTier /*preferred*/) {
  if (bytes == 0)
    bytes = 1;

  HalBlock block;
  block.size = bytes;

  // Allocate GPU VRAM.
  if (hipMalloc(&block.gpu_ptr, bytes) != hipSuccess) {
    fprintf(stderr, "DiscreteHalAllocator::alloc: hipMalloc(%zu) failed\n",
            bytes);
    return block; // gpu_ptr == nullptr signals failure
  }

  // Allocate pinned host RAM up-front (pre-alloc avoids latency at evict time).
  if (hipHostMalloc(&block.cpu_ptr, bytes, hipHostMallocDefault) != hipSuccess) {
    fprintf(stderr,
            "DiscreteHalAllocator::alloc: hipHostMalloc(%zu) failed\n", bytes);
    hipFree(block.gpu_ptr);
    block.gpu_ptr = nullptr;
    return block;
  }

  block.resident = MemTier::GPU;
  block.gpu_valid = true;
  block.cpu_valid = false; // CPU copy not yet populated
  return block;
}

void DiscreteHalAllocator::free(HalBlock &block) {
  if (block.gpu_ptr) {
    hipFree(block.gpu_ptr);
    block.gpu_ptr = nullptr;
  }
  if (block.cpu_ptr) {
    hipHostFree(block.cpu_ptr);
    block.cpu_ptr = nullptr;
  }
  block.size = 0;
}

// Eviction (GPU → CPU): async D2H memcpy on the provided stream.
// VRAM is retained — the GPU page acts as a "clean slot" for future H2D restore.
// After stream completion: cpu_valid = true, resident = CPU.
// IMPORTANT: the caller must ensure the stream has drained (hipStreamSynchronize)
// before reading cpu_ptr from the host; the GPU write completes asynchronously.
void DiscreteHalAllocator::evict_to_cpu(HalBlock &block, void *stream) {
  if (!block.gpu_ptr || !block.cpu_ptr)
    return;

  // Cast to hipStream_t only in real builds; mock hipMemcpyAsync takes void*.
#ifndef HIPDNN_EP_MM_MOCK_HAL
  hipMemcpyAsync(block.cpu_ptr, block.gpu_ptr, block.size,
                 hipMemcpyDeviceToHost,
                 static_cast<hipStream_t>(stream));
#else
  hipMemcpyAsync(block.cpu_ptr, block.gpu_ptr, block.size,
                 hipMemcpyDeviceToHost, stream);
#endif

  block.resident = MemTier::CPU;
  block.cpu_valid = true;
  // gpu_valid is cleared: GPU copy is now stale (CPU is authoritative after sync).
  block.gpu_valid = false;
}

// Restore (CPU → GPU): async H2D memcpy on the provided stream.
// Uses the retained VRAM slot — no hipMalloc needed.
// After stream completion: gpu_valid = true, resident = GPU.
// The cpu_ptr copy remains valid (cpu_valid may stay true if no GPU write
// diverges it; the MM tracks this via resident/gpu_valid flags).
void DiscreteHalAllocator::restore_to_gpu(HalBlock &block, void *stream) {
  if (!block.gpu_ptr || !block.cpu_ptr)
    return;

#ifndef HIPDNN_EP_MM_MOCK_HAL
  hipMemcpyAsync(block.gpu_ptr, block.cpu_ptr, block.size,
                 hipMemcpyHostToDevice,
                 static_cast<hipStream_t>(stream));
#else
  hipMemcpyAsync(block.gpu_ptr, block.cpu_ptr, block.size,
                 hipMemcpyHostToDevice, stream);
#endif

  block.resident = MemTier::GPU;
  block.gpu_valid = true;
}
