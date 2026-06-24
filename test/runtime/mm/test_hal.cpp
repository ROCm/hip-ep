/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit tests for HalAllocator (ApuHalAllocator + DiscreteHalAllocator).
//
// Compiled with HIPDNN_EP_MM_MOCK_HAL so all HIP calls are replaced with
// malloc/free/memcpy. Tests verify the block contract without a GPU.
//
// Run via: ctest -R MemoryManagerUnitTests
//===----------------------------------------------------------------------===//

#include "mm/hal.h"

#include <cassert>
#include <cstdio>
#include <cstring>

// Simple test harness — no GTest needed, matching test_output_allocator.cpp.
// Shared failure counter — defined in test_memory_manager.cpp, extern here.
extern int g_failures;

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++g_failures;                                                             \
    }                                                                           \
  } while (0)

//===----------------------------------------------------------------------===//
// ApuHalAllocator tests
//===----------------------------------------------------------------------===//

static void test_apu_alloc_returns_valid_block() {
  ApuHalAllocator hal;
  HalBlock b = hal.alloc(1024, MemTier::GPU);
  CHECK(b.gpu_ptr != nullptr);
  CHECK(b.cpu_ptr != nullptr);
  // On APU (mock): gpu_ptr == cpu_ptr (same VA)
  CHECK(b.gpu_ptr == b.cpu_ptr);
  CHECK(b.size == 1024);
  CHECK(b.gpu_valid);
  CHECK(b.cpu_valid); // APU: same page, always accessible from both sides
  CHECK(b.resident == MemTier::GPU);
  hal.free(b);
  CHECK(b.gpu_ptr == nullptr);
  CHECK(b.cpu_ptr == nullptr);
}

static void test_apu_alloc_zero_rounds_up() {
  ApuHalAllocator hal;
  HalBlock b = hal.alloc(0, MemTier::GPU);
  // size=0 is rounded to 1 to avoid undefined hipHostMalloc(0)
  CHECK(b.gpu_ptr != nullptr);
  hal.free(b);
}

static void test_apu_evict_to_cpu_is_metadata_only() {
  ApuHalAllocator hal;
  HalBlock b = hal.alloc(64, MemTier::GPU);
  // Write a sentinel through the GPU pointer.
  *reinterpret_cast<uint32_t *>(b.gpu_ptr) = 0xDEADBEEF;
  hal.evict_to_cpu(b, /*stream=*/nullptr);
  // On APU mock: same VA, sentinel visible through cpu_ptr immediately.
  CHECK(*reinterpret_cast<uint32_t *>(b.cpu_ptr) == 0xDEADBEEF);
  CHECK(b.resident == MemTier::CPU);
  CHECK(b.cpu_valid);
  hal.free(b);
}

static void test_apu_restore_to_gpu_is_metadata_only() {
  ApuHalAllocator hal;
  HalBlock b = hal.alloc(64, MemTier::GPU);
  hal.evict_to_cpu(b, nullptr);
  CHECK(b.resident == MemTier::CPU);
  hal.restore_to_gpu(b, nullptr);
  CHECK(b.resident == MemTier::GPU);
  CHECK(b.gpu_valid);
  hal.free(b);
}

static void test_apu_free_null_block_is_noop() {
  ApuHalAllocator hal;
  HalBlock empty{};
  hal.free(empty); // must not crash
  CHECK(empty.gpu_ptr == nullptr);
}

static void test_apu_is_integrated() {
  ApuHalAllocator hal;
  CHECK(hal.is_integrated());
}

//===----------------------------------------------------------------------===//
// DiscreteHalAllocator tests
//===----------------------------------------------------------------------===//

static void test_discrete_alloc_creates_both_pointers() {
  DiscreteHalAllocator hal;
  HalBlock b = hal.alloc(256, MemTier::GPU);
  CHECK(b.gpu_ptr != nullptr);
  CHECK(b.cpu_ptr != nullptr);
  // On discrete (mock): gpu_ptr and cpu_ptr are distinct allocations.
  CHECK(b.gpu_ptr != b.cpu_ptr);
  CHECK(b.size == 256);
  CHECK(b.gpu_valid);
  CHECK(!b.cpu_valid); // CPU copy not yet populated
  CHECK(b.resident == MemTier::GPU);
  hal.free(b);
  CHECK(b.gpu_ptr == nullptr);
  CHECK(b.cpu_ptr == nullptr);
}

static void test_discrete_evict_copies_data_to_cpu() {
  DiscreteHalAllocator hal;
  HalBlock b = hal.alloc(64, MemTier::GPU);
  *reinterpret_cast<uint32_t *>(b.gpu_ptr) = 0xCAFEBABE;
  hal.evict_to_cpu(b, /*stream=*/nullptr);
  // In mock mode: hipMemcpyAsync is synchronous memcpy.
  CHECK(*reinterpret_cast<uint32_t *>(b.cpu_ptr) == 0xCAFEBABE);
  CHECK(b.cpu_valid);
  CHECK(!b.gpu_valid);
  CHECK(b.resident == MemTier::CPU);
  hal.free(b);
}

static void test_discrete_restore_copies_data_back_to_gpu() {
  DiscreteHalAllocator hal;
  HalBlock b = hal.alloc(64, MemTier::GPU);
  hal.evict_to_cpu(b, nullptr);
  *reinterpret_cast<uint32_t *>(b.cpu_ptr) = 0x12345678;
  hal.restore_to_gpu(b, nullptr);
  CHECK(*reinterpret_cast<uint32_t *>(b.gpu_ptr) == 0x12345678);
  CHECK(b.gpu_valid);
  CHECK(b.resident == MemTier::GPU);
  hal.free(b);
}

static void test_discrete_evict_retains_gpu_ptr() {
  // VRAM is NOT freed on eviction — retained for a subsequent restore.
  DiscreteHalAllocator hal;
  HalBlock b = hal.alloc(128, MemTier::GPU);
  void *orig_gpu = b.gpu_ptr;
  hal.evict_to_cpu(b, nullptr);
  CHECK(b.gpu_ptr == orig_gpu); // VRAM pointer unchanged
  hal.free(b);
}

static void test_discrete_roundtrip_data_integrity() {
  DiscreteHalAllocator hal;
  HalBlock b = hal.alloc(sizeof(uint64_t) * 4, MemTier::GPU);
  uint64_t *gpu = reinterpret_cast<uint64_t *>(b.gpu_ptr);
  for (int i = 0; i < 4; ++i)
    gpu[i] = static_cast<uint64_t>(i * 100 + 7);
  hal.evict_to_cpu(b, nullptr);
  // Corrupt the gpu side to prove we're reading from cpu after restore.
  for (int i = 0; i < 4; ++i)
    gpu[i] = 0xDEADDEAD;
  hal.restore_to_gpu(b, nullptr);
  for (int i = 0; i < 4; ++i)
    CHECK(gpu[i] == static_cast<uint64_t>(i * 100 + 7));
  hal.free(b);
}

static void test_discrete_is_not_integrated() {
  DiscreteHalAllocator hal;
  CHECK(!hal.is_integrated());
}

static void test_discrete_free_null_block_is_noop() {
  DiscreteHalAllocator hal;
  HalBlock empty{};
  hal.free(empty); // must not crash
}

//===----------------------------------------------------------------------===//
// Factory test
//===----------------------------------------------------------------------===//

static void test_factory_returns_nonnull() {
  // In mock mode, factory always returns ApuHalAllocator.
  HalAllocator *hal = hal_create_for_device(0);
  CHECK(hal != nullptr);
  delete hal;
}

// Called from main() in test_memory_manager.cpp.
void run_hal_tests() {
  test_apu_alloc_returns_valid_block();
  test_apu_alloc_zero_rounds_up();
  test_apu_evict_to_cpu_is_metadata_only();
  test_apu_restore_to_gpu_is_metadata_only();
  test_apu_free_null_block_is_noop();
  test_apu_is_integrated();

  test_discrete_alloc_creates_both_pointers();
  test_discrete_evict_copies_data_to_cpu();
  test_discrete_restore_copies_data_back_to_gpu();
  test_discrete_evict_retains_gpu_ptr();
  test_discrete_roundtrip_data_integrity();
  test_discrete_is_not_integrated();
  test_discrete_free_null_block_is_noop();

  test_factory_returns_nonnull();
}
