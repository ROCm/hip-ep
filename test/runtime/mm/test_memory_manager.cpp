/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit tests for MemoryManager (Phase 1 scope).
//
// Uses MockHalAllocator (which wraps IGpuHalAllocator under
// HIPDNN_EP_MM_MOCK_HAL) to verify pool management, scratch, workspace,
// host-scalar, seqlens_k cache, and domain-independence contracts.
//
// Run via: ctest -R MemoryManagerUnitTests
//===----------------------------------------------------------------------===//

#include "mm/hal.h"
#include "mm/memory_manager.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Non-static so test_hal.cpp can use it as extern int g_failures.
int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// Helper: create a MemoryManager backed by a fresh IGpuHalAllocator (mock
// mode).
static MemoryManager *make_mm() {
  return new MemoryManager(new IGpuHalAllocator());
}

//===----------------------------------------------------------------------===//
// Pool management
//===----------------------------------------------------------------------===//

static void test_get_pool_base_returns_nonnull() {
  auto mm = make_mm();
  void *p = mm->get_pool_base(0, 1024);
  CHECK(p != nullptr);
  delete mm;
}

static void test_get_pool_base_same_size_returns_same_ptr() {
  auto mm = make_mm();
  void *p1 = mm->get_pool_base(0, 512);
  void *p2 = mm->get_pool_base(0, 512);
  CHECK(p1 == p2); // no realloc when size unchanged
  delete mm;
}

static void test_get_pool_base_growth_is_amortized() {
  auto mm = make_mm();
  mm->get_pool_base(0, 1000);
  size_t before = mm->gpu_bytes_used();
  // Trigger growth: request 1 byte more than current.
  mm->get_pool_base(0, 1001);
  size_t after = mm->gpu_bytes_used();
  // 1.5× amortized: new allocation must be > 1001 bytes.
  CHECK(after > 1001);
  // And must be at least ~1.4× old (allow some floating-point slack).
  CHECK(after >= before * 14 / 10);
  delete mm;
}

static void test_get_pool_base_negative_domain_returns_null() {
  auto mm = make_mm();
  void *p = mm->get_pool_base(-1, 128);
  CHECK(p == nullptr);
  delete mm;
}

static void test_get_pool_base_zero_size_succeeds() {
  // size=0 is reserved to 1 byte internally.
  auto mm = make_mm();
  void *p = mm->get_pool_base(0, 0);
  CHECK(p != nullptr);
  delete mm;
}

static void test_domain_ids_are_independent() {
  auto mm = make_mm();
  void *d0 = mm->get_pool_base(0, 512);
  void *d1 = mm->get_pool_base(1, 256);
  CHECK(d0 != nullptr);
  CHECK(d1 != nullptr);
  CHECK(d0 != d1); // different domains → different allocations
  void *d0b = mm->get_pool_base(0, 512);
  CHECK(d0 == d0b); // same domain, same size → same pointer (no realloc)
  delete mm;
}

static void test_load_pool_plan_sets_buffers() {
  auto mm = make_mm();
  size_t offsets[] = {0, 128, 384};
  mm->load_pool_plan(512, offsets, 3);
  void *base = mm->get_pool_base(0, 512);
  CHECK(base != nullptr);
  // get_buffer_from_pool uses the recorded offsets.
  void *buf0 = mm->get_buffer_from_pool(0);
  void *buf1 = mm->get_buffer_from_pool(1);
  CHECK(buf0 != nullptr);
  CHECK(buf1 != nullptr);
  // buf1 should be 128 bytes past buf0.
  CHECK(reinterpret_cast<char *>(buf1) == reinterpret_cast<char *>(buf0) + 128);
  delete mm;
}

static void test_get_buffer_from_pool_out_of_range_returns_null() {
  auto mm = make_mm();
  size_t offsets[] = {0, 64};
  mm->load_pool_plan(128, offsets, 2);
  mm->get_pool_base(0, 128);              // ensure pool is allocated
  void *p = mm->get_buffer_from_pool(99); // index out of range
  CHECK(p == nullptr);
  delete mm;
}

//===----------------------------------------------------------------------===//
// Workspace
//===----------------------------------------------------------------------===//

static void test_ensure_workspace_returns_nonnull() {
  auto mm = make_mm();
  void *ws = mm->ensure_workspace(1024);
  CHECK(ws != nullptr);
  CHECK(mm->get_workspace() == ws);
  CHECK(mm->get_workspace_size() >= 1024);
  delete mm;
}

static void test_ensure_workspace_idempotent_for_same_size() {
  auto mm = make_mm();
  void *ws1 = mm->ensure_workspace(512);
  void *ws2 = mm->ensure_workspace(512);
  CHECK(ws1 == ws2);
  delete mm;
}

static void test_ensure_workspace_grows_amortized() {
  auto mm = make_mm();
  mm->ensure_workspace(1000);
  size_t size1 = mm->get_workspace_size();
  mm->ensure_workspace(1001);
  size_t size2 = mm->get_workspace_size();
  CHECK(size2 > 1001); // 1.5× growth
  CHECK(size2 >= size1 * 14 / 10);
  delete mm;
}

static void test_ensure_workspace_zero_is_noop() {
  auto mm = make_mm();
  void *ws = mm->ensure_workspace(0);
  // Returns current workspace (null on first call with 0).
  (void)ws; // just must not crash
  delete mm;
}

//===----------------------------------------------------------------------===//
// Host-scalar scratch
//===----------------------------------------------------------------------===//

static void test_get_host_scratch_returns_writable_ptr() {
  auto mm = make_mm();
  void *p = mm->get_host_scratch(64);
  CHECK(p != nullptr);
  *reinterpret_cast<int *>(p) = 42;
  CHECK(*reinterpret_cast<int *>(p) == 42);
  delete mm;
}

static void test_get_host_scratch_idempotent_for_same_size() {
  auto mm = make_mm();
  void *p1 = mm->get_host_scratch(128);
  void *p2 = mm->get_host_scratch(128);
  CHECK(p1 == p2);
  delete mm;
}

static void test_get_host_scratch_zero_is_valid() {
  auto mm = make_mm();
  void *p = mm->get_host_scratch(0);
  CHECK(p != nullptr); // rounds up to 1
  delete mm;
}

//===----------------------------------------------------------------------===//
// QMoE host scratch
//===----------------------------------------------------------------------===//

static void test_ensure_qmoe_host_scratch_returns_nonnull() {
  auto mm = make_mm();
  void *p = mm->ensure_qmoe_host_scratch(256);
  CHECK(p != nullptr);
  CHECK(mm->get_qmoe_host_scratch() == p);
  delete mm;
}

static void test_ensure_qmoe_host_scratch_idempotent() {
  auto mm = make_mm();
  void *p1 = mm->ensure_qmoe_host_scratch(128);
  void *p2 = mm->ensure_qmoe_host_scratch(128);
  CHECK(p1 == p2);
  delete mm;
}

//===----------------------------------------------------------------------===//
// begin_compute / seqlens_k cache
//===----------------------------------------------------------------------===//

static void test_begin_compute_invalidates_seqlens_k_cache() {
  auto mm = make_mm();
  // Manually prime the cache.
  void *fake_ptr = reinterpret_cast<void *>(0xDEAD);
  mm->seqlens_k_cache_set(fake_ptr, 99);
  CHECK(mm->seqlens_k_cache_valid(fake_ptr));
  CHECK(mm->seqlens_k_cached_val() == 99);
  mm->begin_compute();
  CHECK(!mm->seqlens_k_cache_valid(fake_ptr)); // invalidated
  delete mm;
}

static void test_seqlens_k_cache_ptr_keyed() {
  auto mm = make_mm();
  void *ptr_a = reinterpret_cast<void *>(0x1000);
  void *ptr_b = reinterpret_cast<void *>(0x2000);
  mm->seqlens_k_cache_set(ptr_a, 42);
  CHECK(mm->seqlens_k_cache_valid(ptr_a));
  CHECK(!mm->seqlens_k_cache_valid(ptr_b)); // different pointer → miss
  CHECK(mm->seqlens_k_cached_val() == 42);
  delete mm;
}

static void test_seqlens_k_cache_set_replaces() {
  auto mm = make_mm();
  void *ptr = reinterpret_cast<void *>(0xABCD);
  mm->seqlens_k_cache_set(ptr, 10);
  mm->seqlens_k_cache_set(ptr, 20);
  CHECK(mm->seqlens_k_cached_val() == 20); // latest value wins
  delete mm;
}

static void test_end_compute_is_noop_in_phase1() {
  auto mm = make_mm();
  mm->end_compute(); // must not crash; no-op in Phase 1
  delete mm;
}

//===----------------------------------------------------------------------===//
// Stats
//===----------------------------------------------------------------------===//

static void test_gpu_bytes_used_accounts_for_pool_and_workspace() {
  auto mm = make_mm();
  CHECK(mm->gpu_bytes_used() == 0);
  mm->get_pool_base(0, 1024);
  mm->ensure_workspace(512);
  size_t used = mm->gpu_bytes_used();
  CHECK(used >= 1024 + 512);
  delete mm;
}

static void test_cpu_bytes_used_accounts_for_host_scratch() {
  auto mm = make_mm();
  CHECK(mm->cpu_bytes_used() == 0);
  mm->get_host_scratch(128);
  mm->ensure_qmoe_host_scratch(64);
  size_t used = mm->cpu_bytes_used();
  CHECK(used >= 128 + 64);
  delete mm;
}

static void test_num_pool_domains_grows_on_demand() {
  auto mm = make_mm();
  CHECK(mm->num_pool_domains() == 0);
  mm->get_pool_base(0, 64);
  CHECK(mm->num_pool_domains() >= 1);
  mm->get_pool_base(2, 64);
  CHECK(mm->num_pool_domains() >= 3);
  delete mm;
}

//===----------------------------------------------------------------------===//
// Scratch arena
//===----------------------------------------------------------------------===//

static void test_scratch_alloc_returns_aligned_pointer() {
  auto mm = make_mm();
  void *p = mm->scratch_alloc(100);
  CHECK(p != nullptr);
  CHECK((reinterpret_cast<uintptr_t>(p) % 64) == 0);
  void *p2 = mm->scratch_alloc(33);
  CHECK(p2 != nullptr);
  CHECK((reinterpret_cast<uintptr_t>(p2) % 64) == 0);
  delete mm;
}

static void test_scratch_alloc_sequential_non_overlapping() {
  auto mm = make_mm();
  void *a = mm->scratch_alloc(128);
  void *b = mm->scratch_alloc(256);
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  auto a_end = reinterpret_cast<uintptr_t>(a) + 128;
  auto b_start = reinterpret_cast<uintptr_t>(b);
  CHECK(b_start >= a_end);
  delete mm;
}

static void test_scratch_reset_reuses_memory() {
  auto mm = make_mm();
  void *p1 = mm->scratch_alloc(512);
  CHECK(p1 != nullptr);
  CHECK(mm->scratch_offset() > 0);
  mm->scratch_reset();
  CHECK(mm->scratch_offset() == 0);
  void *p2 = mm->scratch_alloc(512);
  CHECK(p2 == p1);
  delete mm;
}

static void test_scratch_alloc_grows_workspace() {
  auto mm = make_mm();
  mm->scratch_alloc(1000);
  size_t ws1 = mm->get_workspace_size();
  CHECK(ws1 >= 1000);
  mm->scratch_alloc(2000);
  size_t ws2 = mm->get_workspace_size();
  CHECK(ws2 >= 1000 + 2000);
  delete mm;
}

static void test_begin_compute_resets_scratch() {
  auto mm = make_mm();
  mm->scratch_alloc(256);
  CHECK(mm->scratch_offset() > 0);
  mm->begin_compute();
  CHECK(mm->scratch_offset() == 0);
  delete mm;
}

static void test_ensure_workspace_preserves_scratch() {
  auto mm = make_mm();
  mm->scratch_alloc(128);
  size_t off = mm->scratch_offset();
  CHECK(off > 0);
  mm->ensure_workspace(4096);
  CHECK(mm->scratch_offset() == off);
  delete mm;
}

static void test_scratch_alloc_zero_returns_valid_ptr() {
  auto mm = make_mm();
  mm->scratch_alloc(64);
  size_t off_before = mm->scratch_offset();
  void *p = mm->scratch_alloc(0);
  CHECK(mm->scratch_offset() == off_before);
  CHECK(p != nullptr);
  delete mm;
}

//===----------------------------------------------------------------------===//
// KV cache buffer tracking
//===----------------------------------------------------------------------===//

static void test_register_kv_buffer_tracks_size() {
  auto mm = make_mm();
  CHECK(mm->kv_bytes_used() == 0);
  CHECK(mm->kv_buffer_count() == 0);
  char dummy[64];
  mm->register_kv_buffer(dummy, 1024);
  CHECK(mm->kv_bytes_used() == 1024);
  CHECK(mm->kv_buffer_count() == 1);
  delete mm;
}

static void test_unregister_kv_buffer_removes_tracking() {
  auto mm = make_mm();
  char dummy[64];
  mm->register_kv_buffer(dummy, 2048);
  CHECK(mm->kv_bytes_used() == 2048);
  mm->unregister_kv_buffer(dummy);
  CHECK(mm->kv_bytes_used() == 0);
  CHECK(mm->kv_buffer_count() == 0);
  delete mm;
}

static void test_kv_buffer_count_multiple() {
  auto mm = make_mm();
  char a[1], b[1], c[1];
  mm->register_kv_buffer(a, 100);
  mm->register_kv_buffer(b, 200);
  mm->register_kv_buffer(c, 300);
  CHECK(mm->kv_buffer_count() == 3);
  CHECK(mm->kv_bytes_used() == 600);
  mm->unregister_kv_buffer(b);
  CHECK(mm->kv_buffer_count() == 2);
  CHECK(mm->kv_bytes_used() == 400);
  delete mm;
}

static void test_register_kv_duplicate_is_idempotent() {
  auto mm = make_mm();
  char dummy[64];
  mm->register_kv_buffer(dummy, 512);
  mm->register_kv_buffer(dummy, 512);
  CHECK(mm->kv_buffer_count() == 1);
  CHECK(mm->kv_bytes_used() == 512);
  delete mm;
}

static void test_kv_bytes_included_in_gpu_bytes() {
  auto mm = make_mm();
  size_t base = mm->gpu_bytes_used();
  char dummy[64];
  mm->register_kv_buffer(dummy, 4096);
  CHECK(mm->gpu_bytes_used() == base + 4096);
  delete mm;
}

//===----------------------------------------------------------------------===//
// BlockPool tests
//===----------------------------------------------------------------------===//

static void test_block_pool_init_succeeds() {
  auto mm = make_mm();
  bool ok = mm->init_block_pool(/*num_blocks=*/64, /*block_size=*/16,
                                /*kv_num_heads=*/8, /*head_dim=*/64,
                                /*elem_size=*/2);
  CHECK(ok);
  CHECK(mm->get_block_pool() != nullptr);
  CHECK(mm->get_block_pool()->num_blocks() == 64);
  CHECK(mm->get_block_pool()->num_free_blocks() == 64);
  delete mm;
}

static void test_block_pool_alloc_and_free() {
  auto mm = make_mm();
  mm->init_block_pool(4, 16, 8, 64, 2);
  auto *pool = mm->get_block_pool();
  int b0 = pool->alloc_block();
  int b1 = pool->alloc_block();
  CHECK(b0 >= 0 && b0 < 4);
  CHECK(b1 >= 0 && b1 < 4);
  CHECK(b0 != b1);
  CHECK(pool->num_free_blocks() == 2);
  pool->free_block(b0);
  CHECK(pool->num_free_blocks() == 3);
  delete mm;
}

static void test_block_pool_exhaustion_returns_minus1() {
  auto mm = make_mm();
  mm->init_block_pool(2, 16, 8, 64, 2);
  auto *pool = mm->get_block_pool();
  pool->alloc_block();
  pool->alloc_block();
  CHECK(pool->num_free_blocks() == 0);
  int bad = pool->alloc_block();
  CHECK(bad == -1);
  delete mm;
}

static void test_block_pool_key_value_bases_differ() {
  auto mm = make_mm();
  mm->init_block_pool(8, 16, 8, 64, 2);
  auto *pool = mm->get_block_pool();
  CHECK(pool->key_cache_base() != nullptr);
  CHECK(pool->value_cache_base() != nullptr);
  CHECK(pool->key_cache_base() != pool->value_cache_base());
  delete mm;
}

static void test_block_pool_included_in_gpu_bytes_used() {
  auto mm = make_mm();
  size_t before = mm->gpu_bytes_used();
  mm->init_block_pool(16, 16, 8, 64, 2);
  size_t after = mm->gpu_bytes_used();
  // 16 blocks × 16 tokens × 8 heads × 64 dim × 2 bytes × 2 (K+V)
  size_t expected = 16ull * 16 * 8 * 64 * 2 * 2;
  CHECK(after == before + expected);
  delete mm;
}

//===----------------------------------------------------------------------===//
// main — runs all MM unit tests (HAL + MemoryManager)
//===----------------------------------------------------------------------===//

// Defined in test_hal.cpp.
void run_hal_tests();

int main() {
  run_hal_tests();
  test_get_pool_base_returns_nonnull();
  test_get_pool_base_same_size_returns_same_ptr();
  test_get_pool_base_growth_is_amortized();
  test_get_pool_base_negative_domain_returns_null();
  test_get_pool_base_zero_size_succeeds();
  test_domain_ids_are_independent();
  test_load_pool_plan_sets_buffers();
  test_get_buffer_from_pool_out_of_range_returns_null();

  test_ensure_workspace_returns_nonnull();
  test_ensure_workspace_idempotent_for_same_size();
  test_ensure_workspace_grows_amortized();
  test_ensure_workspace_zero_is_noop();

  test_get_host_scratch_returns_writable_ptr();
  test_get_host_scratch_idempotent_for_same_size();
  test_get_host_scratch_zero_is_valid();

  test_ensure_qmoe_host_scratch_returns_nonnull();
  test_ensure_qmoe_host_scratch_idempotent();

  test_begin_compute_invalidates_seqlens_k_cache();
  test_seqlens_k_cache_ptr_keyed();
  test_seqlens_k_cache_set_replaces();
  test_end_compute_is_noop_in_phase1();

  test_scratch_alloc_returns_aligned_pointer();
  test_scratch_alloc_sequential_non_overlapping();
  test_scratch_reset_reuses_memory();
  test_scratch_alloc_grows_workspace();
  test_begin_compute_resets_scratch();
  test_ensure_workspace_preserves_scratch();
  test_scratch_alloc_zero_returns_valid_ptr();

  test_register_kv_buffer_tracks_size();
  test_unregister_kv_buffer_removes_tracking();
  test_kv_buffer_count_multiple();
  test_register_kv_duplicate_is_idempotent();
  test_kv_bytes_included_in_gpu_bytes();

  test_block_pool_init_succeeds();
  test_block_pool_alloc_and_free();
  test_block_pool_exhaustion_returns_minus1();
  test_block_pool_key_value_bases_differ();
  test_block_pool_included_in_gpu_bytes_used();

  test_gpu_bytes_used_accounts_for_pool_and_workspace();
  test_cpu_bytes_used_accounts_for_host_scratch();
  test_num_pool_domains_grows_on_demand();

  if (g_failures == 0) {
    std::printf("MemoryManagerUnitTests: all tests passed.\n");
    return 0;
  }
  std::fprintf(stderr, "MemoryManagerUnitTests: %d test(s) FAILED.\n",
               g_failures);
  return 1;
}
