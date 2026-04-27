/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mm/mm.h"
#include <cassert>
#include <cstdio>

#define ASSERT_OK(expr)                                                        \
  do {                                                                         \
    mm_status_t s_ = (expr);                                                   \
    assert(s_ == MM_OK);                                                       \
    (void)s_;                                                                  \
  } while (0)
#define ASSERT_ERR(expr, code)                                                 \
  do {                                                                         \
    mm_status_t s_ = (expr);                                                   \
    assert(s_ == (code));                                                      \
    (void)s_;                                                                  \
  } while (0)

static void test_lifecycle() {
  /* Not initialized yet */
  assert(mm_is_initialized() == 0);

  /* Init with defaults */
  ASSERT_OK(mm_init(NULL));
  assert(mm_is_initialized() == 1);

  /* Double init is an error */
  ASSERT_ERR(mm_init(NULL), MM_ERR_ALREADY_INIT);

  /* Shutdown */
  mm_shutdown();
  assert(mm_is_initialized() == 0);

  /* Re-init after shutdown */
  mm_config_t cfg = mm_config_default();
  cfg.enable_debug_log = 1;
  ASSERT_OK(mm_init(&cfg));
  mm_shutdown();

  printf("  lifecycle: ok\n");
}

static void test_alloc_free() {
  ASSERT_OK(mm_init(NULL));

  /* Basic alloc */
  mm_handle_t h = mm_alloc(1024, NULL, NULL);
  assert(h != MM_HANDLE_INVALID);

  /* get_ptr */
  void *ptr = mm_get_ptr(h);
  assert(ptr != NULL);

  /* query */
  mm_alloc_info_t info;
  ASSERT_OK(mm_query(h, &info));
  assert(info.handle == h);
  assert(info.ptr == ptr);
  assert(info.size >= 1024);
  assert(info.mem_class == MM_CLASS_GENERIC);
  assert(info.lifetime == MM_LIFETIME_TRANSIENT);
  assert(info.device == 0);

  /* free */
  ASSERT_OK(mm_free(h, NULL));

  /* double free */
  ASSERT_ERR(mm_free(h, NULL), MM_ERR_INVALID_HANDLE);

  /* get_ptr after free */
  assert(mm_get_ptr(h) == NULL);

  /* query after free */
  ASSERT_ERR(mm_query(h, &info), MM_ERR_INVALID_HANDLE);

  mm_shutdown();
  printf("  alloc_free: ok\n");
}

static void test_alloc_with_hints() {
  ASSERT_OK(mm_init(NULL));

  mm_alloc_hints_t hints;
  hints.mem_class = MM_CLASS_WEIGHT;
  hints.lifetime = MM_LIFETIME_STATIC;
  hints.alignment = 512;

  mm_handle_t h = mm_alloc(100, &hints, NULL);
  assert(h != MM_HANDLE_INVALID);

  mm_alloc_info_t info;
  ASSERT_OK(mm_query(h, &info));
  assert(info.mem_class == MM_CLASS_WEIGHT);
  assert(info.lifetime == MM_LIFETIME_STATIC);
  assert(info.size >= 100);
  /* size should be aligned to 512 */
  assert(info.size % 512 == 0);

  ASSERT_OK(mm_free(h, NULL));
  mm_shutdown();
  printf("  alloc_with_hints: ok\n");
}

static void test_zero_size_alloc() {
  ASSERT_OK(mm_init(NULL));

  mm_handle_t h = mm_alloc(0, NULL, NULL);
  assert(h == MM_HANDLE_INVALID);

  mm_shutdown();
  printf("  zero_size_alloc: ok\n");
}

static void test_not_initialized() {
  /* All ops should fail gracefully when not initialized */
  assert(mm_alloc(1024, NULL, NULL) == MM_HANDLE_INVALID);
  ASSERT_ERR(mm_free(1, NULL), MM_ERR_NOT_INITIALIZED);
  assert(mm_get_ptr(1) == NULL);

  mm_alloc_info_t info;
  ASSERT_ERR(mm_query(1, &info), MM_ERR_NOT_INITIALIZED);

  printf("  not_initialized: ok\n");
}

static void test_metrics() {
  ASSERT_OK(mm_init(NULL));

  mm_metrics_snapshot_t snap = mm_metrics_snapshot();
  assert(snap.alloc_count == 0);
  assert(snap.free_count == 0);
  assert(snap.active_count == 0);
  assert(snap.total_allocated_bytes == 0);

  mm_handle_t h1 = mm_alloc(1024, NULL, NULL);
  mm_handle_t h2 = mm_alloc(2048, NULL, NULL);
  mm_handle_t h3 = mm_alloc(512, NULL, NULL);

  snap = mm_metrics_snapshot();
  assert(snap.alloc_count == 3);
  assert(snap.free_count == 0);
  assert(snap.active_count == 3);
  assert(snap.total_allocated_bytes > 0);
  assert(snap.peak_allocated_bytes > 0);

  size_t peak_before = snap.peak_allocated_bytes;

  ASSERT_OK(mm_free(h2, NULL));
  snap = mm_metrics_snapshot();
  assert(snap.alloc_count == 3);
  assert(snap.free_count == 1);
  assert(snap.active_count == 2);
  /* peak should not decrease after free */
  assert(snap.peak_allocated_bytes == peak_before);

  /* reset counters */
  mm_metrics_reset();
  snap = mm_metrics_snapshot();
  assert(snap.alloc_count == 0);
  assert(snap.free_count == 0);

  ASSERT_OK(mm_free(h1, NULL));
  ASSERT_OK(mm_free(h3, NULL));
  mm_shutdown();
  printf("  metrics: ok\n");
}

static void test_dump_state() {
  ASSERT_OK(mm_init(NULL));

  mm_handle_t h1 = mm_alloc(256, NULL, NULL);
  mm_alloc_hints_t hints = {MM_CLASS_WEIGHT, MM_LIFETIME_STATIC, 0};
  mm_handle_t h2 = mm_alloc(1024, &hints, NULL);

  /* Just verify it doesn't crash */
  mm_dump_state(stderr);

  ASSERT_OK(mm_free(h1, NULL));
  ASSERT_OK(mm_free(h2, NULL));
  mm_shutdown();
  printf("  dump_state: ok\n");
}

static void test_leak_detection() {
  mm_config_t cfg = mm_config_default();
  cfg.enable_debug_log = 1;
  ASSERT_OK(mm_init(&cfg));

  /* Allocate but don't free — shutdown should warn and clean up */
  mm_alloc(512, NULL, NULL);
  mm_alloc(1024, NULL, NULL);

  fprintf(stderr, "[TEST] Expecting leak warnings below:\n");
  mm_shutdown();

  /* Verify clean state after shutdown */
  assert(mm_is_initialized() == 0);

  printf("  leak_detection: ok\n");
}

int main() {
  printf("test_mm_core:\n");
  test_lifecycle();
  test_alloc_free();
  test_alloc_with_hints();
  test_zero_size_alloc();
  test_not_initialized();
  test_metrics();
  test_dump_state();
  test_leak_detection();
  printf("test_mm_core: PASSED\n");
  return 0;
}
