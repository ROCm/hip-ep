/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * test_mm_pool — Tests for the Static Pool allocator.
 */

#include "mm/mm.h"
#include <cassert>
#include <cstdio>
#include <cstring>

#define ASSERT_OK(expr) do { mm_status_t s_ = (expr); assert(s_ == MM_OK); (void)s_; } while(0)

static void test_basic_pool() {
    ASSERT_OK(mm_init(NULL));

    size_t offsets[] = {0, 256, 512, 1024};
    mm_static_plan_t plan;
    plan.total_size  = 2048;
    plan.offsets     = offsets;
    plan.num_entries = 4;

    mm_pool_t pool = mm_pool_create(&plan);
    assert(pool != NULL);

    /* base pointer */
    void* base = mm_pool_get_base(pool);
    assert(base != NULL);

    /* size and count */
    assert(mm_pool_get_size(pool) >= 2048);
    assert(mm_pool_get_num_entries(pool) == 4);

    /* sub-buffer pointers */
    void* p0 = mm_pool_get_ptr(pool, 0);
    void* p1 = mm_pool_get_ptr(pool, 1);
    void* p2 = mm_pool_get_ptr(pool, 2);
    void* p3 = mm_pool_get_ptr(pool, 3);

    assert(p0 == base);
    assert(p1 == (char*)base + 256);
    assert(p2 == (char*)base + 512);
    assert(p3 == (char*)base + 1024);

    /* out of bounds */
    assert(mm_pool_get_ptr(pool, 4) == NULL);
    assert(mm_pool_get_ptr(pool, 999) == NULL);

    /* write and read back through sub-buffers (mock HAL uses real memory) */
    memset(p0, 0xAA, 256);
    memset(p1, 0xBB, 256);
    unsigned char* c0 = (unsigned char*)p0;
    unsigned char* c1 = (unsigned char*)p1;
    assert(c0[0] == 0xAA && c0[255] == 0xAA);
    assert(c1[0] == 0xBB && c1[255] == 0xBB);

    mm_pool_destroy(pool);
    mm_shutdown();
    printf("  basic_pool: ok\n");
}

static void test_empty_offsets() {
    ASSERT_OK(mm_init(NULL));

    /* Pool with no sub-buffers (just raw allocation) */
    mm_static_plan_t plan;
    plan.total_size  = 4096;
    plan.offsets     = NULL;
    plan.num_entries = 0;

    mm_pool_t pool = mm_pool_create(&plan);
    assert(pool != NULL);

    assert(mm_pool_get_base(pool) != NULL);
    assert(mm_pool_get_size(pool) >= 4096);
    assert(mm_pool_get_num_entries(pool) == 0);
    assert(mm_pool_get_ptr(pool, 0) == NULL);

    mm_pool_destroy(pool);
    mm_shutdown();
    printf("  empty_offsets: ok\n");
}

static void test_null_safety() {
    /* All getters should return safely with NULL */
    assert(mm_pool_get_ptr(NULL, 0) == NULL);
    assert(mm_pool_get_base(NULL) == NULL);
    assert(mm_pool_get_size(NULL) == 0);
    assert(mm_pool_get_num_entries(NULL) == 0);

    /* destroy NULL is a no-op */
    mm_pool_destroy(NULL);

    /* create with NULL plan */
    assert(mm_pool_create(NULL) == NULL);

    /* create with zero size */
    mm_static_plan_t plan = {0, NULL, 0};
    assert(mm_pool_create(&plan) == NULL);

    printf("  null_safety: ok\n");
}

static void test_metrics_tracking() {
    ASSERT_OK(mm_init(NULL));

    mm_metrics_snapshot_t snap0 = mm_metrics_snapshot();

    size_t offsets[] = {0, 1024};
    mm_static_plan_t plan = {2048, offsets, 2};
    mm_pool_t pool = mm_pool_create(&plan);
    assert(pool != NULL);

    mm_metrics_snapshot_t snap1 = mm_metrics_snapshot();
    assert(snap1.alloc_count == snap0.alloc_count + 1);
    assert(snap1.total_allocated_bytes > snap0.total_allocated_bytes);

    mm_pool_destroy(pool);

    mm_metrics_snapshot_t snap2 = mm_metrics_snapshot();
    assert(snap2.free_count == snap0.free_count + 1);

    mm_shutdown();
    printf("  metrics_tracking: ok\n");
}

static void test_multiple_pools() {
    ASSERT_OK(mm_init(NULL));

    size_t off1[] = {0, 512};
    mm_static_plan_t plan1 = {1024, off1, 2};

    size_t off2[] = {0, 256, 768};
    mm_static_plan_t plan2 = {2048, off2, 3};

    mm_pool_t p1 = mm_pool_create(&plan1);
    mm_pool_t p2 = mm_pool_create(&plan2);
    assert(p1 != NULL && p2 != NULL);

    /* Pools have independent base pointers */
    assert(mm_pool_get_base(p1) != mm_pool_get_base(p2));

    /* Each pool's offsets work independently */
    assert(mm_pool_get_ptr(p1, 1) == (char*)mm_pool_get_base(p1) + 512);
    assert(mm_pool_get_ptr(p2, 2) == (char*)mm_pool_get_base(p2) + 768);

    mm_pool_destroy(p1);
    mm_pool_destroy(p2);
    mm_shutdown();
    printf("  multiple_pools: ok\n");
}

int main() {
    printf("test_mm_pool: starting\n");
    test_null_safety();
    test_basic_pool();
    test_empty_offsets();
    test_metrics_tracking();
    test_multiple_pools();
    printf("test_mm_pool: PASSED\n");
    return 0;
}
