/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * test_mm_types — Verify that all public headers compile and types are
 *                 correctly sized.
 */

#include "mm/mm.h"
#include <cassert>
#include <cstdio>

int main() {
    /* Handle type must be 64 bits */
    static_assert(sizeof(mm_handle_t) == 8, "mm_handle_t must be 8 bytes");

    /* Invalid handle sentinel */
    assert(MM_HANDLE_INVALID == 0);

    /* Config defaults */
    mm_config_t cfg = mm_config_default();
    assert(cfg.device_id == 0);
    assert(cfg.default_alignment == 256);
    assert(cfg.enable_debug_log == 0);

    /* Enum values */
    assert(MM_CLASS_GENERIC == 0);
    assert(MM_CLASS_SCRATCH == 4);
    assert(MM_LIFETIME_STATIC == 0);
    assert(MM_LIFETIME_TRANSIENT == 3);

    /* Hints struct zero-init */
    mm_alloc_hints_t hints = {};
    assert(hints.mem_class == MM_CLASS_GENERIC);
    assert(hints.alignment == 0);

    /* Metrics struct zero-init */
    mm_metrics_snapshot_t metrics = {};
    assert(metrics.alloc_count == 0);
    assert(metrics.active_count == 0);

    /* Error string */
    assert(mm_status_string(MM_OK) != nullptr);
    assert(mm_status_string(MM_ERR_OUT_OF_MEMORY) != nullptr);

    printf("test_mm_types: PASSED\n");
    return 0;
}
