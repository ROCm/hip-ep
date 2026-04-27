/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * mm_config.h — Configuration for the Unified Memory Manager.
 *
 * Pass an mm_config_t to mm_init() to control device selection, alignment,
 * and debug logging. Use mm_config_default() for sensible defaults.
 */

#ifndef MM_CONFIG_H
#define MM_CONFIG_H

#include "mm_types.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)  /* getenv deprecation */
#endif
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * mm_config_t — Initialization configuration
 *
 * Fields:
 *   device_id         — GPU device ordinal to use (default: 0).
 *   default_alignment — Default byte alignment for allocations when the
 *                       caller does not specify one in mm_alloc_hints_t.
 *                       Default: 256 (matches GPU cache line size).
 *   enable_debug_log  — If non-zero, print debug messages (alloc/free events,
 *                       leak warnings) to stderr. Default: 0 (off).
 * --------------------------------------------------------------------------- */
typedef struct {
    mm_device_t device_id;
    size_t      default_alignment;
    int         enable_debug_log;
} mm_config_t;

/**
 * Returns an mm_config_t initialized with sensible defaults:
 *   device_id = 0, default_alignment = 256, enable_debug_log = 0.
 * Set environment variable MM_DEBUG_LOG=1 to enable debug logging.
 */
static inline mm_config_t mm_config_default(void) {
    mm_config_t c;
    c.device_id = 0;
    c.default_alignment = 256;
    const char* dbg = getenv("MM_DEBUG_LOG");
    c.enable_debug_log = (dbg && dbg[0] != '0' && dbg[0] != '\0') ? 1 : 0;
    return c;
}

#ifdef __cplusplus
}
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif /* MM_CONFIG_H */
