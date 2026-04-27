/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * mm_error.h — Error codes for the Unified Memory Manager.
 *
 * All UMM API functions that can fail return mm_status_t. Functions that
 * return handles use MM_HANDLE_INVALID to signal failure instead.
 */

#ifndef MM_ERROR_H
#define MM_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * mm_status_t — Error codes
 *
 *   MM_OK                 — Success.
 *   MM_ERR_NOT_INITIALIZED — mm_init() has not been called yet.
 *   MM_ERR_ALREADY_INIT   — mm_init() was called while already initialized.
 *   MM_ERR_INVALID_HANDLE — Handle is not in the handle table (never existed
 *                           or was already freed).
 *   MM_ERR_OUT_OF_MEMORY  — HAL malloc failed (GPU or host OOM).
 *   MM_ERR_INVALID_ARGUMENT — A function argument is invalid (e.g., size == 0).
 *   MM_ERR_HAL_FAILURE    — The HAL backend returned a non-OOM error.
 *   MM_ERR_DOUBLE_FREE    — mm_free() called on an already-freed handle.
 * --------------------------------------------------------------------------- */
typedef enum {
    MM_OK                   =  0,
    MM_ERR_NOT_INITIALIZED  = -1,
    MM_ERR_ALREADY_INIT     = -2,
    MM_ERR_INVALID_HANDLE   = -3,
    MM_ERR_OUT_OF_MEMORY    = -4,
    MM_ERR_INVALID_ARGUMENT = -5,
    MM_ERR_HAL_FAILURE      = -6,
    MM_ERR_DOUBLE_FREE      = -7
} mm_status_t;

/**
 * Returns a human-readable string for the given status code.
 * The returned pointer is to a static string literal (do not free).
 *
 * Example: mm_status_string(MM_ERR_OUT_OF_MEMORY) => "out of memory"
 */
const char* mm_status_string(mm_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* MM_ERROR_H */
