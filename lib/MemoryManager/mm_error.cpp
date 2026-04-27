/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mm/mm_error.h"

const char* mm_status_string(mm_status_t status) {
    switch (status) {
    case MM_OK:                  return "ok";
    case MM_ERR_NOT_INITIALIZED: return "not initialized";
    case MM_ERR_ALREADY_INIT:    return "already initialized";
    case MM_ERR_INVALID_HANDLE:  return "invalid handle";
    case MM_ERR_OUT_OF_MEMORY:   return "out of memory";
    case MM_ERR_INVALID_ARGUMENT:return "invalid argument";
    case MM_ERR_HAL_FAILURE:     return "HAL failure";
    case MM_ERR_DOUBLE_FREE:     return "double free";
    default:                     return "unknown error";
    }
}
