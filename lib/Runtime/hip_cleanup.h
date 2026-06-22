/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_HIP_CLEANUP_H
#define HIPDNN_EP_HIP_CLEANUP_H

#include "debug_log.h"
#include <cstdio>

// Best-effort cleanup: logs errors but continues cleanup.
// Requires hipError_t and hipSuccess to be defined by the including file
// (via runtime_types.h from either the real or mock build).
#define HIP_CLEANUP(expr)                                                      \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      hipdnn_ep_log_emit("Warning: " #expr " failed with error %d\n",          \
                         static_cast<int>(_err));                              \
    }                                                                          \
  } while (0)

#endif // HIPDNN_EP_HIP_CLEANUP_H
