/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_ERROR_CHECK_MACROS_H
#define HIPDNN_EP_ERROR_CHECK_MACROS_H

#include <cstdio>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

//===----------------------------------------------------------------------===//
// Error Checking Macros with Goto Cleanup Pattern
//===----------------------------------------------------------------------===//
//
// These macros provide unified error checking for GPU library calls with
// goto-based cleanup. They replace the CHECK macros that return directly,
// allowing proper resource cleanup on error paths.
//
//===----------------------------------------------------------------------===//

#define HIP_CHECK_GOTO(expr, label)                                            \
  do {                                                                         \
    hipError_t error = static_cast<hipError_t>(expr);                          \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error: %s failed at %s:%d: %s\n", #expr, __FILE__,  \
              __LINE__, hipGetErrorString(error));                             \
      result = -1;                                                             \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define HIPBLAS_CHECK_GOTO(expr, label)                                        \
  do {                                                                         \
    hipblasStatus_t status = (expr);                                           \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "hipBLAS error: %s failed at %s:%d (status=%d)\n",       \
              #expr, __FILE__, __LINE__, status);                              \
      result = -1;                                                             \
      goto label;                                                              \
    }                                                                          \
  } while (0)

// Best-effort cleanup: re-export from shared header for convenience.
#include "../hip_cleanup.h"

#endif // HIPDNN_EP_ERROR_CHECK_MACROS_H
