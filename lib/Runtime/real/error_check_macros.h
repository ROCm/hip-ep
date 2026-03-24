/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_ERROR_CHECK_MACROS_H
#define HIPDNN_EP_ERROR_CHECK_MACROS_H

#include <cstdio>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <miopen/miopen.h>

//==============================================================================
// Error Checking Macros with Goto Cleanup Pattern
//==============================================================================
//
// These macros provide unified error checking for GPU library calls with
// goto-based cleanup. They replace the CHECK macros that return directly,
// allowing proper resource cleanup on error paths.
//
// Usage:
//   miopenTensorDescriptor_t desc = nullptr;
//   int result = 0;
//
//   MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&desc), cleanup);
//   MIOPEN_CHECK_GOTO(miopenSet4dTensorDescriptor(desc, ...), cleanup);
//
//   cleanup:
//     if (desc) miopenDestroyTensorDescriptor(desc);
//     return result;
//
//==============================================================================

#define MIOPEN_CHECK_GOTO(expr, label)                                         \
  do {                                                                         \
    miopenStatus_t status = (expr);                                            \
    if (status != miopenStatusSuccess) {                                       \
      fprintf(stderr, "MIOpen error: %s failed at %s:%d (status=%d)\n", #expr, \
              __FILE__, __LINE__, status);                                     \
      result = -1;                                                             \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define HIP_CHECK_GOTO(expr, label)                                            \
  do {                                                                         \
    hipError_t error = (expr);                                                 \
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

#define KERNEL_CHECK_GOTO(expr, label)                                         \
  do {                                                                         \
    int _rc = (expr);                                                          \
    if (_rc != 0) {                                                            \
      fprintf(stderr, "Kernel error: %s failed at %s:%d (rc=%d)\n", #expr,     \
              __FILE__, __LINE__, _rc);                                        \
      result = _rc;                                                            \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#endif // HIPDNN_EP_ERROR_CHECK_MACROS_H
