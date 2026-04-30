/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_ERROR_CHECK_MACROS_H
#define HIPDNN_EP_ERROR_CHECK_MACROS_H

#include <atomic>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <miopen/miopen.h>

//===----------------------------------------------------------------------===//
// Sticky-error short-circuit
//===----------------------------------------------------------------------===//
//
// Once any HIP / MIOpen / hipBLASLt call returns hipErrorLaunchFailure (719)
// or one of its peers, the GPU device is in a sticky-error state: every
// subsequent API call will fail, and some library error paths (notably
// hipBLASLt's TensileLite epilogue-string lookup on TheRock 7.11) will
// raise 0xC0000005 inside `strlen` instead of returning a clean status
// code, making the failure look like a library-internal corruption.
//
// To convert that catastrophic surface into a recoverable one, the EP
// flips the flag below the first time a launch failure is observed. Every
// `wrap_*` entrypoint then checks the flag at entry via
// `HIPDNN_EP_BAIL_IF_DEAD()` and short-circuits to `return -1` immediately.
// Net effect: the benchmark process exits with a non-zero status and a
// single "device error" log line instead of a 0xC0000005 stack trace
// through hipBLASLt internals -- which makes the failure recoverable for
// the host runtime (clean cleanup of all allocations) and triagable for
// CI (single diagnostic line vs. an opaque segfault).
//
// The flag is a C++17 inline variable so all translation units share a
// single instance.
//===----------------------------------------------------------------------===//
inline std::atomic<bool> g_hipdnn_ep_device_dead{false};

inline bool hipdnn_ep_is_device_dead() {
  return g_hipdnn_ep_device_dead.load(std::memory_order_acquire);
}

inline void hipdnn_ep_mark_device_dead(const char *src, int line,
                                       const char *cause) {
  bool was = g_hipdnn_ep_device_dead.exchange(true, std::memory_order_acq_rel);
  if (!was) {
    fprintf(stderr,
            "*** HIPDNN_EP: device entered sticky error state at %s:%d (%s); "
            "subsequent wrap_* calls will short-circuit ***\n",
            src, line, cause);
  }
}

#define HIPDNN_EP_BAIL_IF_DEAD()                                               \
  do {                                                                         \
    if (hipdnn_ep_is_device_dead()) {                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

//===----------------------------------------------------------------------===//
// Error Checking Macros with Goto Cleanup Pattern
//===----------------------------------------------------------------------===//
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
//   MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(desc, ...),
//   cleanup);
//
//   cleanup:
//     if (desc) miopenDestroyTensorDescriptor(desc);
//     return result;
//
//===----------------------------------------------------------------------===//

#define MIOPEN_CHECK_GOTO(expr, label)                                         \
  do {                                                                         \
    miopenStatus_t status = (expr);                                            \
    if (status != miopenStatusSuccess) {                                       \
      fprintf(stderr, "MIOpen error: %s failed at %s:%d (status=%d)\n", #expr, \
              __FILE__, __LINE__, status);                                     \
      /* MIOpen status 7 (StatusInternalError) on this stack always means     \
         a HIP launch failure underneath -- mark the device dead so          \
         subsequent wrap_* calls short-circuit (see top of this header). */  \
      if (status == 7)                                                         \
        hipdnn_ep_mark_device_dead(__FILE__, __LINE__, "MIOpen status 7");    \
      result = -1;                                                             \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define HIP_CHECK_GOTO(expr, label)                                            \
  do {                                                                         \
    hipError_t error = static_cast<hipError_t>(expr);                          \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error: %s failed at %s:%d: %s\n", #expr, __FILE__,  \
              __LINE__, hipGetErrorString(error));                             \
      if (error == hipErrorLaunchFailure ||                                    \
          error == hipErrorIllegalAddress ||                                   \
          error == hipErrorContextIsDestroyed)                                 \
        hipdnn_ep_mark_device_dead(__FILE__, __LINE__,                         \
                                   hipGetErrorName(error));                    \
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
      if (status == HIPBLAS_STATUS_INTERNAL_ERROR ||                          \
          status == HIPBLAS_STATUS_NOT_INITIALIZED)                           \
        hipdnn_ep_mark_device_dead(__FILE__, __LINE__,                        \
                                   "hipBLAS internal/not-initialised");      \
      result = -1;                                                             \
      goto label;                                                              \
    }                                                                          \
  } while (0)

// Best-effort cleanup: re-export from shared header for convenience.
#include "../hip_cleanup.h"

#endif // HIPDNN_EP_ERROR_CHECK_MACROS_H
