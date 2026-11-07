/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// RyzenAI CoreLib wrappers for builds without the corelib headers (or mock
// builds): every call reports "not handled", so generated code always takes
// the GPU wrapper.

#include "../hipdnn_ep_runtime.h"

extern "C" {

int wrap_corelib_matmul_nbits(RuntimeState *, int, const void *, const void *,
                              const void *, const void *, const void *,
                              const void *, void *, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t) {
  return HIPDNN_EP_CORELIB_NOT_HANDLED;
}

int wrap_corelib_rms_norm(RuntimeState *, void *, void *, void *, int64_t,
                          int64_t, int64_t, int64_t, int64_t, float, int64_t,
                          int64_t) {
  return HIPDNN_EP_CORELIB_NOT_HANDLED;
}

int wrap_corelib_skip_rms_norm(RuntimeState *, void *, void *, void *, void *,
                               void *, void *, int64_t, int64_t, int64_t, float,
                               int64_t) {
  return HIPDNN_EP_CORELIB_NOT_HANDLED;
}

} // extern "C"
