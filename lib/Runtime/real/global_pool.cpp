/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

//===----------------------------------------------------------------------===//
// Global pool (avg / max / lp) — custom HIP kernel
//===----------------------------------------------------------------------===//
//
// Lowering signature:
//   wrap_global_pool(state, input, output,
//                    outer       = N * C,
//                    reduce_size = D_1 * ... * D_k,
//                    data_type, mode, p)
//
// `outer` and `reduce_size` are computed at HIP→LLVM lowering time from the
// input memref descriptor (so dynamic batch / channel / spatial extents are
// honored). The kernel treats the input as a flat `[outer, reduce_size]`
// matrix and computes one reduced value per row. `mode` selects the
// reduction (HIPDNN_EP_GLOBAL_POOL_AVERAGE / _MAX / _LP); `p` is the LP-norm
// exponent and is ignored unless `mode == LP`. ONNX requires `p >= 1` and
// the upstream OnnxToHip pattern enforces it — so by the time we reach the
// runtime, `p` is always sane for LP.
//
// Supports f16 / f32 / bf16 / f64 (the float-type set ONNX permits for
// global pool in any of the three flavors).

static int hipdnn_ep_to_hip_dtype(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

int wrap_global_pool(RuntimeState *state, void *input, void *output,
                     int64_t outer, int64_t reduce_size, int64_t data_type,
                     int64_t mode, int64_t p) {
  // Mode-specific OP_PROFILE label so PERF rows separate avg / max / lp.
  const char *op_label = hipdnn_ep_global_pool_mode_name(mode);
  OP_PROFILE(
      op_label,
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "outer=%lld,reduce=%lld,p=%lld",
                 (long long)outer, (long long)reduce_size, (long long)p);
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    hipdnn_ep_log_emit("[REAL] wrap_global_pool: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    hipdnn_ep_log_emit("[REAL] wrap_global_pool: unsupported data_type %lld\n",
                       (long long)data_type);
    return -1;
  }

  if (mode != HIPDNN_EP_GLOBAL_POOL_AVERAGE &&
      mode != HIPDNN_EP_GLOBAL_POOL_MAX && mode != HIPDNN_EP_GLOBAL_POOL_LP) {
    hipdnn_ep_log_emit("[REAL] wrap_global_pool: unsupported mode %lld\n",
                       (long long)mode);
    return -1;
  }

  if (mode == HIPDNN_EP_GLOBAL_POOL_LP && p < 1) {
    hipdnn_ep_log_emit(
        "[REAL] wrap_global_pool: LP requires p >= 1 (got %lld)\n",
        (long long)p);
    return -1;
  }

  if (outer <= 0 || reduce_size <= 0) {
    // Nothing to do; treat empty reduction as a no-op (matches the kernel's
    // own early-return guard).
    return 0;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  const char *type_name = hipdnn_ep_datatype_name(data_type);
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  RUNTIME_DEBUG_LOG("[REAL] wrap_global_pool: %s outer=%lld, reduce_size=%lld, "
                    "data_type=%s(%lld), elem_size=%lld, p=%lld\n",
                    op_label, (long long)outer, (long long)reduce_size,
                    type_name, (long long)data_type, (long long)elem_size,
                    (long long)p);

  int result =
      hip_global_pool(stream, input, output, outer, reduce_size, hip_dtype,
                      static_cast<int>(mode), static_cast<int>(p));
  if (result != 0) {
    hipdnn_ep_log_emit(
        "[REAL] wrap_global_pool (%s): kernel launch failed (%d)\n", op_label,
        result);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_global_pool (%s): completed successfully\n",
                    op_label);
  return 0;
}
