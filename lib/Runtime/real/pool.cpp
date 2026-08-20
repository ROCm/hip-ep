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
// Pool (MaxPool / AveragePool / LpPool) — custom HIP kernel
//===----------------------------------------------------------------------===//
//
// Lowering signature (matches PoolLowering.cpp):
//   wrap_pool(state, input, output, indices, shape_valid,
//             data_type, pool_mode, spatial_rank,
//             N, C,
//             in0, in1, in2,
//             out0, out1, out2,
//             k0..k2, s0..s2, p0..p2 (pad_begin),
//             dil0..dil2,
//             storage_order, ceil_mode, has_indices,
//             count_include_pad, p)
//
// `pool_mode` (HIPDNN_EP_POOL_*) selects the per-window reduction. `indices`
// is nullable and only consumed for MAX. `storage_order` and `ceil_mode` are
// accepted for ABI completeness but already absorbed at compile time
// (storage_order=1 is rejected at onnx-to-hip; ceil_mode is reflected in the
// output shape the conversion gives us, so the kernel just walks the windows
// it's been told about). `count_include_pad` only affects AVERAGE; `p` only
// affects LP.

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

int wrap_pool(RuntimeState *state, void *input, void *output, void *indices,
              int64_t shape_valid, int64_t data_type, int64_t pool_mode,
              int64_t spatial_rank, int64_t N, int64_t C, int64_t in0,
              int64_t in1, int64_t in2, int64_t out0, int64_t out1,
              int64_t out2, int64_t k0, int64_t k1, int64_t k2, int64_t s0,
              int64_t s1, int64_t s2, int64_t p0, int64_t p1, int64_t p2,
              int64_t dil0, int64_t dil1, int64_t dil2, int64_t storage_order,
              int64_t ceil_mode, int64_t has_indices, int64_t count_include_pad,
              int64_t p) {
  OP_PROFILE(
      hipdnn_ep_pool_mode_name(pool_mode),
      [&] {
        char b[160];
        snprintf(b, sizeof(b),
                 "rank=%lld,N=%lld,C=%lld,in=[%lld,%lld,%lld],out=[%lld,%lld,"
                 "%lld]",
                 (long long)spatial_rank, (long long)N, (long long)C,
                 (long long)in0, (long long)in1, (long long)in2,
                 (long long)out0, (long long)out1, (long long)out2);
        return std::string(b);
      },
      state);

  if (!state) {
    fprintf(stderr, "[REAL] wrap_pool: null state\n");
    return -1;
  }
  if (hipdnn_ep_validate_dynamic_shape(state, shape_valid) != 0) {
    fprintf(stderr, "[REAL] wrap_pool: invalid dynamic output shape\n");
    return -1;
  }
  if (!input || !output) {
    fprintf(stderr, "[REAL] wrap_pool: null argument\n");
    return -1;
  }
  // Indices are MAX-only; ignore any stray pointer for other modes.
  bool want_indices = has_indices && pool_mode == HIPDNN_EP_POOL_MAX;
  if (want_indices && !indices) {
    fprintf(stderr, "[REAL] wrap_pool: has_indices=1 but indices ptr is "
                    "null\n");
    return -1;
  }
  if (pool_mode != HIPDNN_EP_POOL_AVERAGE && pool_mode != HIPDNN_EP_POOL_MAX &&
      pool_mode != HIPDNN_EP_POOL_LP) {
    fprintf(stderr, "[REAL] wrap_pool: unsupported pool_mode %lld\n",
            (long long)pool_mode);
    return -1;
  }
  if (pool_mode == HIPDNN_EP_POOL_LP && p < 1) {
    fprintf(stderr, "[REAL] wrap_pool: LpPool requires p >= 1 (got %lld)\n",
            (long long)p);
    return -1;
  }
  (void)storage_order;
  (void)ceil_mode;

  int hip_dtype = hipdnn_ep_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_pool: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_pool: mode=%s dtype=%s(%lld) spatial_rank=%lld N=%lld "
      "C=%lld "
      "in=[%lld,%lld,%lld] out=[%lld,%lld,%lld] count_include_pad=%lld p=%lld "
      "indices=%s\n",
      hipdnn_ep_pool_mode_name(pool_mode), hipdnn_ep_datatype_name(data_type),
      (long long)data_type, (long long)spatial_rank, (long long)N, (long long)C,
      (long long)in0, (long long)in1, (long long)in2, (long long)out0,
      (long long)out1, (long long)out2, (long long)count_include_pad,
      (long long)p, want_indices ? "yes" : "null");

  int rc = hip_pool(
      stream, input, output, want_indices ? indices : nullptr, hip_dtype,
      static_cast<int>(pool_mode), static_cast<int>(spatial_rank), N, C, in0,
      in1, in2, out0, out1, out2, k0, k1, k2, s0, s1, s2, p0, p1, p2, dil0,
      dil1, dil2, static_cast<int>(count_include_pad), static_cast<int>(p));
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_pool: kernel launch failed (%d)\n", rc);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_pool: completed successfully\n");
  return 0;
}
