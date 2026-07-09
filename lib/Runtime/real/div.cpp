/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Div: y = lhs / rhs with ONNX multidirectional broadcast (rank <= 4).
//
// Operand shapes are passed as 4D (N, C, H, W); the compiler left-pads
// ranks 1-3 with 1. When lhs or rhs does not match the output shape,
// hip_expand materialises the broadcast into the per-session workspace,
// then hip_elementwise_div runs on same-shape buffers. Div is not
// commutative -- operands are never swapped.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int div_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_div(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t lhs_n, int64_t lhs_c, int64_t lhs_h, int64_t lhs_w,
             int64_t rhs_n, int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
             int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
             int64_t data_type) {
  OP_PROFILE(
      "div",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld", (long long)out_n,
                 (long long)out_c, (long long)out_h, (long long)out_w);
        return std::string(b);
      },
      state);

  if (!state || !lhs || !rhs || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_div: null argument\n");
    return -1;
  }

  const int64_t out_vol = out_n * out_c * out_h * out_w;
  if (out_vol <= 0)
    return 0;

  int hip_dtype = div_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_div: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  const bool lhs_eq_out =
      (lhs_n == out_n && lhs_c == out_c && lhs_h == out_h && lhs_w == out_w);
  const bool rhs_eq_out =
      (rhs_n == out_n && rhs_c == out_c && rhs_h == out_h && rhs_w == out_w);

  void *lhs_use = lhs;
  void *rhs_use = rhs;

  if (!lhs_eq_out || !rhs_eq_out) {
    const int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
    const size_t per_side = static_cast<size_t>(out_vol * elem_bytes);
    const int64_t out_shape[4] = {out_n, out_c, out_h, out_w};

    if (!lhs_eq_out) {
      void *lhs_scratch = hipdnn_ep_scratch_alloc(state, per_side);
      if (!lhs_scratch) {
        fprintf(stderr, "[REAL] wrap_div: scratch_alloc failed (%zu bytes)\n",
                per_side);
        return -1;
      }
      const int64_t in_lhs[4] = {lhs_n, lhs_c, lhs_h, lhs_w};
      int rc = hip_expand(stream, lhs, static_cast<uint8_t *>(lhs_scratch),
                          in_lhs, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_div: hip_expand(lhs) failed (%d)\n", rc);
        return -1;
      }
      lhs_use = lhs_scratch;
    }
    if (!rhs_eq_out) {
      void *rhs_scratch = hipdnn_ep_scratch_alloc(state, per_side);
      if (!rhs_scratch) {
        fprintf(stderr, "[REAL] wrap_div: scratch_alloc failed (%zu bytes)\n",
                per_side);
        return -1;
      }
      const int64_t in_rhs[4] = {rhs_n, rhs_c, rhs_h, rhs_w};
      int rc = hip_expand(stream, rhs, static_cast<uint8_t *>(rhs_scratch),
                          in_rhs, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_div: hip_expand(rhs) failed (%d)\n", rc);
        return -1;
      }
      rhs_use = rhs_scratch;
    }

    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_div: broadcast expand lhs%s rhs%s -> out=[%lld,%lld,%lld,"
        "%lld]\n",
        lhs_eq_out ? "(ok)" : "(expanded)", rhs_eq_out ? "(ok)" : "(expanded)",
        (long long)out_n, (long long)out_c, (long long)out_h, (long long)out_w);
  } else {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_div: same-shape out=[%lld,%lld,%lld,%lld], dtype=%s\n",
        (long long)out_n, (long long)out_c, (long long)out_h, (long long)out_w,
        hipdnn_ep_datatype_name(data_type));
  }

  return hip_elementwise_div(stream, lhs_use, rhs_use, output, out_vol,
                             hip_dtype);
}
