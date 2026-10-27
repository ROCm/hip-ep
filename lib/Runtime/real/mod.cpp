/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Mod: y = a % b (element-wise, same-shape, ONNX semantics).
//   fmod == 0 -> Python-style integer modulo (sign follows divisor)
//   fmod == 1 -> C fmod (floating-point only)
//
// Source: onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.cu
//         @ v1.22.2 (BINARY_OP_NAME_EXPR(Mod,  _Mod(a, b)),
//                    BINARY_OP_NAME_EXPR(Fmod, _Fmod(a, b)))
//         + core/providers/cuda/cu_inc/common.cuh _Mod / _Fmod
//
// Operand shapes are passed as 4D (N, C, H, W); the compiler left-pads
// ranks 0-3 with 1. When lhs or rhs does not match the output shape,
// hip_expand materialises the broadcast into the per-session workspace, then
// the flat hip_elementwise_mod runs on same-shape buffers (mirrors wrap_less).
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>

static int mod_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_mod(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t a_n, int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n,
             int64_t b_c, int64_t b_h, int64_t b_w, int64_t out_n,
             int64_t out_c, int64_t out_h, int64_t out_w, int64_t data_type,
             int64_t fmod) {
  OP_PROFILE(
      "mod",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld:%s%s", (long long)out_n,
                 (long long)out_c, (long long)out_h, (long long)out_w,
                 hipdnn_ep_datatype_name(data_type), fmod ? ":fmod" : ":pymod");
        return std::string(b);
      },
      state);

  if (!state || !lhs || !rhs || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_mod: null argument\n");
    return -1;
  }
  const int64_t out_vol = out_n * out_c * out_h * out_w;
  if (out_vol <= 0) {
    return 0;
  }

  int hip_dtype = mod_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_mod: unsupported data_type=%s(%lld)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  // ONNX Mod: fmod=0 requires integer; fmod=1 requires float.
  bool is_int = (data_type == HIPDNN_EP_DATATYPE_INT32 ||
                 data_type == HIPDNN_EP_DATATYPE_INT64);
  bool is_fp = (data_type == HIPDNN_EP_DATATYPE_HALF ||
                data_type == HIPDNN_EP_DATATYPE_FLOAT);
  if (fmod == 0 && !is_int) {
    fprintf(stderr,
            "[REAL] wrap_mod: fmod=0 requires integer data_type, got %s\n",
            hipdnn_ep_datatype_name(data_type));
    return -1;
  }
  if (fmod != 0 && !is_fp) {
    fprintf(stderr,
            "[REAL] wrap_mod: fmod=1 requires float data_type, got %s\n",
            hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  const bool lhs_eq_out =
      (a_n == out_n && a_c == out_c && a_h == out_h && a_w == out_w);
  const bool rhs_eq_out =
      (b_n == out_n && b_c == out_c && b_h == out_h && b_w == out_w);

  void *lhs_use = lhs;
  void *rhs_use = rhs;

  if (!lhs_eq_out || !rhs_eq_out) {
    const int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
    const size_t per_side = static_cast<size_t>(out_vol * elem_bytes);
    const size_t needed = per_side * static_cast<size_t>((!lhs_eq_out ? 1 : 0) +
                                                         (!rhs_eq_out ? 1 : 0));
    if (hipdnn_ep_state_ensure_workspace(state, needed) != 0) {
      fprintf(stderr, "[REAL] wrap_mod: workspace ensure failed (%zu bytes)\n",
              needed);
      return -1;
    }
    uint8_t *ws_byte =
        static_cast<uint8_t *>(hipdnn_ep_state_get_workspace(state));
    const int64_t out_shape[4] = {out_n, out_c, out_h, out_w};

    if (!lhs_eq_out) {
      const int64_t in_a[4] = {a_n, a_c, a_h, a_w};
      int rc =
          hip_expand(stream, lhs, ws_byte, in_a, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_mod: hip_expand(lhs) failed (%d)\n", rc);
        return -1;
      }
      lhs_use = ws_byte;
      ws_byte += per_side;
    }
    if (!rhs_eq_out) {
      const int64_t in_b[4] = {b_n, b_c, b_h, b_w};
      int rc =
          hip_expand(stream, rhs, ws_byte, in_b, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_mod: hip_expand(rhs) failed (%d)\n", rc);
        return -1;
      }
      rhs_use = ws_byte;
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_mod: out=[%lld,%lld,%lld,%lld] lhs%s rhs%s "
                    "data_type=%s fmod=%lld -> hip_elementwise_mod\n",
                    (long long)out_n, (long long)out_c, (long long)out_h,
                    (long long)out_w, lhs_eq_out ? "(ok)" : "(expanded)",
                    rhs_eq_out ? "(ok)" : "(expanded)",
                    hipdnn_ep_datatype_name(data_type), (long long)fmod);
  return hip_elementwise_mod(stream, lhs_use, rhs_use, output, out_vol,
                             hip_dtype, fmod ? 1 : 0);
}
