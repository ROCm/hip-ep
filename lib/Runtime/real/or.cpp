/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Or: y = a | b (element-wise logical OR on bool tensors) with ONNX
// multidirectional broadcast (rank <= 4).
//
// Source:
// onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.{h,cu}
//         @ v1.22.2 (BINARY_OP_NAME_EXPR(Or, (a | b)),
//                    SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Or, bool))
//
// Operand shapes are passed as 4D (N, C, H, W); the compiler left-pads
// ranks 1-3 with 1. When lhs or rhs does not match the output shape,
// hip_expand materialises the broadcast into the per-session workspace,
// then the flat hip_elementwise_or runs on same-shape buffers (mirrors
// wrap_div). Bool is a 1-byte (uint8) stream on the GPU side.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>

int wrap_or(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
            int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
            int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
            int64_t out_h, int64_t out_w, int64_t data_type) {
  OP_PROFILE(
      "or",
      [&] {
        char buf[64];
        snprintf(buf, sizeof(buf), "%lldx%lldx%lldx%lld", (long long)out_n,
                 (long long)out_c, (long long)out_h, (long long)out_w);
        return std::string(buf);
      },
      state);

  if (!state || !a || !b || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_or: null argument\n");
    return -1;
  }

  // ONNX `Or` is bool-only; the wire format is always 1 byte per element,
  // so the HIPDNN dtype (passed as a sentinel 0 for i1) is not consulted.
  (void)data_type;

  const int64_t out_vol = out_n * out_c * out_h * out_w;
  if (out_vol <= 0)
    return 0;

  void *stream = hipdnn_ep_state_get_stream(state);

  const bool lhs_eq_out =
      (a_n == out_n && a_c == out_c && a_h == out_h && a_w == out_w);
  const bool rhs_eq_out =
      (b_n == out_n && b_c == out_c && b_h == out_h && b_w == out_w);

  void *a_use = a;
  void *b_use = b;

  if (!lhs_eq_out || !rhs_eq_out) {
    // Bool payload is 1 byte/element; expand via the uint8 kernel path.
    const size_t per_side = static_cast<size_t>(out_vol);
    const size_t needed = per_side * static_cast<size_t>((!lhs_eq_out ? 1 : 0) +
                                                         (!rhs_eq_out ? 1 : 0));
    if (hipdnn_ep_state_ensure_workspace(state, needed) != 0) {
      fprintf(stderr, "[REAL] wrap_or: workspace ensure failed (%zu bytes)\n",
              needed);
      return -1;
    }
    uint8_t *ws_byte =
        static_cast<uint8_t *>(hipdnn_ep_state_get_workspace(state));
    const int64_t out_shape[4] = {out_n, out_c, out_h, out_w};

    if (!lhs_eq_out) {
      const int64_t in_a[4] = {a_n, a_c, a_h, a_w};
      int rc = hip_expand(stream, a, ws_byte, in_a, 4, out_shape, 4,
                          HIP_DTYPE_UINT8);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_or: hip_expand(a) failed (%d)\n", rc);
        return -1;
      }
      a_use = ws_byte;
      ws_byte += per_side;
    }
    if (!rhs_eq_out) {
      const int64_t in_b[4] = {b_n, b_c, b_h, b_w};
      int rc = hip_expand(stream, b, ws_byte, in_b, 4, out_shape, 4,
                          HIP_DTYPE_UINT8);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_or: hip_expand(b) failed (%d)\n", rc);
        return -1;
      }
      b_use = ws_byte;
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_or: out=[%lld,%lld,%lld,%lld] lhs%s rhs%s "
                    "-> hip_elementwise_or\n",
                    (long long)out_n, (long long)out_c, (long long)out_h,
                    (long long)out_w, lhs_eq_out ? "(ok)" : "(expanded)",
                    rhs_eq_out ? "(ok)" : "(expanded)");
  return hip_elementwise_or(stream, a_use, b_use, output, out_vol);
}
