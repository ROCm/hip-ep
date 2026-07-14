/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Equal: y = (a == b)  -- element-wise, bool (1-byte) output, with ONNX
// multidirectional broadcast (rank <= 4).
//
// `data_type` refers to the INPUT type (the comparison operand type). The
// output is always 1 byte per element.
//
// Source: onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.cu
//         @ v1.22.2 (BINARY_OP_NAME_EXPR2(Equal, (a == b)))
//
// Broadcast handling (operand shapes are 4D, left-padded with 1 by the
// compiler):
//   * same-shape / scalar operand -> handled directly by the kernel
//     (a scalar, num==1, reads through a zero stride).
//   * any other partial broadcast (e.g. [1,8,1] vs [1,1,8]) -> materialised
//     to the output shape via hip_expand into the per-session workspace,
//     then compared on same-shape buffers. Without this, the flat kernel
//     indexes a partially-broadcast operand past its end.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>

static int equal_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_equal(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
               int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
               int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
               int64_t out_h, int64_t out_w, int64_t data_type) {
  OP_PROFILE(
      "equal",
      [&] {
        char buf[80];
        snprintf(buf, sizeof(buf), "%lldx%lldx%lldx%lld:%s", (long long)out_n,
                 (long long)out_c, (long long)out_h, (long long)out_w,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(buf);
      },
      state);

  if (!state || !a || !b || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_equal: null argument\n");
    return -1;
  }

  const int64_t out_vol = out_n * out_c * out_h * out_w;
  if (out_vol <= 0)
    return 0;

  int hip_dtype = equal_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_equal: unsupported input data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  const int64_t a_vol = a_n * a_c * a_h * a_w;
  const int64_t b_vol = b_n * b_c * b_h * b_w;

  // An operand is served directly (no expand) when it already matches the
  // output volume, or when it is a single scalar (the kernel reads it through
  // a zero stride). Anything else is a partial broadcast that must be
  // materialised so the flat kernel never indexes past the operand's end.
  const bool a_direct = (a_vol == out_vol) || (a_vol == 1);
  const bool b_direct = (b_vol == out_vol) || (b_vol == 1);

  void *stream = hipdnn_ep_state_get_stream(state);
  void *a_use = a;
  void *b_use = b;
  int64_t a_num = a_vol;
  int64_t b_num = b_vol;

  if (!a_direct || !b_direct) {
    const int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
    const size_t per_side = static_cast<size_t>(out_vol * elem_bytes);
    const size_t needed = per_side * static_cast<size_t>((!a_direct ? 1 : 0) +
                                                         (!b_direct ? 1 : 0));
    if (hipdnn_ep_state_ensure_workspace(state, needed) != 0) {
      fprintf(stderr, "[REAL] wrap_equal: workspace ensure failed (%zu bytes)\n",
              needed);
      return -1;
    }
    uint8_t *ws_byte = static_cast<uint8_t *>(hipdnn_ep_state_get_workspace(state));
    const int64_t out_shape[4] = {out_n, out_c, out_h, out_w};

    if (!a_direct) {
      const int64_t in_a[4] = {a_n, a_c, a_h, a_w};
      int rc =
          hip_expand(stream, a, ws_byte, in_a, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_equal: hip_expand(a) failed (%d)\n", rc);
        return -1;
      }
      a_use = ws_byte;
      a_num = out_vol;
      ws_byte += per_side;
    }
    if (!b_direct) {
      const int64_t in_b[4] = {b_n, b_c, b_h, b_w};
      int rc =
          hip_expand(stream, b, ws_byte, in_b, 4, out_shape, 4, hip_dtype);
      if (rc != 0) {
        fprintf(stderr, "[REAL] wrap_equal: hip_expand(b) failed (%d)\n", rc);
        return -1;
      }
      b_use = ws_byte;
      b_num = out_vol;
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_equal: out=[%lld,%lld,%lld,%lld] a%s b%s "
                    "input_type=%s -> hip_elementwise_equal\n",
                    (long long)out_n, (long long)out_c, (long long)out_h,
                    (long long)out_w, a_direct ? "(direct)" : "(expanded)",
                    b_direct ? "(direct)" : "(expanded)",
                    hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_equal(stream, a_use, b_use, output, a_num, b_num,
                               out_vol, hip_dtype);
}
