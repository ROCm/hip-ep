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
// Same-shape constraint identical to Div / Equal / Less.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

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
             int64_t num_elements, int64_t data_type, int64_t fmod) {
  OP_PROFILE(
      "mod",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lld:%s%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type), fmod ? ":fmod" : ":pymod");
        return std::string(b);
      },
      state);

  if (!state || !lhs || !rhs || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_mod: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  int hip_dtype = mod_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    hipdnn_ep_log_emit("[REAL] wrap_mod: unsupported data_type=%s(%lld)\n",
                       hipdnn_ep_datatype_name(data_type),
                       (long long)data_type);
    return -1;
  }

  // ONNX Mod: fmod=0 requires integer; fmod=1 requires float.
  bool is_int = (data_type == HIPDNN_EP_DATATYPE_INT32 ||
                 data_type == HIPDNN_EP_DATATYPE_INT64);
  bool is_fp = (data_type == HIPDNN_EP_DATATYPE_HALF ||
                data_type == HIPDNN_EP_DATATYPE_FLOAT);
  if (fmod == 0 && !is_int) {
    hipdnn_ep_log_emit(
        "[REAL] wrap_mod: fmod=0 requires integer data_type, got %s\n",
        hipdnn_ep_datatype_name(data_type));
    return -1;
  }
  if (fmod != 0 && !is_fp) {
    hipdnn_ep_log_emit(
        "[REAL] wrap_mod: fmod=1 requires float data_type, got %s\n",
        hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG("[REAL] wrap_mod: num=%lld, data_type=%s, fmod=%lld -> "
                    "hip_elementwise_mod\n",
                    (long long)num_elements, hipdnn_ep_datatype_name(data_type),
                    (long long)fmod);
  return hip_elementwise_mod(stream, lhs, rhs, output, num_elements, hip_dtype,
                             fmod ? 1 : 0);
}
