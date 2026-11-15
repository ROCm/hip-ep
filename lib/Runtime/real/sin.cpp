/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Sin: y = sin(x) (element-wise).
//
// Source: onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.cu
//         @ v1.22.2 (UNARY_OP_NAME_EXPR(Sin, _Sin(a)),
//                    SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Sin))
//         + core/providers/cuda/cu_inc/common.cuh _Sin<half> / _Sin<float>.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int sin_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  default:
    return -1;
  }
}

int wrap_sin(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "sin",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_sin: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  int hip_dtype = sin_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_sin: unsupported data_type=%s(%lld) "
            "(supported: f16, f32)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_sin: num=%lld, data_type=%s -> hip_elementwise_sin\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));
  int rc = hip_elementwise_sin(stream, input, output, num_elements, hip_dtype);
  if (rc == 0 && _opProf && op_profile_call_capture_enabled(_opProf->ps)) {
    using Json = CallCaptureJson;
    const char *dtype =
        data_type == HIPDNN_EP_DATATYPE_HALF ? "float16" : "float32";
    auto tensor = [dtype, num_elements](const char *role) {
      return Json::object(
          {{"slot", 0},
           {"role", role},
           {"present", true},
           {"dtype", dtype},
           {"shape", Json::dimensions({num_elements})},
           {"layout", Json::object({{"contiguous", true},
                                    {"view", "production_abi_view"},
                                    {"order", "flat"}})}});
    };
    _opProf->capture = op_profile_capture_call(
        _opProf->ps,
        Json::object(
            {{"op_type", "Sin"},
             {"execution",
              Json::object(
                  {{"backend", "HIP-EP"},
                   {"entry", "wrap_sin"},
                   {"variant", "flat_unary"},
                   {"config",
                    Json::object({{"hip_dtype", hip_dtype},
                                  {"kernel_entry", "hip_elementwise_sin"}})}})},
             {"inputs", Json::array({tensor("input")})},
             {"outputs", Json::array({tensor("output")})},
             {"attributes", Json::object({})},
             {"parameters", Json::object({{"data_type", data_type},
                                          {"num_elements", num_elements}})}}));
  }
  return rc;
}
