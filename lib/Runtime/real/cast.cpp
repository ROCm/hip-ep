/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdio>

// Map HIPDNN_EP_DATATYPE_* → hip_dtype_t for custom kernels.
// The two enum systems use different orderings (e.g. bf16=2 vs 5, i64=4 vs 2).
static int hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  default:
    return -1;
  }
}

int wrap_cast(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t src_data_type,
              int64_t dst_data_type) {
  OP_PROFILE(
      "cast",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_cast: null argument (state=%p input=%p output=%p "
        "n=%lld src=%lld dst=%lld)\n",
        (void *)state, input, output, (long long)num_elements,
        (long long)src_data_type, (long long)dst_data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  // Same-dtype fast-path: ONNX exporters (notably the Qwen3.5-VL vision
  // encoder) sometimes emit redundant `Cast(x, to=src_dtype)` nodes that
  // survive into the EP IR (e.g. fp32 LayerNorm scale/bias wrapped in
  // `Cast(fp32_const, to=fp32)` on every transformer block). The custom kernel
  // dispatch table only covers genuine type-changing casts and would return -1
  // here for src==dst, which would silently leave `output` untouched (= the
  // pool's zero-initialized contents) and propagate zeros through downstream
  // ops (canonical symptom: LayerNorm sees a 0 scale/bias and emits a
  // row-broadcast tensor). A device-to-device byte-copy is the correct identity
  // semantics regardless of dtype, so handle this case before the kernel
  // dispatch.
  if (src_data_type == dst_data_type) {
    int64_t elem_size = hipdnn_ep_datatype_size(src_data_type);
    if (elem_size <= 0) {
      fprintf(stderr,
              "[REAL] wrap_cast same-dtype: unsupported data type %lld\n",
              (long long)src_data_type);
      return -1;
    }
    if (input == output) {
      // Identity in-place: nothing to do.
      return 0;
    }
    size_t bytes = static_cast<size_t>(num_elements) * elem_size;
    hipError_t err =
        hipMemcpyAsync(output, input, bytes, hipMemcpyDeviceToDevice,
                       static_cast<hipStream_t>(stream));
    if (err != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_cast same-dtype hipMemcpyAsync failed: %s "
              "(n=%lld dtype=%lld)\n",
              hipGetErrorString(err), (long long)num_elements,
              (long long)src_data_type);
      return -1;
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_cast same-dtype D2D copy: n=%lld dtype=%s(%lld) "
        "bytes=%zu\n",
        (long long)num_elements, hipdnn_ep_datatype_name(src_data_type),
        (long long)src_data_type, bytes);
    return 0;
  }

  int src_hip_dtype = hipdnn_to_hip_dtype(src_data_type);
  int dst_hip_dtype = hipdnn_to_hip_dtype(dst_data_type);

  if (src_hip_dtype < 0 || dst_hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_cast: unsupported data type src=%lld dst=%lld\n",
            (long long)src_data_type, (long long)dst_data_type);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_cast: num_elements=%lld, src=%s(%lld), "
      "dst=%s(%lld)\n",
      (long long)num_elements, hipdnn_ep_datatype_name(src_data_type),
      (long long)src_data_type, hipdnn_ep_datatype_name(dst_data_type),
      (long long)dst_data_type);

  int rc = hip_cast(stream, input, output, num_elements, src_hip_dtype,
                    dst_hip_dtype);
  if (rc == 0 && num_elements > 0 && _opProf &&
      op_profile_call_capture_enabled(_opProf->ps)) {
    using Json = CallCaptureJson;
    auto dtypeName = [](int dtype) -> const char * {
      switch (dtype) {
      case HIP_DTYPE_FLOAT32:
        return "float32";
      case HIP_DTYPE_FLOAT16:
        return "float16";
      case HIP_DTYPE_BFLOAT16:
        return "bfloat16";
      case HIP_DTYPE_INT64:
        return "int64";
      case HIP_DTYPE_INT32:
        return "int32";
      case HIP_DTYPE_INT16:
        return "int16";
      case HIP_DTYPE_INT8:
        return "int8";
      case HIP_DTYPE_UINT8:
        return "uint8";
      default:
        return nullptr;
      }
    };
    auto tensor = [num_elements](const char *role, const char *dtype) {
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
    const char *src_dtype = dtypeName(src_hip_dtype);
    const char *dst_dtype = dtypeName(dst_hip_dtype);
    if (src_dtype && dst_dtype) {
      _opProf->capture = op_profile_capture_call(
          _opProf->ps,
          Json::object(
              {{"op_type", "Cast"},
               {"execution",
                Json::object(
                    {{"backend", "HIP-EP"},
                     {"entry", "wrap_cast"},
                     {"variant", "flat_dtype_cast"},
                     {"config",
                      Json::object({{"input_hip_dtype", src_hip_dtype},
                                    {"output_hip_dtype", dst_hip_dtype},
                                    {"kernel_entry", "hip_cast"}})}})},
               {"inputs", Json::array({tensor("input", src_dtype)})},
               {"outputs", Json::array({tensor("output", dst_dtype)})},
               {"attributes", Json::object({})},
               {"parameters",
                Json::object({{"num_elements", num_elements},
                              {"src_data_type", src_data_type},
                              {"dst_data_type", dst_data_type}})}}));
    }
  }
  return rc;
}
