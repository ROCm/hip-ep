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
  // Host scratch is not GPU-visible at the host address on this target
  // (hipHostGetDevicePointer returns that same address). Copy it into the
  // device mirror and pass the mirror pointer to the kernel.
  if (hipdnn_ep_copy_scratch_to_device(state, stream) != 0)
    return -1;
  void *dev_input = hipdnn_ep_as_device_ptr(state, input);
  void *dev_output = hipdnn_ep_as_device_ptr(state, output);

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
    hipError_t err = hipMemcpyAsync(
        dev_output, dev_input, bytes, hipMemcpyDeviceToDevice,
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
    if (dev_output != output)
      hipdnn_ep_copy_scratch_from_device(state, stream, output, bytes);
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

  int64_t src_size = hipdnn_ep_datatype_size(src_data_type);
  int64_t dst_size = hipdnn_ep_datatype_size(dst_data_type);
  size_t in_bytes = src_size > 0 ? static_cast<size_t>(num_elements) *
                                       static_cast<size_t>(src_size)
                                 : 0;
  size_t out_bytes = dst_size > 0 ? static_cast<size_t>(num_elements) *
                                        static_cast<size_t>(dst_size)
                                  : 0;
  void *in_staged = nullptr;
  void *out_staged = nullptr;
  void *kernel_input = hipdnn_ep_ensure_device_ptr(stream, dev_input, in_bytes,
                                                   /*copy_to_device=*/1,
                                                   &in_staged);
  void *kernel_output = hipdnn_ep_ensure_device_ptr(
      stream, dev_output, out_bytes, /*copy_to_device=*/0, &out_staged);
  int rc = hip_cast(stream, kernel_input, kernel_output, num_elements,
                    src_hip_dtype, dst_hip_dtype);
  if (rc == 0 && out_staged) {
    (void)hipMemcpyAsync(output, out_staged, out_bytes, hipMemcpyDeviceToHost,
                         static_cast<hipStream_t>(stream));
  } else if (rc == 0 && dev_output != output && dst_size > 0) {
    hipdnn_ep_copy_scratch_from_device(state, stream, output, out_bytes);
  }
  if (in_staged || out_staged) {
    (void)hipStreamSynchronize(static_cast<hipStream_t>(stream));
    if (in_staged)
      (void)hipFree(in_staged);
    if (out_staged)
      (void)hipFree(out_staged);
  }
  return rc;
}
