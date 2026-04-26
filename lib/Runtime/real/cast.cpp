/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"

#include <cstdint>
#include <cstdio>
#include <vector>

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
  case 5: // HIPDNN_EP_DATATYPE_INT8
    return HIP_DTYPE_INT8;
  case 6: // HIPDNN_EP_DATATYPE_UINT8 (if defined)
    return HIP_DTYPE_UINT8;
  default:
    return -1;
  }
}

int wrap_cast(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t src_data_type,
              int64_t dst_data_type) {
  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_cast: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

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
  if (rc != 0) {
    fprintf(stderr,
            "[cast] hip_cast FAILED rc=%d src=%lld(%d) dst=%lld(%d) n=%lld "
            "-- attempting host fallback\n",
            rc, (long long)src_data_type, src_hip_dtype,
            (long long)dst_data_type, dst_hip_dtype,
            (long long)num_elements);
    fflush(stderr);
    hipDeviceSynchronize();
    (void)hipGetLastError();

    int64_t src_elem = hipdnn_ep_datatype_size(src_data_type);
    int64_t dst_elem = hipdnn_ep_datatype_size(dst_data_type);
    if (src_elem > 0 && dst_elem > 0) {
      std::vector<uint8_t> h_in(num_elements * src_elem);
      std::vector<uint8_t> h_out(num_elements * dst_elem, 0);
      hipMemcpy(h_in.data(), input, h_in.size(), hipMemcpyDeviceToHost);

      auto as_f64 = [&](int64_t i, int64_t dt, const uint8_t *buf) -> double {
        switch (dt) {
        case HIPDNN_EP_DATATYPE_FLOAT: {
          float v;
          memcpy(&v, buf + i * 4, 4);
          return (double)v;
        }
        case HIPDNN_EP_DATATYPE_INT32: {
          int32_t v;
          memcpy(&v, buf + i * 4, 4);
          return (double)v;
        }
        case HIPDNN_EP_DATATYPE_INT64: {
          int64_t v;
          memcpy(&v, buf + i * 8, 8);
          return (double)v;
        }
        default:
          return 0.0;
        }
      };
      auto write = [&](int64_t i, int64_t dt, uint8_t *buf, double v) {
        switch (dt) {
        case HIPDNN_EP_DATATYPE_FLOAT: {
          float fv = (float)v;
          memcpy(buf + i * 4, &fv, 4);
          break;
        }
        case HIPDNN_EP_DATATYPE_INT32: {
          int32_t iv = (int32_t)v;
          memcpy(buf + i * 4, &iv, 4);
          break;
        }
        case HIPDNN_EP_DATATYPE_INT64: {
          int64_t iv = (int64_t)v;
          memcpy(buf + i * 8, &iv, 8);
          break;
        }
        default:
          break;
        }
      };

      for (int64_t i = 0; i < num_elements; i++)
        write(i, dst_data_type, h_out.data(),
              as_f64(i, src_data_type, h_in.data()));

      if (num_elements <= 1024) {
        fprintf(stderr, "[cast fallback] first 4 input bytes: ");
        for (int b = 0; b < 32 && b < (int)h_in.size(); b++)
          fprintf(stderr, "%02x ", h_in[b]);
        fprintf(stderr, "\n[cast fallback] first 4 output values: ");
        for (int i = 0; i < 4 && i < (int)num_elements; i++)
          fprintf(stderr, "%.6g ", as_f64(i, dst_data_type, h_out.data()));
        fprintf(stderr, "\n");
        fflush(stderr);
      }

      hipMemcpy(output, h_out.data(), h_out.size(), hipMemcpyHostToDevice);
      hipDeviceSynchronize();
      rc = 0;
    }
  }
  nan_trace_check("cast", output, num_elements);
  return rc;
}
