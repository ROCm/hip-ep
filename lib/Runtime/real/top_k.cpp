/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kTopKDeviceMax = 256;
constexpr int kTopKMaxRank = 8;

int top_k_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int64_t topk_host_element_size(int hip_dtype) {
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    return 2;
  case HIP_DTYPE_FLOAT32:
    return 4;
  case HIP_DTYPE_FLOAT64:
    return 8;
  default:
    return 0;
  }
}

double topk_key(const char *element, int hip_dtype) {
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    __half value;
    memcpy(&value, element, sizeof(value));
    return __half2float(value);
  }
  if (hip_dtype == HIP_DTYPE_FLOAT64) {
    double value;
    memcpy(&value, element, sizeof(value));
    return value;
  }
  float value;
  memcpy(&value, element, sizeof(value));
  return value;
}

// The device kernel covers K up to 256 and, above that, full sorts of a
// contiguous last axis. Other large K (partial TopK, a non-last axis) and f64,
// which the device kernel would compare in f32, are selected here.
int host_top_k(hipStream_t stream, const void *x, void *values, void *indices,
               int64_t axis, int64_t largest, int64_t rank,
               const int64_t *x_shape, int64_t num_elements, int64_t k,
               int hip_dtype) {
  const int64_t element_size = topk_host_element_size(hip_dtype);
  if (element_size == 0) {
    fprintf(stderr, "[REAL] wrap_top_k: unsupported host hip_dtype=%d\n",
            hip_dtype);
    return -1;
  }
  if (axis < 0)
    axis += rank;
  if (rank <= 0 || rank > kTopKMaxRank || axis < 0 || axis >= rank || k < 0)
    return -1;
  int64_t axis_size = x_shape[axis];
  if (axis_size < 0 || k > axis_size)
    return -1;
  int64_t num_rows = 1;
  for (int64_t dim = 0; dim < rank; ++dim)
    if (dim != axis)
      num_rows *= x_shape[dim];
  if (num_rows < 0 || axis_size < 0 || num_elements < num_rows * axis_size)
    return -1;

  int64_t in_stride[kTopKMaxRank];
  int64_t out_stride[kTopKMaxRank];
  int64_t out_shape[kTopKMaxRank];
  in_stride[rank - 1] = 1;
  out_stride[rank - 1] = 1;
  for (int64_t dim = 0; dim < rank; ++dim)
    out_shape[dim] = dim == axis ? k : x_shape[dim];
  for (int64_t dim = rank - 2; dim >= 0; --dim) {
    in_stride[dim] = in_stride[dim + 1] * x_shape[dim + 1];
    out_stride[dim] = out_stride[dim + 1] * out_shape[dim + 1];
  }

  std::vector<char> host_x(static_cast<size_t>(num_elements) * element_size);
  std::vector<char> host_values(static_cast<size_t>(num_rows * k) *
                                element_size);
  std::vector<int64_t> host_indices(static_cast<size_t>(num_rows * k));
  if (!host_x.empty() &&
      hipMemcpyAsync(host_x.data(), x, host_x.size(), hipMemcpyDeviceToHost,
                     stream) != hipSuccess)
    return -1;
  if (hipStreamSynchronize(stream) != hipSuccess)
    return -1;

  std::vector<int64_t> order(static_cast<size_t>(axis_size));
  const char *input = host_x.data();
  for (int64_t row = 0; row < num_rows; ++row) {
    int64_t coords[kTopKMaxRank] = {};
    int64_t rest = row;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      if (dim == axis)
        continue;
      coords[dim] = x_shape[dim] > 0 ? rest % x_shape[dim] : 0;
      if (x_shape[dim] > 0)
        rest /= x_shape[dim];
    }
    int64_t base = 0;
    for (int64_t dim = 0; dim < rank; ++dim)
      if (dim != axis)
        base += coords[dim] * in_stride[dim];
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + k, order.end(),
        [&](int64_t lhs, int64_t rhs) {
          double left = topk_key(
              input + (base + lhs * in_stride[axis]) * element_size, hip_dtype);
          double right = topk_key(
              input + (base + rhs * in_stride[axis]) * element_size, hip_dtype);
          if (left != right)
            return largest ? left > right : left < right;
          return lhs < rhs;
        });
    int64_t out_base = 0;
    for (int64_t dim = 0; dim < rank; ++dim)
      if (dim != axis)
        out_base += coords[dim] * out_stride[dim];
    for (int64_t slot = 0; slot < k; ++slot) {
      int64_t index = order[static_cast<size_t>(slot)];
      memcpy(host_values.data() +
                 (out_base + slot * out_stride[axis]) * element_size,
             input + (base + index * in_stride[axis]) * element_size,
             static_cast<size_t>(element_size));
      host_indices[static_cast<size_t>(out_base + slot * out_stride[axis])] =
          index;
    }
  }

  if ((!host_values.empty() &&
       hipMemcpyAsync(values, host_values.data(), host_values.size(),
                      hipMemcpyHostToDevice, stream) != hipSuccess) ||
      (!host_indices.empty() &&
       hipMemcpyAsync(indices, host_indices.data(),
                      host_indices.size() * sizeof(int64_t),
                      hipMemcpyHostToDevice, stream) != hipSuccess) ||
      hipStreamSynchronize(stream) != hipSuccess)
    return -1;
  return 0;
}

} // namespace

int wrap_top_k(RuntimeState *state, void *x, void *k, void *values,
               void *indices, int64_t axis, int64_t largest, int64_t sorted,
               int64_t rank, const int64_t *x_shape, int64_t num_elements,
               int64_t data_type) {
  OP_PROFILE(
      "top_k",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:axis=%lld:%s", (long long)rank,
                 (long long)axis, hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  // main_graph drops this return value, so a nonzero status must also land in
  // the device error flag that inference_compute reads after the stream sync.
  auto finish = [&](int status) {
    if (status != 0)
      hipdnn_ep_state_set_error_flag(state, status);
    return status;
  };

  if (!state || !x || !k || !values || !indices || !x_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_top_k: null argument\n");
    return finish(-1);
  }

  int hip_dtype = top_k_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_top_k: unsupported data_type=%lld\n",
            (long long)data_type);
    return finish(-1);
  }

  int64_t k_val = 0;
  void *stream = hipdnn_ep_state_get_stream(state);
  hipError_t err = hipMemcpy(&k_val, k, sizeof(int64_t), hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_top_k: failed to read K: %s\n",
            hipGetErrorString(err));
    return finish(-1);
  }
  err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_top_k: stream sync failed: %s\n",
            hipGetErrorString(err));
    return finish(-1);
  }

  auto run_on_host = [&] {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_top_k: axis=%lld, k=%lld, dtype=%s -> host\n",
        (long long)axis, (long long)k_val, hipdnn_ep_datatype_name(data_type));
    return finish(host_top_k(static_cast<hipStream_t>(stream), x, values,
                             indices, axis, largest, rank, x_shape,
                             num_elements, k_val, hip_dtype));
  };

  const bool device_dtype =
      hip_dtype == HIP_DTYPE_FLOAT16 || hip_dtype == HIP_DTYPE_FLOAT32;
  if (!device_dtype || !hip_top_k_device_supported(axis, rank, x_shape, k_val))
    return run_on_host();

  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  const size_t needed = hip_top_k_workspace_bytes(axis, rank, x_shape, k_val);
  if (needed > 0 && hipdnn_ep_state_ensure_workspace(state, needed) == 0) {
    workspace = hipdnn_ep_state_get_workspace(state);
    workspace_bytes = hipdnn_ep_state_get_workspace_size(state);
  }
  // The device full sort needs its key workspace; the K <= 256 paths do not.
  if (k_val > kTopKDeviceMax && workspace_bytes < needed)
    return run_on_host();

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_top_k: axis=%lld, k=%lld, rank=%lld, largest=%lld, "
      "sorted=%lld, dtype=%s, workspace=%zu/%zu -> hip_top_k\n",
      (long long)axis, (long long)k_val, (long long)rank, (long long)largest,
      (long long)sorted, hipdnn_ep_datatype_name(data_type), workspace_bytes,
      needed);

  return finish(hip_top_k(stream, x, values, indices, axis, largest, sorted,
                          rank, x_shape, k_val, hip_dtype, workspace,
                          workspace_bytes));
}
