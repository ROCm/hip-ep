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
#include <vector>

namespace {

constexpr int kTopKDeviceMax = 256;
constexpr int kTopKMaxRank = 8;

double topk_key(const char *element, int64_t element_size) {
  if (element_size == 2) {
    __half value;
    memcpy(&value, element, sizeof(value));
    return __half2float(value);
  }
  if (element_size == 8) {
    double value;
    memcpy(&value, element, sizeof(value));
    return value;
  }
  float value;
  memcpy(&value, element, sizeof(value));
  return value;
}

// The device kernel keeps K winners in a fixed 256-slot list. A larger K,
// which the SSD postprocessor uses to sort a whole class, is selected here.
int host_top_k(hipStream_t stream, const void *x, void *values, void *indices,
               int64_t axis, int64_t largest, int64_t rank,
               const int64_t *x_shape, int64_t num_elements, int64_t k,
               int64_t element_size) {
  if (element_size != 2 && element_size != 4 && element_size != 8)
    return -1;
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
          double left =
              topk_key(input + (base + lhs * in_stride[axis]) * element_size,
                       element_size);
          double right =
              topk_key(input + (base + rhs * in_stride[axis]) * element_size,
                       element_size);
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
               int64_t element_size_bytes) {
  OP_PROFILE(
      "top_k",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:axis=%lld", (long long)rank,
                 (long long)axis);
        return std::string(b);
      },
      state);

  (void)num_elements;
  if (!state || !x || !k || !values || !indices || !x_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_top_k: null argument\n");
    return -1;
  }

  int64_t k_val = 0;
  void *stream = hipdnn_ep_state_get_stream(state);
  hipError_t err = hipMemcpy(&k_val, k, sizeof(int64_t), hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_top_k: failed to read K: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_top_k: stream sync failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  if (k_val > kTopKDeviceMax)
    return host_top_k(static_cast<hipStream_t>(stream), x, values, indices,
                      axis, largest, rank, x_shape, num_elements, k_val,
                      element_size_bytes);

  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  const size_t needed = hip_top_k_workspace_bytes(axis, rank, x_shape, k_val);
  if (needed > 0 && hipdnn_ep_state_ensure_workspace(state, needed) == 0) {
    workspace = hipdnn_ep_state_get_workspace(state);
    workspace_bytes = hipdnn_ep_state_get_workspace_size(state);
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_top_k: axis=%lld, k=%lld, rank=%lld, largest=%lld, "
      "sorted=%lld, workspace=%zu/%zu -> hip_top_k\n",
      (long long)axis, (long long)k_val, (long long)rank, (long long)largest,
      (long long)sorted, workspace_bytes, needed);

  return hip_top_k(stream, x, values, indices, axis, largest, sorted, rank,
                   x_shape, k_val, static_cast<int>(element_size_bytes),
                   workspace, workspace_bytes);
}
