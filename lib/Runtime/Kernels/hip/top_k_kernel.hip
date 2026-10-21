/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kMaxRank = 8;

struct TopKGeom {
  int64_t rank;
  int64_t axis;
  int64_t k;
  int64_t in_shape[kMaxRank];
  int64_t in_stride[kMaxRank];
  int64_t out_stride[kMaxRank];
  int64_t num_rows;
  int64_t axis_size;
};

__device__ float topk_elem_to_float(float v) { return v; }

__device__ float topk_elem_to_float(double v) {
  return static_cast<float>(v);
}

__device__ float topk_elem_to_float(__half v) { return __half2float(v); }

__device__ void topk_store_value(float *values, int64_t offset, float v) {
  values[offset] = v;
}

__device__ void topk_store_value(double *values, int64_t offset, float v) {
  values[offset] = static_cast<double>(v);
}

__device__ void topk_store_value(__half *values, int64_t offset, float v) {
  values[offset] = __float2half(v);
}

template <typename T>
__device__ bool topk_better(float candidate, float best, bool largest) {
  return largest ? (candidate > best) : (candidate < best);
}

template <typename T>
__device__ void topk_row(const T *__restrict__ data, T *__restrict__ values,
                         int64_t *__restrict__ indices, TopKGeom geom,
                         int64_t row_id, int largest, int sorted) {
  (void)sorted;
  int64_t coords[kMaxRank] = {0};
  int64_t rem = row_id;
  for (int64_t d = geom.rank - 1; d >= 0; --d) {
    if (d == geom.axis)
      continue;
    int64_t dim = geom.in_shape[d];
    coords[d] = (dim > 0) ? (rem % dim) : 0;
    if (dim > 0)
      rem /= dim;
  }

  int64_t base_offset = 0;
  for (int64_t d = 0; d < geom.rank; ++d) {
    if (d == geom.axis)
      continue;
    base_offset += coords[d] * geom.in_stride[d];
  }

  int64_t k_eff = geom.k;
  if (k_eff > geom.axis_size)
    k_eff = geom.axis_size;
  if (k_eff <= 0)
    return;

  int64_t picked_idx[256];
  float picked_val[256];
  int64_t picked_count = 0;

  for (int64_t j = 0; j < k_eff; ++j) {
    float best_val = largest ? -INFINITY : INFINITY;
    int64_t best_idx = 0;
    bool found = false;

    for (int64_t i = 0; i < geom.axis_size; ++i) {
      bool already = false;
      for (int64_t p = 0; p < picked_count; ++p) {
        if (picked_idx[p] == i) {
          already = true;
          break;
        }
      }
      if (already)
        continue;

      T elem = data[base_offset + i * geom.in_stride[geom.axis]];
      float candidate = topk_elem_to_float(elem);
      if (!found || topk_better<T>(candidate, best_val, largest != 0)) {
        best_val = candidate;
        best_idx = i;
        found = true;
      }
    }

    if (!found)
      break;

    picked_idx[picked_count] = best_idx;
    picked_val[picked_count] = best_val;
    ++picked_count;
  }

  for (int64_t j = 0; j < picked_count; ++j) {
    coords[geom.axis] = j;
    int64_t out_offset = 0;
    for (int64_t d = 0; d < geom.rank; ++d)
      out_offset += coords[d] * geom.out_stride[d];
    topk_store_value(values, out_offset, picked_val[j]);
    indices[out_offset] = picked_idx[j];
  }
}

template <typename T>
__global__ void top_k_kernel(const T *__restrict__ data, T *__restrict__ values,
                             int64_t *__restrict__ indices, TopKGeom geom,
                             int largest, int sorted) {
  int64_t row_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_id >= geom.num_rows)
    return;
  topk_row(data, values, indices, geom, row_id, largest, sorted);
}

// Small last-axis rows use one cooperative block, not one serial thread.
// Equal values select the lower index. Keep the serial path's NaN behavior.
__global__ void top_k_small_f32(const float *data, float *values,
                              int64_t *indices, int width, int k, bool largest) {
  constexpr int threads = 128;
  const int row = blockIdx.x, tid = threadIdx.x;
  const float *input = data + static_cast<int64_t>(row) * width;
  float local[2];
  bool used[2];
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int index = tid + part * threads;
    used[part] = index >= width;
    local[part] = used[part] ? 0.0f : input[index];
  }
  __shared__ float waveValue[4];
  __shared__ int waveIndex[4], waveFirst[4], selected;
  for (int pick = 0; pick < k; ++pick) {
    int first = 256, best = 256;
    float value = 0.0f;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      const int index = tid + part * threads;
      if (!used[part]) {
        first = min(first, index);
        if (!isnan(local[part]) && (best == 256 ||
            (largest ? local[part] > value : local[part] < value))) {
          best = index;
          value = local[part];
        }
      }
    }
    for (int offset = warpSize / 2; offset; offset /= 2) {
      const float otherValue = __shfl_down(value, offset);
      const int otherIndex = __shfl_down(best, offset);
      first = min(first, __shfl_down(first, offset));
      if (otherIndex < 256 && (best == 256 ||
          (largest ? otherValue > value : otherValue < value) ||
          (otherValue == value && otherIndex < best))) {
        best = otherIndex;
        value = otherValue;
      }
    }
    if (tid % warpSize == 0) {
      waveValue[tid / warpSize] = value;
      waveIndex[tid / warpSize] = best;
      waveFirst[tid / warpSize] = first;
    }
    __syncthreads();
    if (tid == 0) {
      best = 256;
      first = 256;
      for (int wave = 0; wave < threads / warpSize; ++wave) {
        first = min(first, waveFirst[wave]);
        const int index = waveIndex[wave];
        const float candidate = waveValue[wave];
        if (index < 256 && (best == 256 ||
            (largest ? candidate > value : candidate < value) ||
            (candidate == value && index < best))) {
          best = index;
          value = candidate;
        }
      }
      // The serial search starts at the first unused index. A NaN there wins.
      selected = isnan(input[first]) ? first : best;
      values[static_cast<int64_t>(row) * k + pick] = input[selected];
      indices[static_cast<int64_t>(row) * k + pick] = selected;
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < 2; ++part)
      used[part] = used[part] || tid + part * threads == selected;
  }
}

template <typename T>
static int launch_top_k(void *stream, const void *data, void *values,
                        void *indices, const TopKGeom &geom, int largest,
                        int sorted) {
  if (geom.num_rows <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (geom.num_rows + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(top_k_kernel<T>), dim3(grid_size),
                   dim3(kBlockSize), 0, hip_stream,
                   static_cast<const T *>(data), static_cast<T *>(values),
                   static_cast<int64_t *>(indices), geom, largest, sorted);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] top_k_kernel launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

} // namespace

extern "C" int hip_top_k(void *stream, const void *data, void *values,
                         void *indices, int64_t axis, int64_t largest,
                         int64_t sorted, int64_t rank,
                         const int64_t *x_shape, int64_t k,
                         int element_size_bytes) {
  if (!data || !values || !indices || !x_shape)
    return -1;
  if (rank <= 0 || rank > kMaxRank)
    return -1;
  if (k <= 0)
    return 0;
  if (k > 256) {
    fprintf(stderr, "[custom_kernels] hip_top_k: K=%lld exceeds max 256\n",
            (long long)k);
    return -1;
  }

  TopKGeom geom{};
  geom.rank = rank;
  geom.axis = axis;
  if (geom.axis < 0)
    geom.axis += rank;
  if (geom.axis < 0 || geom.axis >= rank)
    return -1;

  geom.k = k;
  for (int64_t i = 0; i < rank; ++i)
    geom.in_shape[i] = x_shape[i];

  geom.in_stride[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i)
    geom.in_stride[i] = geom.in_stride[i + 1] * geom.in_shape[i + 1];

  geom.axis_size = geom.in_shape[geom.axis];
  int64_t k_eff = geom.k;
  if (k_eff > geom.axis_size)
    k_eff = geom.axis_size;

  int64_t out_shape[kMaxRank];
  for (int64_t i = 0; i < rank; ++i)
    out_shape[i] = geom.in_shape[i];
  out_shape[geom.axis] = k_eff;

  geom.out_stride[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i)
    geom.out_stride[i] = geom.out_stride[i + 1] * out_shape[i + 1];
  geom.num_rows = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == geom.axis)
      continue;
    geom.num_rows *= geom.in_shape[i];
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_top_k: axis=%lld, k=%lld, rank=%lld, "
      "rows=%lld, axis_size=%lld, elem=%d\n",
      (long long)geom.axis, (long long)geom.k, (long long)rank,
      (long long)geom.num_rows, (long long)geom.axis_size,
      element_size_bytes);

  if (element_size_bytes == 4 && geom.axis == rank - 1 &&
      geom.axis_size >= 1 && geom.axis_size <= 256 && k_eff <= 16 &&
      geom.num_rows > 0 && geom.num_rows <= 65535) {
    hipLaunchKernelGGL(top_k_small_f32, dim3(geom.num_rows), dim3(128), 0,
                      static_cast<hipStream_t>(stream),
                      static_cast<const float *>(data), static_cast<float *>(values),
                      static_cast<int64_t *>(indices), int(geom.axis_size), int(k_eff),
                      largest != 0);
    return hipGetLastError();
  }

  switch (element_size_bytes) {
  case 2:
    return launch_top_k<__half>(stream, data, values, indices, geom,
                                static_cast<int>(largest),
                                static_cast<int>(sorted));
  case 4:
    return launch_top_k<float>(stream, data, values, indices, geom,
                               static_cast<int>(largest),
                               static_cast<int>(sorted));
  case 8:
    return launch_top_k<double>(stream, data, values, indices, geom,
                              static_cast<int>(largest),
                              static_cast<int>(sorted));
  default:
    fprintf(stderr,
            "[custom_kernels] hip_top_k: unsupported element_size=%d\n",
            element_size_bytes);
    return -1;
  }
}
