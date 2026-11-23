/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <limits>

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

__device__ float topk_elem_to_float(__half v) { return __half2float(v); }

__device__ void topk_store_value(float *values, int64_t offset, float v) {
  values[offset] = v;
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

// K == 1 over the contiguous last axis (argmax / argmin of LLM logits): one
// block per row, coalesced strided loads, shared-memory (value, index)
// reduction. Ties keep the lowest index, as topk_row does. The one-thread-per-
// row kernel above scans a 248k-wide vocab row serially (~30 ms per call).
constexpr int kTop1Block = 1024;

template <typename T>
__global__ void __launch_bounds__(kTop1Block)
    top1_last_axis_kernel(const T *__restrict__ data, T *__restrict__ values,
                          int64_t *__restrict__ indices, int64_t axis_size,
                          int largest) {
  __shared__ float s_val[kTop1Block];
  __shared__ int64_t s_idx[kTop1Block];
  const T *row = data + static_cast<int64_t>(blockIdx.x) * axis_size;
  const bool lg = largest != 0;
  float best = lg ? -INFINITY : INFINITY;
  int64_t best_i = axis_size;
  for (int64_t i = threadIdx.x; i < axis_size; i += kTop1Block) {
    float v = topk_elem_to_float(row[i]);
    if (best_i == axis_size || (lg ? v > best : v < best)) {
      best = v;
      best_i = i;
    }
  }
  s_val[threadIdx.x] = best;
  s_idx[threadIdx.x] = best_i;
  __syncthreads();
  for (int s = kTop1Block / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      float ov = s_val[threadIdx.x + s];
      int64_t oi = s_idx[threadIdx.x + s];
      float mv = s_val[threadIdx.x];
      int64_t mi = s_idx[threadIdx.x];
      bool take = (mi == axis_size) ||
                  (oi != axis_size &&
                   ((lg ? ov > mv : ov < mv) || (ov == mv && oi < mi)));
      if (take) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    topk_store_value(values, blockIdx.x, s_val[0]);
    indices[blockIdx.x] = s_idx[0];
  }
}

// 2 <= K <= 32 over the contiguous last axis (draft candidate lists over a
// vocab row): one block per row. Each thread keeps the sorted top-K of its
// strided slice in registers, then K rounds of a block-wide (key, index) max
// over the thread list heads emit the exact top-K in sorted order. Keys are
// negated for largest=0 so both directions select the max; ties keep the
// lowest index, as topk_row does.
constexpr int kTopKSmallBlock = 256;
constexpr int kTopKSmallMax = 32;

template <typename T>
__global__ void __launch_bounds__(kTopKSmallBlock)
    topk_small_last_axis_kernel(const T *__restrict__ data,
                                T *__restrict__ values,
                                int64_t *__restrict__ indices,
                                int64_t axis_size, int k, int largest) {
  __shared__ float s_key[kTopKSmallBlock];
  __shared__ int s_idx[kTopKSmallBlock];
  __shared__ int s_tid[kTopKSmallBlock];
  const T *row = data + static_cast<int64_t>(blockIdx.x) * axis_size;
  const bool lg = largest != 0;
  const int t = threadIdx.x;
  float lkey[kTopKSmallMax];
  int lidx[kTopKSmallMax];
  int cnt = 0;
  float thr = -INFINITY; // key of the K-th entry once the list is full
  for (int64_t i = t; i < axis_size; i += kTopKSmallBlock) {
    float v = topk_elem_to_float(row[i]);
    float key = lg ? v : -v;
    if (cnt == k && !(key > thr))
      continue;
    int pos = cnt < k ? cnt : k - 1;
    while (pos > 0 && lkey[pos - 1] < key) {
      lkey[pos] = lkey[pos - 1];
      lidx[pos] = lidx[pos - 1];
      --pos;
    }
    lkey[pos] = key;
    lidx[pos] = static_cast<int>(i);
    if (cnt < k)
      ++cnt;
    if (cnt == k)
      thr = lkey[k - 1];
  }
  int head = 0;
  const int64_t out_base = static_cast<int64_t>(blockIdx.x) * k;
  for (int j = 0; j < k; ++j) {
    s_key[t] = head < cnt ? lkey[head] : -INFINITY;
    s_idx[t] = head < cnt ? lidx[head] : INT32_MAX;
    s_tid[t] = t;
    __syncthreads();
    for (int s = kTopKSmallBlock / 2; s > 0; s >>= 1) {
      if (t < s) {
        float ok = s_key[t + s];
        int oi = s_idx[t + s];
        float mk = s_key[t];
        int mi = s_idx[t];
        if (ok > mk || (ok == mk && oi < mi)) {
          s_key[t] = ok;
          s_idx[t] = oi;
          s_tid[t] = s_tid[t + s];
        }
      }
      __syncthreads();
    }
    const int win = s_tid[0];
    const float wk = s_key[0];
    const int wi = s_idx[0];
    __syncthreads();
    if (t == win)
      ++head;
    if (t == 0) {
      topk_store_value(values, out_base + j, lg ? wk : -wk);
      indices[out_base + j] = wi;
    }
  }
}

// Long rows (a full vocab) leave one block per row latency-bound, so they go
// through two passes instead: pass 1 splits each row into chunks of
// kTopKChunk elements and writes every chunk's sorted top-K (key, index) to a
// scratch buffer; pass 2 merges the chunks' lists, one block per row. Keys are
// compared as (key desc, index asc) throughout so ties resolve to the lowest
// index exactly as the one-pass kernel does.
constexpr int64_t kTopKChunk = 4096;

__device__ __forceinline__ bool topk_better(float ka, int ia, float kb,
                                            int ib) {
  return ka > kb || (ka == kb && ia < ib);
}

// Block-wide sorted top-K of n (key, index) pairs produced by load(j); the
// j-th result is handed to store(j, key, index).
template <typename Load, typename Store>
__device__ __forceinline__ void topk_block_select(int64_t n, int k, Load load,
                                                  Store store) {
  __shared__ float s_key[kTopKSmallBlock];
  __shared__ int s_idx[kTopKSmallBlock];
  __shared__ int s_tid[kTopKSmallBlock];
  const int t = threadIdx.x;
  float lkey[kTopKSmallMax];
  int lidx[kTopKSmallMax];
  int cnt = 0;
  for (int64_t j = t; j < n; j += kTopKSmallBlock) {
    float key;
    int idx;
    load(j, key, idx);
    if (cnt == k && !topk_better(key, idx, lkey[k - 1], lidx[k - 1]))
      continue;
    int pos = cnt < k ? cnt : k - 1;
    while (pos > 0 && topk_better(key, idx, lkey[pos - 1], lidx[pos - 1])) {
      lkey[pos] = lkey[pos - 1];
      lidx[pos] = lidx[pos - 1];
      --pos;
    }
    lkey[pos] = key;
    lidx[pos] = idx;
    if (cnt < k)
      ++cnt;
  }
  int head = 0;
  for (int j = 0; j < k; ++j) {
    s_key[t] = head < cnt ? lkey[head] : -INFINITY;
    s_idx[t] = head < cnt ? lidx[head] : INT32_MAX;
    s_tid[t] = t;
    __syncthreads();
    for (int s = kTopKSmallBlock / 2; s > 0; s >>= 1) {
      if (t < s && topk_better(s_key[t + s], s_idx[t + s], s_key[t], s_idx[t])) {
        s_key[t] = s_key[t + s];
        s_idx[t] = s_idx[t + s];
        s_tid[t] = s_tid[t + s];
      }
      __syncthreads();
    }
    const int win = s_tid[0];
    const float wk = s_key[0];
    const int wi = s_idx[0];
    __syncthreads();
    if (t == win)
      ++head;
    if (t == 0)
      store(j, wk, wi);
  }
}

template <typename T>
__global__ void __launch_bounds__(kTopKSmallBlock)
    topk_chunk_pass_kernel(const T *__restrict__ data, float *__restrict__ ckey,
                           int *__restrict__ cidx, int64_t axis_size, int k,
                           int largest) {
  const int64_t chunk = blockIdx.x, row = blockIdx.y;
  const int64_t lo = chunk * kTopKChunk;
  const int64_t n = min(kTopKChunk, axis_size - lo);
  const T *src = data + row * axis_size + lo;
  const bool lg = largest != 0;
  const int64_t out = (row * gridDim.x + chunk) * k;
  topk_block_select(
      n, k,
      [&](int64_t j, float &key, int &idx) {
        float v = topk_elem_to_float(src[j]);
        key = lg ? v : -v;
        idx = static_cast<int>(lo + j);
      },
      [&](int j, float key, int idx) {
        ckey[out + j] = key;
        cidx[out + j] = idx;
      });
}

template <typename T>
__global__ void __launch_bounds__(kTopKSmallBlock)
    topk_merge_pass_kernel(const float *__restrict__ ckey,
                           const int *__restrict__ cidx, T *__restrict__ values,
                           int64_t *__restrict__ indices, int n_per_row, int k,
                           int largest) {
  const int64_t row = blockIdx.x;
  const float *rk = ckey + row * n_per_row;
  const int *ri = cidx + row * n_per_row;
  const bool lg = largest != 0;
  topk_block_select(
      n_per_row, k,
      [&](int64_t j, float &key, int &idx) {
        key = rk[j];
        idx = ri[j];
      },
      [&](int j, float key, int idx) {
        topk_store_value(values, row * k + j, lg ? key : -key);
        indices[row * k + j] = idx;
      });
}

static bool topk_small_last_axis_ok(const TopKGeom &geom) {
  return geom.k >= 2 && geom.k <= kTopKSmallMax && geom.k <= geom.axis_size &&
         geom.axis == geom.rank - 1 && geom.axis_size < INT32_MAX;
}

// K above the per-thread list size kTopKSmallMax stays exact in the two-pass
// kernels as long as no thread is handed more than kTopKSmallMax candidates in
// either pass: then every thread list holds its whole share and is never
// truncated (chunk pass: kTopKChunk / block, merge pass: chunks * K / block).
// That covers the low-rank candidate head of EAGLE-4 drafts (K = 128 over a
// 248k vocab: 61 chunks, 31 candidates per thread in the merge).
constexpr int64_t kTopKTwoPassMaxK = 256;

static bool topk_two_pass_ok(const TopKGeom &geom) {
  if (geom.axis_size <= 2 * kTopKChunk || geom.num_rows > 65535)
    return false;
  if (topk_small_last_axis_ok(geom))
    return true;
  const int64_t chunks = (geom.axis_size + kTopKChunk - 1) / kTopKChunk;
  const int64_t per_thread = kTopKSmallBlock * kTopKSmallMax;
  return geom.k > kTopKSmallMax && geom.k <= kTopKTwoPassMaxK &&
         geom.axis == geom.rank - 1 && geom.axis_size < INT32_MAX &&
         kTopKChunk <= per_thread && chunks * geom.k <= per_thread;
}

static bool topk_full_sort_ok(const TopKGeom &geom) {
  return geom.k == geom.axis_size && geom.k > kTopKTwoPassMaxK &&
         geom.axis == geom.rank - 1 && geom.axis_size < INT32_MAX;
}

static int64_t topk_full_sort_width(int64_t axis_size) {
  int64_t width = 1;
  while (width < axis_size) {
    if (width > INT32_MAX / 2)
      return 0;
    width *= 2;
  }
  return width;
}

static bool topk_checked_product(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static size_t topk_full_sort_bytes(const TopKGeom &geom) {
  const int64_t width = topk_full_sort_width(geom.axis_size);
  if (width == 0 || geom.num_rows <= 0)
    return 0;
  int64_t elements = 0;
  if (!topk_checked_product(geom.num_rows, width, elements))
    return 0;
  constexpr uint64_t bytes_per_element = sizeof(float) + sizeof(int32_t);
  if (static_cast<uint64_t>(elements) >
      std::numeric_limits<size_t>::max() / bytes_per_element)
    return 0;
  return static_cast<size_t>(elements) * bytes_per_element;
}

__device__ __forceinline__ bool topk_full_before(float a_key, int32_t a_index,
                                                 float b_key, int32_t b_index,
                                                 int largest) {
  if (a_index == INT32_MAX)
    return false;
  if (b_index == INT32_MAX)
    return true;
  bool a_nan = isnan(a_key);
  bool b_nan = isnan(b_key);
  if (a_nan != b_nan)
    return !a_nan;
  if (a_nan)
    return a_index < b_index;
  if (a_key == b_key)
    return a_index < b_index;
  return largest ? a_key > b_key : a_key < b_key;
}

template <typename T>
__global__ void topk_full_sort_init_kernel(const T *__restrict__ data,
                                           float *__restrict__ keys,
                                           int32_t *__restrict__ indices,
                                           int64_t axis_size,
                                           int64_t padded_width,
                                           int64_t total) {
  int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total)
    return;
  int64_t column = linear % padded_width;
  int64_t row = linear / padded_width;
  if (column < axis_size) {
    keys[linear] = topk_elem_to_float(data[row * axis_size + column]);
    indices[linear] = static_cast<int32_t>(column);
  } else {
    keys[linear] = 0.0f;
    indices[linear] = INT32_MAX;
  }
}

__global__ void topk_full_sort_step_kernel(float *__restrict__ keys,
                                           int32_t *__restrict__ indices,
                                           int64_t padded_width,
                                           int64_t total, int64_t stage,
                                           int64_t stride, int largest) {
  int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total)
    return;
  int64_t column = linear % padded_width;
  int64_t partner_column = column ^ stride;
  if (partner_column <= column)
    return;
  int64_t partner = linear - column + partner_column;
  float a_key = keys[linear];
  float b_key = keys[partner];
  int32_t a_index = indices[linear];
  int32_t b_index = indices[partner];
  bool ascending_half = (column & stage) == 0;
  bool a_before_b =
      topk_full_before(a_key, a_index, b_key, b_index, largest);
  if ((ascending_half && !a_before_b) || (!ascending_half && a_before_b)) {
    keys[linear] = b_key;
    keys[partner] = a_key;
    indices[linear] = b_index;
    indices[partner] = a_index;
  }
}

template <typename T>
__global__ void topk_full_sort_output_kernel(
    const T *__restrict__ data, T *__restrict__ values,
    int64_t *__restrict__ output_indices,
    const int32_t *__restrict__ indices,
    int64_t axis_size, int64_t padded_width, int64_t total) {
  int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total)
    return;
  int64_t column = linear % axis_size;
  int64_t row = linear / axis_size;
  int32_t index = indices[row * padded_width + column];
  values[linear] = data[row * axis_size + index];
  output_indices[linear] = index;
}

template <typename T>
static int launch_top_k_full_sort(hipStream_t stream, const T *data, T *values,
                                  int64_t *indices, const TopKGeom &geom,
                                  int largest, void *workspace) {
  const int64_t width = topk_full_sort_width(geom.axis_size);
  int64_t workspace_elements = 0;
  int64_t output_elements = 0;
  if (!topk_checked_product(geom.num_rows, width, workspace_elements) ||
      !topk_checked_product(geom.num_rows, geom.axis_size, output_elements))
    return -1;
  float *keys = static_cast<float *>(workspace);
  int32_t *workspace_indices =
      reinterpret_cast<int32_t *>(keys + workspace_elements);
  constexpr int block_size = 256;
  const int64_t workspace_blocks =
      (workspace_elements - 1) / block_size + 1;
  const int64_t output_blocks = (output_elements - 1) / block_size + 1;
  if (workspace_blocks > std::numeric_limits<unsigned>::max() ||
      output_blocks > std::numeric_limits<unsigned>::max())
    return -1;
  const unsigned workspace_grid = static_cast<unsigned>(workspace_blocks);
  const unsigned output_grid = static_cast<unsigned>(output_blocks);

  (void)hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(topk_full_sort_init_kernel<T>),
                     dim3(workspace_grid), dim3(block_size), 0, stream, data,
                     keys, workspace_indices, geom.axis_size, width,
                     workspace_elements);
  for (int64_t stage = 2; stage <= width; stage *= 2)
    for (int64_t stride = stage / 2; stride > 0; stride /= 2)
      hipLaunchKernelGGL(topk_full_sort_step_kernel, dim3(workspace_grid),
                         dim3(block_size), 0, stream, keys, workspace_indices,
                         width, workspace_elements, stage, stride, largest);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(topk_full_sort_output_kernel<T>),
                     dim3(output_grid), dim3(block_size), 0, stream, data,
                     values, indices, workspace_indices, geom.axis_size, width,
                     output_elements);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] full-sort top_k launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

// Per-chunk (key, index) lists of the two-pass path: float keys, then int
// indices.
static size_t topk_two_pass_bytes(const TopKGeom &geom) {
  const int64_t chunks = (geom.axis_size + kTopKChunk - 1) / kTopKChunk;
  return static_cast<size_t>(geom.num_rows * chunks * geom.k) *
         (sizeof(float) + sizeof(int));
}

template <typename T>
static int launch_top_k_two_pass(hipStream_t stream, const T *data, T *values,
                                 int64_t *indices, const TopKGeom &geom,
                                 int largest, void *workspace) {
  const int64_t chunks = (geom.axis_size + kTopKChunk - 1) / kTopKChunk;
  const int k = static_cast<int>(geom.k);
  const int64_t pairs = geom.num_rows * chunks * k;
  float *ckey = static_cast<float *>(workspace);
  int *cidx = reinterpret_cast<int *>(ckey + pairs);
  (void)hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(topk_chunk_pass_kernel<T>),
                     dim3(static_cast<unsigned>(chunks),
                          static_cast<unsigned>(geom.num_rows)),
                     dim3(kTopKSmallBlock), 0, stream, data, ckey, cidx,
                     geom.axis_size, k, largest);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(topk_merge_pass_kernel<T>),
                     dim3(static_cast<unsigned>(geom.num_rows)),
                     dim3(kTopKSmallBlock), 0, stream, ckey, cidx, values,
                     indices, static_cast<int>(chunks * k), k, largest);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] two-pass top_k launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

template <typename T>
static int launch_top_k(void *stream, const void *data, void *values,
                        void *indices, const TopKGeom &geom, int largest,
                        int sorted, void *workspace, size_t workspace_bytes) {
  if (geom.num_rows <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  if (topk_full_sort_ok(geom)) {
    const size_t needed = topk_full_sort_bytes(geom);
    if (!workspace || needed == 0 || workspace_bytes < needed) {
      fprintf(stderr,
              "[custom_kernels] full-sort top_k needs %zu workspace bytes, "
              "got %zu\n",
              needed, workspace_bytes);
      return -1;
    }
    return launch_top_k_full_sort(
        hip_stream, static_cast<const T *>(data), static_cast<T *>(values),
        static_cast<int64_t *>(indices), geom, largest, workspace);
  }
  // Without enough workspace a long row still takes the one-pass kernel.
  if (topk_two_pass_ok(geom) && workspace &&
      workspace_bytes >= topk_two_pass_bytes(geom))
    return launch_top_k_two_pass(hip_stream, static_cast<const T *>(data),
                                 static_cast<T *>(values),
                                 static_cast<int64_t *>(indices), geom,
                                 largest, workspace);
  if (topk_small_last_axis_ok(geom)) {
    (void)hipGetLastError();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(topk_small_last_axis_kernel<T>),
                       dim3(static_cast<unsigned>(geom.num_rows)),
                       dim3(kTopKSmallBlock), 0, hip_stream,
                       static_cast<const T *>(data), static_cast<T *>(values),
                       static_cast<int64_t *>(indices), geom.axis_size,
                       static_cast<int>(geom.k), largest);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] topk_small_last_axis_kernel launch failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }
    return 0;
  }
  if (geom.k == 1 && geom.axis == geom.rank - 1 && geom.axis_size > 0) {
    (void)hipGetLastError();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(top1_last_axis_kernel<T>),
                       dim3(static_cast<unsigned>(geom.num_rows)),
                       dim3(kTop1Block), 0, hip_stream,
                       static_cast<const T *>(data), static_cast<T *>(values),
                       static_cast<int64_t *>(indices), geom.axis_size,
                       largest);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "[custom_kernels] top1_last_axis_kernel launch failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }
    return 0;
  }
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

static bool make_topk_geom(int64_t axis, int64_t rank, const int64_t *x_shape,
                           int64_t k, TopKGeom &geom) {
  geom = TopKGeom{};
  geom.rank = rank;
  geom.axis = axis;
  if (geom.axis < 0)
    geom.axis += rank;
  if (geom.axis < 0 || geom.axis >= rank)
    return false;

  geom.k = k;
  for (int64_t i = 0; i < rank; ++i) {
    if (x_shape[i] < 0)
      return false;
    geom.in_shape[i] = x_shape[i];
  }

  geom.in_stride[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i)
    if (!topk_checked_product(geom.in_stride[i + 1],
                              geom.in_shape[i + 1], geom.in_stride[i]))
      return false;

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
    if (!topk_checked_product(geom.out_stride[i + 1], out_shape[i + 1],
                              geom.out_stride[i]))
      return false;
  geom.num_rows = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == geom.axis)
      continue;
    if (!topk_checked_product(geom.num_rows, geom.in_shape[i],
                              geom.num_rows))
      return false;
  }
  return true;
}

} // namespace

extern "C" size_t hip_top_k_workspace_bytes(int64_t axis, int64_t rank,
                                            const int64_t *x_shape,
                                            int64_t k) {
  if (!x_shape || rank <= 0 || rank > kMaxRank || k <= 0)
    return 0;
  TopKGeom geom;
  if (!make_topk_geom(axis, rank, x_shape, k, geom))
    return 0;
  if (topk_full_sort_ok(geom))
    return topk_full_sort_bytes(geom);
  if (!topk_two_pass_ok(geom))
    return 0;
  return topk_two_pass_bytes(geom);
}

extern "C" int hip_top_k_device_supported(int64_t axis, int64_t rank,
                                          const int64_t *x_shape, int64_t k) {
  if (!x_shape || rank <= 0 || rank > kMaxRank)
    return 0;
  if (k <= kTopKTwoPassMaxK)
    return 1;
  TopKGeom geom;
  if (!make_topk_geom(axis, rank, x_shape, k, geom))
    return 0;
  return topk_full_sort_ok(geom) && topk_full_sort_bytes(geom) > 0;
}

extern "C" int hip_top_k(void *stream, const void *data, void *values,
                         void *indices, int64_t axis, int64_t largest,
                         int64_t sorted, int64_t rank,
                         const int64_t *x_shape, int64_t k,
                         int hip_dtype, void *workspace,
                         size_t workspace_bytes) {
  if (!data || !values || !indices || !x_shape)
    return -1;
  if (rank <= 0 || rank > kMaxRank)
    return -1;
  if (k <= 0)
    return 0;

  TopKGeom geom;
  if (!make_topk_geom(axis, rank, x_shape, k, geom))
    return -1;
  if (k > kTopKTwoPassMaxK && !topk_full_sort_ok(geom)) {
    fprintf(stderr,
            "[custom_kernels] hip_top_k: K=%lld exceeds max 256 and is not a "
            "contiguous full-axis sort\n",
            (long long)k);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_top_k: axis=%lld, k=%lld, rank=%lld, "
      "rows=%lld, axis_size=%lld, dtype=%d\n",
      (long long)geom.axis, (long long)geom.k, (long long)rank,
      (long long)geom.num_rows, (long long)geom.axis_size, hip_dtype);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    return launch_top_k<__half>(stream, data, values, indices, geom,
                                static_cast<int>(largest),
                                static_cast<int>(sorted), workspace,
                                workspace_bytes);
  case HIP_DTYPE_FLOAT32:
    return launch_top_k<float>(stream, data, values, indices, geom,
                               static_cast<int>(largest),
                               static_cast<int>(sorted), workspace,
                               workspace_bytes);
  default:
    fprintf(stderr, "[custom_kernels] hip_top_k: unsupported hip_dtype=%d\n",
            hip_dtype);
    return -1;
  }
}
