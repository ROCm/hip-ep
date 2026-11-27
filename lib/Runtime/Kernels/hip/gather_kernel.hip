/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Generic gather kernel for arbitrary axis.
// data    has logical shape [outer, axis_size, inner]
// indices has indices_num elements (flattened)
// output  has logical shape [outer, indices_num, inner]
// Each thread copies one element of the output tensor.
//
// IdxT is the element type of the ONNX `indices` tensor. ONNX permits both
// int32 and int64 index tensors; reading an int32 buffer as int64 fuses
// adjacent indices into huge out-of-range values, which the bounds check then
// maps to 0 -> the whole gather silently returns zeros. Templating on IdxT and
// widening to int64 in-kernel handles both layouts.
template <typename T, typename IdxT>
__global__ void gather_kernel(
    const T* __restrict__ data,
    const IdxT* __restrict__ indices,
    T* __restrict__ output,
    int64_t outer_size,
    int64_t axis_size,
    int64_t inner_size,
    int64_t indices_num,
    int64_t total_elements) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;
  int64_t inner_off = tid % inner_size;
  int64_t idx_off = (tid / inner_size) % indices_num;
  int64_t outer_off = tid / (inner_size * indices_num);
  int64_t index_val = static_cast<int64_t>(indices[idx_off]);
  if (index_val < 0) index_val += axis_size;
  if (index_val < 0 || index_val >= axis_size) {
    output[tid] = T(0);
    return;
  }
  int64_t data_off =
      (outer_off * axis_size + index_val) * inner_size + inner_off;
  output[tid] = data[data_off];
}

// Unsigned division by a launch-invariant divisor as a multiply-high plus
// shift. Exact for dividends and divisors below 2^31.
struct GatherFastDivmod {
  uint32_t divisor;
  uint32_t multiplier;
  uint32_t shift;

  static GatherFastDivmod make(uint32_t d) {
    uint32_t s = 0;
    while ((uint64_t(1) << s) < d)
      ++s;
    uint64_t m = ((uint64_t(1) << 32) * ((uint64_t(1) << s) - d)) / d + 1;
    return {d, static_cast<uint32_t>(m), s};
  }

  __device__ uint32_t div(uint32_t n) const {
    return (__umulhi(n, multiplier) + n) >> shift;
  }
};

// The widest load a row allows: B bytes, a divisor of the row size and of the
// data and output alignment.
template <int B> struct alignas(B) GatherChunk {
  unsigned char bytes[B];
};

// Rows of more than kSmallRowChunks chunks: one thread per B-byte chunk.
template <int B, typename IdxT>
__global__ void gather_row_chunks_kernel(const GatherChunk<B>* __restrict__ data,
                                         const IdxT* __restrict__ indices,
                                         GatherChunk<B>* __restrict__ output,
                                         uint32_t axis_size,
                                         GatherFastDivmod chunks_per_row,
                                         GatherFastDivmod indices_num,
                                         uint32_t total_chunks) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_chunks)
    return;
  uint32_t row = chunks_per_row.div(tid);
  uint32_t chunk = tid - row * chunks_per_row.divisor;
  uint32_t outer = indices_num.div(row);
  uint32_t idx_off = row - outer * indices_num.divisor;
  int64_t index_val = static_cast<int64_t>(indices[idx_off]);
  if (index_val < 0)
    index_val += axis_size;
  if (index_val < 0 || index_val >= axis_size) {
    output[tid] = GatherChunk<B>{};
    return;
  }
  output[tid] = data[(outer * axis_size + static_cast<uint32_t>(index_val)) *
                         chunks_per_row.divisor +
                     chunk];
}

// Rows of N <= kSmallRowChunks chunks: one thread per row reads the index once
// and copies the row.
template <int B, int N, typename IdxT>
__global__ void gather_small_rows_kernel(const GatherChunk<B>* __restrict__ data,
                                         const IdxT* __restrict__ indices,
                                         GatherChunk<B>* __restrict__ output,
                                         uint32_t axis_size,
                                         GatherFastDivmod indices_num,
                                         uint32_t rows) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  uint32_t outer = indices_num.div(row);
  uint32_t idx_off = row - outer * indices_num.divisor;
  int64_t index_val = static_cast<int64_t>(indices[idx_off]);
  if (index_val < 0)
    index_val += axis_size;
  GatherChunk<B>* dst = output + static_cast<size_t>(row) * N;
  if (index_val < 0 || index_val >= axis_size) {
#pragma unroll
    for (int i = 0; i < N; ++i)
      dst[i] = GatherChunk<B>{};
    return;
  }
  const GatherChunk<B>* src =
      data + (outer * axis_size + static_cast<uint32_t>(index_val)) * N;
#pragma unroll
  for (int i = 0; i < N; ++i)
    dst[i] = src[i];
}

constexpr int kGatherBlockSize = 256;
constexpr int64_t kSmallRowChunks = 4;

template <int B, typename IdxT>
static void launch_gather_rows(hipStream_t stream, const void* data,
                               const void* indices, void* output,
                               uint32_t axis_size, uint32_t chunks_per_row,
                               uint32_t indices_num, uint32_t total_chunks) {
  GatherFastDivmod idx_div = GatherFastDivmod::make(indices_num);
  auto src = static_cast<const GatherChunk<B>*>(data);
  auto idx = static_cast<const IdxT*>(indices);
  auto dst = static_cast<GatherChunk<B>*>(output);
  if (chunks_per_row <= kSmallRowChunks) {
    uint32_t rows = total_chunks / chunks_per_row;
    dim3 grid((rows + kGatherBlockSize - 1) / kGatherBlockSize);
    switch (chunks_per_row) {
    case 1:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_small_rows_kernel<B, 1, IdxT>),
                         grid, dim3(kGatherBlockSize), 0, stream, src, idx,
                         dst, axis_size, idx_div, rows);
      return;
    case 2:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_small_rows_kernel<B, 2, IdxT>),
                         grid, dim3(kGatherBlockSize), 0, stream, src, idx,
                         dst, axis_size, idx_div, rows);
      return;
    case 3:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_small_rows_kernel<B, 3, IdxT>),
                         grid, dim3(kGatherBlockSize), 0, stream, src, idx,
                         dst, axis_size, idx_div, rows);
      return;
    default:
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_small_rows_kernel<B, 4, IdxT>),
                         grid, dim3(kGatherBlockSize), 0, stream, src, idx,
                         dst, axis_size, idx_div, rows);
      return;
    }
  }
  dim3 grid((total_chunks + kGatherBlockSize - 1) / kGatherBlockSize);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_row_chunks_kernel<B, IdxT>), grid,
                     dim3(kGatherBlockSize), 0, stream, src, idx, dst,
                     axis_size, GatherFastDivmod::make(chunks_per_row), idx_div,
                     total_chunks);
}

template <typename IdxT>
static void dispatch_gather_rows(int chunk_bytes, hipStream_t stream,
                                 const void* data, const void* indices,
                                 void* output, uint32_t axis_size,
                                 uint32_t chunks_per_row, uint32_t indices_num,
                                 uint32_t total_chunks) {
  switch (chunk_bytes) {
  case 16:
    launch_gather_rows<16, IdxT>(stream, data, indices, output, axis_size,
                                 chunks_per_row, indices_num, total_chunks);
    break;
  case 8:
    launch_gather_rows<8, IdxT>(stream, data, indices, output, axis_size,
                                chunks_per_row, indices_num, total_chunks);
    break;
  case 4:
    launch_gather_rows<4, IdxT>(stream, data, indices, output, axis_size,
                                chunks_per_row, indices_num, total_chunks);
    break;
  case 2:
    launch_gather_rows<2, IdxT>(stream, data, indices, output, axis_size,
                                chunks_per_row, indices_num, total_chunks);
    break;
  default:
    launch_gather_rows<1, IdxT>(stream, data, indices, output, axis_size,
                                chunks_per_row, indices_num, total_chunks);
    break;
  }
}

// Dispatch on the index element type (4 = int32, 8 = int64). The data element
// type is fixed by the caller's `T`.
template <typename T>
static void launch_gather(int grid_size, int block_size, hipStream_t stream,
                          const void* data, const void* indices, void* output,
                          int64_t outer_size, int64_t axis_size,
                          int64_t inner_size, int64_t indices_num,
                          int64_t total_elements,
                          int indices_element_size_bytes) {
  if (indices_element_size_bytes == 4) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_kernel<T, int32_t>),
                       dim3(grid_size), dim3(block_size), 0, stream,
                       static_cast<const T*>(data),
                       static_cast<const int32_t*>(indices),
                       static_cast<T*>(output), outer_size, axis_size,
                       inner_size, indices_num, total_elements);
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_kernel<T, int64_t>),
                       dim3(grid_size), dim3(block_size), 0, stream,
                       static_cast<const T*>(data),
                       static_cast<const int64_t*>(indices),
                       static_cast<T*>(output), outer_size, axis_size,
                       inner_size, indices_num, total_elements);
  }
}

extern "C" int hip_gather(
    void* stream,
    const void* data,
    const void* indices,
    void* output,
    int64_t axis,
    int64_t data_num_elements,
    int64_t indices_num_elements,
    int64_t output_num_elements,
    int64_t axis_size,
    int64_t inner_size,
    int element_size_bytes,
    int indices_element_size_bytes) {
  if (output_num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (output_num_elements + kBlockSize - 1) / kBlockSize);

  // axis_size = data.shape[axis]; inner_size = product(data.shape[axis+1:])
  // outer_size derived from data layout: data has outer*axis_size*inner elems.
  int64_t outer_size = (axis_size > 0 && inner_size > 0)
      ? data_num_elements / (axis_size * inner_size)
      : 0;
  if (outer_size <= 0) {
    fprintf(stderr,
            "[custom_kernels] hip_gather: bad geometry axis=%lld data_num=%lld "
            "axis_size=%lld inner_size=%lld\n",
            (long long)axis, (long long)data_num_elements,
            (long long)axis_size, (long long)inner_size);
    return -1;
  }

  if (indices_element_size_bytes != 4 && indices_element_size_bytes != 8) {
    fprintf(stderr,
            "[custom_kernels] hip_gather: unsupported indices element_size=%d "
            "(expected 4 or 8)\n",
            indices_element_size_bytes);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_gather: axis=%lld, elem_size=%d, "
          "idx_size=%d, data_num=%lld, indices_num=%lld, output_num=%lld, "
          "outer=%lld, axis_size=%lld, inner=%lld, grid=%d, block=%d\n",
          (long long)axis, element_size_bytes, indices_element_size_bytes,
          (long long)data_num_elements, (long long)indices_num_elements,
          (long long)output_num_elements,
          (long long)outer_size, (long long)axis_size, (long long)inner_size,
          grid_size, kBlockSize);

  // HIP holds an error in the last-error slot until something reads it, so
  // without this the check below can report an error left behind by an
  // unrelated earlier API call as this kernel's launch failure.
  (void)hipGetLastError();

  // Copy whole rows in the widest chunk the row size and both pointers allow,
  // with 32-bit offsets when every chunk index fits. Larger tensors keep the
  // 64-bit per-element kernel below.
  const int64_t row_bytes = inner_size * element_size_bytes;
  int chunk_bytes = 16;
  while (chunk_bytes > 1 &&
         (row_bytes % chunk_bytes != 0 ||
          reinterpret_cast<uintptr_t>(data) % chunk_bytes != 0 ||
          reinterpret_cast<uintptr_t>(output) % chunk_bytes != 0))
    chunk_bytes /= 2;
  const int64_t data_chunks = data_num_elements * element_size_bytes / chunk_bytes;
  const int64_t out_chunks = output_num_elements * element_size_bytes / chunk_bytes;
  constexpr int64_t kMax32 = int64_t(1) << 31;
  if ((element_size_bytes == 1 || element_size_bytes == 2 ||
       element_size_bytes == 4 || element_size_bytes == 8) &&
      data_chunks < kMax32 && out_chunks < kMax32 && axis_size < kMax32 &&
      indices_num_elements < kMax32) {
    const uint32_t chunks_per_row =
        static_cast<uint32_t>(row_bytes / chunk_bytes);
    if (indices_element_size_bytes == 4)
      dispatch_gather_rows<int32_t>(
          chunk_bytes, hip_stream, data, indices, output,
          static_cast<uint32_t>(axis_size), chunks_per_row,
          static_cast<uint32_t>(indices_num_elements),
          static_cast<uint32_t>(out_chunks));
    else
      dispatch_gather_rows<int64_t>(
          chunk_bytes, hip_stream, data, indices, output,
          static_cast<uint32_t>(axis_size), chunks_per_row,
          static_cast<uint32_t>(indices_num_elements),
          static_cast<uint32_t>(out_chunks));
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "[custom_kernels] hip_gather launch failed: %s\n",
              hipGetErrorString(err));
    }
    return static_cast<int>(err);
  }

  switch (element_size_bytes) {
    case 1:
      launch_gather<uint8_t>(grid_size, kBlockSize, hip_stream, data, indices,
                             output, outer_size, axis_size, inner_size,
                             indices_num_elements, output_num_elements,
                             indices_element_size_bytes);
      break;
    case 2:
      launch_gather<uint16_t>(grid_size, kBlockSize, hip_stream, data, indices,
                              output, outer_size, axis_size, inner_size,
                              indices_num_elements, output_num_elements,
                              indices_element_size_bytes);
      break;
    case 4:
      launch_gather<uint32_t>(grid_size, kBlockSize, hip_stream, data, indices,
                              output, outer_size, axis_size, inner_size,
                              indices_num_elements, output_num_elements,
                              indices_element_size_bytes);
      break;
    case 8:
      launch_gather<int64_t>(grid_size, kBlockSize, hip_stream, data, indices,
                             output, outer_size, axis_size, inner_size,
                             indices_num_elements, output_num_elements,
                             indices_element_size_bytes);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_gather: unsupported element_size=%d\n",
              element_size_bytes);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_gather launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
