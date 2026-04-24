/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Compile-time limits. Must match HIP_SLICE_MAX_RANK in hip_custom_kernels.h.
static_assert(HIP_SLICE_MAX_RANK >= 1, "rank must be at least 1");

// Metadata passed by value to the kernel. Keeping everything inside a POD
// lets us stick to fixed-size buffers instead of allocating device memory
// for per-launch shape/stride tables.
struct SliceMeta {
  int64_t input_shape[HIP_SLICE_MAX_RANK];
  int64_t input_strides[HIP_SLICE_MAX_RANK];
  int64_t output_strides[HIP_SLICE_MAX_RANK];
  int32_t rank;
  int32_t num_slice_entries;
};

// Element-level slice kernel. One thread per output element; the thread maps
// its flat output index back into a coordinate using `output_strides`, then
// transforms sliced axes via `axes[i] / starts[i] / steps[i]` (reading them
// from device memory) into an input coordinate, and finally computes the
// corresponding linear input offset via `input_strides`.
template <typename T>
__global__ void slice_kernel(
    const T* __restrict__ data,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ axes,   // nullable
    const int64_t* __restrict__ steps,  // nullable
    T* __restrict__ output,
    SliceMeta meta,
    int64_t output_num_elements) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= output_num_elements) return;

  int rank = meta.rank;

  // 1. flat output index -> per-axis output coordinate
  int64_t out_coord[HIP_SLICE_MAX_RANK];
  {
    int64_t rem = tid;
    for (int i = 0; i < rank; ++i) {
      int64_t stride = meta.output_strides[i];
      int64_t c = (stride > 0) ? (rem / stride) : 0;
      out_coord[i] = c;
      rem -= c * stride;
    }
  }

  // 2. default in_coord = out_coord for axes not covered by a slice entry
  int64_t in_coord[HIP_SLICE_MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    in_coord[i] = out_coord[i];
  }

  // 3. apply slice for each (starts[i], ends[i], axes[i], steps[i]) entry
  for (int i = 0; i < meta.num_slice_entries; ++i) {
    int64_t axis = axes ? axes[i] : static_cast<int64_t>(i);
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) continue;

    int64_t step = steps ? steps[i] : 1;
    int64_t start = starts[i];
    int64_t dim_size = meta.input_shape[axis];

    if (start < 0) start += dim_size;
    if (step >= 0) {
      if (start < 0) start = 0;
      if (start > dim_size) start = dim_size;
    } else {
      if (start < 0) start = -1;
      if (start > dim_size - 1) start = dim_size - 1;
    }

    in_coord[axis] = start + out_coord[axis] * step;
  }

  // 4. compute linear input offset
  int64_t in_offset = 0;
  for (int i = 0; i < rank; ++i) {
    in_offset += in_coord[i] * meta.input_strides[i];
  }

  output[tid] = data[in_offset];
}

// Compute row-major strides for a contiguous shape. strides[i] = prod(shape[i+1..]).
// Assumes `rank <= HIP_SLICE_MAX_RANK`.
static void compute_contiguous_strides(const int64_t* shape, int rank,
                                       int64_t* strides) {
  int64_t running = 1;
  for (int i = rank - 1; i >= 0; --i) {
    strides[i] = running;
    running *= shape[i];
  }
}

extern "C" int hip_slice(
    void* stream,
    const void* data,
    const void* starts,
    const void* /*ends*/,       // not used by the kernel; output shape already known
    const void* axes,
    const void* steps,
    void* output,
    const int64_t* input_shape,
    const int64_t* output_shape,
    int64_t rank,
    int64_t num_slice_entries,
    int64_t output_num_elements,
    int element_size_bytes) {
  if (output_num_elements <= 0) return 0;

  if (!data || !output || !starts || !input_shape || !output_shape) {
    fprintf(stderr, "[custom_kernels] hip_slice: null argument\n");
    return -1;
  }

  if (rank <= 0 || rank > HIP_SLICE_MAX_RANK) {
    fprintf(stderr,
            "[custom_kernels] hip_slice: unsupported rank=%lld (max %d)\n",
            (long long)rank, HIP_SLICE_MAX_RANK);
    return -1;
  }

  if (num_slice_entries < 0 || num_slice_entries > rank) {
    fprintf(stderr,
            "[custom_kernels] hip_slice: invalid num_slice_entries=%lld for "
            "rank=%lld\n",
            (long long)num_slice_entries, (long long)rank);
    return -1;
  }

  SliceMeta meta{};
  meta.rank = static_cast<int32_t>(rank);
  meta.num_slice_entries = static_cast<int32_t>(num_slice_entries);
  for (int i = 0; i < static_cast<int>(rank); ++i) {
    meta.input_shape[i] = input_shape[i];
  }
  compute_contiguous_strides(input_shape, static_cast<int>(rank),
                             meta.input_strides);
  compute_contiguous_strides(output_shape, static_cast<int>(rank),
                             meta.output_strides);

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (output_num_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_slice: rank=%lld, slice_entries=%lld, "
      "output_num=%lld, elem_size=%d, grid=%d, block=%d\n",
      (long long)rank, (long long)num_slice_entries,
      (long long)output_num_elements, element_size_bytes, grid_size,
      kBlockSize);

  const int64_t* starts_ptr = static_cast<const int64_t*>(starts);
  const int64_t* axes_ptr = static_cast<const int64_t*>(axes);
  const int64_t* steps_ptr = static_cast<const int64_t*>(steps);

  switch (element_size_bytes) {
    case 1:
      hipLaunchKernelGGL(slice_kernel<uint8_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint8_t*>(data),
                         starts_ptr, axes_ptr, steps_ptr,
                         static_cast<uint8_t*>(output),
                         meta, output_num_elements);
      break;
    case 2:
      hipLaunchKernelGGL(slice_kernel<uint16_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint16_t*>(data),
                         starts_ptr, axes_ptr, steps_ptr,
                         static_cast<uint16_t*>(output),
                         meta, output_num_elements);
      break;
    case 4:
      hipLaunchKernelGGL(slice_kernel<uint32_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint32_t*>(data),
                         starts_ptr, axes_ptr, steps_ptr,
                         static_cast<uint32_t*>(output),
                         meta, output_num_elements);
      break;
    case 8:
      hipLaunchKernelGGL(slice_kernel<uint64_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint64_t*>(data),
                         starts_ptr, axes_ptr, steps_ptr,
                         static_cast<uint64_t*>(output),
                         meta, output_num_elements);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_slice: unsupported element_size=%d\n",
              element_size_bytes);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_slice launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
