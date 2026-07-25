/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"
#include "scatter_reduction.inc.hip"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kMaxRank = 8;

struct ScatterElementsGeom {
  int64_t axis;
  int64_t rank;
  int64_t ishape[kMaxRank];
  int64_t dshape[kMaxRank];
  int64_t dstride[kMaxRank];
};

template <typename T, typename IdxT>
__global__ void scatter_elements_kernel(const IdxT *__restrict__ indices,
                                        const T *__restrict__ updates,
                                        T *__restrict__ output,
                                        int64_t num_updates,
                                        ScatterElementsGeom geom,
                                        int reduction_id) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_updates)
    return;

  int64_t coords[kMaxRank] = {0};
  int64_t rem = tid;
  for (int64_t d = geom.rank - 1; d >= 0; --d) {
    int64_t dim = geom.ishape[d];
    coords[d] = (dim > 0) ? (rem % dim) : 0;
    if (dim > 0)
      rem /= dim;
  }

  int64_t idx_val = static_cast<int64_t>(indices[tid]);
  int64_t axis_size = geom.dshape[geom.axis];
  if (idx_val < 0)
    idx_val += axis_size;
  if (idx_val < 0 || idx_val >= axis_size)
    return;

  int64_t offset = 0;
  for (int64_t d = 0; d < geom.rank; ++d) {
    int64_t c = (d == geom.axis) ? idx_val : coords[d];
    if (c < 0 || c >= geom.dshape[d])
      return;
    offset += c * geom.dstride[d];
  }

  scatter_apply(output + offset, updates[tid], reduction_id);
}

template <typename T>
static int launch_scatter_elements(void *stream, const void *indices,
                                   const void *updates, void *output,
                                   int64_t num_updates,
                                   const ScatterElementsGeom &geom,
                                   int reduction_id,
                                   int indices_element_size_bytes) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (num_updates + kBlockSize - 1) / kBlockSize);

  if (indices_element_size_bytes == 4) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scatter_elements_kernel<T, int32_t>),
                     dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                     static_cast<const int32_t *>(indices),
                     static_cast<const T *>(updates), static_cast<T *>(output),
                     num_updates, geom, reduction_id);
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scatter_elements_kernel<T, int64_t>),
                     dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                     static_cast<const int64_t *>(indices),
                     static_cast<const T *>(updates), static_cast<T *>(output),
                     num_updates, geom, reduction_id);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] scatter_elements_kernel launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

} // namespace

extern "C" int hip_scatter_elements(
    void *stream, const void *data, const void *indices, const void *updates,
    void *output, int64_t axis, int64_t reduction_id, int64_t rank,
    const int64_t *data_shape, const int64_t *indices_shape,
    int64_t num_updates, int element_size_bytes,
    int indices_element_size_bytes) {
  if (!data || !indices || !updates || !output || !data_shape || !indices_shape)
    return -1;
  if (rank <= 0 || rank > kMaxRank)
    return -1;
  if (reduction_id < 0 || reduction_id > SCATTER_MAX)
    return -1;
  if (indices_element_size_bytes != 4 && indices_element_size_bytes != 8)
    return -1;

  int64_t total_data_elements = 1;
  for (int64_t i = 0; i < rank; ++i)
    total_data_elements *= data_shape[i];
  if (total_data_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  hipError_t err = hipMemcpyAsync(
      output, data, static_cast<size_t>(total_data_elements * element_size_bytes),
      hipMemcpyDeviceToDevice, hip_stream);
  if (err != hipSuccess)
    return static_cast<int>(err);

  if (num_updates <= 0)
    return 0;

  ScatterElementsGeom geom{};
  geom.axis = axis;
  geom.rank = rank;
  for (int64_t i = 0; i < rank; ++i) {
    geom.dshape[i] = data_shape[i];
    geom.ishape[i] = indices_shape[i];
  }
  geom.dstride[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i)
    geom.dstride[i] = geom.dstride[i + 1] * geom.dshape[i + 1];

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_scatter_elements: axis=%lld, reduction=%d, "
      "rank=%lld, num=%lld, elem=%d, idx_elem=%d\n",
      (long long)axis, reduction_id, (long long)rank, (long long)num_updates,
      element_size_bytes, indices_element_size_bytes);

  switch (element_size_bytes) {
  case 2:
    return launch_scatter_elements<__half>(
        stream, indices, updates, output, num_updates, geom, reduction_id,
        indices_element_size_bytes);
  case 4:
    return launch_scatter_elements<float>(
        stream, indices, updates, output, num_updates, geom, reduction_id,
        indices_element_size_bytes);
  case 8:
    return launch_scatter_elements<int64_t>(
        stream, indices, updates, output, num_updates, geom, reduction_id,
        indices_element_size_bytes);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_scatter_elements: unsupported "
            "element_size=%d\n",
            element_size_bytes);
    return -1;
  }
}
