/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kMaxRank = 8;

struct GatherElementsGeom {
  int64_t axis;
  int64_t rank;
  int64_t ishape[kMaxRank];
  int64_t dshape[kMaxRank];
  int64_t dstride[kMaxRank];
};

template <typename T, typename IdxT>
__global__ void gather_elements_kernel(const T *__restrict__ data,
                                       const IdxT *__restrict__ indices,
                                       T *__restrict__ output,
                                       int64_t num_elements,
                                       GatherElementsGeom geom) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_elements)
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
  if (idx_val < 0 || idx_val >= axis_size) {
    output[tid] = T(0);
    return;
  }

  int64_t offset = 0;
  for (int64_t d = 0; d < geom.rank; ++d) {
    int64_t c = (d == geom.axis) ? idx_val : coords[d];
    if (c < 0 || c >= geom.dshape[d]) {
      output[tid] = T(0);
      return;
    }
    offset += c * geom.dstride[d];
  }
  output[tid] = data[offset];
}

template <typename T>
static int launch_gather_elements(void *stream, const void *data,
                                  const void *indices, void *output,
                                  int64_t num_elements,
                                  const GatherElementsGeom &geom,
                                  int indices_element_size_bytes) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (num_elements + kBlockSize - 1) / kBlockSize);

  if (indices_element_size_bytes == 4) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_elements_kernel<T, int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const T *>(data),
                       static_cast<const int32_t *>(indices),
                       static_cast<T *>(output), num_elements, geom);
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gather_elements_kernel<T, int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const T *>(data),
                       static_cast<const int64_t *>(indices),
                       static_cast<T *>(output), num_elements, geom);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] gather_elements_kernel launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

} // namespace

extern "C" int hip_gather_elements(
    void *stream, const void *data, const void *indices, void *output,
    int64_t axis, int64_t rank, const int64_t *data_shape,
    const int64_t *indices_shape, int64_t num_elements,
    int element_size_bytes, int indices_element_size_bytes) {
  if (num_elements <= 0)
    return 0;
  if (!data || !indices || !output || !data_shape || !indices_shape)
    return -1;
  if (rank <= 0 || rank > kMaxRank)
    return -1;
  if (indices_element_size_bytes != 4 && indices_element_size_bytes != 8) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_elements: unsupported "
            "indices_element_size=%d\n",
            indices_element_size_bytes);
    return -1;
  }

  GatherElementsGeom geom{};
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
      "[custom_kernels] hip_gather_elements: axis=%lld, rank=%lld, "
      "num=%lld, elem=%d, idx_elem=%d\n",
      (long long)axis, (long long)rank, (long long)num_elements,
      element_size_bytes, indices_element_size_bytes);

  switch (element_size_bytes) {
  case 2:
    return launch_gather_elements<__half>(
        stream, data, indices, output, num_elements, geom,
        indices_element_size_bytes);
  case 4:
    return launch_gather_elements<float>(
        stream, data, indices, output, num_elements, geom,
        indices_element_size_bytes);
  case 8:
    return launch_gather_elements<double>(
        stream, data, indices, output, num_elements, geom,
        indices_element_size_bytes);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_gather_elements: unsupported "
            "element_size=%d\n",
            element_size_bytes);
    return -1;
  }
}
