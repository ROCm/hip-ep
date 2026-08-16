/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>

namespace {

template <typename T>
__global__ void hip_slice_kernel(const T *__restrict__ input,
                                 T *__restrict__ output,
                                 const int64_t *__restrict__ metadata, int rank,
                                 int64_t numOutputElements) {
  int64_t outputIndex =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (outputIndex >= numOutputElements)
    return;

  const int64_t *inputStrides = metadata;
  const int64_t *outputStrides = metadata + rank;
  const int64_t *starts = metadata + 2 * rank;
  const int64_t *steps = metadata + 3 * rank;

  int64_t remaining = outputIndex;
  int64_t inputOffset = 0;
  for (int axis = 0; axis < rank; ++axis) {
    int64_t coordinate = remaining / outputStrides[axis];
    remaining -= coordinate * outputStrides[axis];
    inputOffset += (starts[axis] + coordinate * steps[axis]) *
                   inputStrides[axis];
  }
  output[outputIndex] = input[inputOffset];
}

bool computeStrides(const int64_t *shape, int rank, int64_t *strides,
                    int64_t &elements) {
  elements = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    if (shape[axis] < 0)
      return false;
    strides[axis] = elements;
    if (shape[axis] != 0 &&
        elements > std::numeric_limits<int64_t>::max() / shape[axis])
      return false;
    elements *= shape[axis];
  }
  return true;
}

template <typename T>
void launch(hipStream_t stream, const void *input, void *output,
            const int64_t *metadata, int rank, int64_t elements,
            uint32_t gridSize) {
  constexpr int kBlockSize = 256;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<T>), dim3(gridSize),
                     dim3(kBlockSize), 0, stream,
                     static_cast<const T *>(input), static_cast<T *>(output),
                     metadata, rank, elements);
}

void freeHostMetadata(void *metadata) { std::free(metadata); }

} // namespace

extern "C" int hip_slice(void *stream, const void *input, void *output,
                         const int64_t *input_shape_host,
                         const int64_t *output_shape_host,
                         const int64_t *starts_per_axis_host,
                         const int64_t *steps_per_axis_host, int rank,
                         int hip_dtype, void *metadata_scratch_device,
                         size_t metadata_scratch_bytes) {
  if (rank < 0 ||
      (rank > 0 &&
       (!input_shape_host || !output_shape_host || !starts_per_axis_host ||
        !steps_per_axis_host || !metadata_scratch_device)) ||
      !input || !output)
    return -1;
  if (static_cast<uint64_t>(rank) >
      std::numeric_limits<size_t>::max() / (4 * sizeof(int64_t)))
    return -1;
  size_t requiredBytes =
      static_cast<size_t>(rank) * 4 * sizeof(int64_t);
  if (metadata_scratch_bytes < requiredBytes)
    return -1;

  int64_t *metadata = nullptr;
  if (requiredBytes > 0) {
    metadata = static_cast<int64_t *>(std::malloc(requiredBytes));
    if (!metadata)
      return -1;
  }
  int64_t inputElements = 0;
  int64_t outputElements = 0;
  int64_t *outputStrides = metadata ? metadata + rank : nullptr;
  if (!computeStrides(input_shape_host, rank, metadata, inputElements) ||
      !computeStrides(output_shape_host, rank, outputStrides, outputElements)) {
    std::free(metadata);
    return -1;
  }
  (void)inputElements;
  for (int axis = 0; axis < rank; ++axis) {
    metadata[2 * rank + axis] = starts_per_axis_host[axis];
    metadata[3 * rank + axis] = steps_per_axis_host[axis];
  }
  if (outputElements == 0) {
    std::free(metadata);
    return 0;
  }

  constexpr int64_t kBlockSize = 256;
  uint64_t gridSize =
      (static_cast<uint64_t>(outputElements) + kBlockSize - 1) / kBlockSize;
  if (gridSize == 0 || gridSize > std::numeric_limits<uint32_t>::max()) {
    std::free(metadata);
    return -1;
  }
  if (hip_dtype != HIP_DTYPE_FLOAT16 && hip_dtype != HIP_DTYPE_FLOAT32 &&
      hip_dtype != HIP_DTYPE_INT32 && hip_dtype != HIP_DTYPE_INT64) {
    fprintf(stderr, "[custom_kernels] hip_slice: unsupported dtype=%d\n",
            hip_dtype);
    std::free(metadata);
    return -1;
  }

  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  if (requiredBytes > 0) {
    hipError_t copyError =
        hipMemcpyAsync(metadata_scratch_device, metadata, requiredBytes,
                       hipMemcpyHostToDevice, hipStream);
    if (copyError != hipSuccess) {
      fprintf(stderr, "[custom_kernels] hip_slice metadata H2D failed: %s\n",
              hipGetErrorString(copyError));
      std::free(metadata);
      return static_cast<int>(copyError);
    }
  }

  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    launch<__half>(hipStream, input, output,
                   static_cast<const int64_t *>(metadata_scratch_device), rank,
                   outputElements, static_cast<uint32_t>(gridSize));
    break;
  case HIP_DTYPE_FLOAT32:
    launch<float>(hipStream, input, output,
                  static_cast<const int64_t *>(metadata_scratch_device), rank,
                  outputElements, static_cast<uint32_t>(gridSize));
    break;
  case HIP_DTYPE_INT32:
    launch<int32_t>(hipStream, input, output,
                    static_cast<const int64_t *>(metadata_scratch_device), rank,
                    outputElements, static_cast<uint32_t>(gridSize));
    break;
  case HIP_DTYPE_INT64:
    launch<int64_t>(hipStream, input, output,
                    static_cast<const int64_t *>(metadata_scratch_device), rank,
                    outputElements, static_cast<uint32_t>(gridSize));
    break;
  }

  hipError_t launchError = hipGetLastError();
  if (launchError != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_slice launch failed: %s\n",
            hipGetErrorString(launchError));
    (void)hipStreamSynchronize(hipStream);
    std::free(metadata);
    return static_cast<int>(launchError);
  }
  if (metadata) {
    hipError_t cleanupError =
        hipLaunchHostFunc(hipStream, freeHostMetadata, metadata);
    if (cleanupError != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_slice metadata cleanup enqueue failed: "
              "%s\n",
              hipGetErrorString(cleanupError));
      (void)hipStreamSynchronize(hipStream);
      std::free(metadata);
      return static_cast<int>(cleanupError);
    }
  }
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_slice: dtype=%d rank=%d num_out=%lld\n", hip_dtype,
      rank, (long long)outputElements);
  return 0;
}
