/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ConstantOfShape (scalar splat).
//
// Single-launch fill kernel that writes a scalar value into every element
// of `output`. The value is passed as a raw int64 bit pattern + dtype tag
// so that the wrapper does not need a separate scalar pointer (which would
// require either a tiny D2H or a per-call hipMalloc to stage to GPU).
//
// One thread per output element; bandwidth-bound; no LDS. Block size 256
// matches the rest of the in-tree elementwise kernels (range / nonzero).

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

template <typename T>
__global__ void constant_of_shape_kernel(T *out, int64_t n, T value) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = value;
}

extern "C" int hip_constant_of_shape(void *stream, void *output,
                                     int64_t num_elements, int64_t value_bits,
                                     int hip_dtype) {
  if (!output) {
    fprintf(stderr,
            "[custom_kernels] hip_constant_of_shape: null output pointer\n");
    return -1;
  }
  if (num_elements <= 0) {
    // Empty output -> nothing to do. The dispatcher above us would have
    // skipped this call already, but be defensive.
    return 0;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  const uint64_t grid_size_u64 =
      static_cast<uint64_t>((num_elements + kBlockSize - 1) / kBlockSize);
  if (grid_size_u64 > std::numeric_limits<uint32_t>::max()) {
    fprintf(stderr,
            "[custom_kernels] hip_constant_of_shape: num_elements too large "
            "for grid (%lld)\n",
            (long long)num_elements);
    return -1;
  }
  const uint32_t grid_size = static_cast<uint32_t>(grid_size_u64);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_constant_of_shape: num_elements=%lld dtype=%d "
      "grid=%u block=%d\n",
      (long long)num_elements, hip_dtype, grid_size, kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32: {
    // The compiler-side caller bit-casts a float into the low 32 bits of
    // value_bits via memcpy, matching IEEE-754 layout.
    uint32_t bits = static_cast<uint32_t>(value_bits & 0xFFFFFFFFu);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    hipLaunchKernelGGL(constant_of_shape_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<float *>(output), num_elements, v);
    break;
  }
  case HIP_DTYPE_FLOAT16: {
    // __half has the same layout as uint16_t; bit-cast from the low 16
    // bits of value_bits. `__half_raw` would do the same but adds an
    // include dependency.
    uint16_t bits = static_cast<uint16_t>(value_bits & 0xFFFFu);
    __half v;
    std::memcpy(&v, &bits, sizeof(v));
    hipLaunchKernelGGL(constant_of_shape_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<__half *>(output), num_elements, v);
    break;
  }
  case HIP_DTYPE_BFLOAT16: {
    uint16_t bits = static_cast<uint16_t>(value_bits & 0xFFFFu);
    __hip_bfloat16 v;
    std::memcpy(&v, &bits, sizeof(v));
    hipLaunchKernelGGL(constant_of_shape_kernel<__hip_bfloat16>,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<__hip_bfloat16 *>(output), num_elements, v);
    break;
  }
  case HIP_DTYPE_FLOAT64: {
    uint64_t bits = static_cast<uint64_t>(value_bits);
    double v;
    std::memcpy(&v, &bits, sizeof(v));
    hipLaunchKernelGGL(constant_of_shape_kernel<double>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<double *>(output), num_elements, v);
    break;
  }
  case HIP_DTYPE_INT8: {
    // Also covers ONNX bool — the spec value attribute will have already
    // been clamped to 0 or 1 by the caller.
    int8_t v = static_cast<int8_t>(value_bits & 0xFFLL);
    hipLaunchKernelGGL(constant_of_shape_kernel<int8_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<int8_t *>(output), num_elements, v);
    break;
  }
  case HIP_DTYPE_INT16: {
    int16_t v = static_cast<int16_t>(value_bits & 0xFFFFLL);
    hipLaunchKernelGGL(constant_of_shape_kernel<int16_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<int16_t *>(output), num_elements, v);
    break;
  }
  case HIP_DTYPE_INT32: {
    int32_t v = static_cast<int32_t>(value_bits & 0xFFFFFFFFLL);
    hipLaunchKernelGGL(constant_of_shape_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<int32_t *>(output), num_elements, v);
    break;
  }
  case HIP_DTYPE_INT64: {
    int64_t v = static_cast<int64_t>(value_bits);
    hipLaunchKernelGGL(constant_of_shape_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<int64_t *>(output), num_elements, v);
    break;
  }
  default:
    fprintf(stderr,
            "[custom_kernels] hip_constant_of_shape: unsupported "
            "hip_dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_constant_of_shape launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
