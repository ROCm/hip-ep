/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

template <typename Tin, typename Tout>
__global__ void cast_kernel_t(const Tin *__restrict__ input,
                               Tout *__restrict__ output,
                               int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<Tout>(input[idx]);
  }
}

// Specialisations for f16/bf16 conversions where static_cast doesn't
// compile cleanly across (vendor) HIP toolchains.
__global__ void cast_f32_to_bf16_kernel(const float *__restrict__ input,
                                         __hip_bfloat16 *__restrict__ output,
                                         int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __float2bfloat16(input[idx]);
}

__global__ void cast_bf16_to_f32_kernel(const __hip_bfloat16 *__restrict__ input,
                                         float *__restrict__ output,
                                         int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __bfloat162float(input[idx]);
}

__global__ void cast_f32_to_f16_kernel(const float *__restrict__ input,
                                        __half *__restrict__ output,
                                        int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __float2half(input[idx]);
}

__global__ void cast_f16_to_f32_kernel(const __half *__restrict__ input,
                                        float *__restrict__ output,
                                        int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __half2float(input[idx]);
}

__global__ void cast_i64_to_i32_kernel(
    const int64_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

__global__ void cast_i64_to_f32_kernel(
    const int64_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_f32_to_i64_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

__global__ void cast_i32_to_f32_kernel(
    const int32_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_f32_to_i32_kernel(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

__global__ void cast_i32_to_i64_kernel(
    const int32_t* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

__global__ void cast_i8_to_f32_kernel(
    const int8_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_f32_to_i8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int8_t>(input[idx]);
  }
}

namespace {

size_t dtype_size(int dtype) {
  switch (dtype) {
  case HIP_DTYPE_FLOAT32:
  case HIP_DTYPE_INT32:
    return 4;
  case HIP_DTYPE_FLOAT16:
  case HIP_DTYPE_BFLOAT16:
    return 2;
  case HIP_DTYPE_INT64:
    return 8;
  case HIP_DTYPE_INT8:
  case HIP_DTYPE_UINT8:
    return 1;
  default:
    return 0;
  }
}

// Host-side cast fallback: D2H the input, cast element-by-element on CPU,
// then H2D the output.  Only handles the integer type pairs that fail
// to launch on gfx1201.
template <typename Tin, typename Tout>
int host_cast_fallback(hipStream_t stream, const void *dev_in, void *dev_out,
                       int64_t n) {
  Tin *h_in = static_cast<Tin *>(malloc(n * sizeof(Tin)));
  Tout *h_out = static_cast<Tout *>(malloc(n * sizeof(Tout)));
  if (!h_in || !h_out) {
    free(h_in);
    free(h_out);
    return -1;
  }
  hipError_t err =
      hipMemcpy(h_in, dev_in, n * sizeof(Tin), hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    free(h_in);
    free(h_out);
    return static_cast<int>(err);
  }
  for (int64_t i = 0; i < n; i++)
    h_out[i] = static_cast<Tout>(h_in[i]);
  err = hipMemcpy(dev_out, h_out, n * sizeof(Tout), hipMemcpyHostToDevice);
  free(h_in);
  free(h_out);
  return static_cast<int>(err);
}

} // namespace

extern "C" int hip_cast(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int input_dtype,
    int output_dtype) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError(); // clear stale errors

#define CAST_LAUNCH(_K, _Tin, _Tout)                                            \
  hipLaunchKernelGGL((_K), dim3(grid_size), dim3(kBlockSize), 0, hip_stream,    \
                     static_cast<const _Tin *>(input),                          \
                     static_cast<_Tout *>(output), num_elements)

  // Track whether we attempted a launch (for error checking below).
  bool launched = true;

  if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_INT32) {
    CAST_LAUNCH(cast_i64_to_i32_kernel, int64_t, int32_t);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_INT64) {
    CAST_LAUNCH(cast_f32_to_i64_kernel, float, int64_t);
  } else if (input_dtype == HIP_DTYPE_INT64 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH(cast_i64_to_f32_kernel, int64_t, float);
  } else if (input_dtype == HIP_DTYPE_INT32 &&
             output_dtype == HIP_DTYPE_INT64) {
    CAST_LAUNCH(cast_i32_to_i64_kernel, int32_t, int64_t);
  } else if (input_dtype == HIP_DTYPE_INT32 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH(cast_i32_to_f32_kernel, int32_t, float);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_INT32) {
    CAST_LAUNCH(cast_f32_to_i32_kernel, float, int32_t);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_BFLOAT16) {
    CAST_LAUNCH(cast_f32_to_bf16_kernel, float, __hip_bfloat16);
  } else if (input_dtype == HIP_DTYPE_BFLOAT16 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH(cast_bf16_to_f32_kernel, __hip_bfloat16, float);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_FLOAT16) {
    CAST_LAUNCH(cast_f32_to_f16_kernel, float, __half);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH(cast_f16_to_f32_kernel, __half, float);
  } else if ((input_dtype == HIP_DTYPE_INT8 || input_dtype == HIP_DTYPE_UINT8) &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH(cast_i8_to_f32_kernel, int8_t, float);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             (output_dtype == HIP_DTYPE_INT8 || output_dtype == HIP_DTYPE_UINT8)) {
    CAST_LAUNCH(cast_f32_to_i8_kernel, float, int8_t);
  } else if (input_dtype == output_dtype) {
    launched = false;
    size_t esz = dtype_size(input_dtype);
    if (esz == 0) {
      fprintf(stderr,
              "[custom_kernels] hip_cast identity: unsupported dtype=%d\n",
              input_dtype);
      return -1;
    }
    return static_cast<int>(hipMemcpyAsync(output, input,
                                            num_elements * esz,
                                            hipMemcpyDeviceToDevice,
                                            hip_stream));
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_cast: unsupported conversion "
            "input_dtype=%d -> output_dtype=%d\n",
            input_dtype, output_dtype);
    return -1;
  }
#undef CAST_LAUNCH

  if (!launched)
    return 0;

  hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;

  // GPU kernel launch failed -- fall back to host-side cast.
  fprintf(stderr,
          "[custom_kernels] hip_cast GPU launch failed (%s), "
          "using host fallback for n=%lld src=%d dst=%d\n",
          hipGetErrorString(err), (long long)num_elements,
          input_dtype, output_dtype);

  (void)hipGetLastError(); // clear the error before host fallback

  // Dispatch host fallback for common integer conversions.
  if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_FLOAT32)
    return host_cast_fallback<int64_t, float>(hip_stream, input, output,
                                              num_elements);
  if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_INT64)
    return host_cast_fallback<float, int64_t>(hip_stream, input, output,
                                              num_elements);
  if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_INT32)
    return host_cast_fallback<int64_t, int32_t>(hip_stream, input, output,
                                                num_elements);
  if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_INT64)
    return host_cast_fallback<int32_t, int64_t>(hip_stream, input, output,
                                                num_elements);
  if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_FLOAT32)
    return host_cast_fallback<int32_t, float>(hip_stream, input, output,
                                              num_elements);
  if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_INT32)
    return host_cast_fallback<float, int32_t>(hip_stream, input, output,
                                              num_elements);
  if ((input_dtype == HIP_DTYPE_INT8 || input_dtype == HIP_DTYPE_UINT8) &&
      output_dtype == HIP_DTYPE_FLOAT32)
    return host_cast_fallback<int8_t, float>(hip_stream, input, output,
                                             num_elements);
  if (input_dtype == HIP_DTYPE_FLOAT32 &&
      (output_dtype == HIP_DTYPE_INT8 || output_dtype == HIP_DTYPE_UINT8))
    return host_cast_fallback<float, int8_t>(hip_stream, input, output,
                                             num_elements);

  fprintf(stderr,
          "[custom_kernels] hip_cast: no host fallback for "
          "input_dtype=%d -> output_dtype=%d\n",
          input_dtype, output_dtype);
  return static_cast<int>(err);
}
