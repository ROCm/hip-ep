/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

#include "rtc/hiprtc_module.h"

// The __global__ definitions live in a header so the AOT build below and the
// hipRTC path compile byte-identical source.
#include "hip/rtc/elementwise_sub_device.h"

// Same header, embedded as bytes at build time (see Kernels/CMakeLists.txt).
extern "C" const unsigned char kElementwiseSubDeviceSource[];
extern "C" const size_t kElementwiseSubDeviceSource_size;

namespace {

const char* rtcKernelName(int hip_dtype) {
  switch (hip_dtype) {
    case HIP_DTYPE_INT64:   return "elementwise_sub_i64_kernel";
    case HIP_DTYPE_INT32:   return "elementwise_sub_i32_kernel";
    case HIP_DTYPE_FLOAT32: return "elementwise_sub_f32_kernel";
    case HIP_DTYPE_FLOAT16: return "elementwise_sub_f16_kernel";
    default:                return nullptr;
  }
}

hipdnn_ep::rtc::Module& rtcModule() {
  static hipdnn_ep::rtc::Module module(
      "elementwise_sub",
      reinterpret_cast<const char*>(kElementwiseSubDeviceSource),
      kElementwiseSubDeviceSource_size,
      {"elementwise_sub_i64_kernel", "elementwise_sub_i32_kernel",
       "elementwise_sub_f32_kernel", "elementwise_sub_f16_kernel"});
  return module;
}

// Argument packing is unchecked by the compiler here -- the order and types
// must track the __global__ signatures in the device header by hand.
int launchRtc(int hip_dtype, hipStream_t stream, int grid_size, int block_size,
              const void* lhs, const void* rhs, void* output,
              int64_t num_elements) {
  const char* kernel_name = rtcKernelName(hip_dtype);
  if (!kernel_name) return -1;

  hipFunction_t fn = rtcModule().getFunction(kernel_name);
  if (!fn) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sub: RTC kernel %s unavailable\n",
            kernel_name);
    return -1;
  }

  void* args[] = {&lhs, &rhs, &output, &num_elements};
  hipError_t err = hipModuleLaunchKernel(fn, grid_size, 1, 1, block_size, 1, 1,
                                         0, stream, args, nullptr);
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sub RTC launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

}  // namespace

extern "C" int hip_elementwise_sub(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  if (hipdnn_ep::rtc::enabled() && rtcKernelName(hip_dtype) != nullptr) {
    return launchRtc(hip_dtype, hip_stream, grid_size, kBlockSize, lhs, rhs,
                     output, num_elements);
  }

  switch (hip_dtype) {
    case HIP_DTYPE_INT64:
      hipLaunchKernelGGL(elementwise_sub_i64_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t*>(lhs),
                         static_cast<const int64_t*>(rhs),
                         static_cast<int64_t*>(output), num_elements);
      break;
    case HIP_DTYPE_INT32:
      hipLaunchKernelGGL(elementwise_sub_i32_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int32_t*>(lhs),
                         static_cast<const int32_t*>(rhs),
                         static_cast<int32_t*>(output), num_elements);
      break;
    case HIP_DTYPE_FLOAT32:
      hipLaunchKernelGGL(elementwise_sub_f32_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const float*>(lhs),
                         static_cast<const float*>(rhs),
                         static_cast<float*>(output), num_elements);
      break;
    case HIP_DTYPE_FLOAT16:
      hipLaunchKernelGGL(elementwise_sub_f16_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const __half*>(lhs),
                         static_cast<const __half*>(rhs),
                         static_cast<__half*>(output), num_elements);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_sub: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sub launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
