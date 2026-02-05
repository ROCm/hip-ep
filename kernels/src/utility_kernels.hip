// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Utility HIP Kernels
// ===================
// Helper GPU operations for memory management and data movement.
//
// Functions:
//   - launch_reshape_copy_f32: Device-to-device memory copy for reshape

#include <hip/hip_runtime.h>

//============================================================================
// Utility Launcher Functions (extern "C" for C++ interop)
//============================================================================

extern "C" {

// Reshape is a no-op if in-place, otherwise performs device-to-device copy
void launch_reshape_copy_f32(
    const float* input,
    float* output,
    int64_t size,
    hipStream_t stream) {

  if (input != output) {
    hipMemcpyAsync(output, input, size * sizeof(float), hipMemcpyDeviceToDevice, stream);
  }
}

} // extern "C"
