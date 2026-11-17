// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include <hip/hip_runtime.h>

// Keep the production wrapper and CPU reference in a host C++ translation
// unit: Windows HIP/MSVC std::min overloads otherwise share a mangled name.
static __global__ void holdStreamKernel(const volatile int *release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin)
    __builtin_amdgcn_s_sleep(127);
}
extern "C" void qmoeUtHoldStream(hipStream_t stream, const int *release) {
  holdStreamKernel<<<1, 1, 0, stream>>>(release);
}
