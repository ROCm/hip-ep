/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone, single-kernel HIP source for the DX12 smoke-test driver
 * (../../hip_ep_dx12_driver.cpp). Logic mirrors elementwise_add_kernel<float>
 * in lib/Runtime/Kernels/hip/elementwise_binary_kernel.hip -- the kernel
 * hip-ep's own runtime launches for a HIP-dialect `hip.add` op -- but is
 * pulled out as its own extern "C" translation unit so it can be compiled
 * standalone to a relocatable/linked AMDGPU code object (see
 * build_add_kernel.ps1) instead of being embedded in the host runtime DLL.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

extern "C" __global__ void hip_ep_add_f32(const float* __restrict__ lhs,
                                           const float* __restrict__ rhs,
                                           float* __restrict__ output,
                                           int64_t num_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < num_elements)
    {
        output[idx] = lhs[idx] + rhs[idx];
    }
}