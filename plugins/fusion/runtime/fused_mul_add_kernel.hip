/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- fused_mul_add_kernel.hip - gfx1103 kernel: (b*x)+a ----------------===//
//
// Touchpoint G: device kernel for hip.fused_mul_add on gfx1103 (Radeon 780M).
//
// Computes:  output[i] = (mul_operand[i] * x[i]) + add_operand[i]
//
// Dispatched by wrap_fused_mul_add (runtime/wrap_fused_mul_add.cpp) via
// hipLaunchKernelGGL. Supports f32, f16, bf16 element types.
//
// Compiled with hipcc --offload-arch=gfx1103 (gated by BUILD_HIP_PLUGINS).
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// Block size chosen for gfx1103 (RDNA3) — 256 threads fills one CU wave
// without spills for this register-light kernel.
static constexpr int kBlockSize = 256;

template <typename T>
__global__ void fused_mul_add_kernel(const T *__restrict__ x,
                                     const T *__restrict__ mul_operand,
                                     const T *__restrict__ add_operand,
                                     T *__restrict__ output,
                                     int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = mul_operand[idx] * x[idx] + add_operand[idx];
}

// Data type tag matching HIPDNN_EP_DATATYPE_* in hipdnn_ep_runtime.h
enum FusedMulAddDataType {
  FMAD_F32     = 0,
  FMAD_F16     = 1,
  FMAD_BF16    = 2,
};

// C-callable launcher — called from wrap_fused_mul_add.
// Returns 0 on success, non-zero on hipError.
extern "C" int hip_fused_mul_add_launch(hipStream_t stream,
                                        const void *x,
                                        const void *mul_operand,
                                        const void *add_operand,
                                        void *output,
                                        int64_t num_elements,
                                        int data_type) {
  if (num_elements <= 0)
    return 0;

  dim3 block(kBlockSize);
  dim3 grid((num_elements + kBlockSize - 1) / kBlockSize);

  hipError_t err = hipSuccess;
  switch (data_type) {
  case FMAD_F32:
    hipLaunchKernelGGL(fused_mul_add_kernel<float>, grid, block, 0, stream,
                       static_cast<const float *>(x),
                       static_cast<const float *>(mul_operand),
                       static_cast<const float *>(add_operand),
                       static_cast<float *>(output), num_elements);
    break;
  case FMAD_F16:
    hipLaunchKernelGGL(fused_mul_add_kernel<__half>, grid, block, 0, stream,
                       static_cast<const __half *>(x),
                       static_cast<const __half *>(mul_operand),
                       static_cast<const __half *>(add_operand),
                       static_cast<__half *>(output), num_elements);
    break;
  case FMAD_BF16:
    hipLaunchKernelGGL(fused_mul_add_kernel<hip_bfloat16>, grid, block, 0,
                       stream,
                       static_cast<const hip_bfloat16 *>(x),
                       static_cast<const hip_bfloat16 *>(mul_operand),
                       static_cast<const hip_bfloat16 *>(add_operand),
                       static_cast<hip_bfloat16 *>(output), num_elements);
    break;
  default:
    return -1; // unsupported dtype
  }
  err = hipGetLastError();
  return (err == hipSuccess) ? 0 : static_cast<int>(err);
}
