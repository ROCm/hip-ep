/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPDNN_EP_DYNAMIC_DISPATCH_H
#define HIPDNN_EP_DYNAMIC_DISPATCH_H

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// DynamicDispatch Backend Wrappers
//===----------------------------------------------------------------------===//
//
// These wrappers expose AMD DynamicDispatch (Vitis AI / XRT NPU backend)
// operators through a C ABI that matches the existing GPU backend pattern
// (MIOpen, hipBLASLt). The lowering in lib/Conversion/HipToLLVM emits calls
// to these wrap_dd_* functions when --use-dynamic-dispatch is enabled.
//
// DynamicDispatch uses a C++ object-oriented API with operators inheriting from
// OpInterface. We bridge to C by storing operator instances in RuntimeState's
// op_state slots (similar to GQA/MHA GEMM descriptor caches).
//
// Architecture:
//   HIP Dialect Op → LLVM IR (wrap_dd_* call) → C++ Wrapper → DD Operator
//
//===----------------------------------------------------------------------===//

struct RuntimeState;

// GEMM / MatMul operation via DynamicDispatch combined_gemm operator
// Maps to ryzenai::combined_gemm<InT, WtT, OutT>
//
// Parameters:
//   state: RuntimeState containing XRT context and op_state slots
//   op_state_slot: Op-state slot index for caching DD operator instance
//   input_a: Left matrix (M x K)
//   input_b: Right matrix (K x N) or weights (pre-quantized if using QDQ)
//   bias: Optional bias vector (can be NULL)
//   output: Result matrix (M x N), allocated by caller
//   M, N, K: Matrix dimensions
//   alpha, beta: Scaling factors (typically alpha=1.0, beta=0.0 for C=A*B)
//   transA, transB: Transpose flags (0=no transpose, 1=transpose)
//   data_type: HIPDNN_EP_DATATYPE_* value
//
// Returns: 0 on success, negative on error
int wrap_dd_matmul(struct RuntimeState *state, int32_t op_state_slot,
                   const void *input_a, const void *input_b, const void *bias,
                   void *output, int64_t M, int64_t N, int64_t K, double alpha,
                   double beta, int64_t transA, int64_t transB,
                   int64_t data_type);

// Convolution operation via DynamicDispatch iconv operator
// Maps to ryzenai::iconv<InT, WtT, OutT>
//
// Parameters:
//   state: RuntimeState containing XRT context
//   op_state_slot: Op-state slot index for caching DD operator instance
//   input: Input tensor (N x C x H x W)
//   weights: Filter weights (K x C/group x kH x kW)
//   bias: Optional bias (can be NULL)
//   output: Output tensor (N x K x outH x outW), allocated by caller
//   n, c, h, w: Input dimensions
//   k: Number of output channels (filters)
//   out_h, out_w: Output spatial dimensions
//   kernel_h, kernel_w: Filter dimensions
//   stride_h, stride_w: Convolution strides
//   pad_top, pad_left, pad_bottom, pad_right: Padding
//   dilation_h, dilation_w: Dilation factors
//   group: Number of groups (1=standard conv, c=depthwise)
//   data_type: HIPDNN_EP_DATATYPE_* value
//
// Returns: 0 on success, negative on error
int wrap_dd_conv2d(struct RuntimeState *state, int32_t op_state_slot,
                   const void *input, int64_t n, int64_t c, int64_t h,
                   int64_t w, const void *weights, int64_t k, const void *bias,
                   void *output, int64_t out_h, int64_t out_w,
                   int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                   int64_t stride_w, int64_t pad_top, int64_t pad_left,
                   int64_t pad_bottom, int64_t pad_right, int64_t dilation_h,
                   int64_t dilation_w, int64_t group, int64_t data_type);

#ifdef __cplusplus
}
#endif

#endif // HIPDNN_EP_DYNAMIC_DISPATCH_H
