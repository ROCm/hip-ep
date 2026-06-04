/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_REAL_CONV1D_H
#define HIPDNN_EP_REAL_CONV1D_H

#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// 1D convolution via MIOpen.
//
// The MIOpen forward-convolution API is 4D-only (NCHW). We reinterpret the
// caller's NCL layout as NC[H=1]L and dispatch to the same path the existing
// 2D wrap_miopenConvolutionForward uses. Used by the Whisper encoder's two
// front-end Conv layers (the only ONNX ops with rank-3 input / weight today).
//
// Tensor layouts (NCL, fp16 today):
//   input  : [N, Cin,  Lin]
//   weights: [Cout, Cin, K]
//   bias   : [Cout]  (optional, may be NULL)
//   output : [N, Cout, Lout]      with Lout = (Lin + 2*pad - K) / stride + 1
//
// Workspace is drawn from the per-state `conv_scratch` pool
// (hipdnn_ep_state_ensure_conv_scratch / _get_conv_scratch) -- mirrors the
// qmoe_scratch pattern in runtime_state_internal.h: lazily grown, never
// shrinks, freed in hipdnn_ep_state_cleanup. The MIOpen handle is borrowed
// from the RuntimeState (same accessor used by wrap_miopenConvolutionForward).
//
// `element_size_bytes` is reserved for future fp32 / bf16 support; the
// implementation currently asserts elem_size == 2 (miopenHalf).
//
// Returns 0 on success, non-zero on error.
int wrap_conv1d(RuntimeState *state, const void *input, const void *weights,
                const void *bias, void *output, int64_t N, int64_t Cin,
                int64_t Lin, int64_t Cout, int64_t K, int64_t stride,
                int64_t pad, int64_t element_size_bytes);

#ifdef __cplusplus
}
#endif

#endif // HIPDNN_EP_REAL_CONV1D_H
