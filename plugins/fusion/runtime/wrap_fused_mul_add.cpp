/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- wrap_fused_mul_add.cpp - Runtime wrapper for hip.fused_mul_add -----===//
//
// Touchpoints F + G: C-ABI runtime symbol + device-kernel dispatch.
//
// This file provides the `wrap_fused_mul_add` symbol that the HipToLLVM
// lowering (FusedMulAddLowering.cpp) emits a call to. It:
//   F. Is a `wrap_*` C-ABI symbol linked into the plugin .so and resolved at
//      runtime by the generated code (via addLibrary/addLibraryPath).
//   G. Dispatches to `hip_fused_mul_add_launch` (fused_mul_add_kernel.hip)
//      which contains the actual gfx1103 HIP kernel.
//
// RuntimeState is an opaque pointer matching the in-tree hipdnn_ep_runtime.h
// layout (the compiler only passes it through; we call getter functions).
//
// For the PoC, the state getters are declared extern so the symbol resolves
// from the host process (hip-compiler.dll / libhip-compiler.so) at dlopen
// time — the same pattern used by all in-tree wrap_* functions.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

// Forward-declare the HIP kernel launcher from fused_mul_add_kernel.hip.
// hipStream_t is typedef'd as void* in HIP C headers; use void* here to avoid
// pulling in all HIP headers in the C++ runtime wrapper.
extern "C" int hip_fused_mul_add_launch(void *stream,
                                        const void *x,
                                        const void *mul_operand,
                                        const void *add_operand,
                                        void *output,
                                        int64_t num_elements,
                                        int data_type);

// State getter — resolved from the host process at dlopen time.
// Matches hipdnn_ep_state_get_stream() in hipdnn_ep_runtime.h.
extern "C" void *hipdnn_ep_state_get_stream(void *state);

//===----------------------------------------------------------------------===//
// Public wrap_* symbol — called by the generated LLVM IR
//===----------------------------------------------------------------------===//

/// Touchpoints F + G: runtime wrapper + kernel dispatch.
///
/// \param state   Opaque RuntimeState* (from !hip.context lowering)
/// \param x           pointer to x operand (contiguous, device memory)
/// \param mul_operand pointer to mul operand (b in b*x+a)
/// \param add_operand pointer to add operand (a in b*x+a)
/// \param output      pointer to output buffer
/// \param num_elements total element count (pre-computed by the lowering)
/// \param data_type   HIPDNN_EP_DATATYPE_* enum value (0=f32, 1=f16, 2=bf16)
///
/// \returns 0 on success, non-zero on error.
extern "C" int wrap_fused_mul_add(void *state,
                                  const void *x,
                                  const void *mul_operand,
                                  const void *add_operand,
                                  void *output,
                                  int64_t num_elements,
                                  int64_t data_type) {
  if (!state || !x || !mul_operand || !add_operand || !output)
    return -1;
  if (num_elements <= 0)
    return 0;

  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_fused_mul_add_launch(stream, x, mul_operand, add_operand, output,
                                  num_elements, static_cast<int>(data_type));
}
