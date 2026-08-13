// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file -hipsr-populate-shape-region --verify-diagnostics

// The pass derives a shape from the placeholder's consumer, which needs a
// per-operation recipe. A hipsr.compute result shape follows whatever its body
// does, so there is nothing to dispatch on: the conversion that builds a
// compute populates the region itself, and this pass then skips it.
func.func @compute_consumer(%ctx: !hipsr.context, %input: tensor<2x3xf16>)
    -> tensor<6xf16> {
  // expected-error @+1 {{shape-region population does not support consumer hipsr.compute}}
  %init = hipsr.placeholder(%ctx)
      ins(%input : tensor<2x3xf16>)
      {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x3xf16>
  %out = hipsr.compute(%ctx) ins(%input : tensor<2x3xf16>)
                             outs(%init : tensor<2x3xf16>) {
  ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>,
       %dest: tensor<2x3xf16>):
    %flat = tensor.collapse_shape %in [[0, 1]]
        : tensor<2x3xf16> into tensor<6xf16>
    hipsr.compute_yield %flat : tensor<6xf16>
  } : tensor<6xf16>
  return %out : tensor<6xf16>
}
