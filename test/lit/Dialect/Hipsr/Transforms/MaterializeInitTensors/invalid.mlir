// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Error paths of -hipsr-materialize-init-tensors. Later phases add cases here
// rather than a new file; positive coverage lives in
// materialize-init-tensors.mlir.
//
// The pass consumes shape regions, so it rejects a placeholder whose region was
// never filled instead of silently dropping its shape computation.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-materialize-init-tensors

func.func @unpopulated_shape_region(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                                    %b: tensor<256x512xf16>)
    -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>):
    // expected-error@+1 {{shape region must be populated by -hipsr-populate-shape-region}}
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16>
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %matmul : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}
