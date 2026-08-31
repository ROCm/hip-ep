// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Error paths of -hipsr-materialize-init-tensors. New cases go here rather than
// into a new file; positive coverage lives in materialize-init-tensors.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-materialize-init-tensors

// The allocation is sized from the shape region, so an empty one leaves nothing
// to compute it from. The op verifier allows the region to be empty because
// -hipsr-populate-shape-region fills it later.
func.func @unpopulated_shape_region(%ctx: !hipsr.context, %a: tensor<?x256xf16, #hipsr.mem<device>>,
                                    %b: tensor<256x512xf16, #hipsr.mem<device>>)
    -> tensor<?x512xf16, #hipsr.mem<device>> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16, #hipsr.mem<device>>,
       %domain_b: tensor<256x512xf16, #hipsr.mem<device>>):
    // expected-error@+1 {{shape region must be populated by -hipsr-populate-shape-region}}
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16, #hipsr.mem<device>>
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>)
        outs(%init : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield %matmul : tensor<?x512xf16, #hipsr.mem<device>>
  } -> tensor<?x512xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %0 : tensor<?x512xf16, #hipsr.mem<device>>
}

// -----

// A barrier reads its inputs as buffers, and the pool hands a buffer out only
// after the last allocation in the domain, so an input allocated here has no
// value to name yet. The placeholder verifier allows the edge, because a
// placeholder result is a legal shape-graph input, and
// -hipsr-partition-pool-domains never builds it, because it starts a barrier one
// domain past every input. That leaves the pass to reject it.
func.func @barrier_over_placeholder(%ctx: !hipsr.context, %in: tensor<?x1xf16, #hipsr.mem<device>>)
    -> tensor<?x1xf16, #hipsr.mem<device>> {
  %0 = hipsr.pool_domain(%ctx, %in
      : !hipsr.context, tensor<?x1xf16, #hipsr.mem<device>>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_in: tensor<?x1xf16, #hipsr.mem<device>>):
    %cast_init = hipsr.placeholder(%domain_ctx)
        ins(%domain_in : tensor<?x1xf16, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x1xf32, #hipsr.mem<device>> shape_region {
    ^bb0(%in_shape: tensor<2xindex>):
      hipsr.shape_yield %in_shape : tensor<2xindex>
    }
    %cast = hipsr.cast(%domain_ctx)
        ins(%domain_in : tensor<?x1xf16, #hipsr.mem<device>>)
        outs(%cast_init : tensor<?x1xf32, #hipsr.mem<device>>) : tensor<?x1xf32, #hipsr.mem<device>>
    // expected-error@+1 {{barrier input must be allocated outside this pool domain}}
    %barrier_init = hipsr.placeholder(%domain_ctx)
        ins(%cast_init : tensor<?x1xf32, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<barrier>}
        : tensor<?x1xf16, #hipsr.mem<device>> shape_region {
    ^bb0(%region_ctx: !hipsr.context, %region_cast: tensor<?x1xf32, #hipsr.mem<device>>):
      %cast_shape = shape.shape_of %region_cast
          : tensor<?x1xf32, #hipsr.mem<device>> -> tensor<2xindex>
      hipsr.shape_yield %cast_shape : tensor<2xindex>
    }
    %out = hipsr.cast(%domain_ctx)
        ins(%cast : tensor<?x1xf32, #hipsr.mem<device>>)
        outs(%barrier_init : tensor<?x1xf16, #hipsr.mem<device>>) : tensor<?x1xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield %out : tensor<?x1xf16, #hipsr.mem<device>>
  } -> tensor<?x1xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %0 : tensor<?x1xf16, #hipsr.mem<device>>
}
