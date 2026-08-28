// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Error paths of -hipsr-materialize-init-tensors. Later phases add cases here
// rather than a new file; positive coverage lives in
// materialize-init-tensors.mlir. Cases that only record a gap in the current
// implementation move over to the positive file once that gap is closed.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-materialize-init-tensors

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

// A barrier shape region reads data values rather than shapes, so its arguments
// cannot be replaced with shape.shape_of. Until that path is implemented the
// pass rejects the placeholder instead of building a wrong shape graph.
func.func @barrier_placeholder(%ctx: !hipsr.context, %a: tensor<?x256xf16, #hipsr.mem<device>>,
                               %b: tensor<256x512xf16, #hipsr.mem<device>>) -> tensor<?x512xf16, #hipsr.mem<device>> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16, #hipsr.mem<device>>,
       %domain_b: tensor<256x512xf16, #hipsr.mem<device>>):
    // expected-error@+1 {{barrier placeholders are not materialized yet}}
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>)
        {placeholder_type = #hipsr.placeholder_type<barrier>}
        : tensor<?x512xf16, #hipsr.mem<device>> shape_region {
    ^bb0(%region_ctx: !hipsr.context, %region_a: tensor<?x256xf16, #hipsr.mem<device>>,
         %region_b: tensor<256x512xf16, #hipsr.mem<device>>):
      %a_shape = shape.shape_of %region_a : tensor<?x256xf16, #hipsr.mem<device>> -> !shape.shape
      %b_shape = shape.shape_of %region_b : tensor<256x512xf16, #hipsr.mem<device>> -> !shape.shape
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %result_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield %result_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16, #hipsr.mem<device>>, tensor<256x512xf16, #hipsr.mem<device>>)
        outs(%init : tensor<?x512xf16, #hipsr.mem<device>>) : tensor<?x512xf16, #hipsr.mem<device>>
    hipsr.pool_domain_yield %matmul : tensor<?x512xf16, #hipsr.mem<device>>
  } -> tensor<?x512xf16, #hipsr.mem<device>> {domain_id = 0 : i64}
  return %0 : tensor<?x512xf16, #hipsr.mem<device>>
}
