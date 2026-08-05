// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// -hipsr-materialize-init-tensors resolves as a pass name and runs over a pool
// domain that holds a populated placeholder, leaving the domain and its data op
// intact. The materialization itself (placeholder grouping, shape regions,
// tensor.empty creation) is covered by the sibling files.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -hipsr-materialize-init-tensors | FileCheck %s

// CHECK-LABEL: func.func @matmul_domain(
// CHECK:         hipsr.pool_domain
// CHECK:         hipsr.matmul
// CHECK:         hipsr.pool_domain_yield
// CHECK:         return
func.func @matmul_domain(%ctx: !hipsr.context, %a: tensor<?x256xf16>,
                         %b: tensor<256x512xf16>) -> tensor<?x512xf16> {
  %0 = hipsr.pool_domain(%ctx, %a, %b
      : !hipsr.context, tensor<?x256xf16>, tensor<256x512xf16>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_a: tensor<?x256xf16>,
       %domain_b: tensor<256x512xf16>):
    %init = hipsr.placeholder(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        {placeholder_type = #hipsr.placeholder_type<normal>}
        : tensor<?x512xf16> shape_region {
    ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
      %m_index = shape.const_size 0
      %m = shape.get_extent %a_shape, %m_index
          : !shape.shape, !shape.size -> !shape.size
      %n_index = shape.const_size 1
      %n = shape.get_extent %b_shape, %n_index
          : !shape.shape, !shape.size -> !shape.size
      %result_shape = shape.from_extents %m, %n : !shape.size, !shape.size
      hipsr.shape_yield2 %result_shape : !shape.shape
    }
    %matmul = hipsr.matmul(%domain_ctx)
        ins(%domain_a, %domain_b : tensor<?x256xf16>, tensor<256x512xf16>)
        outs(%init : tensor<?x512xf16>) : tensor<?x512xf16>
    hipsr.pool_domain_yield %matmul : tensor<?x512xf16>
  } -> tensor<?x512xf16>
  return %0 : tensor<?x512xf16>
}
