// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @dequant_bnb4_nf4
// CHECK: tosa.gather
// CHECK: tosa.tile
// CHECK: tosa.mul
// CHECK-NOT: hip.dequant_bnb4
func.func @dequant_bnb4_nf4(%ctx: !hip.context, %packed: tensor<16xui8>,
                            %absmax: tensor<2xf32>, %init: tensor<4x8xf32>)
    -> tensor<4x8xf32> attributes {rock.kernel} {
  %r = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        tensor<16xui8>, tensor<2xf32>) outs(%init : tensor<4x8xf32>)
        {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 1 : i64}
        : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @dequant_bnb4_fp4
// CHECK: tosa.gather
// CHECK-NOT: hip.dequant_bnb4
func.func @dequant_bnb4_fp4(%ctx: !hip.context, %packed: tensor<16xui8>,
                            %absmax: tensor<2xf32>, %init: tensor<4x8xf32>)
    -> tensor<4x8xf32> attributes {rock.kernel} {
  %r = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        tensor<16xui8>, tensor<2xf32>) outs(%init : tensor<4x8xf32>)
        {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 0 : i64}
        : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

func.func @dequant_bnb4_bad_quant(%ctx: !hip.context, %packed: tensor<16xui8>,
                                  %absmax: tensor<2xf32>, %init: tensor<4x8xf32>)
    -> tensor<4x8xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.dequant_bnb4'}}
  %r = hip.dequant_bnb4(%ctx) ins(%packed, %absmax :
        tensor<16xui8>, tensor<2xf32>) outs(%init : tensor<4x8xf32>)
        {K = 8 : i64, N = 4 : i64, block_size = 16 : i64, quant_type = 2 : i64}
        : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
