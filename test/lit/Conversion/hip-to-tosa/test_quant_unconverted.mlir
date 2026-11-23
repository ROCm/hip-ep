// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.quant_conv and hip.quant_matmul exist only inside rocMLIR kernels and
// have no HIP lowering, so a form the converters refuse must fail the pass
// here instead of reaching rocMLIR unconverted. A dynamic shape verifies, but
// the converters need static shapes.
//
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s

func.func @quant_conv_dynamic(%x: tensor<?x2x4x4xi8>, %w: tensor<2x2x3x3xi8>,
                              %e: tensor<?x2x4x4xi32>)
    -> tensor<?x2x4x4xi32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  // expected-error @+1 {{failed to legalize operation 'hip.quant_conv'}}
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<?x2x4x4xi8>,
         tensor<2x2x3x3xi8>) outs(%e : tensor<?x2x4x4xi32>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<?x2x4x4xi32>
  return %r : tensor<?x2x4x4xi32>
}

// -----

func.func @quant_matmul_dynamic(%a: tensor<?x64xi8>, %b: tensor<64x16xi8>,
                                %e: tensor<?x16xi32>)
    -> tensor<?x16xi32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  // expected-error @+1 {{failed to legalize operation 'hip.quant_matmul'}}
  %r = hip.quant_matmul(%ctx) ins(%a, %b : tensor<?x64xi8>, tensor<64x16xi8>)
         outs(%e : tensor<?x16xi32>) : tensor<?x16xi32>
  return %r : tensor<?x16xi32>
}
