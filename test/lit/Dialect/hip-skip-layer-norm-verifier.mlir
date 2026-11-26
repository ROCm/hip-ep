// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @bf16(%ctx: !hip.context, %x: memref<4x8xbf16, 1>,
                %g: memref<8xbf16, 1>, %y: memref<4x8xbf16, 1>) {
  // expected-error @+1 {{expected f16 or f32 elements}}
  hip.skip_layer_norm(%ctx)
      ins(%x, %x, %g : memref<4x8xbf16, 1>, memref<4x8xbf16, 1>,
          memref<8xbf16, 1>)
      outs(%y : memref<4x8xbf16, 1>) {epsilon = 1.0e-05 : f32}
  return
}

// -----

func.func @mixed_types(%ctx: !hip.context, %x: memref<4x8xf16, 1>,
                       %g: memref<8xf32, 1>, %y: memref<4x8xf16, 1>) {
  // expected-error @+1 {{expected all operands to have element type 'f16'}}
  hip.skip_layer_norm(%ctx)
      ins(%x, %x, %g : memref<4x8xf16, 1>, memref<4x8xf16, 1>,
          memref<8xf32, 1>)
      outs(%y : memref<4x8xf16, 1>) {epsilon = 1.0e-05 : f32}
  return
}

// -----

func.func @gamma_not_hidden(%ctx: !hip.context, %x: memref<4x8xf16, 1>,
                            %g: memref<4xf16, 1>, %y: memref<4x8xf16, 1>) {
  // expected-error @+1 {{expected gamma / beta / bias to hold 8 elements}}
  hip.skip_layer_norm(%ctx)
      ins(%x, %x, %g : memref<4x8xf16, 1>, memref<4x8xf16, 1>,
          memref<4xf16, 1>)
      outs(%y : memref<4x8xf16, 1>) {epsilon = 1.0e-05 : f32}
  return
}
