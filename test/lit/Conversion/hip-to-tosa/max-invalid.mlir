// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hip.max forms the conversion rejects.
//
// The pass runs applyFullConversion with the whole hip dialect marked illegal,
// so a rejected op fails legalization rather than surviving in the output.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file --verify-diagnostics %s

// Dynamic shapes give the pattern no static shape to reason about.
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                         %y: tensor<?x8xf16>, %init: tensor<?x8xf16>)
    -> tensor<?x8xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.max'}}
  %r = hip.max(%ctx) ins(%x, %y : tensor<?x8xf16>, tensor<?x8xf16>)
                     outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

// Trailing dim 8 cannot broadcast to 112 after rank equalization.
func.func @incompatible_broadcast(%ctx: !hip.context,
                                  %x: tensor<1x64x112x112xf16>,
                                  %y: tensor<8xf16>,
                                  %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.max'}}
  %r = hip.max(%ctx) ins(%x, %y : tensor<1x64x112x112xf16>, tensor<8xf16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     : tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}
