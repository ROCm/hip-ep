// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hip.add forms the conversion rejects.
//
// The pass runs applyFullConversion with the whole hip dialect marked illegal,
// so a rejected op fails legalization rather than surviving in the output.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file --verify-diagnostics %s

// Dynamic shapes give the pattern no static shape to reason about.
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                         %y: tensor<?x8xf16>, %init: tensor<?x8xf16>)
    -> tensor<?x8xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<?x8xf16>, tensor<?x8xf16>)
                     outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

// Rank equalization prepends 1s, so tensor<8xf16> becomes 1x1x1x8. The
// trailing 8 still cannot broadcast to 112, which the post-equalization check
// catches rather than emitting invalid TOSA.
func.func @incompatible_broadcast(%ctx: !hip.context,
                                  %x: tensor<1x64x112x112xf16>,
                                  %y: tensor<8xf16>,
                                  %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x64x112x112xf16>, tensor<8xf16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     -> tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

// -----

// A mismatched element type is not a broadcast question at all.
func.func @element_type_mismatch(%ctx: !hip.context, %x: tensor<2x8xf16>,
                                 %y: tensor<2x8xf32>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf32>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
