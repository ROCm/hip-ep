// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Elementwise unary forms the conversion rejects.
//
// The pass runs applyFullConversion with the whole hip dialect marked illegal,
// so a rejected op fails legalization rather than surviving in the output.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file --verify-diagnostics %s

// Dynamic shapes give the pattern no static shape to reason about.
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                         %init: tensor<?x8xf16>) -> tensor<?x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.abs'}}
  %r = hip.abs(%ctx) ins(%x : tensor<?x8xf16>)
                     outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

// TOSA unary ops carry SameOperandsAndResultShape, so an operand that would
// have to broadcast to the result is not a 1-1 mapping. Unlike the binary ops
// there is no size-1 broadcast to fall back on.
func.func @shape_mismatch(%ctx: !hip.context, %x: tensor<1x1x32xf16>,
                          %init: tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.exp'}}
  %r = hip.exp(%ctx) ins(%x : tensor<1x1x32xf16>)
                     outs(%init : tensor<1x128x32xf16>) : tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// -----

// tosa.sin takes Tosa_FloatTensor, so an integer operand would fail the TOSA
// verifier. Reject it here instead of emitting invalid TOSA.
func.func @integer_operand(%ctx: !hip.context, %x: tensor<4xi32>,
                           %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.sin'}}
  %r = hip.sin(%ctx) ins(%x : tensor<4xi32>)
                     outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}
