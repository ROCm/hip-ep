// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hip.add forms the conversion rejects.
//
// The pass runs applyFullConversion with the whole hip dialect marked illegal,
// so a rejected op fails legalization rather than surviving in the output.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file --verify-diagnostics %s

// hip broadcasting extends rank; TOSA broadcasts size-1 dimensions only and
// requires every operand to already carry the result's rank.
func.func @rank_extending_broadcast(%ctx: !hip.context,
                                    %x: tensor<1x128x32xf16>,
                                    %y: tensor<32xf16>,
                                    %init: tensor<1x128x32xf16>)
    -> tensor<1x128x32xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<32xf16>)
                     outs(%init : tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// -----

// A rank-0 scalar operand is the same rank mismatch; this is how ReLU shows up
// after onnx-to-hip lowering.
func.func @scalar_operand(%ctx: !hip.context, %x: tensor<1x64x112x112xf16>,
                          %y: tensor<f16>, %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x64x112x112xf16>, tensor<f16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     -> tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

// -----

// Dynamic shapes give the pattern no static shape to reason about.
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                         %y: tensor<?x8xf16>, %init: tensor<?x8xf16>)
    -> tensor<?x8xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<?x8xf16>, tensor<?x8xf16>)
                     outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}
