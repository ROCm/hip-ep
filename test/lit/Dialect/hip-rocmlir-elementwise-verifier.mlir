// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @wide_integer_input(%ctx: !hip.context, %x: tensor<4xi64>)
    -> tensor<4xi64> {
  // expected-error @+1 {{input type 'tensor<4xi64>' is not a kernel type (static, and f16, bf16, f32 or an integer of at most 32 bits)}}
  %r = hip.rocmlir_elementwise(%ctx) ins(%x : tensor<4xi64>) -> tensor<4xi64> {
  ^bb0(%c: !hip.context, %a: tensor<4xi64>):
    hip.yield %a : tensor<4xi64>
  }
  return %r : tensor<4xi64>
}

// -----

func.func @rank_zero_result(%ctx: !hip.context, %x: tensor<f16>)
    -> tensor<f16> {
  // expected-error @+1 {{result type 'tensor<f16>' is rank 0 or not a kernel type (static, and f16, bf16, f32 or an integer of at most 32 bits)}}
  %r = hip.rocmlir_elementwise(%ctx) ins(%x : tensor<f16>) -> tensor<f16> {
  ^bb0(%c: !hip.context, %a: tensor<f16>):
    hip.yield %a : tensor<f16>
  }
  return %r : tensor<f16>
}

// -----

func.func @missing_context_argument(%ctx: !hip.context, %x: tensor<4xf16>)
    -> tensor<4xf16> {
  // expected-error @+1 {{expects a !hip.context block argument followed by one block argument per input}}
  %r = hip.rocmlir_elementwise(%ctx) ins(%x : tensor<4xf16>) -> tensor<4xf16> {
  ^bb0(%a: tensor<4xf16>):
    hip.yield %a : tensor<4xf16>
  }
  return %r : tensor<4xf16>
}

// -----

func.func @yield_type_mismatch(%ctx: !hip.context, %x: tensor<4xf16>)
    -> tensor<4xf16> {
  // expected-error @+1 {{yields types that do not match the results}}
  %r = hip.rocmlir_elementwise(%ctx) ins(%x : tensor<4xf16>) -> tensor<4xf16> {
  ^bb0(%c: !hip.context, %a: tensor<4xf16>):
    %e = tensor.empty() : tensor<8xf16>
    hip.yield %e : tensor<8xf16>
  }
  return %r : tensor<4xf16>
}

// -----

func.func @disallowed_op(%ctx: !hip.context, %x: tensor<4xf16>)
    -> tensor<4xf16> {
  %r = hip.rocmlir_elementwise(%ctx) ins(%x : tensor<4xf16>) -> tensor<4xf16> {
  ^bb0(%c: !hip.context, %a: tensor<4xf16>):
    // expected-error @+1 {{is not allowed in a hip.rocmlir_elementwise body}}
    %s = func.call @f(%a) : (tensor<4xf16>) -> tensor<4xf16>
    hip.yield %s : tensor<4xf16>
  }
  return %r : tensor<4xf16>
}
func.func private @f(tensor<4xf16>) -> tensor<4xf16>

// -----

// f64 and index tensors are not kernel arguments either: rocMLIR kernels take
// f16, bf16, f32 and integers of at most 32 bits, as the fusion assumes.
func.func @f64_input(%ctx: !hip.context, %x: tensor<4xf64>) -> tensor<4xf32> {
  // expected-error @+1 {{input type 'tensor<4xf64>' is not a kernel type}}
  %r = hip.rocmlir_elementwise(%ctx) ins(%x : tensor<4xf64>) -> tensor<4xf32> {
  ^bb0(%c: !hip.context, %a: tensor<4xf64>):
    %e = tensor.empty() : tensor<4xf32>
    %y = hip.cast(%c) ins(%a : tensor<4xf64>) outs(%e : tensor<4xf32>) {to = 1 : i64} : tensor<4xf32>
    hip.yield %y : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----

func.func @index_result(%ctx: !hip.context, %x: tensor<4xf32>) -> tensor<4xindex> {
  // expected-error @+1 {{result type 'tensor<4xindex>' is rank 0 or not a kernel type}}
  %r = hip.rocmlir_elementwise(%ctx) ins(%x : tensor<4xf32>) -> tensor<4xindex> {
  ^bb0(%c: !hip.context, %a: tensor<4xf32>):
    %y = tensor.empty() : tensor<4xindex>
    hip.yield %y : tensor<4xindex>
  }
  return %r : tensor<4xindex>
}
