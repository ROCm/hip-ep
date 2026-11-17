// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @cast_both_int(%a: tensor<8xui16>) -> tensor<8xi16> {
  // expected-error @+1 {{exactly one side must be quantized}}
  %0 = hip.storage_cast %a : tensor<8xui16> to tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @cast_wrong_signedness(%a: tensor<8xi16>)
    -> tensor<8x!quant.uniform<u16:f32, 0.5>> {
  // expected-error @+1 {{storage type 'i16' does not match}}
  %0 = hip.storage_cast %a : tensor<8xi16> to tensor<8x!quant.uniform<u16:f32, 0.5>>
  return %0 : tensor<8x!quant.uniform<u16:f32, 0.5>>
}

// -----

func.func @cast_shape(%a: tensor<8xui16>)
    -> tensor<4x!quant.uniform<u16:f32, 0.5>> {
  // expected-error @+1 {{input and result shapes must match}}
  %0 = hip.storage_cast %a : tensor<8xui16> to tensor<4x!quant.uniform<u16:f32, 0.5>>
  return %0 : tensor<4x!quant.uniform<u16:f32, 0.5>>
}

// -----

func.func @add_mixed(%ctx: !hip.context, %a: tensor<8x!quant.uniform<u16:f32, 0.5>>,
                     %b: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  // expected-error @+1 {{operands must be all quantized or all non-quantized}}
  %s = hip.add(%ctx) ins(%a, %b : tensor<8x!quant.uniform<u16:f32, 0.5>>, tensor<8xf32>)
      outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %s : tensor<8xf32>
}

// -----

func.func @attr_on_quantized(%ctx: !hip.context, %a: tensor<8x!quant.uniform<u16:f32, 0.5>>)
    -> tensor<8x!quant.uniform<u16:f32, 0.5>> {
  %e = tensor.empty() : tensor<8x!quant.uniform<u16:f32, 0.5>>
  // expected-error @+1 {{'quant_types' is only valid on storage-typed operands}}
  %s = hip.add(%ctx) ins(%a, %a : tensor<8x!quant.uniform<u16:f32, 0.5>>, tensor<8x!quant.uniform<u16:f32, 0.5>>)
      outs(%e : tensor<8x!quant.uniform<u16:f32, 0.5>>)
      {quant_types = [!quant.uniform<u16:f32, 0.5>, !quant.uniform<u16:f32, 0.5>, !quant.uniform<u16:f32, 0.5>]}
      -> tensor<8x!quant.uniform<u16:f32, 0.5>>
  return %s : tensor<8x!quant.uniform<u16:f32, 0.5>>
}

// -----

func.func @attr_storage_mismatch(%ctx: !hip.context, %a: memref<8xui8>, %out: memref<8xui16>) {
  // expected-error @+1 {{operand element type 'ui8' does not match the 'quant_types' entry '!quant.uniform<u16:f32, 5.000000e-01>'}}
  hip.add(%ctx) ins(%a, %a : memref<8xui8>, memref<8xui8>) outs(%out : memref<8xui16>)
      {quant_types = [!quant.uniform<u16:f32, 0.5>, !quant.uniform<u16:f32, 0.5>, !quant.uniform<u16:f32, 0.5>]}
  return
}

// -----

func.func @attr_count(%ctx: !hip.context, %a: memref<8xui16>, %out: memref<8xui16>) {
  // expected-error @+1 {{'quant_types' expects 3 entries, got 1}}
  hip.add(%ctx) ins(%a, %a : memref<8xui16>, memref<8xui16>) outs(%out : memref<8xui16>)
      {quant_types = [!quant.uniform<u16:f32, 0.5>]}
  return
}

// -----

// A non-quantized entry must be the operand's own element type.

func.func @float_entry_mismatch(%ctx: !hip.context, %a: memref<8xf32>, %out: memref<8xf32>) {
  // expected-error @+1 {{operand element type 'f32' does not match the 'quant_types' entry 'f16'}}
  hip.add(%ctx) ins(%a, %a : memref<8xf32>, memref<8xf32>) outs(%out : memref<8xf32>)
      {quant_types = [f32, f32, f16]}
  return
}

// -----

func.func @entries_mixed(%ctx: !hip.context, %a: memref<8xui16>, %out: memref<8xui16>) {
  // expected-error @+1 {{'quant_types' must be all quantized or all non-quantized}}
  hip.add(%ctx) ins(%a, %a : memref<8xui16>, memref<8xui16>) outs(%out : memref<8xui16>)
      {quant_types = [!quant.uniform<u16:f32, 0.5>, ui16, !quant.uniform<u16:f32, 0.5>]}
  return
}

// -----

func.func @matmul_mixed(%ctx: !hip.context, %a: tensor<4x4x!quant.uniform<i8:f32, 0.5>>,
                        %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %e = tensor.empty() : tensor<4x4xf32>
  // expected-error @+1 {{operands must be all quantized or all non-quantized}}
  %m = hip.matmul(%ctx) ins(%a, %b : tensor<4x4x!quant.uniform<i8:f32, 0.5>>, tensor<4x4xf32>)
      outs(%e : tensor<4x4xf32>) : tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// -----

func.func @gemm_count(%ctx: !hip.context, %a: memref<4x4xi8>, %b: memref<4x4xi8>,
                      %c: memref<4xi32>, %out: memref<4x4xi8>) {
  // expected-error @+1 {{'quant_types' expects 4 entries, got 3}}
  hip.gemm(%ctx) ins(%a, %b, %c : memref<4x4xi8>, memref<4x4xi8>, memref<4xi32>)
      outs(%out : memref<4x4xi8>)
      {quant_types = [!quant.uniform<i8:f32, 0.5>, !quant.uniform<i8:f32, 0.5>, !quant.uniform<i8:f32, 0.5>]}
  return
}

// -----

func.func @matmul_mixed(%ctx: !hip.context, %a: tensor<4x8x!quant.uniform<i8:f32, 0.5>>,
                        %b: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %e = tensor.empty() : tensor<4x6xf32>
  // expected-error @+1 {{operands must be all quantized or all non-quantized}}
  %m = hip.matmul(%ctx) ins(%a, %b : tensor<4x8x!quant.uniform<i8:f32, 0.5>>, tensor<8x6xf32>)
      outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  return %m : tensor<4x6xf32>
}

// -----

// The bias counts as an operand: a float C with quantized A and B is mixed.

func.func @gemm_mixed_bias(%ctx: !hip.context, %a: tensor<4x8x!quant.uniform<i8:f32, 0.5>>,
                           %b: tensor<8x6x!quant.uniform<i8:f32, 0.5>>, %c: tensor<6xf32>)
    -> tensor<4x6x!quant.uniform<i8:f32, 0.5>> {
  %e = tensor.empty() : tensor<4x6x!quant.uniform<i8:f32, 0.5>>
  // expected-error @+1 {{operands must be all quantized or all non-quantized}}
  %g = hip.gemm(%ctx) ins(%a, %b, %c : tensor<4x8x!quant.uniform<i8:f32, 0.5>>, tensor<8x6x!quant.uniform<i8:f32, 0.5>>, tensor<6xf32>)
      outs(%e : tensor<4x6x!quant.uniform<i8:f32, 0.5>>) : tensor<4x6x!quant.uniform<i8:f32, 0.5>>
  return %g : tensor<4x6x!quant.uniform<i8:f32, 0.5>>
}

// -----

func.func @gemm_attr_count(%ctx: !hip.context, %a: memref<4x8xi8>, %b: memref<8x6xi8>,
                           %c: memref<6xi32>, %out: memref<4x6xi8>) {
  // expected-error @+1 {{'quant_types' expects 4 entries, got 3}}
  hip.gemm(%ctx) ins(%a, %b, %c : memref<4x8xi8>, memref<8x6xi8>, memref<6xi32>)
      outs(%out : memref<4x6xi8>)
      {quant_types = [!quant.uniform<i8:f32, 0.5>, !quant.uniform<i8:f32, 0.5>, !quant.uniform<i8:f32, 0.5>]}
  return
}

// -----

func.func @matmul_bias_rank2(%ctx: !hip.context, %a: memref<4x8xf32>, %b: memref<8x6xf32>,
                             %bias: memref<4x6xf32>, %out: memref<4x6xf32>) {
  // expected-error @+1 {{bias must be rank 1}}
  hip.matmul(%ctx) ins(%a, %b, %bias : memref<4x8xf32>, memref<8x6xf32>, memref<4x6xf32>)
      outs(%out : memref<4x6xf32>)
  return
}

// -----

func.func @matmul_bias_length(%ctx: !hip.context, %a: memref<4x8xf32>, %b: memref<8x6xf32>,
                              %bias: memref<5xf32>, %out: memref<4x6xf32>) {
  // expected-error @+1 {{bias length 5 does not match N = 6}}
  hip.matmul(%ctx) ins(%a, %b, %bias : memref<4x8xf32>, memref<8x6xf32>, memref<5xf32>)
      outs(%out : memref<4x6xf32>)
  return
}

// -----

func.func @matmul_bias_rank3_weight(%ctx: !hip.context, %a: memref<2x4x8xf32>, %b: memref<2x8x6xf32>,
                                    %bias: memref<6xf32>, %out: memref<2x4x6xf32>) {
  // expected-error @+1 {{bias requires a rank-2 B}}
  hip.matmul(%ctx) ins(%a, %b, %bias : memref<2x4x8xf32>, memref<2x8x6xf32>, memref<6xf32>)
      outs(%out : memref<2x4x6xf32>)
  return
}

// -----

func.func @matmul_bias_trans_a(%ctx: !hip.context, %a: memref<2x8x4xf32>, %b: memref<8x6xf32>,
                               %bias: memref<6xf32>, %out: memref<2x4x6xf32>) {
  // expected-error @+1 {{bias with a rank > 2 A requires transA = 0}}
  hip.matmul(%ctx) ins(%a, %b, %bias : memref<2x8x4xf32>, memref<8x6xf32>, memref<6xf32>)
      outs(%out : memref<2x4x6xf32>) {transA = 1 : i64}
  return
}
