// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-legalize-quant-types moves quantized element types into `quant_types`
// and retypes tensors to storage integers, so bufferization sees no quant
// types.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(func.func(hip-legalize-quant-types))' %s \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @quantized_add
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<8xui16>, %[[B:.*]]: tensor<8xui16>)
// CHECK-NEXT:    %[[E:.*]] = tensor.empty() : tensor<8xui16>
// CHECK-NEXT:    %[[S:.*]] = hip.add(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<8xui16>, tensor<8xui16>) outs(%[[E]] : tensor<8xui16>) {quant_types = [!quant.uniform<u16:f32, 2.500000e-01:32768>, !quant.uniform<u16:f32, 5.000000e-01:32768>, !quant.uniform<u16:f32, 1.250000e-01:32768>]} -> tensor<8xui16>
// CHECK-NEXT:    return %[[S]] : tensor<8xui16>
func.func @quantized_add(%ctx: !hip.context, %a: tensor<8xui16>,
                         %b: tensor<8xui16>) -> tensor<8xui16> {
  %qa = hip.storage_cast %a : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.25:32768>>
  %qb = hip.storage_cast %b : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.5:32768>>
  %e = tensor.empty() : tensor<8x!quant.uniform<u16:f32, 0.125:32768>>
  %s = hip.add(%ctx) ins(%qa, %qb : tensor<8x!quant.uniform<u16:f32, 0.25:32768>>,
                                    tensor<8x!quant.uniform<u16:f32, 0.5:32768>>)
      outs(%e : tensor<8x!quant.uniform<u16:f32, 0.125:32768>>)
      -> tensor<8x!quant.uniform<u16:f32, 0.125:32768>>
  %r = hip.storage_cast %s : tensor<8x!quant.uniform<u16:f32, 0.125:32768>> to tensor<8xui16>
  return %r : tensor<8xui16>
}

// -----

// A float add is the same generic op: it gets `quant_types` with its float
// types, meaning not quantized.

// CHECK-LABEL: func.func @float_add
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) {quant_types = [f32, f32, f32]} -> tensor<8xf32>
func.func @float_add(%ctx: !hip.context, %a: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.add(%ctx) ins(%a, %a : tensor<8xf32>, tensor<8xf32>)
      outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %s : tensor<8xf32>
}

// -----

// An integer add is not quantized either; the entries tell it apart from a
// quantized ui16 add once both are plain integer memrefs.

// CHECK-LABEL: func.func @int_add
// CHECK:         {quant_types = [i32, i32, i32]}
func.func @int_add(%ctx: !hip.context, %a: tensor<8xi32>) -> tensor<8xi32> {
  %e = tensor.empty() : tensor<8xi32>
  %s = hip.add(%ctx) ins(%a, %a : tensor<8xi32>, tensor<8xi32>)
      outs(%e : tensor<8xi32>) -> tensor<8xi32>
  return %s : tensor<8xi32>
}

// -----

// Running the pass again must not overwrite the recorded types with the
// (now storage-typed) tensors.

// CHECK-LABEL: func.func @already_legalized
// CHECK:         {quant_types = [!quant.uniform<u16:f32, 5.000000e-01:3>, !quant.uniform<u16:f32, 5.000000e-01:3>, !quant.uniform<u16:f32, 5.000000e-01:3>]}
func.func @already_legalized(%ctx: !hip.context, %a: tensor<8xui16>) -> tensor<8xui16> {
  %e = tensor.empty() : tensor<8xui16>
  %s = hip.add(%ctx) ins(%a, %a : tensor<8xui16>, tensor<8xui16>)
      outs(%e : tensor<8xui16>)
      {quant_types = [!quant.uniform<u16:f32, 0.5:3>, !quant.uniform<u16:f32, 0.5:3>, !quant.uniform<u16:f32, 0.5:3>]}
      -> tensor<8xui16>
  return %s : tensor<8xui16>
}

// -----

// The quantized add kernel takes one storage type for all three tensors.

func.func @no_kernel(%ctx: !hip.context, %a: tensor<8xui16>,
                     %b: tensor<8xui8>) -> tensor<8xui16> {
  %qa = hip.storage_cast %a : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.25>>
  %qb = hip.storage_cast %b : tensor<8xui8> to tensor<8x!quant.uniform<u8:f32, 0.5>>
  %e = tensor.empty() : tensor<8x!quant.uniform<u16:f32, 0.125>>
  // expected-error @+1 {{has no kernel for its quantized types}}
  %s = hip.add(%ctx) ins(%qa, %qb : tensor<8x!quant.uniform<u16:f32, 0.25>>,
                                    tensor<8x!quant.uniform<u8:f32, 0.5>>)
      outs(%e : tensor<8x!quant.uniform<u16:f32, 0.125>>)
      -> tensor<8x!quant.uniform<u16:f32, 0.125>>
  %r = hip.storage_cast %s : tensor<8x!quant.uniform<u16:f32, 0.125>> to tensor<8xui16>
  return %r : tensor<8xui16>
}
