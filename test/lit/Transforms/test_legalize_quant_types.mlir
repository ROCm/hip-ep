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

// -----

// Matmul and gemm are quant-capable like add: quantized ones get their quant
// types in `quant_types`, with storage-integer tensors.

// CHECK-LABEL: func.func @quantized_matmul
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8x8xui16>, tensor<8x8xi8>) outs(%{{.*}} : tensor<8x8xui16>) {quant_types = [!quant.uniform<u16:f32, 2.500000e-01:32768>, !quant.uniform<i8:f32, 1.250000e-01:3>, !quant.uniform<u16:f32, 5.000000e-01:32768>]} : tensor<8x8xui16>
func.func @quantized_matmul(%ctx: !hip.context, %a: tensor<8x8xui16>,
                            %b: tensor<8x8xi8>) -> tensor<8x8xui16> {
  %qa = hip.storage_cast %a : tensor<8x8xui16> to tensor<8x8x!quant.uniform<u16:f32, 0.25:32768>>
  %qb = hip.storage_cast %b : tensor<8x8xi8> to tensor<8x8x!quant.uniform<i8:f32, 0.125:3>>
  %e = tensor.empty() : tensor<8x8x!quant.uniform<u16:f32, 0.5:32768>>
  %mm = hip.matmul(%ctx) ins(%qa, %qb : tensor<8x8x!quant.uniform<u16:f32, 0.25:32768>>,
                                        tensor<8x8x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e : tensor<8x8x!quant.uniform<u16:f32, 0.5:32768>>)
      : tensor<8x8x!quant.uniform<u16:f32, 0.5:32768>>
  %r = hip.storage_cast %mm : tensor<8x8x!quant.uniform<u16:f32, 0.5:32768>> to tensor<8x8xui16>
  return %r : tensor<8x8xui16>
}

// -----

// A gemm with a bias records four entries, without one three.

// CHECK-LABEL: func.func @quantized_gemm_bias
// CHECK:         {alpha = 2.000000e+00 : f32, quant_types = [!quant.uniform<i8:f32, 2.500000e-01>, !quant.uniform<i8:f32, 1.250000e-01>, !quant.uniform<i32:f32, 3.125000e-02>, !quant.uniform<i8:f32, 5.000000e-01>]}
func.func @quantized_gemm_bias(%ctx: !hip.context, %a: tensor<4x8xi8>,
                               %b: tensor<8x6xi8>, %c: tensor<6xi32>) -> tensor<4x6xi8> {
  %qa = hip.storage_cast %a : tensor<4x8xi8> to tensor<4x8x!quant.uniform<i8:f32, 0.25>>
  %qb = hip.storage_cast %b : tensor<8x6xi8> to tensor<8x6x!quant.uniform<i8:f32, 0.125>>
  %qc = hip.storage_cast %c : tensor<6xi32> to tensor<6x!quant.uniform<i32:f32, 0.03125>>
  %e = tensor.empty() : tensor<4x6x!quant.uniform<i8:f32, 0.5>>
  %g = hip.gemm(%ctx) ins(%qa, %qb, %qc : tensor<4x8x!quant.uniform<i8:f32, 0.25>>,
                                          tensor<8x6x!quant.uniform<i8:f32, 0.125>>,
                                          tensor<6x!quant.uniform<i32:f32, 0.03125>>)
      outs(%e : tensor<4x6x!quant.uniform<i8:f32, 0.5>>) {alpha = 2.0 : f32}
      : tensor<4x6x!quant.uniform<i8:f32, 0.5>>
  %r = hip.storage_cast %g : tensor<4x6x!quant.uniform<i8:f32, 0.5>> to tensor<4x6xi8>
  return %r : tensor<4x6xi8>
}

// -----

// CHECK-LABEL: func.func @quantized_gemm_no_bias
// CHECK:         {quant_types = [!quant.uniform<i8:f32, 2.500000e-01>, !quant.uniform<i8:f32, 1.250000e-01>, !quant.uniform<i8:f32, 5.000000e-01>]}
func.func @quantized_gemm_no_bias(%ctx: !hip.context, %a: tensor<4x8xi8>,
                                  %b: tensor<8x6xi8>) -> tensor<4x6xi8> {
  %qa = hip.storage_cast %a : tensor<4x8xi8> to tensor<4x8x!quant.uniform<i8:f32, 0.25>>
  %qb = hip.storage_cast %b : tensor<8x6xi8> to tensor<8x6x!quant.uniform<i8:f32, 0.125>>
  %e = tensor.empty() : tensor<4x6x!quant.uniform<i8:f32, 0.5>>
  %g = hip.gemm(%ctx) ins(%qa, %qb : tensor<4x8x!quant.uniform<i8:f32, 0.25>>,
                                     tensor<8x6x!quant.uniform<i8:f32, 0.125>>)
      outs(%e : tensor<4x6x!quant.uniform<i8:f32, 0.5>>)
      : tensor<4x6x!quant.uniform<i8:f32, 0.5>>
  %r = hip.storage_cast %g : tensor<4x6x!quant.uniform<i8:f32, 0.5>> to tensor<4x6xi8>
  return %r : tensor<4x6xi8>
}

// -----

// Float matmul and gemm get their float types.

// CHECK-LABEL: func.func @float_matmul_gemm
// CHECK:         hip.matmul{{.*}}{quant_types = [f16, f16, f16]}
// CHECK:         hip.gemm{{.*}}{quant_types = [f32, f32, f32]}
func.func @float_matmul_gemm(%ctx: !hip.context, %a: tensor<4x8xf16>, %b: tensor<8x6xf16>,
                             %c: tensor<4x8xf32>, %d: tensor<8x6xf32>)
    -> (tensor<4x6xf16>, tensor<4x6xf32>) {
  %e0 = tensor.empty() : tensor<4x6xf16>
  %m = hip.matmul(%ctx) ins(%a, %b : tensor<4x8xf16>, tensor<8x6xf16>)
      outs(%e0 : tensor<4x6xf16>) : tensor<4x6xf16>
  %e1 = tensor.empty() : tensor<4x6xf32>
  %g = hip.gemm(%ctx) ins(%c, %d : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e1 : tensor<4x6xf32>) : tensor<4x6xf32>
  return %m, %g : tensor<4x6xf16>, tensor<4x6xf32>
}

// -----

// B wider than 8 bit has no matmul kernel.

func.func @matmul_no_kernel(%ctx: !hip.context, %a: tensor<8x8xui16>,
                            %b: tensor<8x8xui16>) -> tensor<8x8xui16> {
  %qa = hip.storage_cast %a : tensor<8x8xui16> to tensor<8x8x!quant.uniform<u16:f32, 0.25>>
  %qb = hip.storage_cast %b : tensor<8x8xui16> to tensor<8x8x!quant.uniform<u16:f32, 0.25>>
  %e = tensor.empty() : tensor<8x8x!quant.uniform<u16:f32, 0.5>>
  // expected-error @+1 {{has no kernel for its quantized types}}
  %mm = hip.matmul(%ctx) ins(%qa, %qb : tensor<8x8x!quant.uniform<u16:f32, 0.25>>,
                                        tensor<8x8x!quant.uniform<u16:f32, 0.25>>)
      outs(%e : tensor<8x8x!quant.uniform<u16:f32, 0.5>>)
      : tensor<8x8x!quant.uniform<u16:f32, 0.5>>
  %r = hip.storage_cast %mm : tensor<8x8x!quant.uniform<u16:f32, 0.5>> to tensor<8x8xui16>
  return %r : tensor<8x8xui16>
}

// -----

// Matmul and gemm are quant-capable too: float ones get their float types,
// quantized ones the quant types (a bias-free gemm has three entries).

// CHECK-LABEL: func.func @float_matmul
// CHECK:         {quant_types = [f32, f32, f32]}
func.func @float_matmul(%ctx: !hip.context, %a: tensor<4x8xf32>,
                        %b: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %e = tensor.empty() : tensor<4x6xf32>
  %m = hip.matmul(%ctx) ins(%a, %b : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  return %m : tensor<4x6xf32>
}

// -----

// CHECK-LABEL: func.func @quantized_matmul
// CHECK-SAME:    (%{{.*}}: !hip.context, %[[A:.*]]: tensor<4x8xui16>, %[[B:.*]]: tensor<8x6xi8>)
// CHECK:         hip.matmul(%{{.*}}) ins(%[[A]], %[[B]] : tensor<4x8xui16>, tensor<8x6xi8>) outs(%{{.*}} : tensor<4x6xui16>) {quant_types = [!quant.uniform<u16:f32, 2.500000e-01:32768>, !quant.uniform<i8:f32, 1.250000e-01:3>, !quant.uniform<u16:f32, 5.000000e-01:32768>]} : tensor<4x6xui16>
func.func @quantized_matmul(%ctx: !hip.context, %a: tensor<4x8xui16>,
                            %b: tensor<8x6xi8>) -> tensor<4x6xui16> {
  %qa = hip.storage_cast %a : tensor<4x8xui16> to tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>
  %qb = hip.storage_cast %b : tensor<8x6xi8> to tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>
  %e = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %m = hip.matmul(%ctx) ins(%qa, %qb : tensor<4x8x!quant.uniform<u16:f32, 0.25:32768>>, tensor<8x6x!quant.uniform<i8:f32, 0.125:3>>)
      outs(%e : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>) : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>>
  %r = hip.storage_cast %m : tensor<4x6x!quant.uniform<u16:f32, 0.5:32768>> to tensor<4x6xui16>
  return %r : tensor<4x6xui16>
}

// -----

// CHECK-LABEL: func.func @quantized_gemm_bias
// CHECK:         {quant_types = [!quant.uniform<i8:f32, 5.000000e-01>, !quant.uniform<i8:f32, 2.500000e-01>, !quant.uniform<i32:f32, 1.250000e-01>, !quant.uniform<i8:f32, 1.000000e+00>]}
func.func @quantized_gemm_bias(%ctx: !hip.context, %a: tensor<4x8xi8>,
                               %b: tensor<8x6xi8>, %c: tensor<6xi32>) -> tensor<4x6xi8> {
  %qa = hip.storage_cast %a : tensor<4x8xi8> to tensor<4x8x!quant.uniform<i8:f32, 0.5>>
  %qb = hip.storage_cast %b : tensor<8x6xi8> to tensor<8x6x!quant.uniform<i8:f32, 0.25>>
  %qc = hip.storage_cast %c : tensor<6xi32> to tensor<6x!quant.uniform<i32:f32, 0.125>>
  %e = tensor.empty() : tensor<4x6x!quant.uniform<i8:f32, 1.0>>
  %g = hip.gemm(%ctx) ins(%qa, %qb, %qc : tensor<4x8x!quant.uniform<i8:f32, 0.5>>, tensor<8x6x!quant.uniform<i8:f32, 0.25>>, tensor<6x!quant.uniform<i32:f32, 0.125>>)
      outs(%e : tensor<4x6x!quant.uniform<i8:f32, 1.0>>) : tensor<4x6x!quant.uniform<i8:f32, 1.0>>
  %r = hip.storage_cast %g : tensor<4x6x!quant.uniform<i8:f32, 1.0>> to tensor<4x6xi8>
  return %r : tensor<4x6xi8>
}

// -----

// CHECK-LABEL: func.func @float_gemm_no_bias
// CHECK:         {quant_types = [f16, f16, f16]}
func.func @float_gemm_no_bias(%ctx: !hip.context, %a: tensor<4x8xf16>,
                              %b: tensor<8x6xf16>) -> tensor<4x6xf16> {
  %e = tensor.empty() : tensor<4x6xf16>
  %g = hip.gemm(%ctx) ins(%a, %b : tensor<4x8xf16>, tensor<8x6xf16>)
      outs(%e : tensor<4x6xf16>) : tensor<4x6xf16>
  return %g : tensor<4x6xf16>
}

// -----

// A 16-bit weight has no matmul kernel.

func.func @matmul_no_kernel(%ctx: !hip.context, %a: tensor<4x8xui16>,
                            %b: tensor<8x6xui16>) -> tensor<4x6xui16> {
  %qa = hip.storage_cast %a : tensor<4x8xui16> to tensor<4x8x!quant.uniform<u16:f32, 0.25>>
  %qb = hip.storage_cast %b : tensor<8x6xui16> to tensor<8x6x!quant.uniform<u16:f32, 0.125>>
  %e = tensor.empty() : tensor<4x6x!quant.uniform<u16:f32, 0.5>>
  // expected-error @+1 {{has no kernel for its quantized types}}
  %m = hip.matmul(%ctx) ins(%qa, %qb : tensor<4x8x!quant.uniform<u16:f32, 0.25>>, tensor<8x6x!quant.uniform<u16:f32, 0.125>>)
      outs(%e : tensor<4x6x!quant.uniform<u16:f32, 0.5>>) : tensor<4x6x!quant.uniform<u16:f32, 0.5>>
  %r = hip.storage_cast %m : tensor<4x6x!quant.uniform<u16:f32, 0.5>> to tensor<4x6xui16>
  return %r : tensor<4x6xui16>
}
