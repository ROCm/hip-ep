// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Quantized element types on hip.add and hip.storage_cast: round trip and
// folding.

// RUN: hip-mlir-opt --split-input-file %s | hip-mlir-opt --split-input-file | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --canonicalize %s | FileCheck %s --check-prefix=FOLD

!q = !quant.uniform<u16:f32, 5.000000e-01:3>

// CHECK-LABEL: func.func @quantized_add
// CHECK:         hip.storage_cast %{{.*}} : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 5.000000e-01:3>>
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8x!quant.uniform<u16:f32, 5.000000e-01:3>>, tensor<8x!quant.uniform<u16:f32, 5.000000e-01:3>>)
// CHECK:         hip.storage_cast %{{.*}} : tensor<8x!quant.uniform<u16:f32, 5.000000e-01:3>> to tensor<8xui16>
func.func @quantized_add(%ctx: !hip.context, %a: tensor<8xui16>,
                         %b: tensor<8xui16>) -> tensor<8xui16> {
  %qa = hip.storage_cast %a : tensor<8xui16> to tensor<8x!q>
  %qb = hip.storage_cast %b : tensor<8xui16> to tensor<8x!q>
  %e = tensor.empty() : tensor<8x!q>
  %s = hip.add(%ctx) ins(%qa, %qb : tensor<8x!q>, tensor<8x!q>)
      outs(%e : tensor<8x!q>) -> tensor<8x!q>
  %r = hip.storage_cast %s : tensor<8x!q> to tensor<8xui16>
  return %r : tensor<8xui16>
}

// -----

// After hip-legalize-quant-types: storage tensors, types in `quant_types`.
// Signed storage is signless iN.

// CHECK-LABEL: func.func @legalized_add
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : memref<8xi8>, memref<8xi8>) outs(%{{.*}} : memref<8xi8>) {quant_types = [!quant.uniform<i8:f32, 2.500000e-01:-5>, !quant.uniform<i8:f32, 5.000000e-01:3>, !quant.uniform<i8:f32, 1.250000e-01:7>]}
func.func @legalized_add(%ctx: !hip.context, %a: memref<8xi8>,
                         %b: memref<8xi8>, %out: memref<8xi8>) {
  hip.add(%ctx) ins(%a, %b : memref<8xi8>, memref<8xi8>) outs(%out : memref<8xi8>)
      {quant_types = [!quant.uniform<i8:f32, 0.25:-5>,
                      !quant.uniform<i8:f32, 0.5:3>,
                      !quant.uniform<i8:f32, 0.125:7>]}
  return
}

// -----

// A cast pair that restores the original type folds away.

// FOLD-LABEL: func.func @fold_cast_pair
// FOLD-NEXT:    return %arg0, %arg1
func.func @fold_cast_pair(%a: tensor<8xui16>, %b: tensor<8x!quant.uniform<u16:f32, 0.5:3>>)
    -> (tensor<8xui16>, tensor<8x!quant.uniform<u16:f32, 0.5:3>>) {
  %0 = hip.storage_cast %a : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.5:3>>
  %1 = hip.storage_cast %0 : tensor<8x!quant.uniform<u16:f32, 0.5:3>> to tensor<8xui16>
  %2 = hip.storage_cast %b : tensor<8x!quant.uniform<u16:f32, 0.5:3>> to tensor<8xui16>
  %3 = hip.storage_cast %2 : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.5:3>>
  return %1, %3 : tensor<8xui16>, tensor<8x!quant.uniform<u16:f32, 0.5:3>>
}

// -----

// Different quant parameters on the two sides: a requantization, not a view.

// FOLD-LABEL: func.func @no_fold_requant
// FOLD-COUNT-2: hip.storage_cast
func.func @no_fold_requant(%b: tensor<8x!quant.uniform<u16:f32, 0.5:3>>)
    -> tensor<8x!quant.uniform<u16:f32, 0.25:3>> {
  %0 = hip.storage_cast %b : tensor<8x!quant.uniform<u16:f32, 0.5:3>> to tensor<8xui16>
  %1 = hip.storage_cast %0 : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 0.25:3>>
  return %1 : tensor<8x!quant.uniform<u16:f32, 0.25:3>>
}
