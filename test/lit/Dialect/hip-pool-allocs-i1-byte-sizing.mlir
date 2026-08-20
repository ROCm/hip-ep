// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt --hip-pool-allocs='alignment=1' %s | FileCheck %s

// HIP kernels address i1 storage as one byte per element. This allocation is
// larger than the default pool alignment so a packed-bit size cannot hide.
// CHECK-LABEL: func.func @static_i1
// CHECK: %[[BYTES:.*]] = arith.constant 300 : index
// CHECK: %[[POOL:.*]] = hip.get_pool(%{{.*}}, %[[BYTES]]) : memref<?xi8>
// CHECK: memref.view %[[POOL]]{{.*}} to memref<300xi1>
func.func @static_i1(
    %ctx: !hip.context, %input: memref<300xi1>) -> memref<300xi1> {
  %temporary = memref.alloc() : memref<300xi1>
  hip.not(%ctx) ins(%input : memref<300xi1>)
                outs(%temporary : memref<300xi1>)
  return %temporary : memref<300xi1>
}

// A dynamic i1 shape multiplies its runtime extent by the four-element static
// suffix, not by a packed-bit fraction.
// CHECK-LABEL: func.func @dynamic_i1_static_suffix
// CHECK-SAME: %[[N:[a-zA-Z0-9_]+]]: index
// CHECK: %[[COEFF:.*]] = arith.constant 4 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[N]], %[[COEFF]] : index
// CHECK: hip.get_pool(%{{.*}}, %[[BYTES]]) : memref<?xi8>
func.func @dynamic_i1_static_suffix(
    %ctx: !hip.context, %input: memref<?x4xi1>,
    %n: index) -> memref<?x4xi1> {
  %temporary = memref.alloc(%n) : memref<?x4xi1>
  hip.not(%ctx) ins(%input : memref<?x4xi1>)
                outs(%temporary : memref<?x4xi1>)
  return %temporary : memref<?x4xi1>
}

// Non-i1 element sizing is unchanged: 4xf32 contributes 16 bytes per dynamic
// element.
// CHECK-LABEL: func.func @dynamic_f32_static_suffix
// CHECK-SAME: %[[N:[a-zA-Z0-9_]+]]: index
// CHECK: %[[COEFF:.*]] = arith.constant 16 : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[N]], %[[COEFF]] : index
// CHECK: hip.get_pool(%{{.*}}, %[[BYTES]]) : memref<?xi8>
func.func @dynamic_f32_static_suffix(
    %ctx: !hip.context, %input: memref<?x4xf32>,
    %n: index) -> memref<?x4xf32> {
  %temporary = memref.alloc(%n) : memref<?x4xf32>
  hip.miopen.softmax(%ctx) ins(%input : memref<?x4xf32>)
                           outs(%temporary : memref<?x4xf32>)
  return %temporary : memref<?x4xf32>
}
