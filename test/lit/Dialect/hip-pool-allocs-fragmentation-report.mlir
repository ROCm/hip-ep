// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test for the debug-only fragmentation probe of --hip-pool-allocs
// (emit-fragmentation-report option).
//
// The probe compares each part of a domain's packed footprint to its max-load
// lower bound (the peak sum of concurrently-live sizes). It does not affect
// generated IR: RUN line 1 checks the report and RUN line 2 checks the layout.
//
// Static and comparable common-factor cases check numeric zero excess.
// Runtime-aligned and mixed-factor cases check unavailable coefficients.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs='lifetime-only=true emit-fragmentation-report=true' \
// RUN:   --verify-diagnostics %s
// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Static allocs: A (512 B) spans the whole function; B and C (256 B each) have
// disjoint lifetimes, so C reuses B's slot. Peak concurrent = A+B = 768 B, and
// best-fit also uses 768 B.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_static
// CHECK: hip.get_pool
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 768/768 B (0 frag); dyn-pool 0/0 units (0 frag); small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_static(%ctx: !hip.context,
                              %arg: memref<1x128xf32>) -> memref<1x128xf32> {
  %a = memref.alloc() : memref<1x128xf32>
  %b = memref.alloc() : memref<1x64xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<1x128xf32>)
                           outs(%a : memref<1x128xf32>)
  hip.miopen.softmax(%ctx) ins(%b : memref<1x64xf32>)
                           outs(%b : memref<1x64xf32>)
  // The final use of %b precedes the allocation of %c.
  hip.miopen.softmax(%ctx) ins(%b : memref<1x64xf32>)
                           outs(%b : memref<1x64xf32>)

  %c = memref.alloc() : memref<1x64xf32>
  hip.miopen.softmax(%ctx) ins(%c : memref<1x64xf32>)
                           outs(%c : memref<1x64xf32>)
  // %a remains live across the lifetimes of %b and %c.
  hip.miopen.softmax(%ctx) ins(%a : memref<1x128xf32>)
                           outs(%a : memref<1x128xf32>)
  return %arg : memref<1x128xf32>
}

//===----------------------------------------------------------------------===//
// Two dynamic allocations use the same operand %d and have coefficients 32768
// and 16384. Their disjoint lifetimes permit one slab with coefficient 32768,
// equal to the max-load lower bound.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_dyn
// CHECK: hip.get_pool
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 0/0 B (0 frag); dyn-pool 32768/32768 units (0 frag); small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_dyn(%ctx: !hip.context,
                           %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  %a = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  %b = memref.alloc(%d) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)
  return %arg : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// A non-alignment-multiple dynamic slab is rounded up in byte space at runtime,
// so no compile-time coefficient comparison is available.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_runtime_aligned
// CHECK: %[[DIM:.*]] = memref.dim
// CHECK: %[[COEFF:.*]] = arith.constant 8 : index
// CHECK: %[[WIDTH:.*]] = arith.muli %[[DIM]], %[[COEFF]] : index
// CHECK: %[[C255:.*]] = arith.constant 255 : index
// CHECK: %[[C256:.*]] = arith.constant 256 : index
// CHECK: %[[ADJUSTED:.*]] = arith.addi %[[WIDTH]], %[[C255]] : index
// CHECK: %[[QUOTIENT:.*]] = arith.divui %[[ADJUSTED]], %[[C256]] : index
// CHECK: %[[ALIGNED:.*]] = arith.muli %[[QUOTIENT]], %[[C256]] : index
// CHECK: hip.get_pool(%{{.*}}, %[[ALIGNED]])
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 0/0 B (0 frag); dyn-pool coefficient unavailable (runtime alignment); small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_runtime_aligned(
    %ctx: !hip.context, %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  // The aligned slab width alignUp(8 * %d, 256) cannot generally be expressed
  // as C * %d for a compile-time coefficient C.
  %small = memref.alloc(%d) : memref<?x2xf32>
  hip.miopen.softmax(%ctx) ins(%small : memref<?x2xf32>)
                           outs(%small : memref<?x2xf32>)
  return %arg : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// A slab may contain an unaligned member without requiring runtime alignment
// when all members share F and the maximum coefficient is alignment-multiple.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_aligned_effective_width
// CHECK-NOT: arith.divui
// CHECK: hip.get_pool
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 0/0 B (0 frag); dyn-pool 32768/32768 units (0 frag); small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_aligned_effective_width(
    %ctx: !hip.context, %arg: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d = memref.dim %arg, %c0 : memref<?xf32>

  %large = memref.alloc(%d) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%large : memref<?x8192xf32>)
                           outs(%large : memref<?x8192xf32>)

  %small = memref.alloc(%d) : memref<?x2xf32>
  hip.miopen.softmax(%ctx) ins(%small : memref<?x2xf32>)
                           outs(%small : memref<?x2xf32>)
  return %arg : memref<?xf32>
}

//===----------------------------------------------------------------------===//
// Groups with different runtime factors have no common coefficient unit.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @frag_report_mixed_factors
// CHECK: %[[WIDTH:.*]] = arith.maxui
// CHECK: hip.get_pool(%{{.*}}, %[[WIDTH]])
// CHECK-NOT: hip.get_pool

// expected-remark@+1 {{static 0/0 B (0 frag); dyn-pool coefficient unavailable (mixed runtime factors); small-buckets 0/0 bins (0 excess)}}
func.func @frag_report_mixed_factors(
    %ctx: !hip.context, %arg0: memref<?xf32>,
    %arg1: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg0, %c0 : memref<?xf32>
  %d1 = memref.dim %arg1, %c0 : memref<?xf32>

  %a = memref.alloc(%d0) : memref<?x8192xf32>
  hip.miopen.softmax(%ctx) ins(%a : memref<?x8192xf32>)
                           outs(%a : memref<?x8192xf32>)

  %b = memref.alloc(%d1) : memref<?x4096xf32>
  hip.miopen.softmax(%ctx) ins(%b : memref<?x4096xf32>)
                           outs(%b : memref<?x4096xf32>)
  return %arg0 : memref<?xf32>
}
