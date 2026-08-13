// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Where (elementwise selection) is correctly lowered
// to hip.where operation in tensor-first mode.
//
// SCOPE: this is a static IR-rewrite test driven by `hip-mlir-opt | FileCheck`.
// It runs the `--convert-onnx-to-hip` pass and inspects the textual IR ONLY.
// No HIP kernel is compiled or launched, and no runtime code path is exercised
// by these checks. Therefore the set of element types covered here intentionally
// matches the ONNX `Where` type constraint (NumPy-style typing) rather than the
// narrower set currently dispatched by the runtime kernel. The conversion is
// designed to be type-agnostic so it remains correct as the kernel grows
// support for more element types; runtime-level type coverage is gated
// independently by the EP capability check and the kernel's dtype switch.
//
// This test validates:
// - Ternary elementwise lowering (onnx.Where -> hip.where)
// - Three-input operand handling (condition, x, y)
// - Bool (i1) condition + matching x/y/output element type
// - Same-rank operands and multidirectional broadcasting (NumPy-style)
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// - Type-agnostic lowering across the full ONNX `Where` T constraint
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    return %arg0 : tensor<2x4xf32>
  }

  // Test 1: Same-shape Where (2x4 f32)
  func.func @test_where_same_shape(%cond: tensor<2x4xi1>,
                                   %x: tensor<2x4xf32>,
                                   %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_where_same_shape
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<2x4xi1>, %[[X:.*]]: tensor<2x4xf32>, %[[Y:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // CHECK: tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>) outs({{.*}} : tensor<2x4xf32>)
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<2x4xf32>
  }

  // Test 2: Multidirectional broadcasting (NumPy-style)
  // condition [1,4] + x [2,1] + y [2,4] -> output [2,4]
  func.func @test_where_broadcast(%cond: tensor<1x4xi1>,
                                  %x: tensor<2x1xf32>,
                                  %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_where_broadcast
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<1x4xi1>, %[[X:.*]]: tensor<2x1xf32>, %[[Y:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<1x4xi1>, tensor<2x1xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // CHECK: tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<1x4xi1>, tensor<2x1xf32>, tensor<2x4xf32>) outs({{.*}} : tensor<2x4xf32>)

    return %output : tensor<2x4xf32>
  }

  // Test 3: i64 element type (X/Y are int64)
  func.func @test_where_i64(%cond: tensor<3x5xi1>,
                            %x: tensor<3x5xi64>,
                            %y: tensor<3x5xi64>) -> tensor<3x5xi64> {
    // CHECK-LABEL: func.func @test_where_i64
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<3x5xi1>, %[[X:.*]]: tensor<3x5xi64>, %[[Y:.*]]: tensor<3x5xi64>) -> tensor<3x5xi64>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<3x5xi1>, tensor<3x5xi64>, tensor<3x5xi64>) -> tensor<3x5xi64>

    // CHECK: tensor.empty() : tensor<3x5xi64>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<3x5xi1>, tensor<3x5xi64>, tensor<3x5xi64>) outs({{.*}} : tensor<3x5xi64>)

    return %output : tensor<3x5xi64>
  }

  // Test 4: Dynamic shapes
  func.func @test_where_dynamic(%cond: tensor<?x?xi1>,
                                %x: tensor<?x?xf16>,
                                %y: tensor<?x?xf16>) -> tensor<?x?xf16> {
    // CHECK-LABEL: func.func @test_where_dynamic
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<?x?xi1>, %[[X:.*]]: tensor<?x?xf16>, %[[Y:.*]]: tensor<?x?xf16>) -> tensor<?x?xf16>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<?x?xi1>, tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>

    // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf16>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<?x?xi1>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
    // CHECK-NOT: hip.alloc

    return %output : tensor<?x?xf16>
  }

  // Test 5: An earlier dynamic operand may resolve to the broadcast value 1.
  // The destination must select X's runtime extent rather than unconditionally
  // using condition's extent.
  func.func @test_where_dynamic_unit_broadcast(
      %cond: tensor<?xi1>, %x: tensor<?xf16>, %y: tensor<?xf16>)
      -> tensor<?xf16> {
    // CHECK-LABEL: func.func @test_where_dynamic_unit_broadcast
    // CHECK-DAG: %[[COND_DIM:.*]] = tensor.dim %[[COND:.*]], %{{.*}} : tensor<?xi1>
    // CHECK-DAG: %[[X_DIM:.*]] = tensor.dim %[[X:.*]], %{{.*}} : tensor<?xf16>
    // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
    // CHECK: %[[COND_IS_ONE:.*]] = arith.cmpi eq, %[[COND_DIM]], %[[ONE]] : index
    // CHECK: %[[EXTENT:.*]] = arith.select %[[COND_IS_ONE]], %[[X_DIM]], %[[COND_DIM]] : index
    // CHECK: tensor.empty(%{{.*}}) : tensor<?xf16>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<?xi1>, tensor<?xf16>, tensor<?xf16>) -> tensor<?xf16>
    return %output : tensor<?xf16>
  }

  // --------------------------------------------------------------------------
  // Tests 6-9: type-agnostic conversion coverage.
  //
  // These cases confirm the OnnxToHip rewrite preserves the operand/result
  // element type for the full ONNX `Where` T constraint (uint8, int8, double,
  // bool, ...) -- not just the subset the runtime kernel currently dispatches.
  // They are pure FileCheck assertions on the rewritten IR; nothing here gets
  // lowered to LLVM or executed. Whether a given element type is actually
  // supported end-to-end is decided later by the EP capability check and the
  // kernel's dtype switch, which are exercised by other test layers.
  // --------------------------------------------------------------------------

  // Test 6: Signed 8-bit integer X/Y.
  func.func @test_where_i8(%cond: tensor<2x4xi1>,
                           %x: tensor<2x4xi8>,
                           %y: tensor<2x4xi8>) -> tensor<2x4xi8> {
    // CHECK-LABEL: func.func @test_where_i8
    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<2x4xi1>, tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>

    // CHECK: tensor.empty() : tensor<2x4xi8>
    // CHECK: hip.where(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4xi1>, tensor<2x4xi8>, tensor<2x4xi8>) outs(%{{.*}} : tensor<2x4xi8>)

    return %output : tensor<2x4xi8>
  }

  // Test 7: Unsigned 8-bit integer X/Y (verifies the ui8 path is not rejected).
  func.func @test_where_ui8(%cond: tensor<2x4xi1>,
                            %x: tensor<2x4xui8>,
                            %y: tensor<2x4xui8>) -> tensor<2x4xui8> {
    // CHECK-LABEL: func.func @test_where_ui8
    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<2x4xi1>, tensor<2x4xui8>, tensor<2x4xui8>) -> tensor<2x4xui8>

    // CHECK: tensor.empty() : tensor<2x4xui8>
    // CHECK: hip.where(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4xi1>, tensor<2x4xui8>, tensor<2x4xui8>) outs(%{{.*}} : tensor<2x4xui8>)

    return %output : tensor<2x4xui8>
  }

  // Test 8: f64 (double) X/Y.
  func.func @test_where_f64(%cond: tensor<2x4xi1>,
                            %x: tensor<2x4xf64>,
                            %y: tensor<2x4xf64>) -> tensor<2x4xf64> {
    // CHECK-LABEL: func.func @test_where_f64
    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<2x4xi1>, tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x4xf64>

    // CHECK: tensor.empty() : tensor<2x4xf64>
    // CHECK: hip.where(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4xi1>, tensor<2x4xf64>, tensor<2x4xf64>) outs(%{{.*}} : tensor<2x4xf64>)

    return %output : tensor<2x4xf64>
  }

  // Test 9: bool (i1) X/Y -- where(cond, true_v, false_v) over bool tensors.
  func.func @test_where_bool(%cond: tensor<2x4xi1>,
                             %x: tensor<2x4xi1>,
                             %y: tensor<2x4xi1>) -> tensor<2x4xi1> {
    // CHECK-LABEL: func.func @test_where_bool
    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<2x4xi1>, tensor<2x4xi1>, tensor<2x4xi1>) -> tensor<2x4xi1>

    // CHECK: tensor.empty() : tensor<2x4xi1>
    // CHECK: hip.where(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4xi1>, tensor<2x4xi1>, tensor<2x4xi1>) outs(%{{.*}} : tensor<2x4xi1>)

    return %output : tensor<2x4xi1>
  }

  // --------------------------------------------------------------------------
  // Tests 10-11: rank-0 (scalar) coverage.
  //
  // ONNX Where permits rank-0 tensors under NumPy-style multidirectional
  // broadcasting. The runtime kernel handles scalars through natural loop
  // degeneracy (see `hip_elementwise_where` in
  // lib/Runtime/Kernels/hip/elementwise_where_kernel.hip), and the
  // lowering uses `std::max(rank, 1)` for the alloca size so rank-0 stack
  // arrays remain valid. These cases exercise the conversion side: an empty
  // tensor.empty() plus a hip.where on rank-0 operands.
  // --------------------------------------------------------------------------

  // Test 10: All-scalar Where -- cond / x / y / out are all rank-0 f32.
  func.func @test_where_scalar(%cond: tensor<i1>,
                               %x: tensor<f32>,
                               %y: tensor<f32>) -> tensor<f32> {
    // CHECK-LABEL: func.func @test_where_scalar
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<i1>, %[[X:.*]]: tensor<f32>, %[[Y:.*]]: tensor<f32>) -> tensor<f32>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

    // CHECK: tensor.empty() : tensor<f32>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<i1>, tensor<f32>, tensor<f32>) outs({{.*}} : tensor<f32>)

    return %output : tensor<f32>
  }

  // Test 11: Scalar broadcast against a rank-2 tensor.
  // cond: i1, x: f32 (scalar), y: 2x4xf32 -> 2x4xf32. Verifies scalar
  // broadcast path on the conversion side.
  func.func @test_where_scalar_broadcast(%cond: tensor<i1>,
                                         %x: tensor<f32>,
                                         %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @test_where_scalar_broadcast
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<i1>, %[[X:.*]]: tensor<f32>, %[[Y:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>

    %output = "onnx.Where"(%cond, %x, %y) :
        (tensor<i1>, tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // CHECK: tensor.empty() : tensor<2x4xf32>
    // CHECK: hip.where(%[[CTX]]) ins(%[[COND]], %[[X]], %[[Y]] : tensor<i1>, tensor<f32>, tensor<2x4xf32>) outs({{.*}} : tensor<2x4xf32>)

    return %output : tensor<2x4xf32>
  }
}
