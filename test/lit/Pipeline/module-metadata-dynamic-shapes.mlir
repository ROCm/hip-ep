// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: module metadata with dynamic shapes.
//
// Verifies that dynamic dimensions (?) are preserved as -1 (i.e.
// ShapedType::kDynamic) in the metadata shape arrays, AND that the new
// `hipdnn.output_dim_specs` + `hipdnn.dyn_dim_slots_count` attributes
// (added by the `ComposeDimSpecs` pass) describe how the EP can resolve
// each dynamic output dim at compute time.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s --check-prefix=PRE
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --hip-compose-dim-specs %s | FileCheck %s --check-prefix=POST

// PRE: module attributes {
// PRE-SAME: hipdnn.input_count = 1 : i64
// PRE-SAME: hipdnn.input_element_sizes = array<i64: 4>
// PRE-SAME: hipdnn.input_shapes = [array<i64: -9223372036854775808, 3, -9223372036854775808>]
// PRE-SAME: hipdnn.output_count = 1 : i64
// PRE-SAME: hipdnn.output_element_sizes = array<i64: 4>
// PRE-SAME: hipdnn.output_shapes = [array<i64: -9223372036854775808, 3, -9223372036854775808>]

// After ComposeDimSpecs runs, the module gains an `hipdnn.output_dim_specs`
// attribute and a `hipdnn.dyn_dim_slots_count` counter.  For the trivial
// identity graph there are no Category-C slots (count = 0), and each
// dynamic output dim should resolve to an `InputDim` leaf pointing back at
// the EP-relative input + dim.
// POST: hipdnn.dyn_dim_slots_count = 0 : i32
// POST-SAME: hipdnn.output_dim_specs

module {
  func.func @main_graph(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"} : (tensor<?x3x?xf32>) -> ()
  }
}
