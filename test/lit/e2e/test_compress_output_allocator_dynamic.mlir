// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Regression guard for a data-dependent Compress graph output.
//
// Compress keeps one slice per TRUE condition entry, so its extent is only
// known once the condition has been scanned. The EP allocates graph outputs
// in-graph through hip.alloc_output and ORT rejects a request that does not
// match the shape it computed for the run, so sizing the output from the
// condition LENGTH (the upper bound) fails on any model that pads its input
// and drops the pad slices with Compress -- e.g. a variable-resolution vision
// encoder padded to a fixed patch capacity.
//
// This test pins the fix: the dynamic hip.alloc_output extent must come from
// the scanned count (hip.nonzero -> hip.readback_dim), not from the condition
// buffer's extent.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.nonzero
// CHECK:         %[[N:.*]] = hip.readback_dim
// CHECK:         hip.alloc_output(%{{.*}}, %[[N]]) {out_idx = 0 : i64} : memref<?x2xf16>
// CHECK:         hip.compress

module {
  func.func @main_graph(%input: tensor<?x2xf16> {onnx.name = "input"},
                        %condition: tensor<?xi1> {onnx.name = "condition"})
      -> (tensor<?x2xf16> {onnx.name = "output"}) {
    %0 = "onnx.Compress"(%input, %condition) {
      axis = 0 : si64,
      onnx_node_name = "compress_node"
    } : (tensor<?x2xf16>, tensor<?xi1>) -> tensor<?x2xf16>
    "onnx.Return"(%0) : (tensor<?x2xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
