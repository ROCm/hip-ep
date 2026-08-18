// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: hipsr pipeline on a dynamically shaped embedding graph.
//
// The input is the ONNX-MLIR form of a text-embedding graph that scatters image
// features into token embeddings. Every batch and sequence extent stays
// symbolic, and NonZero makes the scatter index count data-dependent, so the
// pipeline must never freeze an extent to a constant. The embedding table is a
// two-gigabyte external constant carried as an offset/size reference rather
// than inline data.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hipsr-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[IDS:.*]]: tensor<?x?xi64> {onnx.name = "input_ids"},
// CHECK-SAME:      %[[FEATURES:.*]]: tensor<?x4096xf16> {onnx.name = "image_features"})
// CHECK-SAME:      -> (tensor<?x?x4096xf16> {onnx.name = "inputs_embeds"})

// The embedding table stays a reference to external data.
// CHECK: %[[TABLE:.*]] = "onnx.Constant"() {location = "embedding.onnx.data"
// CHECK-SAME: offset = 0 : i64, size = 2034237440 : i64} : () -> tensor<248320x4096xf16>

// CHECK: %[[FLAT:.*]] = "onnx.Reshape"(%[[FEATURES]], %{{.*}}) {{.*}} -> tensor<?xf16>
// CHECK: %[[MASK:.*]] = "onnx.Equal"(%[[IDS]], %{{.*}}) {{.*}} -> tensor<?x?xui8>
// CHECK: %[[EMBEDS:.*]] = "onnx.Gather"(%[[TABLE]], %[[IDS]]) {{.*}} -> tensor<?x?x4096xf16>
// CHECK: %[[SHAPE:.*]] = "onnx.Shape"(%[[EMBEDS]]) {{.*}} -> tensor<3xi64>

// NonZero yields a data-dependent trailing extent that feeds the scatter.
// CHECK: %[[NONZERO:.*]] = "onnx.NonZero"(%{{.*}}) {{.*}} -> tensor<3x?xi64>
// CHECK: %[[INDICES:.*]] = "onnx.Transpose"(%[[NONZERO]]) {{.*}} -> tensor<?x3xi64>
// CHECK: %[[UPDATES:.*]] = "onnx.Slice"(%[[FLAT]], {{.*}}) {{.*}} -> tensor<?xf16>
// CHECK: %[[OUT:.*]] = "onnx.ScatterND"(%[[EMBEDS]], %[[INDICES]], %[[UPDATES]])
// CHECK-SAME: -> tensor<?x?x4096xf16>
// CHECK: "onnx.Return"(%[[OUT]])

module {
  func.func @main_graph(%arg0: tensor<?x?xi64> {onnx.name = "input_ids"}, %arg1: tensor<?x4096xf16> {onnx.name = "image_features"}) -> (tensor<?x?x4096xf16> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {location = "embedding.onnx.data", node.outputs = ["embed_tokens.weight"], offset = 0 : i64, size = 2034237440 : i64} : () -> tensor<248320x4096xf16>
    %2 = "onnx.Constant"() {node.outputs = ["/Constant_output_0"], value = dense<248056> : tensor<i64>} : () -> tensor<i64>
    %3 = "onnx.Constant"() {node.outputs = ["/Constant_1_output_0"], value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %4 = "onnx.Constant"() {node.outputs = ["/Constant_3_output_0"], value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %5 = "onnx.Constant"() {node.outputs = ["/Constant_4_output_0"], value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %6 = "onnx.Reshape"(%arg1, %3) {allowzero = 0 : si64, node.outputs = ["/Reshape_output_0"], onnx_node_name = "/Reshape"} : (tensor<?x4096xf16>, tensor<1xi64>) -> tensor<?xf16>
    %7 = "onnx.Equal"(%arg0, %2) {node.outputs = ["/Equal_output_0"], onnx_node_name = "/Equal"} : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xui8>
    %8 = "onnx.Unsqueeze"(%7, %3) {node.outputs = ["/Unsqueeze_output_0"], onnx_node_name = "/Unsqueeze"} : (tensor<?x?xui8>, tensor<1xi64>) -> tensor<?x?x1xui8>
    %9 = "onnx.Gather"(%1, %arg0) {axis = 0 : si64, node.outputs = ["/embed_tokens/Gather_output_0"], onnx_node_name = "/embed_tokens/Gather"} : (tensor<248320x4096xf16>, tensor<?x?xi64>) -> tensor<?x?x4096xf16>
    %10 = "onnx.Shape"(%9) {node.outputs = ["/Shape_1_output_0"], onnx_node_name = "/Shape_1", start = 0 : si64} : (tensor<?x?x4096xf16>) -> tensor<3xi64>
    %11 = "onnx.Expand"(%8, %10) {node.outputs = ["/Expand_output_0"], onnx_node_name = "/Expand"} : (tensor<?x?x1xui8>, tensor<3xi64>) -> tensor<?x?x?xui8>
    %12 = "onnx.Expand"(%11, %10) {node.outputs = ["/Expand_1_output_0"], onnx_node_name = "/Expand_1"} : (tensor<?x?x?xui8>, tensor<3xi64>) -> tensor<?x?x?xui8>
    %13 = "onnx.NonZero"(%12) {node.outputs = ["/NonZero_output_0"], onnx_node_name = "/NonZero"} : (tensor<?x?x?xui8>) -> tensor<3x?xi64>
    %14 = "onnx.Transpose"(%13) {node.outputs = ["/Transpose_output_0"], onnx_node_name = "/Transpose", perm = [1, 0]} : (tensor<3x?xi64>) -> tensor<?x3xi64>
    %15 = "onnx.Shape"(%14) {node.outputs = ["/Shape_2_output_0"], onnx_node_name = "/Shape_2", start = 0 : si64} : (tensor<?x3xi64>) -> tensor<2xi64>
    %16 = "onnx.Gather"(%15, %4) {axis = 0 : si64, node.outputs = ["/Gather_output_0"], onnx_node_name = "/Gather"} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    %17 = "onnx.Unsqueeze"(%16, %5) {node.outputs = ["/Unsqueeze_1_output_0"], onnx_node_name = "/Unsqueeze_1"} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %18 = "onnx.Slice"(%6, %5, %17, %5) {node.outputs = ["/Slice_output_0"], onnx_node_name = "/Slice"} : (tensor<?xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf16>
    %19 = "onnx.ScatterND"(%9, %14, %18) {node.outputs = ["inputs_embeds"], onnx_node_name = "/ScatterND", reduction = "none"} : (tensor<?x?x4096xf16>, tensor<?x3xi64>, tensor<?xf16>) -> tensor<?x?x4096xf16>
    "onnx.Return"(%19) : (tensor<?x?x4096xf16>) -> ()
  }
}
