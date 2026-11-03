// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/tensor_hip.mlir --hip-tensor-to-bufferized-pipeline | FileCheck %s --check-prefix=INLINE --implicit-check-not=tensor. --implicit-check-not=hip.constant
// RUN: mkdir -p %t/out
// RUN: hip-mlir-opt %t/tensor_hip.mlir --hip-tensor-to-bufferized-pipeline='externalize-min-num-elements=1 externalize-output-dir=%t/out' | FileCheck %s --check-prefix=EXTERNAL
// RUN: test -f %t/out/model.constants.bin
// RUN: hip-mlir-opt %t/empty.mlir --hip-tensor-to-bufferized-pipeline='externalize-min-num-elements=4 skip-constant-data=true' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=TENSOR-PIPELINE
// RUN: hip-mlir-opt %t/onnx.mlir --onnx-to-hip-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ONNX-PIPELINE

// --hip-tensor-to-bufferized-pipeline starts from the frontend boundary:
// tensor-form HIP IR with the !hip.context argument, the graph metadata and
// hip.constant carriers, and no source-dialect operation. The input below is
// what convert-onnx-to-hip produces for `y = x + w`, but nothing here needs an
// ONNX operation or pass.

// Default options keep the constant inline; the result is bufferized and the
// graph output is allocated through the output allocator.
// INLINE:       memref.global "private" constant @{{.*}} : memref<2x4xf32> = dense<
// INLINE-LABEL: func.func @main_graph
// INLINE-SAME:    (%[[CTX:.*]]: !hip.context, %[[X:.*]]: memref<2x4xf32> {onnx.name = "x"}) -> memref<2x4xf32>
// INLINE:         %[[W:.*]] = memref.get_global
// INLINE:         %[[Y:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<2x4xf32>
// INLINE:         hip.add(%[[CTX]]) ins(%[[X]], %[[W]] : memref<2x4xf32>, memref<2x4xf32>) outs(%[[Y]] : memref<2x4xf32>)
// INLINE:         return %[[Y]]
// No tensor op or hip.constant carrier is left anywhere in the output
// (--implicit-check-not on the RUN line).

// The pipeline options reach constant externalization: at the production
// threshold the constant is written to the constants file and read back at run
// time.
// EXTERNAL:       module attributes {hip.constants_file = "model.constants.bin"
// EXTERNAL-LABEL: func.func @main_graph
// EXTERNAL:         %[[W:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) : memref<2x4xf32>
// EXTERNAL:         hip.add(%{{.*}}) ins(%{{.*}}, %[[W]] : memref<2x4xf32>, memref<2x4xf32>)
// EXTERNAL-NOT:   memref.global "private" constant

// The tensor pipeline begins with HIP shape refinement and fusion, and
// forwards its options to hip-externalize-constants.
// TENSOR-PIPELINE:      builtin.module(
// TENSOR-PIPELINE-NEXT:   hip-infer-shapes,
// TENSOR-PIPELINE-NEXT:   hip-fusion-transform,
// TENSOR-PIPELINE-NEXT:   {{.*}}hip-externalize-constants{externalize-min-num-elements=4 externalize-output-dir= skip-constant-data=true}

// The ONNX pipeline is its head followed by that pipeline: after conversion
// (and any plugin passes at the AfterConvertOnnxToHip slot) comes the tensor
// pipeline's opening sequence.
// ONNX-PIPELINE:      convert-onnx-to-hip,
// ONNX-PIPELINE:      hip-infer-shapes,
// ONNX-PIPELINE-NEXT: hip-fusion-transform,

//--- tensor_hip.mlir
module attributes {hipdnn.input_count = 1 : i64, hipdnn.input_element_sizes = array<i64: 4>, hipdnn.input_shapes = [array<i64: 2, 4>], hipdnn.output_count = 1 : i64, hipdnn.output_element_sizes = array<i64: 4>, hipdnn.output_shapes = [array<i64: 2, 4>]} {
  func.func @main_graph(%ctx: !hip.context,
                        %x: tensor<2x4xf32> {onnx.name = "x"})
      -> tensor<2x4xf32> {
    %w = hip.constant {serialization_order = 0 : i64, source_name = "w",
                       symbol_name_hint = "w",
                       value = dense<[[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>}
        : tensor<2x4xf32>
    %e = tensor.empty() : tensor<2x4xf32>
    %y = hip.add(%ctx) ins(%x, %w : tensor<2x4xf32>, tensor<2x4xf32>)
        outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }
}

//--- empty.mlir
module {
}

//--- onnx.mlir
module {
  func.func @main_graph(%x: tensor<2x4xf32> {onnx.name = "x"})
      -> (tensor<2x4xf32> {onnx.name = "y"}) {
    %y = "onnx.Relu"(%x) {onnx_node_name = "relu"}
        : (tensor<2x4xf32>) -> tensor<2x4xf32>
    "onnx.Return"(%y) : (tensor<2x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
