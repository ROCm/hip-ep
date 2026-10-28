// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// REQUIRES: hip_static_plugins
//
// A module whose `hip.compiler.frontend_pipeline` attribute names a plugin
// frontend pipeline is compiled by that plugin's passes followed by the host's
// tensor-to-bufferized and HIP-to-LLVM pipelines, not by the ONNX pipeline.
// The sample registers "hip-ep-sample" (see sample_plugin.cpp). Every case
// stops at or before LLVM IR emission, so no GPU is needed.
//
// hip-compiler writes the constants file to the working directory, and the IR
// dump records each pass that changed the module.
// RUN: split-file %s %t
// RUN: cd %t && env HIPDNN_EP_IR_DUMP_PATH=%t/selected.dump HIPDNN_EP_IR_DUMP_SINGLE=1 HIPDNN_EP_IR_DUMP_AFTER_ONLY=1 hip-compiler selected.mlir -o selected.bc
// RUN: FileCheck %s --check-prefix=SELECTED < %t/selected.dump
// RUN: FileCheck %s --check-prefix=CONSTANTS < %t/model.constants.json
// RUN: not hip-compiler %t/non_string.mlir -o %t/out.bc 2>&1 | FileCheck %s --check-prefix=NOT-STRING
// RUN: not hip-compiler %t/empty_string.mlir -o %t/out.bc 2>&1 | FileCheck %s --check-prefix=NOT-STRING
// RUN: not hip-compiler %t/unregistered.mlir -o %t/out.bc 2>&1 | FileCheck %s --check-prefix=UNREGISTERED
// RUN: not hip-compiler %t/duplicate.mlir -o %t/out.bc 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not hip-compiler %t/rejected.mlir -o %t/out.bc 2>&1 | FileCheck %s --check-prefix=REJECTED
// RUN: not hip-compiler %t/rejected_silently.mlir -o %t/out.bc 2>&1 | FileCheck %s --check-prefix=REJECTED-SILENTLY

// The sample's lowering pass runs first, then the tensor-to-bufferized
// pipeline, which externalizes the lowering's carrier together with the
// input's, and the HIP-to-LLVM pipeline, which emits the entry point. If the
// ONNX pipeline ran instead, the lowering pass and its carrier would be absent.
// SELECTED:      IR Dump After {{.*}}SampleFrontendPass
// SELECTED-NOT:  hip_ep_sample.frontend_input
// SELECTED:      hip.constant {source_name = "sample_frontend_weight"
// SELECTED:      IR Dump After {{.*}}ExternalizeConstantsPass
// SELECTED-NOT:  = hip.constant
// SELECTED:      IR Dump After {{.*}}GenerateInterfacePass
// SELECTED:      llvm.func @inference_compute(
// CONSTANTS-DAG: "name": "hip_ext_constant_w_0"
// CONSTANTS-DAG: "name": "hip_ext_constant_sample_frontend_weight_1"

// NOT-STRING: error: module attribute 'hip.compiler.frontend_pipeline' must be a nonempty string
// UNREGISTERED: error: no compiler plugin registered frontend pipeline 'hip-ep-sample-unregistered'
// DUPLICATE: error: multiple compiler plugins registered frontend pipeline 'hip-ep-sample-duplicate'
// REJECTED: error: hip-ep-sample cannot lower this module
// REJECTED-SILENTLY: error: frontend pipeline 'hip-ep-sample' rejected the input module

//--- selected.mlir
// Tensor-form HIP IR for `y = x + w`, marked for the sample's lowering pass;
// nothing in it is ONNX.
module attributes {hip.compiler.frontend_pipeline = "hip-ep-sample", hipdnn.input_count = 1 : i64, hipdnn.input_element_sizes = array<i64: 4>, hipdnn.input_shapes = [array<i64: 2>], hipdnn.output_count = 1 : i64, hipdnn.output_element_sizes = array<i64: 4>, hipdnn.output_shapes = [array<i64: 2>]} {
  func.func @main_graph(%ctx: !hip.context, %x: tensor<2xf32> {onnx.name = "x"})
      -> tensor<2xf32> attributes {hip_ep_sample.frontend_input} {
    %w = hip.constant {serialization_order = 0 : i64, source_name = "w",
                       symbol_name_hint = "w",
                       value = dense<[1.0, 2.0]> : tensor<2xf32>}
        : tensor<2xf32>
    %e = tensor.empty() : tensor<2xf32>
    %y = hip.add(%ctx) ins(%x, %w : tensor<2xf32>, tensor<2xf32>)
        outs(%e : tensor<2xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }
}

//--- non_string.mlir
module attributes {hip.compiler.frontend_pipeline = 1 : i64} {
}

//--- empty_string.mlir
module attributes {hip.compiler.frontend_pipeline = ""} {
}

//--- unregistered.mlir
module attributes {hip.compiler.frontend_pipeline = "hip-ep-sample-unregistered"} {
}

//--- duplicate.mlir
module attributes {hip.compiler.frontend_pipeline = "hip-ep-sample-duplicate"} {
}

//--- rejected.mlir
module attributes {hip.compiler.frontend_pipeline = "hip-ep-sample", hip_ep_sample.reject} {
}

//--- rejected_silently.mlir
module attributes {hip.compiler.frontend_pipeline = "hip-ep-sample", hip_ep_sample.reject_silently} {
}
