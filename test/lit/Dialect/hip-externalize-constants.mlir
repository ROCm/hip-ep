// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: mkdir -p %t && hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=4 externalize-output-dir=%t' %s | FileCheck %s

// Spike: prove the neutral hip.constant carrier + the standalone, dialect-
// agnostic hip-externalize-constants pass. There is NO onnx op anywhere in this
// file -- the pass keys only on hip.constant, so this also demonstrates that a
// downstream plugin (which never touches the onnx dialect) can emit hip.constant
// and get its weights externalized identically to in-tree ops.

// Module-level externalization metadata is stamped (same contract that
// convert-onnx-to-hip stamps today, so generate-interface + runtime are
// unaffected by the split).
// CHECK-DAG: hip.constants_file = "model.constants.bin"
// CHECK-DAG: hipdnn.constants = [{kind = 0 : i64, offset = 0 : i64, size = 32 : i64}]

// The large constant (8 elements >= threshold 4) becomes an extern
// memref.global carrying hip.external_data (the exact form
// hip-resolve-extern-constants consumes).
// CHECK: memref.global "private" @hip_ext_constant_0
// CHECK-SAME: hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 32 : i64}

// CHECK-LABEL: func.func @large_const
// CHECK: %[[G:.*]] = memref.get_global @hip_ext_constant_0 : memref<2x4xf32>
// CHECK: bufferization.to_tensor %[[G]] restrict : memref<2x4xf32> to tensor<2x4xf32>
// CHECK-NOT: hip.constant
func.func @large_const() -> tensor<2x4xf32> {
  %0 = hip.constant {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// The small constant (2 elements < threshold) folds to an inline arith.constant.
// CHECK-LABEL: func.func @small_const
// CHECK: arith.constant
// CHECK-NOT: hip.constant
func.func @small_const() -> tensor<2xf32> {
  %0 = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
  return %0 : tensor<2xf32>
}
