// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Equal becomes hipsr.equal with a ui8 mask, the byte the runtime writes.
// Both operands must be on the device, so a host constant gets a device copy.
// Rejected forms are in equal-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// An embedding graph holds the id it compares against as a rank-0 constant,
// which the constant conversion leaves on the host, so it gets a device
// constant of the same rank. The ids are already on the device and stay as
// they are. The placeholder gets no shape region here: hipsr.equal is DPS, so
// hipsr-populate-shape-region fills it in later.
// CHECK-LABEL: func.func @host_scalar_on_device(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[IDS:.+]]: tensor<?x?xi64, #hipsr.mem<device>>) -> tensor<?x?xui8, #hipsr.mem<device>> {
// The host constant has no use left and waits for canonicalization.
// CHECK-NEXT:    arith.constant dense<248056> : tensor<i64>
// CHECK-NEXT:    %[[TOKEN:.+]] = hipsr.constant {value = dense<248056> : tensor<i64>} : tensor<i64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[IDS]], %[[TOKEN]] : tensor<?x?xi64, #hipsr.mem<device>>, tensor<i64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?xui8, #hipsr.mem<device>>
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.equal(%[[CTX]]) ins(%[[IDS]], %[[TOKEN]] : tensor<?x?xi64, #hipsr.mem<device>>, tensor<i64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?xui8, #hipsr.mem<device>>) : tensor<?x?xui8, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x?xui8, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @host_scalar_on_device(%ctx: !hipsr.context, %ids: tensor<?x?xi64>)
    -> tensor<?x?xi1> {
  %0 = "onnx.Constant"() {value = dense<248056> : tensor<i64>}
      : () -> tensor<i64>
  %1 = "onnx.Equal"(%ids, %0) : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi1>
  "onnx.Return"(%1) : (tensor<?x?xi1>) -> ()
}
