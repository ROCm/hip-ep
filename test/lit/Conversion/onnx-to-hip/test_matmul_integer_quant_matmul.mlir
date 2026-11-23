// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip=conv-integer-quant-conv --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --split-input-file %s | FileCheck %s --check-prefix=OFF

// hip.quant_matmul is outlined only from @main_graph's entry block, so each
// case below is its own @main_graph. The last case is a different function.
//
// With zp offsets offA/offB (128 for ui8, 0 for i8), za = azp - offA and
// zb = bzp - offB:
//   sum_k (a - azp)(b - bzp)
//     = qmm(a, b') - za * colsum(b') - zb * rowsum(a') + za * zb * K

// A signed int8 MatMulInteger with no zero points is one quant_matmul. There
// is no rebase and no correction. Without the option it stays on hip.matmul.
module {
  func.func @main_graph(%a: tensor<4x8xi8>, %b: tensor<8x6xi8>)
      -> tensor<4x6xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (tensor<4x8xi8>, tensor<8x6xi8>)
        -> tensor<4x6xi32>
    return %y : tensor<4x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.cast
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<4x8xi8>
// CHECK-SAME: tensor<8x6xi8>
// CHECK-SAME: tensor<4x6xi32>
// CHECK-NOT: hip.quant_matmul
// CHECK-NOT: hip.matmul
// CHECK-NOT: onnx.MatMulInteger

// OFF-LABEL: func.func @main_graph
// OFF-NOT: hip.quant_matmul
// OFF: hip.matmul

// -----

// A runtime a_zero_point on signed A adds the column-sum term
// za * colsum(B), computed by a quant_matmul against a ones row. There is no
// row-sum term because b_zero_point is omitted.
module {
  func.func @main_graph(%a: tensor<4x8xi8>, %b: tensor<8x6xi8>,
                        %azp: tensor<i8>) -> tensor<4x6xi32> {
    %y = "onnx.MatMulInteger"(%a, %b, %azp) : (
        tensor<4x8xi8>, tensor<8x6xi8>, tensor<i8>) -> tensor<4x6xi32>
    return %y : tensor<4x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<4x8xi8>
// CHECK-SAME: tensor<8x6xi8>
// CHECK-SAME: tensor<4x6xi32>
// CHECK: hip.constant {{.*}}dense<1> : tensor<1x8xi8>
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<1x8xi8>
// CHECK-SAME: tensor<1x6xi32>
// CHECK: hip.mul
// CHECK: hip.sub
// CHECK-SAME: tensor<4x6xi32>
// CHECK-NOT: hip.quant_matmul
// CHECK-NOT: hip.matmul

// -----

// A constant zero point equal to its rebase offset has no term: a signed
// zero point of 0 is the same as omitting it.
module {
  func.func @main_graph(%a: tensor<4x8xi8>, %b: tensor<8x6xi8>)
      -> tensor<4x6xi32> {
    %azp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
    %bzp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
    %y = "onnx.MatMulInteger"(%a, %b, %azp, %bzp) : (
        tensor<4x8xi8>, tensor<8x6xi8>, tensor<i8>, tensor<i8>)
        -> tensor<4x6xi32>
    return %y : tensor<4x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.quant_matmul
// CHECK-NOT: hip.quant_matmul
// CHECK-NOT: hip.sub
// CHECK-NOT: hip.matmul

// -----

// uint8 with both zero points omitted. A stays ui8 and the quant op applies
// its own 128 offset. B is rebased by 128 into i8. An omitted uint8 zero
// point is numeric 0, which is -128 after the rebase, so both correction
// terms and the za * zb * K term are required.
module {
  func.func @main_graph(%a: tensor<4x8xui8>, %b: tensor<8x6xui8>)
      -> tensor<4x6xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (tensor<4x8xui8>, tensor<8x6xui8>)
        -> tensor<4x6xi32>
    return %y : tensor<4x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// The omitted zero points are materialized as 0 and rebased by 128.
// CHECK: hip.constant {{.*}}dense<0> : tensor<ui8>
// CHECK: hip.constant {{.*}}dense<0> : tensor<ui8>
// CHECK: hip.constant {{.*}}dense<128> : tensor<i32>
// B is rebased: cast to i32, subtract 128, cast to i8.
// CHECK: hip.cast{{.*}}tensor<8x6xui8>
// CHECK: hip.sub{{.*}}tensor<8x6xi32>
// CHECK: hip.cast{{.*}}outs({{.*}} : tensor<8x6xi8>)
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<4x8xui8>, tensor<8x6xi8>
// CHECK-SAME: tensor<4x6xi32>
// colsum(B') with a ones row, then rowsum(A') with a ones column.
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<1x8xi8>, tensor<8x6xi8>
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<4x8xui8>, tensor<8x1xi8>
// CHECK-SAME: tensor<4x1xi32>
// za * zb * K, with K = 8.
// CHECK: hip.constant {{.*}}dense<8> : tensor<i32>
// CHECK: hip.add
// CHECK-SAME: tensor<4x6xi32>
// CHECK-NOT: hip.matmul

// -----

// A batch on A folds into M, a per-column b_zero_point becomes [1, N], and
// the result is expanded back. The row-sum term is [M, 1] * [1, N].
module {
  func.func @main_graph(%a: tensor<2x4x8xi8>, %b: tensor<8x6xi8>,
                        %bzp: tensor<6xi8>) -> tensor<2x4x6xi32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.MatMulInteger"(%a, %b, %none, %bzp) : (
        tensor<2x4x8xi8>, tensor<8x6xi8>, none, tensor<6xi8>)
        -> tensor<2x4x6xi32>
    return %y : tensor<2x4x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK: tensor.collapse_shape
// CHECK-SAME: tensor<2x4x8xi8> into tensor<8x8xi8>
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<8x8xi8>
// CHECK-SAME: tensor<8x6xi32>
// CHECK: hip.quant_matmul
// CHECK-SAME: tensor<8x8xi8>
// CHECK-SAME: tensor<8x1xi8>
// CHECK-SAME: tensor<8x1xi32>
// CHECK: hip.mul
// CHECK-SAME: tensor<8x6xi32>
// CHECK: hip.sub
// CHECK: tensor.expand_shape
// CHECK-SAME: tensor<8x6xi32> into tensor<2x4x6xi32>
// CHECK-NOT: hip.matmul

// -----

// A per-row a_zero_point would need its own M-wide term, so it stays on
// the cast/sub/hip.matmul lowering.
module {
  func.func @main_graph(%a: tensor<4x8xi8>, %b: tensor<8x6xi8>,
                        %azp: tensor<4xi8>) -> tensor<4x6xi32> {
    %y = "onnx.MatMulInteger"(%a, %b, %azp) : (
        tensor<4x8xi8>, tensor<8x6xi8>, tensor<4xi8>) -> tensor<4x6xi32>
    return %y : tensor<4x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.quant_matmul
// CHECK: hip.matmul

// -----

// A dynamic M stays on hip.matmul: the quant path needs static shapes.
module {
  func.func @main_graph(%a: tensor<?x8xi8>, %b: tensor<8x6xi8>)
      -> tensor<?x6xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (tensor<?x8xi8>, tensor<8x6xi8>)
        -> tensor<?x6xi32>
    return %y : tensor<?x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.quant_matmul
// CHECK: hip.matmul

// -----

// A rank-3 B stays on hip.matmul: hip.quant_matmul is rank 2.
module {
  func.func @main_graph(%a: tensor<2x4x8xi8>, %b: tensor<2x8x6xi8>)
      -> tensor<2x4x6xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (
        tensor<2x4x8xi8>, tensor<2x8x6xi8>) -> tensor<2x4x6xi32>
    return %y : tensor<2x4x6xi32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.quant_matmul
// CHECK: hip.matmul

// -----

// Outside @main_graph's entry block hip.quant_matmul has no lowering, so a
// MatMulInteger in any other function stays on hip.matmul.
module {
  func.func @main_graph(%x: tensor<1x4xui8>) -> tensor<1x4xui8> {
    return %x : tensor<1x4xui8>
  }
  func.func @not_main(%a: tensor<4x8xi8>, %b: tensor<8x6xi8>)
      -> tensor<4x6xi32> {
    %y = "onnx.MatMulInteger"(%a, %b) : (tensor<4x8xi8>, tensor<8x6xi8>)
        -> tensor<4x6xi32>
    return %y : tensor<4x6xi32>
  }
}

// CHECK-LABEL: func.func @not_main
// CHECK-NOT: hip.quant_matmul
// CHECK: hip.matmul
