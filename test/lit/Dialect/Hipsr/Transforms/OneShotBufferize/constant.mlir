// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.constant bufferizes like arith.constant: the result is an
// identity-layout memref that nobody may write, and the blob attributes stay.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// The #hipsr.mem<device> encoding becomes the memref memory space, and the
// layout is the static identity instead of the fully dynamic default. index,
// offset, and size describe the blob, not the type, so they stay unchanged.
// CHECK-LABEL: func.func @constant_result(
// CHECK-SAME: %{{.+}}: !hipsr.context)
// CHECK-SAME: -> (memref<3x1xf16, #hipsr.mem<device>>, memref<4x2xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[INLINE:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x1xf16>} : memref<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[EXTERNALIZED:.+]] = hipsr.constant {index = 3 : i64, offset = 128 : i64, size = 32 : i64, value = dense<{{.*}}> : tensor<4x2xf32>} : memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT: return %[[INLINE]], %[[EXTERNALIZED]] : memref<3x1xf16, #hipsr.mem<device>>, memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @constant_result(%ctx: !hipsr.context)
    -> (tensor<3x1xf16, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) {
  %inline = hipsr.constant {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : tensor<3x1xf16, #hipsr.mem<device>>
  %externalized = hipsr.constant {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>,
                                  index = 3 : i64, offset = 128 : i64, size = 32 : i64}
      : tensor<4x2xf32, #hipsr.mem<device>>
  return %inline, %externalized : tensor<3x1xf16, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>
}

// -----

// The two ways a DPS op can take a constant. The matmul only reads it, so it
// uses the blob buffer directly. The cast writes its destination, and
// isWritable returns false, so the analysis copies the constant first.
// CHECK-LABEL: func.func @constant_as_operand(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[A:.+]]: memref<2x3xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[IN:.+]]: memref<4x2xf32, #hipsr.mem<device>>)
// CHECK-SAME: -> (memref<2x1xf16, #hipsr.mem<device>>, memref<4x2xf32, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[W:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x1xf16>} : memref<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc() {{.*}}: memref<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[W]] : memref<2x3xf16, #hipsr.mem<device>>, memref<3x1xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<2x1xf16, #hipsr.mem<device>>)
// CHECK-NEXT: %[[BLOB:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<4x2xf32>} : memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[COPY:.+]] = memref.alloc() {{.*}}: memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT: memref.copy %[[BLOB]], %[[COPY]] : memref<4x2xf32, #hipsr.mem<device>> to memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins(%[[IN]] : memref<4x2xf32, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[COPY]] : memref<4x2xf32, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]], %[[COPY]] : memref<2x1xf16, #hipsr.mem<device>>, memref<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @constant_as_operand(%ctx: !hipsr.context,
                               %a: tensor<2x3xf16, #hipsr.mem<device>>,
                               %in: tensor<4x2xf32, #hipsr.mem<device>>)
    -> (tensor<2x1xf16, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) {
  %w = hipsr.constant {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : tensor<3x1xf16, #hipsr.mem<device>>
  %init = tensor.empty() : tensor<2x1xf16, #hipsr.mem<device>>
  %0 = hipsr.matmul(%ctx) ins(%a, %w : tensor<2x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>)
      outs(%init : tensor<2x1xf16, #hipsr.mem<device>>)
      : tensor<2x1xf16, #hipsr.mem<device>>
  %blob = hipsr.constant {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>}
      : tensor<4x2xf32, #hipsr.mem<device>>
  %1 = hipsr.cast(%ctx) ins(%in : tensor<4x2xf32, #hipsr.mem<device>>)
      outs(%blob : tensor<4x2xf32, #hipsr.mem<device>>)
      : tensor<4x2xf32, #hipsr.mem<device>>
  return %0, %1 : tensor<2x1xf16, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>
}
