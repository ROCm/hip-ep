// RUN: onnx-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_sqrt(%arg0: memref<128xf32, 1>, %ctx: !hip.context) {
    // CHECK-LABEL: llvm.func @test_sqrt
    // CHECK-SAME: %[[ARG0:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    %output = memref.alloc() : memref<128xf32, 1>

    hip.sqrt(%ctx) ins(%arg0 : memref<128xf32, 1>)
                   outs(%output : memref<128xf32, 1>)

    // CHECK: %[[INPUT_PTR:.*]] = llvm.extractvalue %[[ARG0]][1] : !llvm.struct
    // CHECK: %[[OUTPUT_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct

    // CHECK: %[[NUM_ELEMENTS:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %[[DATA_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_sqrt(%[[CTX]], %[[INPUT_PTR]], %[[OUTPUT_PTR]], %[[NUM_ELEMENTS]], %[[DATA_TYPE]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

    return
  }

  func.func @test_sqrt_f16(%arg0: memref<1x128x512xf16, 1>, %ctx: !hip.context) {
    // CHECK-LABEL: llvm.func @test_sqrt_f16
    // CHECK-SAME: %[[ARG0:.*]]: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>

    %output = memref.alloc() : memref<1x128x512xf16, 1>

    hip.sqrt(%ctx) ins(%arg0 : memref<1x128x512xf16, 1>)
                   outs(%output : memref<1x128x512xf16, 1>)

    // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[DIM0:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[DIM1:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %[[DIM2:.*]] = llvm.mlir.constant(512 : i64) : i64

    // CHECK: %[[MUL1:.*]] = llvm.mul %[[C1]], %[[DIM0]] : i64
    // CHECK: %[[MUL2:.*]] = llvm.mul %[[MUL1]], %[[DIM1]] : i64
    // CHECK: %[[NUM_ELEMENTS:.*]] = llvm.mul %[[MUL2]], %[[DIM2]] : i64

    // CHECK: %[[DATA_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_sqrt(%{{.*}}, %{{.*}}, %{{.*}}, %[[NUM_ELEMENTS]], %[[DATA_TYPE]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

    return
  }

  func.func @test_sqrt_dynamic(%arg0: memref<?x?xbf16, 1>, %ctx: !hip.context) {
    // CHECK-LABEL: llvm.func @test_sqrt_dynamic
    // CHECK-SAME: %[[ARG0:.*]]: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim0 = memref.dim %arg0, %c0 : memref<?x?xbf16, 1>
    %dim1 = memref.dim %arg0, %c1 : memref<?x?xbf16, 1>
    %output = memref.alloc(%dim0, %dim1) : memref<?x?xbf16, 1>

    hip.sqrt(%ctx) ins(%arg0 : memref<?x?xbf16, 1>)
                   outs(%output : memref<?x?xbf16, 1>)

    // CHECK: %[[DIM0:.*]] = llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct
    // CHECK: %[[DIM1:.*]] = llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct
    // CHECK: %[[NUM_ELEMENTS:.*]] = llvm.mul %{{.*}}, %[[DIM1]] : i64

    // CHECK: %[[DATA_TYPE:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_sqrt(%{{.*}}, %{{.*}}, %{{.*}}, %[[NUM_ELEMENTS]], %[[DATA_TYPE]])
    // CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32

    return
  }

  // CHECK: llvm.func @wrap_sqrt(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
}
