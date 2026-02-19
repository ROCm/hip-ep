// Test: two chained hip.miopen.add in DPS with 3D tensors
//
//   add0: A[B,S,D] + B[B,S,D] -> tmp
//   add1: tmp + C[B,S,D] -> D[B,S,D]
//
// Compile pipeline:
//   hip-opt test_add.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o add.ll

module {
  func.func @two_adds(
      %A: memref<?x?x?xf32, 1>,
      %B: memref<?x?x?xf32, 1>,
      %C: memref<?x?x?xf32, 1>,
      %D: memref<?x?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = memref.dim %A, %c0 : memref<?x?x?xf32, 1>
    %d1 = memref.dim %A, %c1 : memref<?x?x?xf32, 1>
    %d2 = memref.dim %A, %c2 : memref<?x?x?xf32, 1>

    %tmp = hip.alloc(%handle, %d0, %d1, %d2) : memref<?x?x?xf32, 1>

    hip.miopen.add(%handle)
        ins(%A, %B : memref<?x?x?xf32, 1>, memref<?x?x?xf32, 1>)
        outs(%tmp : memref<?x?x?xf32, 1>)

    hip.miopen.add(%handle)
        ins(%tmp, %C : memref<?x?x?xf32, 1>, memref<?x?x?xf32, 1>)
        outs(%D : memref<?x?x?xf32, 1>)

    hip.free(%handle, %tmp) : memref<?x?x?xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
