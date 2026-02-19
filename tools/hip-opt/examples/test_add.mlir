// Test: two chained hip.miopen.add in Destination-Passing Style (DPS)
//
//   add0: A + B -> tmp      (intermediate, allocated internally)
//   add1: tmp + C -> D      (final output, provided by caller)
//
// Compile pipeline:
//   hip-opt test_add.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o add.ll

module {
  func.func @two_adds(
      %A: memref<?x?xf32, 1>,
      %B: memref<?x?xf32, 1>,
      %C: memref<?x?xf32, 1>,
      %D: memref<?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    // Intermediate buffer has same shape as A
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = memref.dim %A, %c0 : memref<?x?xf32, 1>
    %d1 = memref.dim %A, %c1 : memref<?x?xf32, 1>

    %tmp = hip.alloc(%handle, %d0, %d1) : memref<?x?xf32, 1>

    hip.miopen.add(%handle)
        ins(%A, %B : memref<?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%tmp : memref<?x?xf32, 1>)

    hip.miopen.add(%handle)
        ins(%tmp, %C : memref<?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%D : memref<?x?xf32, 1>)

    hip.free(%handle, %tmp) : memref<?x?xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
