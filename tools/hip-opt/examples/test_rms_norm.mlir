// Test: two chained hip.miopen.rms_norm in Destination-Passing Style (DPS)
//
//   norm0: tmp = RMSNorm(A, W0)     (intermediate, allocated internally)
//   norm1: B   = RMSNorm(tmp, W1)   (final output, provided by caller)
//
// All data tensors are [N,D], weight tensors are [D].
//
// Compile pipeline:
//   hip-opt test_rms_norm.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o rms_norm.ll

module {
  func.func @two_rms_norms(
      %A:  memref<?x?xf32, 1>,
      %W0: memref<?xf32, 1>,
      %W1: memref<?xf32, 1>,
      %B:  memref<?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    // Intermediate buffer has same shape as A: [N, D]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %N = memref.dim %A, %c0 : memref<?x?xf32, 1>
    %D = memref.dim %A, %c1 : memref<?x?xf32, 1>

    %tmp = hip.alloc(%handle, %N, %D) : memref<?x?xf32, 1>

    hip.miopen.rms_norm(%handle)
        ins(%A, %W0 : memref<?x?xf32, 1>, memref<?xf32, 1>)
        outs(%tmp : memref<?x?xf32, 1>)

    hip.miopen.rms_norm(%handle)
        ins(%tmp, %W1 : memref<?x?xf32, 1>, memref<?xf32, 1>)
        outs(%B : memref<?x?xf32, 1>)

    hip.free(%handle, %tmp) : memref<?x?xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
