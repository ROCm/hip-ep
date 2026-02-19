// Test: two chained hip.hipblaslt.matmul in Destination-Passing Style (DPS)
//
//   matmul0: A[M,K] x B0[K,N] -> tmp[M,N]   (intermediate, allocated internally)
//   matmul1: tmp[M,N] x B1[N,P] -> C[M,P]    (final output, provided by caller)
//
// %tmp is the internally managed intermediate buffer (hip.alloc / hip.free).
// %C is the caller-provided output destination -- no alloc/free needed.
//
// Compile pipeline:
//   hip-opt test_gemm.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o gemm.ll

module {
  func.func @two_matmuls(
      %A:  memref<?x?xf32, 1>,
      %B0: memref<?x?xf32, 1>,
      %B1: memref<?x?xf32, 1>,
      %C:  memref<?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    // Extract dimensions for the intermediate buffer: tmp is [M, N]
    //   M = dim 0 of A,  N = dim 1 of B0
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = memref.dim %A, %c0 : memref<?x?xf32, 1>
    %N = memref.dim %B0, %c1 : memref<?x?xf32, 1>

    %tmp = hip.alloc(%handle, %M, %N) : memref<?x?xf32, 1>

    hip.hipblaslt.matmul(%handle)
        ins(%A, %B0 : memref<?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%tmp : memref<?x?xf32, 1>)

    hip.hipblaslt.matmul(%handle)
        ins(%tmp, %B1 : memref<?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%C : memref<?x?xf32, 1>)

    hip.free(%handle, %tmp) : memref<?x?xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
