// Test: two chained hip.hipblaslt.matmul in DPS with 3D tensors
//
//   matmul0: A[B,S,K] @ B0[K,N] -> tmp[B,S,N]  (B0 broadcast across batch)
//   matmul1: tmp[B,S,N] @ B1[N,P] -> C[B,S,P]   (B1 broadcast across batch)
//
// Compile pipeline:
//   hip-opt test_gemm.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o gemm.ll

module {
  func.func @two_matmuls(
      %A:  memref<?x?x?xf32, 1>,
      %B0: memref<?x?xf32, 1>,
      %B1: memref<?x?xf32, 1>,
      %C:  memref<?x?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %B = memref.dim %A, %c0 : memref<?x?x?xf32, 1>
    %S = memref.dim %A, %c1 : memref<?x?x?xf32, 1>
    %N = memref.dim %B0, %c1 : memref<?x?xf32, 1>

    %tmp = hip.alloc(%handle, %B, %S, %N) : memref<?x?x?xf32, 1>

    hip.hipblaslt.matmul(%handle)
        ins(%A, %B0 : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%tmp : memref<?x?x?xf32, 1>)

    hip.hipblaslt.matmul(%handle)
        ins(%tmp, %B1 : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%C : memref<?x?x?xf32, 1>)

    hip.free(%handle, %tmp) : memref<?x?x?xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
