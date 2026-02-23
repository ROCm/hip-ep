// Test: two chained hip.hipblaslt.matmul in DPS with 3D tensors
//
//   matmul0: A[B,S,K] @ B0[K,N] -> tmp[B,S,N]  (B0 broadcast across batch)
//   matmul1: tmp[B,S,N] @ B1[N,P] -> C[B,S,P]   (B1 broadcast across batch)
//
// Compile pipeline (hip-compiler handles this automatically):
//   hip-opt test_gemm.mlir \
//       --one-shot-bufferize="bufferize-function-boundaries" \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o gemm.ll

module {
  func.func @two_matmuls(
      %A:  tensor<?x?x?xf32>,
      %B0: tensor<?x?xf32>,
      %B1: tensor<?x?xf32>,
      %C:  tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %B = tensor.dim %A, %c0 : tensor<?x?x?xf32>
    %S = tensor.dim %A, %c1 : tensor<?x?x?xf32>
    %N = tensor.dim %B0, %c1 : tensor<?x?xf32>

    %tmp_init = tensor.empty(%B, %S, %N) : tensor<?x?x?xf32>

    %tmp = hip.hipblaslt.matmul(%handle)
        ins(%A, %B0 : tensor<?x?x?xf32>, tensor<?x?xf32>)
        outs(%tmp_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    %C_out = hip.hipblaslt.matmul(%handle)
        ins(%tmp, %B1 : tensor<?x?x?xf32>, tensor<?x?xf32>)
        outs(%C : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    hip.destroy_handle(%handle) : !hip.handle
    return %C_out : tensor<?x?x?xf32>
  }
}
