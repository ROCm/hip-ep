// Test: two chained hip.miopen.rms_norm in DPS with 3D tensors
//
//   norm0: tmp[B,S,D] = RMSNorm(A[B,S,D], W0[D])
//   norm1: B[B,S,D]   = RMSNorm(tmp, W1[D])
//
// Compile pipeline (hip-compiler handles this automatically):
//   hip-opt test_rms_norm.mlir \
//       --one-shot-bufferize="bufferize-function-boundaries" \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o rms_norm.ll

module {
  func.func @two_rms_norms(
      %A:  tensor<?x?x?xf32>,
      %W0: tensor<?xf32>,
      %W1: tensor<?xf32>,
      %B:  tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %A, %c0 : tensor<?x?x?xf32>
    %d1 = tensor.dim %A, %c1 : tensor<?x?x?xf32>
    %d2 = tensor.dim %A, %c2 : tensor<?x?x?xf32>

    %tmp_init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>

    %tmp = hip.miopen.rms_norm(%handle)
        ins(%A, %W0 : tensor<?x?x?xf32>, tensor<?xf32>)
        outs(%tmp_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    %B_out = hip.miopen.rms_norm(%handle)
        ins(%tmp, %W1 : tensor<?x?x?xf32>, tensor<?xf32>)
        outs(%B : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    hip.destroy_handle(%handle) : !hip.handle
    return %B_out : tensor<?x?x?xf32>
  }
}
