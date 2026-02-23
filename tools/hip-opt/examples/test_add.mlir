// Test: two chained hip.miopen.add in DPS with 3D tensors
//
//   add0: A[B,S,D] + B[B,S,D] -> tmp
//   add1: tmp + C[B,S,D] -> D[B,S,D]
//
// Compile pipeline (hip-compiler handles this automatically):
//   hip-opt test_add.mlir \
//       --one-shot-bufferize="bufferize-function-boundaries" \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o add.ll

module {
  func.func @two_adds(
      %A: tensor<?x?x?xf32>,
      %B: tensor<?x?x?xf32>,
      %C: tensor<?x?x?xf32>,
      %D: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %A, %c0 : tensor<?x?x?xf32>
    %d1 = tensor.dim %A, %c1 : tensor<?x?x?xf32>
    %d2 = tensor.dim %A, %c2 : tensor<?x?x?xf32>

    %tmp_init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>

    %tmp = hip.miopen.add(%handle)
        ins(%A, %B : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%tmp_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    %D_out = hip.miopen.add(%handle)
        ins(%tmp, %C : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%D : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    hip.destroy_handle(%handle) : !hip.handle
    return %D_out : tensor<?x?x?xf32>
  }
}
