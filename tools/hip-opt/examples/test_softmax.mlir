// Test: two chained hip.miopen.softmax in DPS with 3D tensors
//
//   softmax0: tmp[B,S,D] = softmax(A[B,S,D])
//   softmax1: B[B,S,D]   = softmax(tmp)
//
// Applying softmax twice: the second softmax on already-normalized rows
// should produce the same result (softmax is idempotent on uniform rows,
// but not in general -- the test verifies correctness, not idempotency).
//
// Compile pipeline (hip-compiler handles this automatically):
//   hip-opt test_softmax.mlir \
//       --one-shot-bufferize="bufferize-function-boundaries" \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o softmax.ll

module {
  func.func @two_softmaxes(
      %A: tensor<?x?x?xf32>,
      %B: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %A, %c0 : tensor<?x?x?xf32>
    %d1 = tensor.dim %A, %c1 : tensor<?x?x?xf32>
    %d2 = tensor.dim %A, %c2 : tensor<?x?x?xf32>

    %tmp_init = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>

    %tmp = hip.miopen.softmax(%handle)
        ins(%A : tensor<?x?x?xf32>)
        outs(%tmp_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    %B_out = hip.miopen.softmax(%handle)
        ins(%tmp : tensor<?x?x?xf32>)
        outs(%B : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    hip.destroy_handle(%handle) : !hip.handle
    return %B_out : tensor<?x?x?xf32>
  }
}
