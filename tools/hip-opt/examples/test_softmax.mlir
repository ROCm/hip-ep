// Test: two chained hip.miopen.softmax in DPS with 3D tensors
//
//   softmax0: tmp[B,S,D] = softmax(A[B,S,D])
//   softmax1: B[B,S,D]   = softmax(tmp)
//
// Applying softmax twice: the second softmax on already-normalized rows
// should produce the same result (softmax is idempotent on uniform rows,
// but not in general -- the test verifies correctness, not idempotency).
//
// Compile pipeline:
//   hip-opt test_softmax.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o softmax.ll

module {
  func.func @two_softmaxes(
      %A: memref<?x?x?xf32, 1>,
      %B: memref<?x?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = memref.dim %A, %c0 : memref<?x?x?xf32, 1>
    %d1 = memref.dim %A, %c1 : memref<?x?x?xf32, 1>
    %d2 = memref.dim %A, %c2 : memref<?x?x?xf32, 1>

    %tmp = hip.alloc(%handle, %d0, %d1, %d2) : memref<?x?x?xf32, 1>

    hip.miopen.softmax(%handle)
        ins(%A : memref<?x?x?xf32, 1>)
        outs(%tmp : memref<?x?x?xf32, 1>)

    hip.miopen.softmax(%handle)
        ins(%tmp : memref<?x?x?xf32, 1>)
        outs(%B : memref<?x?x?xf32, 1>)

    hip.free(%handle, %tmp) : memref<?x?x?xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
