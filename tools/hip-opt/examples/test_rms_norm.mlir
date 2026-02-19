// Test: two chained hip.miopen.rms_norm in DPS with 3D tensors
//
//   norm0: tmp[B,S,D] = RMSNorm(A[B,S,D], W0[D])
//   norm1: B[B,S,D]   = RMSNorm(tmp, W1[D])
//
// Compile pipeline:
//   hip-opt test_rms_norm.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o rms_norm.ll

module {
  func.func @two_rms_norms(
      %A:  memref<?x?x?xf32, 1>,
      %W0: memref<?xf32, 1>,
      %W1: memref<?xf32, 1>,
      %B:  memref<?x?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = memref.dim %A, %c0 : memref<?x?x?xf32, 1>
    %d1 = memref.dim %A, %c1 : memref<?x?x?xf32, 1>
    %d2 = memref.dim %A, %c2 : memref<?x?x?xf32, 1>

    %tmp = hip.alloc(%handle, %d0, %d1, %d2) : memref<?x?x?xf32, 1>

    hip.miopen.rms_norm(%handle)
        ins(%A, %W0 : memref<?x?x?xf32, 1>, memref<?xf32, 1>)
        outs(%tmp : memref<?x?x?xf32, 1>)

    hip.miopen.rms_norm(%handle)
        ins(%tmp, %W1 : memref<?x?x?xf32, 1>, memref<?xf32, 1>)
        outs(%B : memref<?x?x?xf32, 1>)

    hip.free(%handle, %tmp) : memref<?x?x?xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
