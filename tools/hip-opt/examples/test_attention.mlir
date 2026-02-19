// Test: single-head attention composed from individual ops (DPS, 3D)
//
//   Q[B,S,D] = X[B,S,D] @ Wq[D,D]        (matmul, Wq broadcast)
//   K[B,S,D] = X[B,S,D] @ Wk[D,D]        (matmul, Wk broadcast)
//   V[B,S,D] = X[B,S,D] @ Wv[D,D]        (matmul, Wv broadcast)
//   KT[B,D,S] = transpose(K, 1, 2)        (swap dims 1 and 2)
//   scores[B,S,S] = Q[B,S,D] @ KT[B,D,S] (batched matmul)
//   scaled[B,S,S] = scores * scale         (element-wise mul)
//   attn[B,S,S] = softmax(scaled)          (row-wise softmax)
//   out[B,S,D] = attn[B,S,S] @ V[B,S,D]  (batched matmul)
//
// scale is pre-filled with 1/sqrt(D) from the C++ driver.
//
// Compile pipeline:
//   hip-opt test_attention.mlir \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o attention.ll

module {
  func.func @attention(
      %X:     memref<?x?x?xf32, 1>,
      %Wq:    memref<?x?xf32, 1>,
      %Wk:    memref<?x?xf32, 1>,
      %Wv:    memref<?x?xf32, 1>,
      %scale: memref<?x?x?xf32, 1>,
      %out:   memref<?x?x?xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %B = memref.dim %X, %c0 : memref<?x?x?xf32, 1>
    %S = memref.dim %X, %c1 : memref<?x?x?xf32, 1>
    %D = memref.dim %X, %c2 : memref<?x?x?xf32, 1>

    // Q = X @ Wq  [B,S,D]  (Wq is 2D, broadcast across batch)
    %Q = hip.alloc(%handle, %B, %S, %D) : memref<?x?x?xf32, 1>
    hip.hipblaslt.matmul(%handle)
        ins(%X, %Wq : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%Q : memref<?x?x?xf32, 1>)

    // K = X @ Wk  [B,S,D]
    %K = hip.alloc(%handle, %B, %S, %D) : memref<?x?x?xf32, 1>
    hip.hipblaslt.matmul(%handle)
        ins(%X, %Wk : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%K : memref<?x?x?xf32, 1>)

    // V = X @ Wv  [B,S,D]
    %V = hip.alloc(%handle, %B, %S, %D) : memref<?x?x?xf32, 1>
    hip.hipblaslt.matmul(%handle)
        ins(%X, %Wv : memref<?x?x?xf32, 1>, memref<?x?xf32, 1>)
        outs(%V : memref<?x?x?xf32, 1>)

    // KT = transpose(K, 1, 2)  [B,D,S]
    %KT = hip.alloc(%handle, %B, %D, %S) : memref<?x?x?xf32, 1>
    hip.transpose(%handle, %c1, %c2)
        ins(%K : memref<?x?x?xf32, 1>)
        outs(%KT : memref<?x?x?xf32, 1>)

    // scores = Q @ KT  [B,S,S]  (batched matmul, both 3D)
    %scores = hip.alloc(%handle, %B, %S, %S) : memref<?x?x?xf32, 1>
    hip.hipblaslt.matmul(%handle)
        ins(%Q, %KT : memref<?x?x?xf32, 1>, memref<?x?x?xf32, 1>)
        outs(%scores : memref<?x?x?xf32, 1>)

    // scaled = scores * scale  [B,S,S]  (scale pre-filled with 1/sqrt(D))
    %scaled = hip.alloc(%handle, %B, %S, %S) : memref<?x?x?xf32, 1>
    hip.miopen.mul(%handle)
        ins(%scores, %scale : memref<?x?x?xf32, 1>, memref<?x?x?xf32, 1>)
        outs(%scaled : memref<?x?x?xf32, 1>)

    // attn = softmax(scaled)  [B,S,S]  (softmax over last dim)
    %attn = hip.alloc(%handle, %B, %S, %S) : memref<?x?x?xf32, 1>
    hip.miopen.softmax(%handle)
        ins(%scaled : memref<?x?x?xf32, 1>)
        outs(%attn : memref<?x?x?xf32, 1>)

    // out = attn @ V  [B,S,D]  (batched matmul, both 3D)
    hip.hipblaslt.matmul(%handle)
        ins(%attn, %V : memref<?x?x?xf32, 1>, memref<?x?x?xf32, 1>)
        outs(%out : memref<?x?x?xf32, 1>)

    // Free intermediates
    hip.free(%handle, %Q) : memref<?x?x?xf32, 1>
    hip.free(%handle, %K) : memref<?x?x?xf32, 1>
    hip.free(%handle, %V) : memref<?x?x?xf32, 1>
    hip.free(%handle, %KT) : memref<?x?x?xf32, 1>
    hip.free(%handle, %scores) : memref<?x?x?xf32, 1>
    hip.free(%handle, %scaled) : memref<?x?x?xf32, 1>
    hip.free(%handle, %attn) : memref<?x?x?xf32, 1>

    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
