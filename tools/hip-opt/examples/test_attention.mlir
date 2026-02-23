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
// Compile pipeline (hip-compiler handles this automatically):
//   hip-opt test_attention.mlir \
//       --one-shot-bufferize="bufferize-function-boundaries" \
//       --convert-hip-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm \
//       --convert-func-to-llvm --reconcile-unrealized-casts \
//     | mlir-translate --mlir-to-llvmir -o attention.ll

module {
  func.func @attention(
      %X:     tensor<?x?x?xf32>,
      %Wq:    tensor<?x?xf32>,
      %Wk:    tensor<?x?xf32>,
      %Wv:    tensor<?x?xf32>,
      %scale: tensor<?x?x?xf32>,
      %out:   tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %B = tensor.dim %X, %c0 : tensor<?x?x?xf32>
    %S = tensor.dim %X, %c1 : tensor<?x?x?xf32>
    %D = tensor.dim %X, %c2 : tensor<?x?x?xf32>

    // Q = X @ Wq  [B,S,D]  (Wq is 2D, broadcast across batch)
    %Q_init = tensor.empty(%B, %S, %D) : tensor<?x?x?xf32>
    %Q = hip.hipblaslt.matmul(%handle)
        ins(%X, %Wq : tensor<?x?x?xf32>, tensor<?x?xf32>)
        outs(%Q_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // K = X @ Wk  [B,S,D]
    %K_init = tensor.empty(%B, %S, %D) : tensor<?x?x?xf32>
    %K = hip.hipblaslt.matmul(%handle)
        ins(%X, %Wk : tensor<?x?x?xf32>, tensor<?x?xf32>)
        outs(%K_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // V = X @ Wv  [B,S,D]
    %V_init = tensor.empty(%B, %S, %D) : tensor<?x?x?xf32>
    %V = hip.hipblaslt.matmul(%handle)
        ins(%X, %Wv : tensor<?x?x?xf32>, tensor<?x?xf32>)
        outs(%V_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // KT = transpose(K, 1, 2)  [B,D,S]
    %KT_init = tensor.empty(%B, %D, %S) : tensor<?x?x?xf32>
    %KT = hip.transpose(%handle, %c1, %c2)
        ins(%K : tensor<?x?x?xf32>)
        outs(%KT_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // scores = Q @ KT  [B,S,S]  (batched matmul, both 3D)
    %scores_init = tensor.empty(%B, %S, %S) : tensor<?x?x?xf32>
    %scores = hip.hipblaslt.matmul(%handle)
        ins(%Q, %KT : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%scores_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // scaled = scores * scale  [B,S,S]  (scale pre-filled with 1/sqrt(D))
    %scaled_init = tensor.empty(%B, %S, %S) : tensor<?x?x?xf32>
    %scaled = hip.miopen.mul(%handle)
        ins(%scores, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%scaled_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // attn = softmax(scaled)  [B,S,S]  (softmax over last dim)
    %attn_init = tensor.empty(%B, %S, %S) : tensor<?x?x?xf32>
    %attn = hip.miopen.softmax(%handle)
        ins(%scaled : tensor<?x?x?xf32>)
        outs(%attn_init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    // out = attn @ V  [B,S,D]  (batched matmul, both 3D)
    %result = hip.hipblaslt.matmul(%handle)
        ins(%attn, %V : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%out : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    hip.destroy_handle(%handle) : !hip.handle
    return %result : tensor<?x?x?xf32>
  }
}
