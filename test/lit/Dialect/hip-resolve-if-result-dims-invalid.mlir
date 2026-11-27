// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// `--hip-resolve-if-result-dims` diagnoses a branch symbol that names a
// declaration: there is no body to size the output from. It also diagnoses a
// shape built from a captured graph input, and does so without re-traversing
// a failing operand once per use.

// RUN: hip-mlir-opt --hip-resolve-if-result-dims --split-input-file --verify-diagnostics %s

module {
  func.func @declared_branch(%ctx: !hip.context, %c: i1, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    // expected-error @+1 {{cannot size If output 0: branch decl_then is a declaration without a body}}
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @decl_then else @decl_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    return %init : tensor<?x8x64xf32>
  }

  func.func private @decl_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32>

  func.func private @decl_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// -----

// Each add uses the previous value twice, so without caching the failure the
// traversal is exponential in the chain length.
module {
  func.func @shared_failing_chain(%ctx: !hip.context, %c: i1,
                                  %x: tensor<?x?xf32>, %s: tensor<3xi64>)
      -> tensor<?x8x64xf32> {
    // expected-error @+1 {{dim 0 of If output 0 is not host shape arithmetic on the If inputs in chain_then}}
    %d0 = hip.if_result_dim(%c) captures(%x, %s : tensor<?x?xf32>, tensor<3xi64>)
        then @chain_then else @chain_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    return %init : tensor<?x8x64xf32>
  }

  func.func private @chain_then(%ctx: !hip.context, %x: tensor<?x?xf32>,
                                %s: tensor<3xi64>) -> tensor<?x8x64xf32> {
    %init = tensor.empty() : tensor<3xi64>
    %v1 = hip.add(%ctx) ins(%s, %s : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v2 = hip.add(%ctx) ins(%v1, %v1 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v3 = hip.add(%ctx) ins(%v2, %v2 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v4 = hip.add(%ctx) ins(%v3, %v3 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v5 = hip.add(%ctx) ins(%v4, %v4 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v6 = hip.add(%ctx) ins(%v5, %v5 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v7 = hip.add(%ctx) ins(%v6, %v6 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v8 = hip.add(%ctx) ins(%v7, %v7 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v9 = hip.add(%ctx) ins(%v8, %v8 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v10 = hip.add(%ctx) ins(%v9, %v9 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v11 = hip.add(%ctx) ins(%v10, %v10 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v12 = hip.add(%ctx) ins(%v11, %v11 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v13 = hip.add(%ctx) ins(%v12, %v12 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v14 = hip.add(%ctx) ins(%v13, %v13 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v15 = hip.add(%ctx) ins(%v14, %v14 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v16 = hip.add(%ctx) ins(%v15, %v15 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v17 = hip.add(%ctx) ins(%v16, %v16 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v18 = hip.add(%ctx) ins(%v17, %v17 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v19 = hip.add(%ctx) ins(%v18, %v18 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v20 = hip.add(%ctx) ins(%v19, %v19 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v21 = hip.add(%ctx) ins(%v20, %v20 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v22 = hip.add(%ctx) ins(%v21, %v21 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v23 = hip.add(%ctx) ins(%v22, %v22 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v24 = hip.add(%ctx) ins(%v23, %v23 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v25 = hip.add(%ctx) ins(%v24, %v24 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v26 = hip.add(%ctx) ins(%v25, %v25 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v27 = hip.add(%ctx) ins(%v26, %v26 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v28 = hip.add(%ctx) ins(%v27, %v27 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v29 = hip.add(%ctx) ins(%v28, %v28 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v30 = hip.add(%ctx) ins(%v29, %v29 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v31 = hip.add(%ctx) ins(%v30, %v30 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v32 = hip.add(%ctx) ins(%v31, %v31 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v33 = hip.add(%ctx) ins(%v32, %v32 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v34 = hip.add(%ctx) ins(%v33, %v33 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v35 = hip.add(%ctx) ins(%v34, %v34 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v36 = hip.add(%ctx) ins(%v35, %v35 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v37 = hip.add(%ctx) ins(%v36, %v36 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v38 = hip.add(%ctx) ins(%v37, %v37 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v39 = hip.add(%ctx) ins(%v38, %v38 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %v40 = hip.add(%ctx) ins(%v39, %v39 : tensor<3xi64>, tensor<3xi64>)
        outs(%init : tensor<3xi64>) -> tensor<3xi64>
    %y = tensor.reshape %x(%v40)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @chain_else(%ctx: !hip.context, %x: tensor<?x?xf32>,
                                %s: tensor<3xi64>) -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}
