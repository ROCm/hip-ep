// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// `--hip-resolve-if-result-dims` diagnoses a branch symbol that names a
// declaration: there is no body to size the output from.

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
