// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @ngram_pad_id_out_of_i32(%ctx: !hip.context,
                                   %ids: tensor<1x4xi32>,
                                   %mult: tensor<3xi32>,
                                   %vocab: tensor<4xi32>,
                                   %hash: tensor<1x4x4xi32>) -> tensor<1x4x4xi32> {
  // expected-error @+1 {{pad_id 2147483648 does not fit in i32}}
  %r = hip.ngram_hash_mapping(%ctx) ins(%ids, %mult, %vocab :
      tensor<1x4xi32>, tensor<3xi32>, tensor<4xi32>)
    outs(%hash : tensor<1x4x4xi32>)
    {max_ngram_size = 3 : i64, n_head_per_ngram = 2 : i64,
     pad_id = 2147483648 : i64} : tensor<1x4x4xi32>
  return %r : tensor<1x4x4xi32>
}
