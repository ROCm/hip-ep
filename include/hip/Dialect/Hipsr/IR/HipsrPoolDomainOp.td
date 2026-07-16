//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.pool_domain op: a region that groups the ops sharing one memory
// pool. Its terminator, hipsr.pool_domain_yield, lives in
// HipsrPoolDomainYieldOp.td.
//

#ifndef HIPSR_POOL_DOMAIN_OP
#define HIPSR_POOL_DOMAIN_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"
include "hip/Dialect/Hipsr/IR/HipsrTraits.td"

//===----------------------------------------------------------------------===//
// Hipsr_PoolDomainOp
//===----------------------------------------------------------------------===//

def Hipsr_PoolDomainOp : Hipsr_Op<"pool_domain", [
    IsolatedFromAboveButAllowOperands,
    SingleBlockImplicitTerminator<"PoolDomainYieldOp">
  ]> {
  let summary = "Group operations that share one memory pool";
  let description = [{
    Groups a sequence of operations so they can share one memory pool.

    The body is a single block ending in `hipsr.pool_domain_yield`, whose
    values become this op's results. `IsolatedFromAboveButAllowOperands` lets
    the body read this op's operands but nothing else outside, so the block
    takes no arguments. Operands and results are `AnyType`, so the op is
    unchanged by bufferization (tensors just become memrefs).

    Example:
    ```mlir
    %out:2 = hipsr.pool_domain(%ctx, %in : !hipsr.context, tensor<3x4xf32>) {
      %c1 = arith.constant 1 : index
      %n = tensor.dim %in, %c1 : tensor<3x4xf32>
      %idx = tensor.empty(%n) : tensor<2x?xi64>
      %cnt = tensor.empty() : tensor<i32>
      hipsr.pool_domain_yield %idx, %cnt : tensor<2x?xi64>, tensor<i32>
    } -> tensor<2x?xi64>, tensor<i32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  // Operands in parens, results after `->`; both optional so an empty domain
  // prints cleanly.
  let assemblyFormat = [{
    `(` ($operands^ `:` type($operands))? `)` $body
    (`->` type($results)^)? attr-dict
  }];

  let hasVerifier = 1;
}

#endif // HIPSR_POOL_DOMAIN_OP
