//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.pool_domain_yield op: the terminator of a hipsr.pool_domain
// region. The hipsr.pool_domain op itself lives in HipsrPoolDomainOp.td.
//

#ifndef HIPSR_POOL_DOMAIN_YIELD_OP
#define HIPSR_POOL_DOMAIN_YIELD_OP

include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Hipsr/IR/HipsrOps.td"

//===----------------------------------------------------------------------===//
// Hipsr_PoolDomainYieldOp
//===----------------------------------------------------------------------===//

def Hipsr_PoolDomainYieldOp : Hipsr_Op<"pool_domain_yield",
    [Pure, Terminator, HasParent<"PoolDomainOp">]> {
  let summary = "Yield values out of a hipsr.pool_domain region";
  let description = [{
    Ends the body of a `hipsr.pool_domain`. Its operands are passed out as the
    parent op's results, matched by the parent's verifier. Operands are
    `AnyType` to match the parent's results.

    Example:
    ```mlir
    hipsr.pool_domain_yield %out, %count : tensor<2x?xi64>, tensor<i32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  // Operands are optional, so the empty implicit terminator prints as just the
  // mnemonic.
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [
    // No-arg build for the implicit terminator the parent auto-inserts.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // HIPSR_POOL_DOMAIN_YIELD_OP
