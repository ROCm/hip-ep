//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.pool_domain_yield op: the terminator of a hipsr.pool_domain
// region. The hipsr.pool_domain op itself lives in HipsrPoolDomainOp.td.
//

#ifndef HIPSR_POOL_DOMAIN_YIELD_OP
#define HIPSR_POOL_DOMAIN_YIELD_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Hipsr_PoolDomainYieldOp
//===----------------------------------------------------------------------===//

def Hipsr_PoolDomainYieldOp : Hipsr_Op<"pool_domain_yield",
    [Pure, ReturnLike, Terminator, HasParent<"PoolDomainOp">]> {
  let summary = "Yield values out of a hipsr.pool_domain region";
  let description = [{
    Ends the body of a `hipsr.pool_domain`. Its operands match the parent op's
    results one for one. `ReturnLike` marks this as the region exit so MLIR's
    verifier can check the values against the parent's
    `RegionBranchOpInterface`.

    Operands are `AnyType` to match the parent op's results.

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
    // SingleBlockImplicitTerminator requires a no-argument builder.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // HIPSR_POOL_DOMAIN_YIELD_OP
