//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
#ifndef HIPSR_POOL_DOMAIN_YIELD_OP
#define HIPSR_POOL_DOMAIN_YIELD_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Hipsr_PoolDomainYieldOp : Hipsr_Op<"pool_domain_yield",
    [Pure, ReturnLike, Terminator, HasParent<"PoolDomainOp">]> {
  let summary = "Yield values out of a hipsr.pool_domain region";
  let description = [{
    Terminates a `hipsr.pool_domain`. Its operands correspond positionally to
    the parent op's results.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [
    // SingleBlockImplicitTerminator requires a no-argument builder.
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // HIPSR_POOL_DOMAIN_YIELD_OP
