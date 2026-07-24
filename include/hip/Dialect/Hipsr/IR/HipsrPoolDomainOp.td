//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
#ifndef HIPSR_POOL_DOMAIN_OP
#define HIPSR_POOL_DOMAIN_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Hipsr_PoolDomainOp : Hipsr_Op<"pool_domain", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface, [
      "getEntrySuccessorOperands", "getRegionInvocationBounds"
    ]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"PoolDomainYieldOp">
  ]> {
  let summary = "Group operations that share one memory pool";
  let description = [{
    The body executes once under standard MLIR isolation. Operands correspond
    positionally to entry-block arguments, and yielded values become results.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` ($operands^ `:` type($operands))? `)` $body
    (`->` type($results)^)? attr-dict
  }];
}

#endif // HIPSR_POOL_DOMAIN_OP
