//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.pool_domain op: a region that groups the ops sharing one memory
// pool. Its terminator, hipsr.pool_domain_yield, lives in
// HipsrPoolDomainYieldOp.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_PoolDomainOp
//===----------------------------------------------------------------------===//

def Hipsr_PoolDomainOp : Hipsr_Op<"pool_domain", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"PoolDomainYieldOp">
  ]> {
  let summary = "Group operations that share one memory pool";
  let description = [{
    Groups operations that share one memory pool.

    `IsolatedFromAbove` and `RegionBranchOpInterface` make every value crossing
    the boundary explicit and verifiable. This gives transformations complete
    dependency information when treating the group as one unit.
    `RecursiveMemoryEffects` keeps nested effects visible to analyses.
    `AnyType` lets the group stay in place during bufferization.

    `domain_id` selects the runtime pool this domain draws from;
    `hipsr-pool-alloc` conveys it unchanged to the `hipsr.get_pool` it emits.
    Domains with different ids grow independently, and a repeated id means the
    author wants those domains to share one pool.

    Example:
    ```mlir
    %out = hipsr.pool_domain(%ctx, %input : !hipsr.context, tensor<?xf32>) {
    ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<?xf32>):
      %c0 = arith.constant 0 : index
      %size = tensor.dim %domain_input, %c0 : tensor<?xf32>
      %buffer = tensor.empty(%size) : tensor<?xf32>
      hipsr.pool_domain_yield %buffer : tensor<?xf32>
    } -> tensor<?xf32> {domain_id = 0 : i64}
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands, I64Attr:$domain_id);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  // Operands in parens, results after `->`; both optional so an empty domain
  // prints cleanly.
  let assemblyFormat = [{
    `(` ($operands^ `:` type($operands))? `)` $body
    (`->` type($results)^)? attr-dict
  }];
}
