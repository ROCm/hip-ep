//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.compute op: a DPS op whose region groups a multi-operation
// computation. Its terminator, hipsr.compute_yield, lives in
// HipsrComputeYieldOp.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_ComputeOp
//===----------------------------------------------------------------------===//

def Hipsr_ComputeOp : Hipsr_Op<"compute", [
    AttrSizedOperandSegments,
    AutomaticAllocationScope,
    DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                              ["getDpsInitsMutable"]>,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ComputeYieldOp">
  ]> {
  let summary = "Group a multi-operation computation as one DPS op";
  let description = [{
    Groups the operations that implement one high-level computation, so an ONNX
    op that needs several tensor or shape operations still appears as a single
    hipsr op. Without this grouping the intermediate values would sit in the
    enclosing block, where the `hipsr.placeholder` of a later op would have to
    take one as an input; a placeholder accepts only block arguments and
    constant or placeholder results.

    Destination-passing style: the `outs` inits are the destinations, and each
    tensor init is tied to the matching result. The inits come from
    `hipsr.placeholder`, which owns the output-shape computation; this op has no
    shape region of its own.

    `IsolatedFromAbove` keeps every value crossing the boundary explicit, so
    stock MLIR passes stay correct over the body. `ctx`, the inputs, and the
    inits therefore enter the body as entry-block arguments, in that order.
    `RecursiveMemoryEffects` keeps nested effects visible to analyses, and the
    body is an allocation scope so allocations made inside it end there.

    Example:
    ```mlir
    %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                               outs(%init : tensor<6xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>, %dest: tensor<6xf16>):
      %flat = tensor.collapse_shape %in [[0, 1]]
          : tensor<2x3xf16> into tensor<6xf16>
      %filled = tensor.insert_slice %flat into %dest[0] [6] [1]
          : tensor<6xf16> into tensor<6xf16>
      hipsr.compute_yield %filled : tensor<6xf16>
    } : tensor<6xf16>
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Variadic<AnyType>:$inputs,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$inits
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region SizedRegion<1>:$body);

  // Both operand lists are optional groups so an empty one prints as `ins()`.
  // The result types trail the body after `:`, as in the other hipsr DPS ops.
  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` ($inputs^ `:` type($inputs))? `)`
    `outs` `(` ($inits^ `:` type($inits))? `)`
    $body (`:` type($result_tensors)^)? attr-dict
  }];
}
