//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.compute op: a non-DPS op whose region groups a multi-operation
// computation. Its terminator, hipsr.compute_yield, lives in
// HipsrComputeYieldOp.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_ComputeOp
//===----------------------------------------------------------------------===//

def Hipsr_ComputeOp : Hipsr_Op<"compute", [
    AttrSizedOperandSegments,
    AutomaticAllocationScope,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    RecursivelySpeculatable,
    SingleBlockImplicitTerminator<"ComputeYieldOp">
  ]> {
  let summary = "Group a multi-operation computation as one non-DPS op";
  let description = [{
    Groups the ops that implement one high-level computation, so an ONNX op
    that lowers to several tensor or shape ops still appears as a single hipsr
    op. Without the grouping those intermediates sit in the enclosing block,
    where a later `hipsr.placeholder` may not take them as inputs. `ctx`, the
    inputs, and the outputs enter the isolated body as entry-block arguments,
    in that order.

    This is not a DPS op: an `outs` entry supplies the destination buffer for
    the result in the same position, but its type may differ from that
    result's. Result types
    come from `hipsr.compute_yield`; the `outs` values come from
    `hipsr.placeholder`, which owns the output-shape computation, so this op
    has no shape region of its own. Operands and results accept ranked tensors
    or memrefs (with or without a `#hipsr.mem<...>` space). A bufferized
    compute has no results and yields nothing.

    Each result is held in the buffer its `outs` entry names: the yielded value
    has to be that entry seen through producers that keep a whole buffer --
    reshapes, casts, destination-passing writes.

    Example (the body flattens its destination, so the result type differs
    from `outs`):
    ```mlir
    %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                               outs(%init : tensor<2x3xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>,
         %dest: tensor<2x3xf16>):
      %written = tensor.insert_slice %in into %dest[0, 0] [2, 3] [1, 1]
          : tensor<2x3xf16> into tensor<2x3xf16>
      %flat = tensor.collapse_shape %written [[0, 1]]
          : tensor<2x3xf16> into tensor<6xf16>
      hipsr.compute_yield %flat : tensor<6xf16>
    } : tensor<6xf16>
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$inputs,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$outputs
  );
  // Named after linalg's $result_tensors, which is also the pre-bufferization
  // form; a bufferized compute holds memrefs here.
  let results = (outs Variadic<Hipsr_TensorOrDeviceMemRef>:$result_tensors);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` ($inputs^ `:` type($inputs))? `)`
    `outs` `(` ($outputs^ `:` type($outputs))? `)`
    $body (`:` type($result_tensors)^)? attr-dict
  }];

  let hasVerifier = 1;
}
