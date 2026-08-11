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

    This is not a DPS op: an `outs` entry only supplies the destination buffer
    for the result in the same position, and its type may differ from that
    result's, which is what lets a body reshape its data. Result types come
    from `hipsr.compute_yield`; the `outs` values come from
    `hipsr.placeholder`, which owns the output-shape computation, so this op
    has no shape region of its own. A bufferized compute writes into device
    memrefs, has no results, and yields nothing.

    Example (the body flattens, so the result type differs from `outs`):
    ```mlir
    %out = hipsr.compute(%ctx) ins(%data : tensor<2x3xf16>)
                               outs(%init : tensor<2x3xf16>) {
    ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x3xf16>,
         %dest: tensor<2x3xf16>):
      %flat = tensor.collapse_shape %in [[0, 1]]
          : tensor<2x3xf16> into tensor<6xf16>
      hipsr.compute_yield %flat : tensor<6xf16>
    } : tensor<6xf16>
    ```

    A bufferized compute takes and produces device memrefs. Because `outs` is
    not tied to a result, a result's buffer cannot be the `outs` buffer
    whenever the two types differ; it is the buffer of the value yielded in
    the same position, so the results survive bufferization as memrefs and
    `hipsr.compute_yield` carries them out of the body. That mirrors how
    `scf.execute_region` bufferizes. A compute that had no results still ends
    up with none, writing only through its `outs` buffers.

    ```mlir
    %out = hipsr.compute(%ctx) ins(%data : memref<2x3xf16, #hipsr.mem<device>>)
                               outs(%init : memref<2x3xf16, #hipsr.mem<device>>) {
    ^bb0(%body_ctx: !hipsr.context, %in: memref<2x3xf16, #hipsr.mem<device>>,
         %dest: memref<2x3xf16, #hipsr.mem<device>>):
      %flat = memref.collapse_shape %in [[0, 1]]
          : memref<2x3xf16, #hipsr.mem<device>>
          into memref<6xf16, #hipsr.mem<device>>
      hipsr.compute_yield %flat : memref<6xf16, #hipsr.mem<device>>
    } : memref<6xf16, #hipsr.mem<device>>
    ```
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$inputs,
    Variadic<Hipsr_TensorOrDeviceMemRef>:$outputs
  );
  // Named after linalg's $result_tensors, which is also the pre-bufferization
  // form; a bufferized compute holds device memrefs here.
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
