//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_AllocOutputOp : Hipsr_Op<"alloc_output",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Obtain a graph-output buffer from the EP output allocator";
  let description = [{
    Allocates the buffer for graph output `out_idx` via the EP-supplied output
    allocator, at the point in the graph where the output shape is known.
    Operands: runtime context + one Index per dynamic dim of the result memref
    (same operand convention as hip.alloc). The `out_idx` attribute identifies
    which graph output this buffer is.


    Example:
    ```mlir
    %out = hipsr.alloc_output(%ctx, %M, %N) {out_idx = 0 : i64} : memref<?x?xf16>
    ```
  }];

  let arguments = (ins Hipsr_ContextType:$ctx, Variadic<Index>:$dynamicSizes,
                       I64Attr:$out_idx);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = [{
    `(` $ctx (`,` $dynamicSizes^ )? `)` attr-dict `:` type($memref)
  }];
  let hasVerifier = 1;
}
