//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.preserve_shape op: an SSA edge from a shape computation to the data
// value it describes, so the pair survives bufferization.
//

include "mlir/Dialect/Shape/IR/ShapeBase.td"

//===----------------------------------------------------------------------===//
// Hipsr_PreserveShapeOp
//===----------------------------------------------------------------------===//

def Hipsr_PreserveShapeOp : Hipsr_Op<"preserve_shape",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tie a shape computation to the data value it describes";
  let description = [{
    Records the association between a shape value and the data value it
    describes. This is a metadata-only operation: it does not read or write
    memory, produce results, or generate code.

    ```mlir
    %shape = scf.execute_region -> !shape.shape { ... }
    %d0 = ...
    %init = tensor.empty(%d0) : tensor<?x2048xf16>
    hipsr.preserve_shape %shape, %init : tensor<?x2048xf16>
    ```

    ```mlir
    %alloc = memref.alloc(%d0) : memref<?x2048xf16>
    hipsr.preserve_shape %shape, %alloc : memref<?x2048xf16>
    ```

    `$shape` has the same rank as `$data`.
  }];

  let arguments = (ins Shape_ShapeType:$shape, Hipsr_TensorOrMemRef:$data);
  let results = (outs);

  let assemblyFormat = "$shape `,` $data attr-dict `:` type($data)";

  let hasVerifier = 1;
}
