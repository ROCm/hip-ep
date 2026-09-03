//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.preserve_shape op: an SSA edge from a shape computation to the data
// value it describes, so the pair survives bufferization.
//

//===----------------------------------------------------------------------===//
// Hipsr_PreserveShapeOp
//===----------------------------------------------------------------------===//

def Hipsr_PreserveShapeOp : Hipsr_Op<"preserve_shape",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tie a shape computation to the data value it describes";
  let description = [{
    Ties a shape value to the data value it describes, so later passes keep the
    pair together. Metadata only: it reads and writes no memory, returns no
    result, and generates no code.

    `$shape` holds one extent per dimension of `$data`, in one of three forms:

    ```mlir
    // As -hipsr-materialize-init-tensors emits it.
    %shape = scf.execute_region -> !shape.shape { ... }
    %init = tensor.empty(%d0) : tensor<?x2048xf16>
    hipsr.preserve_shape %shape, %init : !shape.shape, tensor<?x2048xf16>

    // After -hipsr-convert-shape-to-extent.
    %shape = tensor.from_elements %d0, %c2048 : tensor<2xindex>
    hipsr.preserve_shape %shape, %init : tensor<2xindex>, tensor<?x2048xf16>

    // After bufferization.
    hipsr.preserve_shape %shape, %alloc : memref<2xindex>, memref<?x2048xf16>
    ```
  }];

  let arguments = (ins Hipsr_ShapeValue:$shape, Hipsr_TensorOrMemRef:$data);
  let results = (outs);

  // `$shape` has more than one possible type, so print it.
  let assemblyFormat = "$shape `,` $data attr-dict `:` type($shape) `,` type($data)";

  let hasVerifier = 1;
}
