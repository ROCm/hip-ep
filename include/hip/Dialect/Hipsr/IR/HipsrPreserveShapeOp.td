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
    Records that `$shape` is the shape of `$data`. Pure metadata: it reads
    nothing, writes nothing, produces nothing, and generates no code.

    It exists because `-one-shot-bufferize` erases the only data-flow path
    between a shape computation and the buffer it sized. A DPS op bufferizes in
    place, so its init tensor and its result share one `memref.alloc` and the
    init's SSA value disappears; the shape computation that sized it is then
    unreachable and DCE deletes it, leaving nothing for
    `-hip-use-output-allocator` to recover a graph output's declared extents
    from. Holding both values in one op keeps the shape reachable.

    No pass is wired to it yet: nothing creates these ops, nothing reads them,
    and nothing erases them either. That last one matters, because the effect
    this op reports keeps DCE off it while it has no lowering and no
    `BufferizableOpInterface` model -- so one written by hand and left in place
    will fail `-one-shot-bufferize` with `op was not bufferized`, and will still
    be standing at the hipsr-to-LLVM conversion. Whatever starts creating these
    ops has to come with a pass that erases them.

    The design behind the op has `-hipsr-materialize-init-tensors` creating one
    per `tensor.empty` it builds and `-hip-use-output-allocator` reading them
    back to size `hip.alloc_output`.

    Valid both before bufferization (`$data` is a ranked tensor) and after it
    (`$data` is a memref), so that the association can outlive the type change
    once a model is attached:

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

    `$shape` has the same rank as `$data`: it is the shape the data value was
    built from, not an unrelated one.
  }];

  let arguments = (ins Shape_ShapeType:$shape, Hipsr_TensorOrMemRef:$data);
  let results = (outs);

  let assemblyFormat = "$shape `,` $data attr-dict `:` type($data)";

  let hasVerifier = 1;
}
