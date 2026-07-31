//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.shape_yield op: the terminator of a placeholder shape region.
//

include "mlir/Dialect/Shape/IR/ShapeBase.td"

//===----------------------------------------------------------------------===//
// Hipsr_ShapeYieldOp
//===----------------------------------------------------------------------===//

def Hipsr_ShapeYieldOp : Hipsr_Op<
    "shape_yield", [Pure, Terminator, HasParent<"PlaceholderOp">]> {
  let summary = "Yield result shapes from a placeholder shape region";
  let description = [{
    Ends a `hipsr.placeholder` shape region and yields one `!shape.shape`
    value for each placeholder result. Element types and ranked tensor types
    remain on the placeholder results.

    A scalar result still has one shape operand whose extent list is empty.
    For example:
    ```mlir
    hipsr.shape_yield %shape : !shape.shape
    hipsr.shape_yield %values_shape, %indices_shape
        : !shape.shape, !shape.shape
    ```
  }];

  let arguments = (ins Variadic<Shape_ShapeType>:$shapes);
  let assemblyFormat = "$shapes attr-dict `:` type($shapes)";
  let hasVerifier = 1;

  // Required by SingleBlockImplicitTerminator for an empty staging block.
  let builders = [
    OpBuilder<(ins), [{
      $_state.addOperands(::mlir::ValueRange{});
    }]>
  ];
}
