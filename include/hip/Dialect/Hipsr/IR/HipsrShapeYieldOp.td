//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.shape_yield op: the terminator of a hipsr shape region. Compute
// ops and the shared Hipsr_Op base class live in HipsrOps.td.
//

#ifndef HIPSR_SHAPE_YIELD_OP
#define HIPSR_SHAPE_YIELD_OP

include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Hipsr/IR/HipsrOps.td"

//===----------------------------------------------------------------------===//
// Hipsr_ShapeYieldOp
//===----------------------------------------------------------------------===//

def Hipsr_ShapeYieldOp : Hipsr_Op<"shape_yield", [Pure, Terminator]> {
  let summary = "Yield the result shapes from a hipsr shape region";
  let description = [{
    Ends a shape region and yields the output shape of each of the enclosing
    op's results. Every shape region ends with exactly one `shape_yield`.

    A result is described by three lists that line up by index:

    - `shapes[i]`        : result i's dimensions, as a group of `index` values.
    - `ranks[i]`         : result i's rank (how many dims are in `shapes[i]`).
    - `element_types[i]` : result i's element type.

    An empty dim group means a rank-0 (scalar) result. Each dim group prints in
    parentheses; the element types print as a bracketed list.

    Examples:
    ```mlir
    // one 2-D f16 result
    hipsr.shape_yield (%m, %n) : [f16]
    // a 2-D f16 result and a 2-D i64 result
    hipsr.shape_yield (%d0, %d1), (%N, %cap) : [f16, i64]
    // one scalar (rank-0) result
    hipsr.shape_yield () : [f32]
    ```
  }];

  // `shapes` is a list of dim groups, one per result. `ranks` holds each
  // group's size; it must be declared because VariadicOfVariadic looks it up by
  // name (ODS does not add it for us) and fills it in from the shape groups.
  // `element_types` holds each result's element type.
  let arguments = (ins
    VariadicOfVariadic<Index, "ranks">:$shapes,
    DenseI32ArrayAttr:$ranks,
    TypeArrayAttr:$element_types
  );

  // Dim groups print in parens; the element-type list prints in brackets. Dims
  // are always `index`, so their types are left out (like scf.yield).
  let assemblyFormat = "$shapes `:` $element_types attr-dict";

  let hasVerifier = 1;

  let builders = [
    // Custom builder: takes element types as a TypeRange and wraps them into the
    // attribute (the ODS-generated builder takes a TypeArrayAttr). Both args
    // default to empty so a no-argument call builds an empty `shape_yield`.
    OpBuilder<(ins CArg<"::llvm::ArrayRef<::mlir::ValueRange>", "{}">:$shapes,
                   CArg<"::mlir::TypeRange", "{}">:$elementTypes), [{
      build($_builder, $_state, shapes,
            $_builder.getTypeArrayAttr(elementTypes));
    }]>
  ];
}

#endif // HIPSR_SHAPE_YIELD_OP
