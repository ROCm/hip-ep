//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ShapeRegionInterface. The pool-domain barriers are separate interfaces:
// StartBarrierInterface (HipsrStartBarrierInterface.td) and EndBarrierInterface
// (HipsrEndBarrierInterface.td).
//

#ifndef HIPSR_SHAPE_REGION_INTERFACE
#define HIPSR_SHAPE_REGION_INTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ShapeRegionInterface
//===----------------------------------------------------------------------===//

def ShapeRegionInterface : OpInterface<"ShapeRegionInterface"> {
  let cppNamespace = "::mlir::hipsr";

  let description = [{
    Interface for ops that compute their own output shape in an attached region.

    An implementing op carries an optional shape region (region 0): either empty
    (no shape computation yet) or a single block that computes the output
    dimensions and ends with `hipsr.shape_yield`.

    The region is `IsolatedFromAbove`, so it reads the operands it needs as
    entry-block arguments rather than capturing the op's operands, which keeps
    stock MLIR passes (-cse, -canonicalize, LICM) correct over it. Which
    operands become args is keyed on the op's barrier category (see
    `getShapeRegionArgOperands`). The `hipsr-populate-shape-region` pass creates
    the entry block with those arguments and hands it to `populateShapeRegion`.

    An implementing op MUST also carry the `IsolatedFromAbove` and `SingleBlock`
    traits (an interface cannot auto-attach traits), e.g.

      def Hipsr_MatMulOp : Hipsr_Op<"matmul", [
          ShapeRegionInterface, IsolatedFromAbove, SingleBlock]> { ... }

    Op categories (barrier markers are separate interfaces) and their
    entry-block args:
    - Regular (MatMul, Add): one shape from the input shapes; args = data ins.
    - StartBarrier (Expand): one shape from input data; args = ctx + data ins.
    - EndBarrier (NonZero): two shapes (capacity + real) over two regions;
      args = data ins + outs.
  }];

  let methods = [
    InterfaceMethod<
      [{Fill `shapeBlock` (already created by the pass, with the op's
        category-specific args -- see `getShapeRegionArgOperands`) with the ops
        computing the output shape. Read operands as the block's arguments --
        the region is isolated, so the op's operands are off limits.}],
      "void", "populateShapeRegion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Block &":$shapeBlock)
    >,
    InterfaceMethod<
      [{Fill the capacity shape region's block (region 1). Only EndBarrier ops
        override this; same block-argument contract as `populateShapeRegion`.}],
      "void", "populateCapacityShapeRegion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Block &":$capacityBlock),
      /*methodBody=*/[{}],
      // Default no-op: ops without a capacity region (Regular/StartBarrier)
      // inherit it instead of each supplying an empty override.
      /*defaultImplementation=*/[{ return; }]
    >,
  ];

  let verify = [{
    return ::mlir::hipsr::verifyShapeRegionStructure($_op);
  }];
}

#endif // HIPSR_SHAPE_REGION_INTERFACE
