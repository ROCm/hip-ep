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

    The region is `IsolatedFromAbove`, so it reads its inputs as entry-block
    arguments (block arg `i` mirrors DPS input `i`) rather than the op's
    operands, which keeps stock MLIR passes (-cse, -canonicalize, LICM) correct
    over it. The `hipsr-populate-shape-region` pass creates the entry block with
    those arguments and hands it to `populateShapeRegion`.

    An implementing op MUST also carry the `IsolatedFromAbove` and `SingleBlock`
    traits (an interface cannot auto-attach traits), e.g.

      def Hipsr_MatMulOp : Hipsr_Op<"matmul", [
          ShapeRegionInterface, IsolatedFromAbove, SingleBlock]> { ... }

    Op categories (barrier markers are separate interfaces):
    - Regular (MatMul, Add): one shape, from the input shapes.
    - StartBarrier (Expand): one shape, from the input data.
    - EndBarrier (NonZero): two shapes (capacity + real) over two regions.
  }];

  let methods = [
    InterfaceMethod<
      [{Fill `shapeBlock` (already created by the pass, with one argument per
        DPS input) with the ops computing the output shape. Read inputs as the
        block's arguments -- the region is isolated, so the operands are off
        limits.}],
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
