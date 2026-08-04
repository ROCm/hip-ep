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

    The shape region (region 0) is optional: empty, or one block ending in
    `hipsr.shape_yield`. It is `IsolatedFromAbove`, reading its inputs as
    entry-block args (keyed on barrier category, see `getShapeRegionArgOperands`),
    so stock MLIR passes (-cse, -canonicalize, LICM) stay correct over it. The
    `hipsr-populate-shape-region` pass fills it via `populateShapeRegion`.

    An implementing op MUST also carry the `IsolatedFromAbove` and `SingleBlock`
    traits, e.g.

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
      [{Emit the output-shape computation into `shapeBlock` (already created by
        the pass with the op's args -- see `getShapeRegionArgOperands`), reading
        inputs as the block's arguments.}],
      "void", "populateShapeRegion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Block &":$shapeBlock),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return; }]
    >,
    InterfaceMethod<
      [{Like `populateShapeRegion`, but for the capacity region (region 1); only
        EndBarrier ops override it.}],
      "void", "populateCapacityShapeRegion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Block &":$capacityBlock),
      /*methodBody=*/[{}],
      // Default no-op for Regular/StartBarrier ops, which have no capacity
      // region.
      /*defaultImplementation=*/[{ return; }]
    >,
  ];

  let verify = [{
    return ::mlir::hipsr::verifyShapeRegionStructure($_op);
  }];
}

#endif // HIPSR_SHAPE_REGION_INTERFACE
