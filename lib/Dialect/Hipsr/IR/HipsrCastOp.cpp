/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrCastOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrCastOp.cpp.inc"

// DestinationStyleOpInterface: the single init operand is the DPS out.
MutableOperandRange CastOp::getDpsInitsMutable() { return getInitMutable(); }

// Single source of truth for the output shape: identical to the input shape.
// Uses the shape dialect so it works uniformly for both tensor and memref
// inputs (Hipsr_TensorOrDeviceMemRef); static dimensions fold automatically.
//
// Before (empty region, as emitted by convert-onnx-to-hipsr):
//   %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
//                         outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
//
// After (arg 0 is the shape-unused ctx; the input is arg 1):
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init)
//       : tensor<?x8xf16> shape_region {
//   ^bb0(%ctxarg: !hipsr.context, %in: tensor<?x8xf32>):
//     %shape = shape.shape_of %in
//     %d0 = shape.get_extent %shape, 0 ; %d1 = shape.get_extent %shape, 1
//     hipsr.shape_yield (%d0, %d1) : [f16]
//   }
void CastOp::populateShapeRegion(OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  // The populate pass creates the entry block first. If it is missing, bail in
  // release (the empty region then fails the verifier) rather than crash.
  Block *body = getShapeBlock();
  assert(body &&
         "populateShapeRegion called before the entry block was created");
  if (!body)
    return;
  builder.setInsertionPointToStart(body);

  Location loc = getLoc();
  // Isolated region: read the input from the block arg, never the operand.
  Value input = ShapeRegionArgs{*body}.getInput();
  auto shapedTy = cast<ShapedType>(input.getType());
  Value shape = builder.create<shape::ShapeOfOp>(loc, input);
  SmallVector<Value> dims;
  dims.reserve(shapedTy.getRank());
  for (int64_t i : llvm::seq<int64_t>(0, shapedTy.getRank())) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    dims.push_back(builder.create<shape::GetExtentOp>(loc, shape, idx));
  }
  // Output element type comes from `init`, not getResult(0): the result is
  // optional (the post-bufferization memref form has no result), but `init`
  // is always present and carries the same element type.
  Type elemTy = cast<ShapedType>(getInit().getType()).getElementType();
  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{ValueRange(dims)},
                               TypeRange{elemTy});
}
