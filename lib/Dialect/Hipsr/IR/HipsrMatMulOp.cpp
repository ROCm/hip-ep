/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.cpp.inc"

// DestinationStyleOpInterface: the single init operand is the DPS out.
MutableOperandRange MatMulOp::getDpsInitsMutable() { return getInitMutable(); }

// A and B must be at least 1-D: matmul needs a contraction dim, and the shape
// region below reads the last one or two dims of each operand. A rank-0
// (scalar) operand has neither.
LogicalResult MatMulOp::verify() {
  if (cast<ShapedType>(getA().getType()).getRank() < 1)
    return emitOpError("operand A must be at least 1-D");
  if (cast<ShapedType>(getB().getType()).getRank() < 1)
    return emitOpError("operand B must be at least 1-D");
  return success();
}

// Fills the shape region with the ONNX/NumPy `matmul` output shape. Uses the
// shape dialect so it works for both tensor and memref inputs; static dims
// fold, so a fully static matmul canonicalizes the dim arithmetic away.
//
// ONNX/NumPy matmul output shape (each maps to a branch below):
//   - both >= 2-D: (M,K) x (K,N) -> (M,N); leading dims are batch dims and
//     broadcast (a dim of 1 takes the other side).
//   - 1-D A (K): acts as (1,K); the leading 1 is dropped from the result.
//   - 1-D B (K): acts as (K,1); the trailing 1 is dropped from the result.
//   - both 1-D: scalar result.
// The 1-D promotions and batch-dim count come from the static ranks. This
// computes only the output shape; validity checks (K equality, batch
// broadcastability) are intentionally omitted and can be added later.
//
// Before (region omitted, as emitted by convert-onnx-to-hipsr):
//   %0 = hipsr.matmul ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
//                     outs(%init : tensor<?x?xf16>) -> tensor<?x?xf16>
//
// After (2-D case; batched adds one dim per broadcast batch axis):
//   %0 = hipsr.matmul ins(%A, %B) outs(%init) -> tensor<?x?xf16> shape_region {
//     %shA = shape.shape_of %A ; %shB = shape.shape_of %B
//     %m = shape.get_extent %shA, 0 ; %n = shape.get_extent %shB, 1
//     hipsr.shape_yield (%m, %n) : [f16]
//   }
void MatMulOp::populateShapeRegion(OpBuilder &builder, Region &shapeRegion) {
  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&shapeRegion);
  builder.setInsertionPointToStart(body);

  Location loc = getLoc();
  Value shA = builder.create<shape::ShapeOfOp>(loc, getA());
  Value shB = builder.create<shape::ShapeOfOp>(loc, getB());
  int64_t aRank = cast<ShapedType>(getA().getType()).getRank();
  int64_t bRank = cast<ShapedType>(getB().getType()).getRank();

  auto extent = [&](Value shape, int64_t idx) -> Value {
    Value c = builder.create<arith::ConstantIndexOp>(loc, idx);
    return builder.create<shape::GetExtentOp>(loc, shape, c);
  };
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  // Broadcast two batch dims: a dim of 1 takes the other side.
  auto broadcastDim = [&](Value da, Value db) -> Value {
    Value aIs1 =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, da, one);
    return builder.create<arith::SelectOp>(loc, aIs1, db, da);
  };

  // A 1-D operand acts as rank 2 for the multiply (A: (1,K), B: (K,1)); the
  // effective rank drives the batch-dim count below.
  bool aIs1D = aRank == 1;
  bool bIs1D = bRank == 1;
  int64_t aEffRank = aIs1D ? 2 : aRank;
  int64_t bEffRank = bIs1D ? 2 : bRank;

  // M from A, N from B. A 1-D operand contributes neither (its promoted unit
  // dim is stripped from the result).
  Value m = aIs1D ? Value() : extent(shA, aRank - 2);
  Value n = bIs1D ? Value() : extent(shB, bRank - 1);

  // Batch dims: the leading dims of each operand, right-aligned and broadcast.
  // The result's batch rank is the larger of the two.
  int64_t aBatch = aEffRank - 2;
  int64_t bBatch = bEffRank - 2;
  int64_t batchRank = std::max(aBatch, bBatch);
  SmallVector<Value> dims;
  dims.reserve(batchRank + 2);
  for (int64_t i : llvm::seq<int64_t>(0, batchRank)) {
    // Right-align: a shorter operand has an implicit 1 at its missing leading
    // axes, so a negative index means "that side is 1, take the other".
    int64_t aAxis = i - (batchRank - aBatch);
    int64_t bAxis = i - (batchRank - bBatch);
    if (aAxis < 0)
      dims.push_back(extent(shB, bAxis));
    else if (bAxis < 0)
      dims.push_back(extent(shA, aAxis));
    else
      dims.push_back(broadcastDim(extent(shA, aAxis), extent(shB, bAxis)));
  }
  // Append M then N, skipping the promoted (later-stripped) unit dims.
  if (m)
    dims.push_back(m);
  if (n)
    dims.push_back(n);

  // Single result: one dim group and its (output) element type.
  Type elemTy = cast<ShapedType>(getResult(0).getType()).getElementType();
  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{ValueRange(dims)},
                               TypeRange{elemTy});
}
