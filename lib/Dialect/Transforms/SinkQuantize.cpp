/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "SinkQuantize.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/SetVector.h>

#include <cstdint>
#include <optional>

namespace mlir::hip {
namespace {

// Per-tensor quantization parameters, compared by value.
struct QuantParams {
  APFloat scale;
  int64_t zeroPoint;
  Type type;

  bool operator==(const QuantParams &other) const {
    return scale.bitwiseIsEqual(other.scale) && zeroPoint == other.zeroPoint &&
           type == other.type;
  }
};

DenseElementsAttr getScalarAttr(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return {};
  auto attr = dyn_cast_if_present<DenseElementsAttr>(constant.getValueAttr());
  if (!attr || attr.getNumElements() != 1)
    return {};
  return attr;
}

std::optional<QuantParams> getParams(Value scale, Value zeroPoint, Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  DenseElementsAttr scaleAttr = getScalarAttr(scale);
  if (!intType || intType.getWidth() != 8 || !scaleAttr ||
      !scaleAttr.getElementType().isF32())
    return std::nullopt;
  int64_t zp = 0;
  if (zeroPoint) {
    DenseElementsAttr zpAttr = getScalarAttr(zeroPoint);
    if (!zpAttr || !isa<IntegerType>(zpAttr.getElementType()))
      return std::nullopt;
    APInt v = *zpAttr.getValues<APInt>().begin();
    zp = zpAttr.getElementType().isUnsignedInteger() ? v.getZExtValue()
                                                     : v.getSExtValue();
  }
  return QuantParams{*scaleAttr.getValues<APFloat>().begin(), zp, type};
}

// A per-tensor dequantize of an 8-bit tensor with inline parameters.
DequantizeLinearOp matchDequantize(Value value,
                                   std::optional<QuantParams> &params) {
  auto dq = value.getDefiningOp<DequantizeLinearOp>();
  if (!dq || dq->getNumResults() != 1 || dq.getPackedInt4() ||
      dq.getBlockSize() != 0)
    return {};
  auto dataType = dyn_cast<RankedTensorType>(dq.getInput().getType());
  if (!dataType || !dataType.hasStaticShape())
    return {};
  params =
      getParams(dq.getScale(), dq.getZeroPoint(), dataType.getElementType());
  return params ? dq : DequantizeLinearOp{};
}

bool isStaticEmpty(Value value) {
  auto empty = value.getDefiningOp<tensor::EmptyOp>();
  return empty && empty.getType().hasStaticShape();
}

// An interior node: a single-result op that only moves elements and whose
// result has no other reader, since the original stays live otherwise.
ElementMoveOpInterface getInterior(Value value) {
  auto op = value.getDefiningOp<ElementMoveOpInterface>();
  if (!op || op->getNumResults() != 1 || !value.hasOneUse() ||
      !op.movesElements())
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return {};
  return op;
}

// Checks the tree under `value` and records its interior ops, leaves first.
bool planTree(Value value, SmallVectorImpl<Operation *> &interior) {
  std::optional<QuantParams> params;
  if (matchDequantize(value, params) || isStaticEmpty(value))
    return true;
  ElementMoveOpInterface op = getInterior(value);
  if (!op)
    return false;
  for (OpOperand *operand : op.getMovedOperands())
    if (!planTree(operand->get(), interior))
      return false;
  interior.push_back(op);
  return true;
}

class Emitter {
public:
  Emitter(RewriterBase &rewriter, QuantizeLinearOp q, QuantParams outParams)
      : rewriter(rewriter), q(q), outParams(outParams) {}

  // The 8-bit value that equals q(value). planTree accepted `value`.
  Value emit(Value value) {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    Value result;
    std::optional<QuantParams> params;
    if (DequantizeLinearOp dq = matchDequantize(value, params)) {
      result = *params == outParams ? dq.getInput() : requantize(dq);
    } else if (auto empty = value.getDefiningOp<tensor::EmptyOp>()) {
      result = tensor::EmptyOp::create(
          rewriter, empty.getLoc(), empty.getType().getShape(), outParams.type);
    } else {
      auto op = cast<ElementMoveOpInterface>(value.getDefiningOp());
      SmallVector<Value> moved;
      for (OpOperand *operand : op.getMovedOperands())
        moved.push_back(emit(operand->get()));
      result = op.cloneWithElementType(rewriter, moved, outParams.type);
    }
    mapping.map(value, result);
    return result;
  }

private:
  Value requantize(DequantizeLinearOp dq) {
    Location loc = q.getLoc();
    auto type =
        cast<RankedTensorType>(dq.getInput().getType()).clone(outParams.type);
    Value init = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                         type.getElementType());
    return QuantizeLinearOp::create(
               rewriter, loc, TypeRange{type}, q.getCtx(), dq.getResult(0),
               q.getScale(), q.getZeroPoint(), init, q.getAxisAttr(),
               q.getBlockSizeAttr(), q.getPrecisionAttr(), q.getSaturateAttr())
        .getResult(0);
  }

  RewriterBase &rewriter;
  QuantizeLinearOp q;
  QuantParams outParams;
  IRMapping mapping;
};

// Before (a and b are u8; a has the quantize's parameters, b does not):
//   %da = hip.dequantize_linear(%ctx) ins(%a, %s) zero_point(%z)
//   %db = hip.dequantize_linear(%ctx) ins(%b, %s2) zero_point(%z2)
//   %e  = tensor.empty() : tensor<1x8x4x4xf32>
//   %c0 = tensor.insert_slice %da into %e[0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
//   %c1 = tensor.insert_slice %db into %c0[0, 4, 0, 0] [1, 4, 4, 4] [1, 1, 1,
//   1] %t  = hip.transpose(%ctx) ins(%c1) outs(%te) {perm = [0, 2, 3, 1]} %y  =
//   hip.quantize_linear(%ctx) ins(%t, %s) zero_point(%z)
// After:
//   %qb = hip.quantize_linear(%ctx) ins(%db, %s) zero_point(%z)
//   %e  = tensor.empty() : tensor<1x8x4x4xui8>
//   %c0 = tensor.insert_slice %a into %e[0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
//   %c1 = tensor.insert_slice %qb into %c0[0, 4, 0, 0] [1, 4, 4, 4] [1, 1, 1,
//   1] %y  = hip.transpose(%ctx) ins(%c1) outs(%te8) {perm = [0, 2, 3, 1]}
//
// A per-tensor quantize is elementwise and monotonic, so it commutes with an
// op that only moves or selects elements, and q(dq(a)) with one scale, zero
// point and type returns a: the round trip error is far below half a unit.
// The whole tree is checked before the first edit.
bool sinkQuantize(RewriterBase &rewriter, QuantizeLinearOp q) {
  if (q->getNumResults() != 1 || q.getSaturate() == 0 || q.getPackedInt4() ||
      q.getBlockSize() != 0)
    return false;
  auto resultType = dyn_cast<RankedTensorType>(q.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  std::optional<QuantParams> outParams =
      getParams(q.getScale(), q.getZeroPoint(), resultType.getElementType());
  if (!outParams || !getInterior(q.getInput()))
    return false;
  SmallVector<Operation *> interior;
  if (!planTree(q.getInput(), interior))
    return false;

  SetVector<Operation *> leaves;
  for (Operation *op : interior)
    for (OpOperand *operand :
         cast<ElementMoveOpInterface>(op).getMovedOperands())
      if (Operation *def = operand->get().getDefiningOp();
          def && !llvm::is_contained(interior, def))
        leaves.insert(def);

  rewriter.setInsertionPoint(q);
  Emitter emitter(rewriter, q, *outParams);
  rewriter.replaceOp(q, emitter.emit(q.getInput()));
  for (Operation *op : llvm::reverse(interior))
    rewriter.eraseOp(op);
  for (Operation *leaf : leaves)
    if (leaf->use_empty())
      rewriter.eraseOp(leaf);
  return true;
}

} // namespace

void sinkQuantizeThroughElementMoves(func::FuncOp funcOp) {
  SmallVector<QuantizeLinearOp> quantizes;
  funcOp.walk([&](QuantizeLinearOp q) { quantizes.push_back(q); });
  IRRewriter rewriter(funcOp.getContext());
  for (QuantizeLinearOp q : quantizes)
    sinkQuantize(rewriter, q);
}

} // namespace mlir::hip
