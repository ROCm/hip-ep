/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "RebiasUnsignedQuantize.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>

#include <cstdint>
#include <optional>

namespace mlir::hip {
namespace {

constexpr int64_t kUnsignedByteOffset = 128;

bool isStaticUnsignedByte(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() &&
         type.getElementType().isUnsignedInteger(8);
}

bool isStaticEmpty(Value value) {
  auto empty = value.getDefiningOp<tensor::EmptyOp>();
  return empty && empty.getType().hasStaticShape();
}

// An absent zero point is 0; otherwise it must be one inline integer.
std::optional<int64_t> getScalarZeroPoint(Value zeroPoint) {
  if (!zeroPoint)
    return 0;
  auto constant = zeroPoint.getDefiningOp<ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto attr = dyn_cast_if_present<DenseElementsAttr>(constant.getValueAttr());
  if (!attr || attr.getNumElements() != 1 ||
      !attr.getElementType().isUnsignedInteger(8))
    return std::nullopt;
  return static_cast<int64_t>(
      (*attr.getValues<APInt>().begin()).getZExtValue());
}

bool isRebiasableQuantize(QuantizeLinearOp q) {
  return q->getNumResults() == 1 && !q.getPackedInt4() &&
         getScalarZeroPoint(q.getZeroPoint()).has_value();
}

bool isRebiasableDequantize(DequantizeLinearOp dq) {
  return dq->getNumResults() == 1 && !dq.getPackedInt4() &&
         getScalarZeroPoint(dq.getZeroPoint()).has_value();
}

// The data operand of a QLinear anchor and its zero point.
OpOperand *getQLinearDataOperand(Operation *op) {
  if (auto conv = dyn_cast<QLinearConvOp>(op))
    return &conv.getInputMutable();
  if (auto matmul = dyn_cast<QLinearMatMulOp>(op))
    return &matmul.getAMutable();
  return nullptr;
}

OpOperand *getQLinearDataZeroPoint(Operation *op) {
  if (auto conv = dyn_cast<QLinearConvOp>(op))
    return &conv.getInputZeroPointMutable();
  return &cast<QLinearMatMulOp>(op).getAZeroPointMutable();
}

OpOperand *getQLinearOutputZeroPoint(Operation *op) {
  if (auto conv = dyn_cast<QLinearConvOp>(op))
    return &conv.getOutputZeroPointMutable();
  return &cast<QLinearMatMulOp>(op).getYZeroPointMutable();
}

// One connected set of ui8 values, with the ops that produce and read them.
struct Tree {
  SetVector<Value> values;
  SetVector<Operation *> quantizes;
  SetVector<Operation *> dequantizes;
  SetVector<Operation *> qlinearResults;
  SetVector<OpOperand *> qlinearInputs;
  SetVector<Operation *> interior;
  SetVector<OpOperand *> emptyOperands;
  bool feedsAnchor = false;
};

class TreeBuilder {
public:
  TreeBuilder(Tree &tree, function_ref<bool(Operation *)> isIntegerAnchor)
      : tree(tree), isIntegerAnchor(isIntegerAnchor) {}

  // Collects the whole tree that holds `root`, so a declined tree is walked
  // once. False when any value in it has a producer or a reader that cannot
  // take i8.
  bool build(Value root) {
    bool accepted = true;
    SmallVector<Value> worklist{root};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      if (!tree.values.insert(value))
        continue;
      if (!isStaticUnsignedByte(value) || !addProducer(value, worklist))
        accepted = false;
      for (OpOperand &use : value.getUses())
        if (!addReader(use, worklist))
          accepted = false;
    }
    return accepted;
  }

private:
  bool addProducer(Value value, SmallVectorImpl<Value> &worklist) {
    Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    if (auto q = dyn_cast<QuantizeLinearOp>(def)) {
      if (!isRebiasableQuantize(q))
        return false;
      tree.quantizes.insert(q);
      return true;
    }
    if (isa<QLinearConvOp, QLinearMatMulOp>(def)) {
      if (def->getNumResults() != 1 || !isIntegerAnchor(def) ||
          !getScalarZeroPoint(getQLinearOutputZeroPoint(def)->get()))
        return false;
      tree.qlinearResults.insert(def);
      return true;
    }
    auto move = dyn_cast<ElementMoveOpInterface>(def);
    if (!move || def->getNumResults() != 1 || !move.movesElements())
      return false;
    tree.interior.insert(def);
    for (OpOperand *operand : move.getMovedOperands()) {
      if (isStaticEmpty(operand->get()))
        tree.emptyOperands.insert(operand);
      else
        worklist.push_back(operand->get());
    }
    return true;
  }

  bool addReader(OpOperand &use, SmallVectorImpl<Value> &worklist) {
    Operation *user = use.getOwner();
    if (auto dq = dyn_cast<DequantizeLinearOp>(user)) {
      if (&use != &dq.getInputMutable() || !isRebiasableDequantize(dq))
        return false;
      tree.dequantizes.insert(dq);
      for (Operation *reader : dq->getResult(0).getUsers())
        tree.feedsAnchor |= isIntegerAnchor(reader);
      return true;
    }
    if (isa<QLinearConvOp, QLinearMatMulOp>(user)) {
      if (&use != getQLinearDataOperand(user) || !isIntegerAnchor(user) ||
          !getScalarZeroPoint(getQLinearDataZeroPoint(user)->get()))
        return false;
      tree.qlinearInputs.insert(&use);
      tree.feedsAnchor = true;
      return true;
    }
    auto move = dyn_cast<ElementMoveOpInterface>(user);
    if (!move || user->getNumResults() != 1 || !move.movesElements() ||
        !llvm::is_contained(move.getMovedOperands(), &use))
      return false;
    worklist.push_back(user->getResult(0));
    return true;
  }

  Tree &tree;
  function_ref<bool(Operation *)> isIntegerAnchor;
};

Type getSignedByte(MLIRContext *ctx) { return IntegerType::get(ctx, 8); }

RankedTensorType toSignedByte(Type type) {
  auto tensorType = cast<RankedTensorType>(type);
  return tensorType.clone(getSignedByte(type.getContext()));
}

Value createEmpty(OpBuilder &builder, Location loc, Value like) {
  RankedTensorType type = toSignedByte(like.getType());
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType());
}

// A new i8 zero point, `zeroPoint` - 128, in the shape of the old one.
void rebiasZeroPoint(OpBuilder &builder, Operation *op,
                     MutableOperandRange zeroPoint) {
  OperandRange current = zeroPoint;
  int64_t value = *getScalarZeroPoint(current.empty() ? Value() : current[0]);
  auto type = current.empty()
                  ? RankedTensorType::get({}, getSignedByte(op->getContext()))
                  : toSignedByte(current[0].getType());
  builder.setInsertionPoint(op);
  auto attr = DenseElementsAttr::get(
      type, builder.getIntegerAttr(type.getElementType(),
                                   value - kUnsignedByteOffset));
  zeroPoint.assign(ConstantOp::create(builder, op->getLoc(), type, attr));
}

// The result becomes i8, and each init that is not in the tree (the
// destination of an insert_slice is) takes a new i8 empty tensor.
void retypeResult(OpBuilder &builder, Operation *op, const Tree &tree) {
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    builder.setInsertionPoint(op);
    for (OpOperand &init : dps.getDpsInitsMutable())
      if (!tree.values.contains(init.get()) &&
          !tree.emptyOperands.contains(&init))
        init.set(createEmpty(builder, op->getLoc(), init.get()));
  }
  Value result = op->getResult(0);
  result.setType(toSignedByte(result.getType()));
}

void rewriteTree(const Tree &tree) {
  OpBuilder builder(tree.values.front().getContext());
  for (OpOperand *operand : tree.emptyOperands) {
    builder.setInsertionPoint(operand->getOwner());
    operand->set(
        createEmpty(builder, operand->getOwner()->getLoc(), operand->get()));
  }
  for (Operation *op : tree.interior)
    retypeResult(builder, op, tree);
  for (Operation *op : tree.quantizes) {
    retypeResult(builder, op, tree);
    rebiasZeroPoint(builder, op,
                    cast<QuantizeLinearOp>(op).getZeroPointMutable());
  }
  for (Operation *op : tree.qlinearResults) {
    retypeResult(builder, op, tree);
    OpOperand *zeroPoint = getQLinearOutputZeroPoint(op);
    rebiasZeroPoint(builder, op,
                    MutableOperandRange(op, zeroPoint->getOperandNumber(), 1));
  }
  for (Operation *op : tree.dequantizes)
    rebiasZeroPoint(builder, op,
                    cast<DequantizeLinearOp>(op).getZeroPointMutable());
  for (OpOperand *input : tree.qlinearInputs) {
    Operation *op = input->getOwner();
    OpOperand *zeroPoint = getQLinearDataZeroPoint(op);
    rebiasZeroPoint(builder, op,
                    MutableOperandRange(op, zeroPoint->getOperandNumber(), 1));
  }
}

} // namespace

// Before (z is a ui8 zero point):
//   %e  = tensor.empty() : tensor<1x8x4x4xui8>
//   %a  = hip.quantize_linear(%ctx) ins(%x, %s) zero_point(%z)
//           outs(%ea : tensor<1x4x4x4xui8>)
//   %c  = tensor.insert_slice %a into %e[0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
//   %dq = hip.dequantize_linear(%ctx) ins(%c, %s) zero_point(%z)
//   %y  = hip.conv(%ctx) ins(%dq, %dw) ...
// After (z8 = z - 128 as i8):
//   %e  = tensor.empty() : tensor<1x8x4x4xi8>
//   %a  = hip.quantize_linear(%ctx) ins(%x, %s) zero_point(%z8)
//           outs(%ea8 : tensor<1x4x4x4xi8>)
//   %c  = tensor.insert_slice %a into %e[0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
//   %dq = hip.dequantize_linear(%ctx) ins(%c, %s) zero_point(%z8)
//   %y  = hip.conv(%ctx) ins(%dq, %dw) ...
//
// The rewrite is exact: clamp(round(x / s) + z - 128, -128, 127) equals
// clamp(round(x / s) + z, 0, 255) - 128, and (q - 128) - (z - 128) = q - z,
// while the ops between them only move elements. A tree that holds a function
// argument, a return, or any other reader keeps ui8, and so does a tree that
// reaches no integer anchor, since only an anchor gains from signed bytes.
void rebiasUnsignedQuantizedTrees(
    func::FuncOp funcOp, function_ref<bool(Operation *)> isIntegerAnchor) {
  SmallVector<Value> roots;
  funcOp.walk([&](Operation *op) {
    if (isa<QuantizeLinearOp, QLinearConvOp, QLinearMatMulOp>(op) &&
        op->getNumResults() == 1 && isStaticUnsignedByte(op->getResult(0)))
      roots.push_back(op->getResult(0));
  });
  DenseSet<Operation *> visited;
  for (Value root : roots) {
    if (!visited.insert(root.getDefiningOp()).second)
      continue;
    Tree tree;
    bool accepted = TreeBuilder(tree, isIntegerAnchor).build(root);
    for (Operation *op : tree.quantizes)
      visited.insert(op);
    for (Operation *op : tree.qlinearResults)
      visited.insert(op);
    if (accepted && tree.feedsAnchor)
      rewriteTree(tree);
  }
}

} // namespace mlir::hip
