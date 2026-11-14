/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipElementMoveInterface.cpp - ElementMoveOpInterface ---------------===//
//
// Provides the TableGen-generated `ElementMoveOpInterface` bindings, the HIP
// op implementations, and the external models on the tensor view ops. See
// docs/design/element-move-interface.md.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::hip;

namespace mlir::hip {
#include "hip/Dialect/IR/HipElementMoveInterface.cpp.inc"
} // namespace mlir::hip

namespace {

// Clones `op` with `moved` in place of its moved operands and its single
// result retyped to `elementType`.
Value cloneRetyped(OpBuilder &builder, Operation *op,
                   ArrayRef<OpOperand *> movedOperands, ValueRange moved,
                   Type elementType) {
  auto resultType =
      cast<RankedTensorType>(op->getResult(0).getType()).clone(elementType);
  Operation *clone = builder.clone(*op);
  for (auto [operand, value] : llvm::zip_equal(movedOperands, moved))
    clone->setOperand(operand->getOperandNumber(), value);
  clone->getResult(0).setType(resultType);
  return clone->getResult(0);
}

// As cloneRetyped, for a HIP DPS op whose init is not a moved operand: the
// clone writes into a new empty tensor of the new type.
Value cloneHipDpsRetyped(OpBuilder &builder, DestinationStyleOpInterface op,
                         ArrayRef<OpOperand *> movedOperands, ValueRange moved,
                         Type elementType) {
  OpOperand &init = op.getDpsInitsMutable()[0];
  auto initType = cast<RankedTensorType>(init.get().getType());
  Value empty = tensor::EmptyOp::create(builder, op.getLoc(),
                                        initType.getShape(), elementType);
  SmallVector<OpOperand *> operands(movedOperands);
  operands.push_back(&init);
  SmallVector<Value> values(moved);
  values.push_back(empty);
  return cloneRetyped(builder, op, operands, values, elementType);
}

// insert_slice moves the elements of both the source and the destination.
struct InsertSliceElementMove
    : ElementMoveOpInterface::ExternalModel<InsertSliceElementMove,
                                            tensor::InsertSliceOp> {
  bool movesElements(Operation *) const { return true; }
  SmallVector<OpOperand *> getMovedOperands(Operation *op) const {
    auto insert = cast<tensor::InsertSliceOp>(op);
    return {&insert.getSourceMutable(), &insert.getDestMutable()};
  }
  Value cloneWithElementType(Operation *op, OpBuilder &builder,
                             ValueRange moved, Type elementType) const {
    return cloneRetyped(builder, op, getMovedOperands(op), moved, elementType);
  }
};

// A view op with one source operand.
template <typename OpTy>
struct ViewElementMove
    : ElementMoveOpInterface::ExternalModel<ViewElementMove<OpTy>, OpTy> {
  bool movesElements(Operation *) const { return true; }
  SmallVector<OpOperand *> getMovedOperands(Operation *op) const {
    return {&op->getOpOperand(0)};
  }
  Value cloneWithElementType(Operation *op, OpBuilder &builder,
                             ValueRange moved, Type elementType) const {
    return cloneRetyped(builder, op, getMovedOperands(op), moved, elementType);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HIP ops
//===----------------------------------------------------------------------===//

bool TransposeOp::movesElements() { return true; }

SmallVector<OpOperand *> TransposeOp::getMovedOperands() {
  return {&getInputMutable()};
}

Value TransposeOp::cloneWithElementType(OpBuilder &builder, ValueRange moved,
                                        Type elementType) {
  return cloneHipDpsRetyped(builder, *this, getMovedOperands(), moved,
                            elementType);
}

// Only nearest copies input elements; the runtime kernel takes i8 and ui8
// for nearest and rejects them for linear.
bool ResizeOp::movesElements() { return getMode() == 0; }

SmallVector<OpOperand *> ResizeOp::getMovedOperands() {
  return {&getInputMutable()};
}

Value ResizeOp::cloneWithElementType(OpBuilder &builder, ValueRange moved,
                                     Type elementType) {
  return cloneHipDpsRetyped(builder, *this, getMovedOperands(), moved,
                            elementType);
}

void mlir::hip::registerElementMoveExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *) {
    tensor::InsertSliceOp::attachInterface<InsertSliceElementMove>(*ctx);
    tensor::ExtractSliceOp::attachInterface<
        ViewElementMove<tensor::ExtractSliceOp>>(*ctx);
    tensor::ExpandShapeOp::attachInterface<
        ViewElementMove<tensor::ExpandShapeOp>>(*ctx);
    tensor::CollapseShapeOp::attachInterface<
        ViewElementMove<tensor::CollapseShapeOp>>(*ctx);
  });
}
