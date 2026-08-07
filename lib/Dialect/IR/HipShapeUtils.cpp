/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtils.cpp - Core and broadcast shape helpers ---------------===//
//
// Category implementation for the public shape helpers declared in
// `hip/Dialect/IR/HipShapeUtils.h`.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipShapeUtils.h"
#include "HipShapeUtilsInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::hip;

namespace mlir::hip::detail {

/// Pretty-print a shape vector with `?` for kDynamic. Used in diagnostics.
std::string formatShape(ArrayRef<int64_t> shape) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "[";
  llvm::interleaveComma(shape, os, [&](int64_t d) {
    if (ShapedType::isDynamic(d))
      os << "?";
    else
      os << d;
  });
  os << "]";
  return out;
}

} // namespace mlir::hip::detail

namespace {

FailureOr<OpFoldResult>
broadcastDim(OpBuilder &b, Location loc, OpFoldResult lhs, OpFoldResult rhs,
             function_ref<InFlightDiagnostic()> emitError) {
  if (lhs == rhs)
    return lhs;

  std::optional<int64_t> lhsStatic = getConstantIntValue(lhs);
  std::optional<int64_t> rhsStatic = getConstantIntValue(rhs);

  if (lhsStatic && rhsStatic) {
    if (*lhsStatic == 1)
      return rhs;
    if (*rhsStatic == 1 || *lhsStatic == *rhsStatic)
      return lhs;
    emitError() << "incompatible broadcast dimensions " << *lhsStatic << " and "
                << *rhsStatic;
    return failure();
  }

  // Under the ONNX broadcastability precondition, a dynamic extent paired
  // with a known non-unit extent must be either 1 or that known extent.
  if (lhsStatic)
    return *lhsStatic == 1 ? rhs : lhs;
  if (rhsStatic)
    return *rhsStatic == 1 ? lhs : rhs;

  Value lhsValue = getValueOrCreateConstantIndexOp(b, loc, lhs);
  Value rhsValue = getValueOrCreateConstantIndexOp(b, loc, rhs);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value lhsIsOne =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lhsValue, one);
  return OpFoldResult(
      arith::SelectOp::create(b, loc, lhsIsOne, rhsValue, lhsValue));
}

} // namespace

/// NumPy-broadcast result shape of `shapes` (right-aligned) from static extents
/// only. Folds `OpTrait::util::getBroadcastedShape` pairwise so static
/// broadcast validation is identical to the matmul batch path.
FailureOr<SmallVector<int64_t>>
mlir::hip::inferBroadcastShape(ArrayRef<ArrayRef<int64_t>> shapes,
                               function_ref<InFlightDiagnostic()> emitError) {
  if (shapes.empty()) {
    emitError() << "broadcast requires at least one input shape";
    return failure();
  }

  SmallVector<int64_t> result(shapes.front());
  for (ArrayRef<int64_t> shape : shapes.drop_front()) {
    SmallVector<int64_t> merged;
    if (!OpTrait::util::getBroadcastedShape(result, shape, merged)) {
      emitError() << "incompatible broadcast shapes "
                  << mlir::hip::detail::formatShape(result) << " and "
                  << mlir::hip::detail::formatShape(shape);
      return failure();
    }
    result = std::move(merged);
  }
  return result;
}

namespace mlir::hip::detail {

/// NumPy-broadcast result shape from already-reified operand shapes. Callers
/// must have validated broadcastability against the static shapes first, since
/// this materializes index SSA as it folds.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastShape(OpBuilder &b, Location loc,
                    ArrayRef<SmallVector<OpFoldResult>> inputShapes,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (inputShapes.empty()) {
    emitError() << "broadcast requires at least one input shape";
    return failure();
  }

  size_t resultRank = 0;
  for (const SmallVector<OpFoldResult> &shape : inputShapes)
    resultRank = std::max(resultRank, shape.size());

  SmallVector<OpFoldResult> result(resultRank, b.getIndexAttr(1));
  for (const SmallVector<OpFoldResult> &shape : inputShapes) {
    size_t pad = resultRank - shape.size();
    for (size_t i : llvm::seq<size_t>(0, resultRank)) {
      OpFoldResult inputDim =
          i < pad ? OpFoldResult(b.getIndexAttr(1)) : shape[i - pad];
      FailureOr<OpFoldResult> merged =
          broadcastDim(b, loc, result[i], inputDim, emitError);
      if (failed(merged))
        return failure();
      result[i] = *merged;
    }
  }
  return result;
}

} // namespace mlir::hip::detail

namespace mlir::hip::detail {

FailureOr<OpFoldResult> scaleAndOffsetDim(OpBuilder &b, Location loc,
                                          OpFoldResult dim, int64_t scale,
                                          int64_t offset) {
  if (std::optional<int64_t> constant = getConstantIntValue(dim)) {
    APInt value = APInt(128, *constant, /*isSigned=*/true) *
                      APInt(128, scale, /*isSigned=*/true) +
                  APInt(128, offset, /*isSigned=*/true);
    if (!value.isSignedIntN(64))
      return failure();
    return OpFoldResult(b.getIndexAttr(value.getSExtValue()));
  }

  Value value = getValueOrCreateConstantIndexOp(b, loc, dim);
  if (scale != 1)
    value = arith::MulIOp::create(
        b, loc, value, arith::ConstantIndexOp::create(b, loc, scale));
  if (offset != 0)
    value = arith::AddIOp::create(
        b, loc, value, arith::ConstantIndexOp::create(b, loc, offset));
  return OpFoldResult(value);
}

} // namespace mlir::hip::detail

LogicalResult mlir::hip::verifyDpsComputeOp(Operation *op,
                                            ArrayRef<Value> dataOperands,
                                            unsigned numInits) {
  if (dataOperands.empty())
    return op->emitOpError("expected at least one data operand");

  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return op->emitOpError(
        "DPS verification requires DestinationStyleOpInterface");
  SmallVector<Value> inits = dpsOp.getDpsInits();
  if (inits.size() != numInits)
    return op->emitOpError("expected ")
           << numInits << " DPS init operand(s), got " << inits.size();

  auto isTensor = [](Value value) {
    return isa<RankedTensorType>(value.getType());
  };
  auto isMemref = [](Value value) { return isa<MemRefType>(value.getType()); };
  if (!isTensor(dataOperands.front()) && !isMemref(dataOperands.front()))
    return op->emitOpError("data operands must be ranked tensors or memrefs");
  bool tensorMode = isTensor(dataOperands.front());
  for (Value value : dataOperands) {
    if (!isTensor(value) && !isMemref(value))
      return op->emitOpError("data operands must be ranked tensors or memrefs");
    if (isTensor(value) != tensorMode)
      return op->emitOpError(
          "all data operands must be the same kind (all tensor or all memref)");
  }

  if (!tensorMode) {
    if (op->getNumResults() != 0)
      return op->emitOpError("memref mode must have zero results, got ")
             << op->getNumResults();
    return success();
  }

  if (op->getNumResults() != numInits)
    return op->emitOpError("tensor mode requires ")
           << numInits << " result(s), got " << op->getNumResults();
  for (unsigned index = 0; index < numInits; ++index) {
    Type resultType = op->getResult(index).getType();
    Type initType = inits[index].getType();
    if (!isa<RankedTensorType>(resultType))
      return op->emitOpError("result #") << index << " must be a ranked tensor";
    if (resultType != initType)
      return op->emitOpError("result type #")
             << index << " (" << resultType << ") must match DPS init type #"
             << index << " (" << initType << ")";
  }
  return success();
}

LogicalResult mlir::hip::verifyHipOpShape(
    Operation *op,
    function_ref<SmallVector<SmallVector<int64_t>>()> computeExpected) {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  SmallVector<SmallVector<int64_t>> expected = computeExpected();
  if (expected.empty())
    return failure();

  auto inits = dpsOp.getDpsInits();
  assert(expected.size() == inits.size() &&
         "shape helper must produce one expected shape per DPS init operand");
  if (expected.size() != inits.size())
    return failure();

  for (auto [i, init] : llvm::enumerate(inits)) {
    auto initType = dyn_cast<ShapedType>(init.getType());
    if (!initType)
      return op->emitOpError("init #") << i << " is not a shaped type";
    ArrayRef<int64_t> actualShape = initType.getShape();
    ArrayRef<int64_t> expectedShape = expected[i];
    if (actualShape.size() != expectedShape.size())
      return op->emitOpError("rank mismatch on result #")
             << i << ": expected rank " << expectedShape.size() << " "
             << detail::formatShape(expectedShape) << " but outs has rank "
             << actualShape.size() << " " << detail::formatShape(actualShape);
    for (size_t dim : llvm::seq<size_t>(0, actualShape.size())) {
      if (ShapedType::isDynamic(actualShape[dim]) ||
          ShapedType::isDynamic(expectedShape[dim]))
        continue;
      if (actualShape[dim] != expectedShape[dim])
        return op->emitOpError("dim ")
               << dim << " of result #" << i << " mismatch: expected "
               << expectedShape[dim] << " "
               << detail::formatShape(expectedShape) << " but outs has "
               << actualShape[dim] << " " << detail::formatShape(actualShape);
    }
  }
  return success();
}

LogicalResult mlir::hip::verifyHipOpShape(
    Operation *op, function_ref<FailureOr<SmallVector<int64_t>>()> inferShape,
    unsigned initIndex) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return op->emitOpError(
        "shape verification requires DestinationStyleOpInterface");
  auto inits = dpsOp.getDpsInits();
  if (initIndex >= inits.size())
    return op->emitOpError("shape verification requested DPS init #")
           << initIndex << " but op has " << inits.size() << " init(s)";

  // The shape helper has already emitted a diagnostic on failure.
  FailureOr<SmallVector<int64_t>> expected = inferShape();
  if (failed(expected))
    return failure();

  auto initType = dyn_cast<ShapedType>(inits[initIndex].getType());
  if (!initType)
    return op->emitOpError("init operand is not a shaped type");
  ArrayRef<int64_t> actual = initType.getShape();
  if (actual.size() != expected->size())
    return op->emitOpError("rank mismatch on result: expected rank ")
           << expected->size() << " " << detail::formatShape(*expected)
           << " but outs has rank " << actual.size() << " "
           << detail::formatShape(actual);

  for (size_t d : llvm::seq<size_t>(0, actual.size())) {
    // kDynamic on either side is a wildcard.
    if (ShapedType::isDynamic(actual[d]) ||
        ShapedType::isDynamic((*expected)[d]))
      continue;
    if (actual[d] != (*expected)[d])
      return op->emitOpError("dim ")
             << d << " of result mismatch: expected " << (*expected)[d] << " "
             << detail::formatShape(*expected) << " but outs has " << actual[d]
             << " " << detail::formatShape(actual);
  }
  return success();
}

LogicalResult mlir::hip::verifyBroadcastDpsOp(Operation *op,
                                              ValueRange operands) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return op->emitOpError(
        "broadcast verification requires DestinationStyleOpInterface");
  SmallVector<Value> inits = dpsOp.getDpsInits();
  SmallVector<Value> dataOperands(operands.begin(), operands.end());
  llvm::append_range(dataOperands, inits);
  if (failed(verifyDpsComputeOp(op, dataOperands, /*numInits=*/1)))
    return failure();

  SmallVector<ArrayRef<int64_t>> shapes;
  shapes.reserve(operands.size());
  for (Value operand : operands)
    shapes.push_back(cast<ShapedType>(operand.getType()).getShape());
  return verifyHipOpShape(op, [&] {
    return inferBroadcastShape(
        shapes, [&]() { return op->emitOpError("broadcast verification: "); });
  });
}

OpFoldResult mlir::hip::reifyDimOrConstant(OpBuilder &b, Location loc,
                                           int64_t staticDim, Value source,
                                           int64_t sourceDim) {
  if (!ShapedType::isDynamic(staticDim))
    return b.getIndexAttr(staticDim);
  // Reify interface restricts callers to tensor results; if a memref
  // reify path is added later, also add a memref.dim branch + LIT case.
  return tensor::getMixedSize(b, loc, source, sourceDim);
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyElementwiseSameShape(OpBuilder &b, Location loc, Value source) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType)
    return failure();
  ArrayRef<int64_t> shape = sourceType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(shape.size());
  for (size_t i : llvm::seq<size_t>(0, shape.size()))
    dims.push_back(reifyDimOrConstant(b, loc, shape[i], source, i));
  return dims;
}

LogicalResult
mlir::hip::reifyElementwiseSameShapeFor(OpBuilder &b, Location loc,
                                        Value source, Operation *op,
                                        ReifiedRankedShapedTypeDims &reified) {
  if (op->getNumResults() == 0)
    return failure();
  FailureOr<SmallVector<OpFoldResult>> dims =
      reifyElementwiseSameShape(b, loc, source);
  if (failed(dims))
    return failure();
  reified.assign({std::move(*dims)});
  return success();
}

LogicalResult mlir::hip::verifySameShapeDpsOp(Operation *op, Value source,
                                              unsigned initIndex) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return op->emitOpError(
        "same-shape verification requires DestinationStyleOpInterface");

  // Generated same-shape verifiers have no leaf-specific operand list. Recover
  // every shaped DPS input here so non-source operands participate in the same
  // all-tensor-or-all-memref and result/init checks as dedicated verifiers.
  // Non-shaped DPS inputs such as !hip.context are intentionally ignored.
  SmallVector<Value> dataOperands;
  for (OpOperand *operand : dpsOp.getDpsInputOperands()) {
    Value value = operand->get();
    if (isa<RankedTensorType, MemRefType>(value.getType()))
      dataOperands.push_back(value);
  }
  SmallVector<Value> inits = dpsOp.getDpsInits();
  llvm::append_range(dataOperands, inits);
  if (failed(verifyDpsComputeOp(op, dataOperands, /*numInits=*/1)))
    return failure();

  auto sourceType = dyn_cast<ShapedType>(source.getType());
  if (!sourceType || !sourceType.hasRank())
    return op->emitOpError(
        "same-shape source operand must be a ranked shaped type");
  SmallVector<int64_t> expected(sourceType.getShape());
  return verifyHipOpShape(
      op, [&]() -> FailureOr<SmallVector<int64_t>> { return expected; },
      initIndex);
}

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyBroadcastResultShape(
    OpBuilder &b, Location loc, ValueRange operands,
    function_ref<InFlightDiagnostic()> emitError) {
  if (operands.empty()) {
    emitError() << "broadcast requires at least one operand";
    return failure();
  }

  SmallVector<ArrayRef<int64_t>> staticShapes;
  staticShapes.reserve(operands.size());
  for (Value operand : operands) {
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType) {
      emitError() << "broadcast operand must be a ranked tensor";
      return failure();
    }
    staticShapes.push_back(operandType.getShape());
  }
  // Validate broadcastability before emitting any `tensor.dim`, so a failure
  // leaves the IR unchanged (see the contract in HipShapeUtils.h).
  if (failed(mlir::hip::inferBroadcastShape(staticShapes, emitError)))
    return failure();

  SmallVector<SmallVector<OpFoldResult>> shapes;
  shapes.reserve(operands.size());
  for (size_t i : llvm::seq<size_t>(0, operands.size())) {
    Value operand = operands[i];
    bool reused = false;
    // Reuse the first mixed shape for repeated SSA operands (e.g. x*x) so
    // broadcastDim sees identical OpFoldResults and emits no redundant
    // tensor.dim/cmpi/select chain. Broadcast arity is normally 2-3, so a
    // linear scan is simpler than maintaining a side map.
    for (size_t j : llvm::seq<size_t>(0, i)) {
      if (operand == operands[j]) {
        shapes.push_back(shapes[j]);
        reused = true;
        break;
      }
    }
    if (reused)
      continue;
    shapes.push_back(tensor::getMixedSizes(b, loc, operand));
  }
  return detail::reifyBroadcastShape(b, loc, shapes, emitError);
}

bool mlir::hip::parseDenseIntElements(DenseElementsAttr dense,
                                      SmallVectorImpl<int64_t> &out,
                                      std::optional<int64_t> expectedRank) {
  out.clear();
  if (!dense)
    return false;
  auto tensorType = dyn_cast<RankedTensorType>(dense.getType());
  if (!tensorType || tensorType.getRank() > 1 ||
      (expectedRank && tensorType.getRank() != *expectedRank))
    return false;
  Type elementType = tensorType.getElementType();
  if (!elementType.isInteger(32) && !elementType.isInteger(64))
    return false;
  for (APInt value : dense.getValues<APInt>())
    out.push_back(value.getSExtValue());
  return true;
}

bool mlir::hip::matchConstantIntTensor(Value value,
                                       SmallVectorImpl<int64_t> &out,
                                       std::optional<int64_t> expectedRank) {
  out.clear();
  if (!value)
    return false;
  IntegerAttr intAttr;
  DenseIntElementsAttr denseAttr;
  if (matchPattern(value, m_Constant(&intAttr))) {
    if (expectedRank && *expectedRank != 0)
      return false;
    out.push_back(intAttr.getInt());
    return true;
  }
  if (matchPattern(value, m_Constant(&denseAttr)))
    return parseDenseIntElements(denseAttr, out, expectedRank);
  return false;
}

LogicalResult
mlir::hip::reifyBroadcastShapeFor(OpBuilder &b, Location loc,
                                  ValueRange operands, Operation *op,
                                  ReifiedRankedShapedTypeDims &reified) {
  if (op->getNumResults() == 0)
    return failure();
  for (Value v : operands)
    if (!isa<RankedTensorType>(v.getType()))
      return failure();
  FailureOr<SmallVector<OpFoldResult>> dims = reifyBroadcastResultShape(
      b, loc, operands, [&]() { return op->emitOpError(); });
  if (failed(dims))
    return failure();
  reified.assign({std::move(*dims)});
  return success();
}
