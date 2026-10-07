/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReshapeShapeFold.cpp - Materialize proven Reshape shapes -----------===//
//
// Consume a completed function-level ShapeProvenanceAnalysis. Accepted payloads
// are rebuilt from constants, canonical tensor dimensions, and proven host
// scalar SSA values, then stamped for the dynamic runtime-shaped Reshape
// fallback. That path revalidates every consumed marker claim before its first
// mutation. Earlier shape-independent/static rewrites do not consume markers
// and may bypass validation. Unknown or partially proven payloads remain
// untouched; a dynamic fallback that needs them uses synchronized readback.
//
//===----------------------------------------------------------------------===//

#include "ShapeProvenanceAnalysis.h"

#include "OnnxToHipUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <utility>

#define DEBUG_TYPE "reshape-shape-fold"

STATISTIC(NumReshapeShapeFolds,
          "Number of Reshape target shapes proven to be host-side");

namespace mlir {
namespace hip {

namespace {

mlir::FailureOr<mlir::Value>
materializeHostShape(mlir::IRRewriter &rewriter, mlir::Location loc,
                     llvm::ArrayRef<ShapeProvenanceExpr> expressions,
                     mlir::Type elementType) {
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType);
  if (!integerType)
    return mlir::failure();

  // Validate the complete plan before creating IR. A failed materialization
  // must not leave a partially constructed shape expression behind.
  for (const ShapeProvenanceExpr &expression : expressions) {
    if (expression.kind == ShapeProvenanceExpr::Kind::TensorDim) {
      auto shapedType =
          mlir::dyn_cast<mlir::ShapedType>(expression.value.getType());
      if (!shapedType || !shapedType.hasRank() || expression.dimension < 0 ||
          expression.dimension >= shapedType.getRank())
        return mlir::failure();
    }
    if (expression.kind == ShapeProvenanceExpr::Kind::HostScalar &&
        !expression.value.getType().isIntOrIndex())
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> elements;
  elements.reserve(expressions.size());
  for (const ShapeProvenanceExpr &expression : expressions) {
    mlir::Value element;
    switch (expression.kind) {
    case ShapeProvenanceExpr::Kind::Constant:
      element = mlir::arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(integerType, expression.constant));
      break;
    case ShapeProvenanceExpr::Kind::TensorDim: {
      mlir::Value dim = mlir::tensor::DimOp::create(
          rewriter, loc, expression.value, expression.dimension);
      element =
          mlir::arith::IndexCastOp::create(rewriter, loc, integerType, dim);
      break;
    }
    case ShapeProvenanceExpr::Kind::HostScalar:
      element = expression.value;
      if (element.getType() == integerType)
        break;
      if (element.getType().isIndex()) {
        element = mlir::arith::IndexCastOp::create(rewriter, loc, integerType,
                                                   element);
        break;
      }
      auto sourceType = mlir::dyn_cast<mlir::IntegerType>(element.getType());
      if (!sourceType)
        return mlir::failure();
      if (sourceType.getWidth() < integerType.getWidth())
        element =
            mlir::arith::ExtSIOp::create(rewriter, loc, integerType, element);
      else if (sourceType.getWidth() > integerType.getWidth())
        element =
            mlir::arith::TruncIOp::create(rewriter, loc, integerType, element);
      break;
    }
    elements.push_back(element);
  }

  auto shapeType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(elements.size())}, integerType);
  return mlir::tensor::FromElementsOp::create(rewriter, loc, shapeType,
                                              elements)
      .getResult();
}

struct ReshapeMaterializationPlan {
  mlir::Operation *reshapeOp;
  mlir::Type shapeElementType;
  llvm::SmallVector<ShapeProvenanceExpr> expressions;
  llvm::SmallVector<int64_t> inputDimMap;
  bool noMinusOne;
};

std::optional<ReshapeMaterializationPlan>
buildMaterializationPlan(mlir::Operation *reshapeOp,
                         const ShapeProvenanceAnalysis &analysis) {
  if (reshapeOp->getNumOperands() != 2 || reshapeOp->getNumResults() != 1 ||
      reshapeOp->hasAttr(kHostShapeOperandAttr))
    return std::nullopt;

  mlir::Value shapeOperand = reshapeOp->getOperand(1);
  auto shapeOperandType =
      mlir::dyn_cast<mlir::RankedTensorType>(shapeOperand.getType());
  if (!shapeOperandType || shapeOperandType.getRank() != 1 ||
      !shapeOperandType.getElementType().isInteger(64))
    return std::nullopt;
  // A literal target remains directly visible after the first lowering sweep
  // as a dense hip.constant carrier. Leave it untouched so Reshape conversion
  // can validate and materialize it without provenance markers.
  llvm::SmallVector<int64_t> constantTarget;
  if (extractConstantIntTensor(shapeOperand, constantTarget,
                               /*expectedRank=*/1))
    return std::nullopt;
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(reshapeOp->getResult(0).getType());
  if (!resultType)
    return std::nullopt;

  mlir::FailureOr<llvm::SmallVector<ShapeProvenanceExpr>> expressions =
      analysis.getPayload(shapeOperand);
  if (mlir::failed(expressions) ||
      expressions->size() != static_cast<size_t>(resultType.getRank()))
    return std::nullopt;

  unsigned minusOneCount =
      llvm::count_if(*expressions, [](const ShapeProvenanceExpr &expression) {
        return expression.proof == ShapeValueProof::MinusOne;
      });
  if (llvm::any_of(*expressions,
                   [](const ShapeProvenanceExpr &expression) {
                     return expression.proof == ShapeValueProof::Invalid;
                   }) ||
      minusOneCount > 1)
    return std::nullopt;
  if (llvm::any_of(*expressions, [](const ShapeProvenanceExpr &expression) {
        return expression.proof == ShapeValueProof::Unknown;
      }))
    return std::nullopt;

  int64_t allowzero = 0;
  if (auto attr = reshapeOp->getAttrOfType<mlir::IntegerAttr>("allowzero"))
    allowzero = attr.getSInt();
  if (allowzero != 0 && allowzero != 1)
    return std::nullopt;
  if (allowzero == 1 && minusOneCount != 0 &&
      llvm::any_of(*expressions, [](const ShapeProvenanceExpr &expression) {
        return expression.proof != ShapeValueProof::MinusOne &&
               expression.proof != ShapeValueProof::Positive;
      }))
    return std::nullopt;

  mlir::Value data = reshapeOp->getOperand(0);
  auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  if (!dataType)
    return std::nullopt;
  if (allowzero == 0) {
    for (auto [index, expression] : llvm::enumerate(*expressions)) {
      if (index < static_cast<size_t>(dataType.getRank()))
        continue;
      if (expression.kind != ShapeProvenanceExpr::Kind::Constant ||
          expression.constant <= 0)
        return std::nullopt;
    }
  }

  // Rewrite every proven equivalent dimension onto the actual Reshape data
  // operand. The marker's input-dimension map can then be revalidated
  // structurally at the conversion boundary without retaining or rerunning the
  // solver.
  llvm::SmallVector<int64_t> inputDimMap(expressions->size(), -1);
  for (auto [index, expression] : llvm::enumerate(*expressions)) {
    if (index >= static_cast<size_t>(dataType.getRank()) ||
        expression.kind != ShapeProvenanceExpr::Kind::TensorDim ||
        !analysis.isEquivalentToDimension(expression, data,
                                          static_cast<int64_t>(index)))
      continue;
    inputDimMap[index] = static_cast<int64_t>(index);
    expression.value = data;
    expression.dimension = static_cast<int64_t>(index);
  }

  bool noMinusOne =
      llvm::all_of(*expressions, [](const ShapeProvenanceExpr &expression) {
        return expression.proof == ShapeValueProof::Positive ||
               expression.proof == ShapeValueProof::NonNegative;
      });
  return ReshapeMaterializationPlan{
      reshapeOp, shapeOperandType.getElementType(), std::move(*expressions),
      std::move(inputDimMap), noMinusOne};
}

bool applyMaterializationPlan(const ReshapeMaterializationPlan &plan,
                              mlir::IRRewriter &rewriter) {
  mlir::Operation *reshapeOp = plan.reshapeOp;
  rewriter.setInsertionPoint(reshapeOp);
  mlir::FailureOr<mlir::Value> newShape = materializeHostShape(
      rewriter, reshapeOp->getLoc(), plan.expressions, plan.shapeElementType);
  if (mlir::failed(newShape))
    return false;

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] rewrote " << reshapeOp->getName()
                          << " shape operand from function-level dataflow over "
                          << plan.expressions.size() << " entries\n");

  rewriter.modifyOpInPlace(reshapeOp, [&] {
    reshapeOp->setOperand(1, *newShape);
    reshapeOp->setAttr(kHostShapeOperandAttr, rewriter.getUnitAttr());
    reshapeOp->setAttr(kHostShapeInputDimMapAttr,
                       rewriter.getDenseI64ArrayAttr(plan.inputDimMap));
    if (plan.noMinusOne)
      reshapeOp->setAttr(kHostShapeNoMinusOneAttr, rewriter.getUnitAttr());
  });
  ++NumReshapeShapeFolds;
  return true;
}

} // namespace

bool hasEligibleReshapeShapeProvenanceCandidate(mlir::func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](mlir::Operation *op) {
    if (found || op->getName().getStringRef() != "onnx.Reshape" ||
        op->hasAttr(kHostShapeOperandAttr) || op->getNumOperands() != 2 ||
        op->getNumResults() != 1)
      return;

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto shapeType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    if (!inputType || !outputType || !shapeType || inputType == outputType ||
        (inputType.hasStaticShape() && outputType.hasStaticShape()) ||
        shapeType.getRank() != 1 || !shapeType.getElementType().isInteger(64))
      return;
    if (!shapeType.isDynamicDim(0) &&
        shapeType.getDimSize(0) != outputType.getRank())
      return;
    found = true;
  });
  return found;
}

mlir::LogicalResult
materializeReshapeShapeOperands(mlir::func::FuncOp funcOp,
                                const ShapeProvenanceAnalysis &analysis) {
  llvm::SmallVector<mlir::Operation *> reshapes;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Reshape")
      reshapes.push_back(op);
  });

  // Query the complete analysis before mutating any analyzed IR. Plans contain
  // non-owning Value handles, but these mutations only insert scalar shape ops
  // and update Reshape operands; they do not erase referenced values.
  llvm::SmallVector<ReshapeMaterializationPlan> plans;
  for (mlir::Operation *reshape : reshapes)
    if (std::optional<ReshapeMaterializationPlan> plan =
            buildMaterializationPlan(reshape, analysis))
      plans.push_back(std::move(*plan));

  mlir::IRRewriter rewriter(funcOp.getContext());
  for (const ReshapeMaterializationPlan &plan : plans) {
    if (!applyMaterializationPlan(plan, rewriter)) {
      plan.reshapeOp->emitError(
          "failed to materialize proven Reshape target shape");
      return mlir::failure();
    }
  }
  return mlir::success();
}

} // namespace hip
} // namespace mlir
