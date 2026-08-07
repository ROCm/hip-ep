/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ShapeProvenanceAnalysis.cpp - Host shape dataflow ------------------===//

#include "ShapeProvenanceAnalysis.h"

#include "OnnxToHipUtils.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <optional>
#include <utility>

namespace mlir {
namespace hip {

namespace {

struct ShapeExprNode {
  ShapeProvenanceExpr::Kind kind;
  int64_t constant;
  mlir::Value value;
  int64_t dimension;
  ShapeValueProof proof;
};

using Expr = const ShapeExprNode *;
using ExprVector = llvm::SmallVector<Expr>;

/// Monotonic sparse-dataflow fact for one SSA value.
///
/// `initialized == false` is lattice bottom. A missing optional is an unknown
/// vector; a null entry in `dimensions` is one unknown axis after a CFG join.
/// Payload proofs are intentionally all-or-nothing because partially reading a
/// shape tensor would mix host SSA with synchronized device payload semantics.
struct ShapeFact {
  bool initialized = false;
  std::optional<ExprVector> dimensions;
  std::optional<ExprVector> payload;

  static ShapeFact join(const ShapeFact &lhs, const ShapeFact &rhs) {
    if (!lhs.initialized)
      return rhs;
    if (!rhs.initialized)
      return lhs;

    ShapeFact result;
    result.initialized = true;
    if (lhs.dimensions && rhs.dimensions &&
        lhs.dimensions->size() == rhs.dimensions->size()) {
      ExprVector dimensions;
      dimensions.reserve(lhs.dimensions->size());
      for (auto [lhsExpr, rhsExpr] :
           llvm::zip(*lhs.dimensions, *rhs.dimensions))
        dimensions.push_back(lhsExpr == rhsExpr ? lhsExpr : nullptr);
      result.dimensions = std::move(dimensions);
    }
    if (lhs.payload && rhs.payload && *lhs.payload == *rhs.payload)
      result.payload = lhs.payload;
    return result;
  }

  bool operator==(const ShapeFact &other) const {
    return initialized == other.initialized && dimensions == other.dimensions &&
           payload == other.payload;
  }

  void print(llvm::raw_ostream &os) const {
    if (!initialized) {
      os << "uninitialized";
      return;
    }
    os << "dims=";
    printExprVector(os, dimensions);
    os << ", payload=";
    printExprVector(os, payload);
  }

private:
  static void printExprVector(llvm::raw_ostream &os,
                              const std::optional<ExprVector> &expressions) {
    if (!expressions) {
      os << "unknown";
      return;
    }
    os << '[';
    llvm::interleaveComma(*expressions, os, [&](Expr expr) {
      if (!expr) {
        os << '?';
        return;
      }
      switch (expr->kind) {
      case ShapeProvenanceExpr::Kind::Constant:
        os << expr->constant;
        break;
      case ShapeProvenanceExpr::Kind::TensorDim:
        os << "dim(" << expr->value << ", " << expr->dimension << ')';
        break;
      case ShapeProvenanceExpr::Kind::HostScalar:
        os << "host(" << expr->value << ')';
        break;
      }
    });
    os << ']';
  }
};

using ShapeLattice = mlir::dataflow::Lattice<ShapeFact>;

enum class TransferKind {
  Shape,
  Gather,
  Slice,
  Concat,
  Identity,
  Unsqueeze,
  Reshape,
  Constant,
  Add,
  Cast,
  MatMul,
};

struct TransferRegistration {
  llvm::StringLiteral name;
  TransferKind kind;
};

/// Keep generic ONNX operation-name dispatch in one table. Transfer functions
/// remain separate below, so adding a producer cannot silently change the
/// conservative fallback for unrelated operations.
///
/// TODO(shape-provenance): If the project adopts standard onnx-mlir, replace
/// name-keyed dispatch with typed operations or external models and reuse ONNX
/// shape helpers for rank-1 payload semantics. ReifyRankedShapedTypeOpInterface
/// exposes shaped-result extents, while ValueBoundsOpInterface can reason about
/// dimensions and scalar/index bounds; neither by itself transports the integer
/// entries of a shape tensor. The Shape dialect can carry such payloads when
/// they are represented as explicit shape values. Retire a local transfer only
/// when one of those typed/helper or explicit-shape contracts represents the
/// same payload fact.
static constexpr TransferRegistration kTransferRegistry[] = {
    {"onnx.Shape", TransferKind::Shape},
    {"onnx.Gather", TransferKind::Gather},
    {"onnx.Slice", TransferKind::Slice},
    {"onnx.Concat", TransferKind::Concat},
    {"onnx.Identity", TransferKind::Identity},
    {"onnx.Unsqueeze", TransferKind::Unsqueeze},
    {"onnx.Reshape", TransferKind::Reshape},
    {"onnx.Constant", TransferKind::Constant},
    {"onnx.Add", TransferKind::Add},
    {"onnx.Cast", TransferKind::Cast},
    {"onnx.CastLike", TransferKind::Cast},
    {"onnx.MatMul", TransferKind::MatMul},
};

std::optional<TransferKind> lookupTransferKind(llvm::StringRef name) {
  for (const TransferRegistration &registration : kTransferRegistry)
    if (registration.name == name)
      return registration.kind;
  return std::nullopt;
}

ShapeValueProof getConstantProof(int64_t value) {
  if (value > 0)
    return ShapeValueProof::Positive;
  if (value == 0)
    return ShapeValueProof::NonNegative;
  if (value == -1)
    return ShapeValueProof::MinusOne;
  return ShapeValueProof::Invalid;
}

int64_t normalizeAndClampIndex(int64_t index, int64_t length) {
  assert(length >= 0 && "shape-vector length must be nonnegative");
  if (index < 0) {
    if (index < -length)
      return 0;
    index += length;
  }
  return std::clamp(index, int64_t(0), length);
}

bool isScalarIntegerOrIndex(mlir::Type type) {
  return !mlir::isa<mlir::ShapedType>(type) && type.isIntOrIndex();
}

} // namespace

class ShapeProvenanceAnalysis::Impl {
public:
  explicit Impl(mlir::func::FuncOp funcOp)
      : funcOp(funcOp),
        solver(mlir::DataFlowConfig().setInterprocedural(false)) {}

  mlir::LogicalResult run() {
    mlir::dataflow::loadBaselineAnalyses(solver);
    solver.load<DataFlow>(*this);
    return solver.initializeAndRun(funcOp);
  }

  mlir::FailureOr<llvm::SmallVector<ShapeProvenanceExpr>>
  getPayload(mlir::Value value) const {
    const ShapeFact *fact = lookupFact(value);
    if (!fact || !fact->payload)
      return mlir::failure();

    llvm::SmallVector<ShapeProvenanceExpr> result;
    result.reserve(fact->payload->size());
    for (Expr expr : *fact->payload) {
      if (!expr)
        return mlir::failure();
      result.push_back({expr->kind, expr->constant, expr->value,
                        expr->dimension, expr->proof});
    }
    return result;
  }

  bool isEquivalentToDimension(const ShapeProvenanceExpr &expr,
                               mlir::Value value, int64_t dimension) const {
    if (expr.kind != ShapeProvenanceExpr::Kind::TensorDim)
      return false;
    Expr lhs = getDimensionExpr(expr.value, expr.dimension);
    Expr rhs = getDimensionExpr(value, dimension);
    return lhs && lhs == rhs;
  }

private:
  class DataFlow final
      : public mlir::dataflow::SparseForwardDataFlowAnalysis<ShapeLattice> {
  public:
    DataFlow(mlir::DataFlowSolver &solver, Impl &impl)
        : SparseForwardDataFlowAnalysis(solver), impl(impl) {}

    mlir::LogicalResult
    visitOperation(mlir::Operation *op,
                   llvm::ArrayRef<const ShapeLattice *> operands,
                   llvm::ArrayRef<ShapeLattice *> results) override {
      if (llvm::any_of(operands, [](const ShapeLattice *operand) {
            return !operand->getValue().initialized;
          }))
        return mlir::success();

      llvm::SmallVector<ShapeFact> resultFacts =
          impl.transfer(op, operands, results.size());
      assert(resultFacts.size() == results.size());
      for (auto [result, fact] : llvm::zip(results, resultFacts))
        propagateIfChanged(result, result->join(fact));
      return mlir::success();
    }

    void setToEntryState(ShapeLattice *lattice) override {
      propagateIfChanged(lattice, lattice->join(impl.getConservativeFact(
                                      lattice->getAnchor())));
    }

  private:
    Impl &impl;
  };

  Expr getConstant(int64_t value) {
    auto key = std::make_pair(value, getConstantProof(value));
    auto [it, inserted] = constants.try_emplace(key, nullptr);
    if (inserted)
      it->second = createExpr(ShapeProvenanceExpr::Kind::Constant, value, {}, 0,
                              key.second);
    return it->second;
  }

  Expr getTensorDim(mlir::Value value, int64_t dimension) {
    auto key = std::make_pair(value, dimension);
    auto [it, inserted] = tensorDims.try_emplace(key, nullptr);
    if (inserted)
      it->second = createExpr(ShapeProvenanceExpr::Kind::TensorDim, 0, value,
                              dimension, ShapeValueProof::NonNegative);
    return it->second;
  }

  Expr getHostScalar(mlir::Value value, ShapeValueProof proof) {
    auto key = std::make_pair(value, proof);
    auto [it, inserted] = hostScalars.try_emplace(key, nullptr);
    if (inserted)
      it->second =
          createExpr(ShapeProvenanceExpr::Kind::HostScalar, 0, value, 0, proof);
    return it->second;
  }

  Expr createExpr(ShapeProvenanceExpr::Kind kind, int64_t constant,
                  mlir::Value value, int64_t dimension, ShapeValueProof proof) {
    expressions.push_back({kind, constant, value, dimension, proof});
    return &expressions.back();
  }

  ShapeFact getConservativeFact(mlir::Value value) {
    ShapeFact fact;
    fact.initialized = true;
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!type)
      return fact;

    ExprVector dimensions;
    dimensions.reserve(type.getRank());
    for (int64_t dimension : llvm::seq<int64_t>(type.getRank())) {
      dimensions.push_back(type.isDynamicDim(dimension)
                               ? getTensorDim(value, dimension)
                               : getConstant(type.getDimSize(dimension)));
    }
    fact.dimensions = std::move(dimensions);
    return fact;
  }

  const ShapeFact *lookupFact(mlir::Value value) const {
    const ShapeLattice *lattice = solver.lookupState<ShapeLattice>(value);
    if (!lattice || !lattice->getValue().initialized)
      return nullptr;
    return &lattice->getValue();
  }

  Expr getDimensionExpr(mlir::Value value, int64_t dimension) const {
    const ShapeFact *fact = lookupFact(value);
    if (!fact || !fact->dimensions || dimension < 0 ||
        dimension >= static_cast<int64_t>(fact->dimensions->size()))
      return nullptr;
    return (*fact->dimensions)[dimension];
  }

  static const ShapeFact &
  getOperandFact(llvm::ArrayRef<const ShapeLattice *> operands,
                 unsigned index) {
    return operands[index]->getValue();
  }

  llvm::SmallVector<ShapeFact>
  transfer(mlir::Operation *op, llvm::ArrayRef<const ShapeLattice *> operands,
           size_t resultCount) {
    llvm::SmallVector<ShapeFact> facts;
    facts.reserve(resultCount);
    for (mlir::Value result : op->getResults())
      facts.push_back(getConservativeFact(result));
    if (facts.empty())
      return facts;

    if (transferStandardTensorOperation(op, operands, facts))
      return facts;
    if (transferScalarOperation(op, operands, facts))
      return facts;
    if (mlir::isa<mlir::arith::ConstantOp>(op)) {
      transferConstant(op, facts.front());
      return facts;
    }
    if (auto fromElements = mlir::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
      transferFromElements(fromElements, operands, facts.front());
      return facts;
    }

    std::optional<TransferKind> kind =
        lookupTransferKind(op->getName().getStringRef());
    if (!kind)
      return facts;

    switch (*kind) {
    case TransferKind::Shape:
      transferShape(op, operands, facts.front());
      break;
    case TransferKind::Gather:
      transferGather(op, operands, facts.front());
      break;
    case TransferKind::Slice:
      transferSlice(op, operands, facts.front());
      break;
    case TransferKind::Concat:
      transferConcat(op, operands, facts.front());
      break;
    case TransferKind::Identity:
      transferValuePreserving(op, operands, facts.front(),
                              /*preservePayload=*/true,
                              /*preserveDimensions=*/true);
      break;
    case TransferKind::Unsqueeze:
    case TransferKind::Reshape:
      transferValuePreserving(op, operands, facts.front(),
                              /*preservePayload=*/true,
                              /*preserveDimensions=*/false);
      break;
    case TransferKind::Constant:
      transferConstant(op, facts.front());
      break;
    case TransferKind::Add:
      transferAdd(op, operands, facts.front());
      break;
    case TransferKind::Cast:
      transferValuePreserving(op, operands, facts.front(),
                              /*preservePayload=*/false,
                              /*preserveDimensions=*/true);
      break;
    case TransferKind::MatMul:
      transferMatMul(op, operands, facts.front());
      break;
    }
    return facts;
  }

  /// Propagate dimensions through standard tensor descriptor operations when
  /// the mapping is exact. Collapse groups with multiple non-unit expressions
  /// remain rooted at the result because representing products would require a
  /// symbolic arithmetic expression, not a dimension identity.
  bool
  transferStandardTensorOperation(mlir::Operation *op,
                                  llvm::ArrayRef<const ShapeLattice *> operands,
                                  llvm::SmallVectorImpl<ShapeFact> &facts) {
    if (mlir::isa<mlir::tensor::CastOp>(op)) {
      transferValuePreserving(op, operands, facts.front(),
                              /*preservePayload=*/false,
                              /*preserveDimensions=*/true);
      return true;
    }

    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op)) {
      if (!facts.front().dimensions)
        return true;
      llvm::SmallVector<mlir::OpFoldResult> mixedShape =
          expand.getMixedOutputShape();
      for (auto [dimension, mixedSize] : llvm::enumerate(mixedShape)) {
        if (std::optional<int64_t> constant =
                mlir::getConstantIntValue(mixedSize)) {
          (*facts.front().dimensions)[dimension] = getConstant(*constant);
          continue;
        }
        mlir::Value value = llvm::cast<mlir::Value>(mixedSize);
        auto operandIt = llvm::find(op->getOperands(), value);
        if (operandIt == op->getOperands().end())
          continue;
        unsigned operandIndex =
            static_cast<unsigned>(operandIt - op->getOperands().begin());
        const auto &payload = getOperandFact(operands, operandIndex).payload;
        if (payload && payload->size() == 1 && payload->front())
          (*facts.front().dimensions)[dimension] = payload->front();
      }
      return true;
    }

    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op)) {
      const auto &sourceDimensions = getOperandFact(operands, 0).dimensions;
      if (!sourceDimensions || !facts.front().dimensions)
        return true;
      for (auto [resultDim, group] :
           llvm::enumerate(collapse.getReassociationIndices())) {
        Expr source = nullptr;
        bool exact = true;
        for (int64_t sourceDim : group) {
          Expr candidate = (*sourceDimensions)[sourceDim];
          if (!candidate) {
            exact = false;
            break;
          }
          if (candidate->kind == ShapeProvenanceExpr::Kind::Constant &&
              candidate->constant == 1)
            continue;
          if (source) {
            exact = false;
            break;
          }
          source = candidate;
        }
        if (exact && source)
          (*facts.front().dimensions)[resultDim] = source;
      }
      return true;
    }

    return false;
  }

  bool transferScalarOperation(mlir::Operation *op,
                               llvm::ArrayRef<const ShapeLattice *> operands,
                               llvm::SmallVectorImpl<ShapeFact> &facts) {
    if (op->getNumResults() != 1 ||
        !isScalarIntegerOrIndex(op->getResult(0).getType()))
      return false;

    mlir::Value result = op->getResult(0);
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
      if (integer)
        if (std::optional<int64_t> value = integer.getValue().trySExtValue())
          facts.front().payload = ExprVector{getConstant(*value)};
      return true;
    }

    if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(op)) {
      if (std::optional<int64_t> index = dim.getConstantIndex()) {
        Expr expr = nullptr;
        const ShapeFact &sourceFact = getOperandFact(operands, 0);
        if (sourceFact.dimensions && *index >= 0 &&
            *index < static_cast<int64_t>(sourceFact.dimensions->size()))
          expr = (*sourceFact.dimensions)[*index];
        if (expr)
          facts.front().payload = ExprVector{expr};
      } else {
        facts.front().payload =
            ExprVector{getHostScalar(result, ShapeValueProof::NonNegative)};
      }
      return true;
    }

    if (!mlir::isa<mlir::arith::IndexCastOp, mlir::arith::ExtSIOp,
                   mlir::arith::TruncIOp, mlir::arith::AddIOp,
                   mlir::arith::SubIOp, mlir::arith::MulIOp,
                   mlir::arith::DivSIOp, mlir::arith::DivUIOp,
                   mlir::arith::CmpIOp, mlir::arith::SelectOp,
                   mlir::arith::AndIOp, mlir::arith::OrIOp>(op))
      return false;

    if (llvm::any_of(operands, [](const ShapeLattice *operand) {
          const auto &payload = operand->getValue().payload;
          return !payload || payload->size() != 1 || !payload->front();
        }))
      return true;

    if (auto indexCast = mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
      if (isLosslessShapeIndexCast(indexCast))
        facts.front().payload = getOperandFact(operands, 0).payload;
      return true;
    }

    ShapeValueProof proof = ShapeValueProof::Unknown;
    if (mlir::isa<mlir::arith::ExtSIOp>(op)) {
      ShapeValueProof operandProof =
          getOperandFact(operands, 0).payload->front()->proof;
      if (operandProof == ShapeValueProof::Positive ||
          operandProof == ShapeValueProof::NonNegative)
        proof = operandProof;
    }
    facts.front().payload = ExprVector{getHostScalar(result, proof)};
    return true;
  }

  void transferFromElements(mlir::tensor::FromElementsOp fromElements,
                            llvm::ArrayRef<const ShapeLattice *> operands,
                            ShapeFact &result) {
    if (fromElements.getElements().size() != operands.size())
      return;
    ExprVector payload;
    payload.reserve(operands.size());
    for (const ShapeLattice *operand : operands) {
      const auto &operandPayload = operand->getValue().payload;
      if (!operandPayload || operandPayload->size() != 1 ||
          !operandPayload->front())
        return;
      payload.push_back(operandPayload->front());
    }
    result.payload = std::move(payload);
  }

  void transferConstant(mlir::Operation *op, ShapeFact &result) {
    llvm::SmallVector<int64_t> constants;
    if (!extractConstantIntTensor(op->getResult(0), constants,
                                  /*expectedRank=*/std::nullopt,
                                  CompileTimeConstantScope::InlineOnly))
      return;
    ExprVector payload;
    payload.reserve(constants.size());
    for (int64_t constant : constants)
      payload.push_back(getConstant(constant));
    result.payload = std::move(payload);
  }

  void transferShape(mlir::Operation *op,
                     llvm::ArrayRef<const ShapeLattice *> operands,
                     ShapeFact &result) {
    if (op->getNumOperands() != 1)
      return;
    const ShapeFact &source = getOperandFact(operands, 0);
    if (!source.dimensions)
      return;

    int64_t rank = static_cast<int64_t>(source.dimensions->size());
    int64_t start = 0;
    int64_t end = rank;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("start"))
      start = attr.getSInt();
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("end"))
      end = attr.getSInt();
    start = normalizeAndClampIndex(start, rank);
    end = normalizeAndClampIndex(end, rank);
    if (start >= end)
      return;

    ExprVector payload(source.dimensions->begin() + start,
                       source.dimensions->begin() + end);
    if (llvm::is_contained(payload, nullptr))
      return;
    result.payload = std::move(payload);
  }

  void transferConcat(mlir::Operation *op,
                      llvm::ArrayRef<const ShapeLattice *> operands,
                      ShapeFact &result) {
    int64_t axis = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = attr.getSInt();
    if (axis != 0)
      return;

    ExprVector payload;
    for (const ShapeLattice *operand : operands) {
      const auto &operandPayload = operand->getValue().payload;
      if (!operandPayload)
        return;
      llvm::append_range(payload, *operandPayload);
    }
    result.payload = std::move(payload);
  }

  void transferGather(mlir::Operation *op,
                      llvm::ArrayRef<const ShapeLattice *> operands,
                      ShapeFact &result) {
    if (op->getNumOperands() != 2)
      return;
    auto dataType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!dataType || dataType.getRank() != 1)
      return;
    int64_t axis = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = attr.getSInt();
    if (axis != 0)
      return;

    llvm::SmallVector<int64_t> indices;
    if (!extractConstantIntTensor(op->getOperand(1), indices,
                                  /*expectedRank=*/std::nullopt,
                                  CompileTimeConstantScope::InlineOnly) ||
        indices.size() != 1)
      return;
    const auto &source = getOperandFact(operands, 0).payload;
    if (!source || source->empty())
      return;
    int64_t index = indices.front();
    if (index < 0)
      index += static_cast<int64_t>(source->size());
    if (index < 0 || index >= static_cast<int64_t>(source->size()))
      return;
    result.payload = ExprVector{(*source)[index]};
  }

  void transferSlice(mlir::Operation *op,
                     llvm::ArrayRef<const ShapeLattice *> operands,
                     ShapeFact &result) {
    if (op->getNumOperands() < 3 || op->getNumOperands() > 5)
      return;
    auto dataType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!dataType || dataType.getRank() != 1)
      return;
    const auto &source = getOperandFact(operands, 0).payload;
    if (!source)
      return;

    llvm::SmallVector<int64_t> starts, ends, axes, steps;
    if (!extractConstantIntTensor(op->getOperand(1), starts,
                                  /*expectedRank=*/std::nullopt,
                                  CompileTimeConstantScope::InlineOnly) ||
        !extractConstantIntTensor(op->getOperand(2), ends,
                                  /*expectedRank=*/std::nullopt,
                                  CompileTimeConstantScope::InlineOnly) ||
        starts.size() != 1 || ends.size() != 1)
      return;
    if (op->getNumOperands() >= 4 &&
        !extractConstantIntTensor(op->getOperand(3), axes,
                                  /*expectedRank=*/std::nullopt,
                                  CompileTimeConstantScope::InlineOnly))
      return;
    if (op->getNumOperands() == 5 &&
        !extractConstantIntTensor(op->getOperand(4), steps,
                                  /*expectedRank=*/std::nullopt,
                                  CompileTimeConstantScope::InlineOnly))
      return;
    if (axes.size() > 1 || steps.size() > 1)
      return;
    int64_t axis = axes.empty() ? 0 : axes.front();
    int64_t step = steps.empty() ? 1 : steps.front();
    if ((axis != 0 && axis != -1) || step != 1)
      return;

    int64_t length = static_cast<int64_t>(source->size());
    int64_t start = starts.front();
    int64_t end = ends.front();
    start = normalizeAndClampIndex(start, length);
    end = normalizeAndClampIndex(end, length);
    if (end < start)
      end = start;
    result.payload = ExprVector(source->begin() + start, source->begin() + end);
  }

  void transferValuePreserving(mlir::Operation *op,
                               llvm::ArrayRef<const ShapeLattice *> operands,
                               ShapeFact &result, bool preservePayload,
                               bool preserveDimensions) {
    if (op->getNumOperands() < 1)
      return;
    const ShapeFact &source = getOperandFact(operands, 0);
    if (preservePayload)
      result.payload = source.payload;
    if (preserveDimensions && source.dimensions && result.dimensions &&
        source.dimensions->size() == result.dimensions->size())
      result.dimensions = source.dimensions;
  }

  void transferAdd(mlir::Operation *op,
                   llvm::ArrayRef<const ShapeLattice *> operands,
                   ShapeFact &result) {
    if (op->getNumOperands() != 2 || !result.dimensions)
      return;
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return;

    for (int64_t dimension : llvm::seq<int64_t>(resultType.getRank())) {
      if (!resultType.isDynamicDim(dimension))
        continue;
      Expr source = nullptr;
      bool ambiguous = false;
      for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
        auto operandType =
            mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
        if (!operandType) {
          ambiguous = true;
          break;
        }
        int64_t pad = resultType.getRank() - operandType.getRank();
        if (dimension < pad)
          continue;
        int64_t operandDim = dimension - pad;
        if (operandType.getDimSize(operandDim) == 1)
          continue;
        const auto &operandDimensions =
            getOperandFact(operands, operandIndex).dimensions;
        Expr candidate =
            operandDimensions ? (*operandDimensions)[operandDim] : nullptr;
        if (!candidate || (source && source != candidate)) {
          ambiguous = true;
          break;
        }
        source = candidate;
      }
      if (source && !ambiguous)
        (*result.dimensions)[dimension] = source;
    }
  }

  void transferMatMul(mlir::Operation *op,
                      llvm::ArrayRef<const ShapeLattice *> operands,
                      ShapeFact &result) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1 ||
        operands.size() != 2 || !result.dimensions)
      return;
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto aType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    if (!resultType || !aType || !bType || resultType.getRank() < 2 ||
        aType.getRank() < 2 || bType.getRank() < 2)
      return;

    int64_t resultRank = resultType.getRank();
    int64_t resultBatchRank = resultRank - 2;
    int64_t aBatchRank = aType.getRank() - 2;
    int64_t bBatchRank = bType.getRank() - 2;
    if (aBatchRank > resultBatchRank || bBatchRank > resultBatchRank ||
        result.dimensions->size() != static_cast<size_t>(resultRank))
      return;

    auto getOperandDimension = [&](unsigned operandIndex,
                                   int64_t operandDimension) -> Expr {
      if (operandIndex >= operands.size() || operandDimension < 0)
        return nullptr;
      const auto &dimensions =
          getOperandFact(operands, operandIndex).dimensions;
      if (!dimensions ||
          operandDimension >= static_cast<int64_t>(dimensions->size()))
        return nullptr;
      return (*dimensions)[operandDimension];
    };

    for (int64_t dimension : llvm::seq<int64_t>(resultRank)) {
      if (!resultType.isDynamicDim(dimension))
        continue;
      Expr source = nullptr;
      bool ambiguous = false;
      if (dimension == resultRank - 2) {
        source = getOperandDimension(0, aType.getRank() - 2);
      } else if (dimension == resultRank - 1) {
        source = getOperandDimension(1, bType.getRank() - 1);
      } else {
        auto mergeBatchDimension = [&](Expr candidate) {
          if (!candidate || (source && source != candidate)) {
            ambiguous = true;
            return;
          }
          source = candidate;
        };

        int64_t aPad = resultBatchRank - aBatchRank;
        int64_t bPad = resultBatchRank - bBatchRank;
        if (dimension >= aPad) {
          int64_t aDim = dimension - aPad;
          if (aType.getDimSize(aDim) != 1)
            mergeBatchDimension(getOperandDimension(0, aDim));
        }
        if (!ambiguous && dimension >= bPad) {
          int64_t bDim = dimension - bPad;
          if (bType.getDimSize(bDim) != 1)
            mergeBatchDimension(getOperandDimension(1, bDim));
        }
      }
      if (source && !ambiguous)
        (*result.dimensions)[dimension] = source;
    }
  }

  mlir::func::FuncOp funcOp;
  mlir::DataFlowSolver solver;
  // Lattice facts compare interned expression pointers. deque preserves their
  // addresses as new expressions are discovered by the solver worklist.
  std::deque<ShapeExprNode> expressions;
  llvm::DenseMap<std::pair<int64_t, ShapeValueProof>, Expr> constants;
  llvm::DenseMap<std::pair<mlir::Value, int64_t>, Expr> tensorDims;
  llvm::DenseMap<std::pair<mlir::Value, ShapeValueProof>, Expr> hostScalars;
};

ShapeProvenanceAnalysis::ShapeProvenanceAnalysis(mlir::func::FuncOp funcOp)
    : impl(std::make_unique<Impl>(funcOp)) {}

ShapeProvenanceAnalysis::~ShapeProvenanceAnalysis() = default;

mlir::LogicalResult ShapeProvenanceAnalysis::run() { return impl->run(); }

mlir::FailureOr<llvm::SmallVector<ShapeProvenanceExpr>>
ShapeProvenanceAnalysis::getPayload(mlir::Value value) const {
  return impl->getPayload(value);
}

bool ShapeProvenanceAnalysis::isEquivalentToDimension(
    const ShapeProvenanceExpr &expr, mlir::Value value,
    int64_t dimension) const {
  return impl->isEquivalentToDimension(expr, value, dimension);
}

} // namespace hip
} // namespace mlir
