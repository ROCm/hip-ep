/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GroupNormDecompositionFusion.cpp - Reshape+InstanceNorm -> GroupNorm
//===//
//
// Exporters without a GroupNormalization op (ONNX opset < 18, and later
// opsets on some paths) spell group normalization of an NCHW tensor as
//
//   Before (rooted on onnx.InstanceNormalization):
//     %g   = onnx.Constant dense<[0, G, -1]>        : tensor<3xi64>
//     %r   = onnx.Reshape(%x, %g)                   : tensor<NxGxLxT>
//     %n   = onnx.InstanceNormalization(%r, %ones, %zeros)   // length G
//     %s   = onnx.Shape(%x)                         : tensor<Rxi64>
//     %b   = onnx.Reshape(%n, %s)                   : tensor<NxCx...xT>
//     %m   = onnx.Mul(%b, %gamma)                   // gamma [C,1,..] (opt.)
//     %a   = onnx.Add(%m, %beta)                    // beta  [C,1,..] (opt.)
//     %sg  = onnx.Sigmoid(%a)                       // SiLU tail (opt.)
//     %y   = onnx.Mul(%a, %sg)
//
//   After:
//     %gc  = onnx.Constant / onnx.Reshape(%gamma) : tensor<CxT>
//     %bc  = onnx.Constant / onnx.Reshape(%beta)  : tensor<CxT>
//     %y   = onnx.Custom(%x, %gc, %bc) <{function_name = "GroupNorm"}>
//              {domain_name = "com.microsoft", groups = G, activation = 1,
//               channels_last = 0, epsilon = <InstanceNormalization's>}
//
// The InstanceNormalization then normalizes each (n, group) slice over all
// of its channels and pixels, which is GroupNorm. The match proves that from
// the reshape operands, not from the types, so dynamic batch and spatial
// extents are fine:
//
//   * the first shape is the constant [0, G, -1] (allowzero 0): axis 0 keeps
//     N and the last axis takes the rest; a static N or the static product
//     may stand in for 0 or -1;
//   * the second shape is Shape(%x), its ReshapeShapeFold form
//     from_elements(index_cast(tensor.dim %x, i) or the static extent), or a
//     constant equal to %x's static shape (0 allowed for axis 0 only);
//   * the InstanceNormalization scale and bias are constant splats of 1 and
//     0; any other per-group affine would have to be folded into gamma/beta;
//   * gamma and beta must broadcast per channel: shape [C,1,...] of rank R-1
//     or [1,C,1,...] of rank R, with %x's element type.
//
// Every intermediate must have one use. The channel extent must be static
// and divisible by G. Anything else is left for the per-op conversions.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "group-norm-decomposition-fusion"

STATISTIC(NumGroupNormFused,
          "Number of Reshape + InstanceNormalization chains fused to "
          "com.microsoft GroupNorm");

namespace mlir {
namespace hip {

namespace {

bool isOnnxOp(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

mlir::DenseElementsAttr inlineConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!isOnnxOp(def, "onnx.Constant"))
    return {};
  return def->getAttrOfType<mlir::DenseElementsAttr>("value");
}

std::optional<llvm::SmallVector<int64_t>> intConstant(mlir::Value v) {
  mlir::DenseElementsAttr dense = inlineConstant(v);
  if (!dense || !mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return std::nullopt;
  llvm::SmallVector<int64_t> values;
  for (const llvm::APInt &value : dense.getValues<llvm::APInt>())
    values.push_back(value.getSExtValue());
  return values;
}

bool isFloatSplat(mlir::Value v, double expected) {
  mlir::DenseElementsAttr dense = inlineConstant(v);
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::FloatType>(dense.getElementType()))
    return false;
  return dense.getSplatValue<llvm::APFloat>().convertToDouble() == expected;
}

bool allowsZeroCopy(mlir::Operation *reshape) {
  auto allowZero = reshape->getAttrOfType<mlir::IntegerAttr>("allowzero");
  return !allowZero || allowZero.getInt() == 0;
}

bool hasSingleUser(mlir::Value v, mlir::Operation *user) {
  return v.hasOneUse() && *v.getUsers().begin() == user;
}

// The one user of `v`, when `v` has exactly one use.
mlir::Operation *soleUser(mlir::Value v) {
  return v.hasOneUse() ? *v.getUsers().begin() : nullptr;
}

// Element `i` of the second reshape's shape is %x's extent on axis i.
bool isExtentOf(mlir::Value element, mlir::Value x, int64_t axis,
                mlir::RankedTensorType xType) {
  if (auto cast = element.getDefiningOp<mlir::arith::IndexCastOp>()) {
    auto dim = cast.getIn().getDefiningOp<mlir::tensor::DimOp>();
    if (!dim || dim.getSource() != x)
      return false;
    std::optional<int64_t> index = dim.getConstantIndex();
    return index && *index == axis;
  }
  if (xType.isDynamicDim(axis))
    return false;
  llvm::APInt value;
  return mlir::matchPattern(element, mlir::m_ConstantInt(&value)) &&
         value.getSExtValue() == xType.getDimSize(axis);
}

// The second reshape restores exactly %x's shape.
bool restoresShapeOf(mlir::Operation *reshape, mlir::Value x,
                     mlir::RankedTensorType xType) {
  mlir::Value shape = reshape->getOperand(1);
  int64_t rank = xType.getRank();
  mlir::Operation *def = shape.getDefiningOp();
  if (isOnnxOp(def, "onnx.Shape")) {
    if (def->getOperand(0) != x)
      return false;
    auto start = def->getAttrOfType<mlir::IntegerAttr>("start");
    auto end = def->getAttrOfType<mlir::IntegerAttr>("end");
    return (!start || start.getInt() == 0) && (!end || end.getInt() >= rank);
  }
  if (auto fromElements =
          mlir::dyn_cast_or_null<mlir::tensor::FromElementsOp>(def)) {
    if (static_cast<int64_t>(fromElements.getElements().size()) != rank)
      return false;
    for (int64_t axis : llvm::seq<int64_t>(0, rank))
      if (!isExtentOf(fromElements.getElements()[axis], x, axis, xType))
        return false;
    return true;
  }
  std::optional<llvm::SmallVector<int64_t>> values = intConstant(shape);
  if (!values || static_cast<int64_t>(values->size()) != rank)
    return false;
  for (int64_t axis : llvm::seq<int64_t>(0, rank)) {
    int64_t value = (*values)[axis];
    // 0 copies the InstanceNormalization input's extent, which is N only on
    // axis 0.
    if (axis == 0 && value == 0 && allowsZeroCopy(reshape))
      continue;
    if (xType.isDynamicDim(axis) || value != xType.getDimSize(axis))
      return false;
  }
  return true;
}

// G when the first reshape is [0, G, -1] (or the static equivalents) on %x.
std::optional<int64_t> groupsOf(mlir::Operation *reshape,
                                mlir::RankedTensorType xType) {
  std::optional<llvm::SmallVector<int64_t>> values =
      intConstant(reshape->getOperand(1));
  if (!values || values->size() != 3)
    return std::nullopt;
  int64_t batch = (*values)[0];
  int64_t groups = (*values)[1];
  int64_t rest = (*values)[2];
  int64_t channels = xType.getDimSize(1);
  if (groups <= 0 || channels % groups != 0)
    return std::nullopt;
  bool keepsBatch = (batch == 0 && allowsZeroCopy(reshape)) ||
                    (!xType.isDynamicDim(0) && batch == xType.getDimSize(0));
  if (!keepsBatch)
    return std::nullopt;
  if (rest == -1)
    return groups;
  int64_t perBatch = 1;
  for (int64_t axis : llvm::seq<int64_t>(1, xType.getRank())) {
    if (xType.isDynamicDim(axis))
      return std::nullopt;
    perBatch *= xType.getDimSize(axis);
  }
  if (rest * groups != perBatch)
    return std::nullopt;
  return groups;
}

// `v` is ranked like %x, with %x's element type and no extent that
// contradicts %x's.
bool isShapedLike(mlir::Value v, mlir::RankedTensorType xType) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!type || type.getElementType() != xType.getElementType())
    return false;
  return mlir::succeeded(mlir::verifyCompatibleShape(type, xType));
}

// The constant operand of a commutative binary op whose other operand is
// `input`, or null.
mlir::Value otherOperand(mlir::Operation *op, mlir::Value input) {
  if (!op || op->getNumOperands() != 2)
    return {};
  if (op->getOperand(0) == input)
    return op->getOperand(1);
  if (op->getOperand(1) == input)
    return op->getOperand(0);
  return {};
}

// `v` broadcasts as one value per channel of %x: [C,1,...] of rank R-1 or
// [1,C,1,...] of rank R.
bool isPerChannel(mlir::Value v, mlir::RankedTensorType xType) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!type || !type.hasStaticShape() ||
      type.getElementType() != xType.getElementType())
    return false;
  int64_t rank = xType.getRank();
  int64_t lead = rank - type.getRank();
  if (lead != 0 && lead != 1)
    return false;
  for (int64_t i : llvm::seq<int64_t>(0, type.getRank())) {
    int64_t axis = i + lead;
    int64_t want = axis == 1 ? xType.getDimSize(1) : 1;
    if (type.getDimSize(i) != want)
      return false;
  }
  return true;
}

struct GroupNormDecompositionFusion : public mlir::RewritePattern {
  GroupNormDecompositionFusion(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.InstanceNormalization", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *norm,
                  mlir::PatternRewriter &rewriter) const override {
    if (norm->getNumOperands() != 3 || norm->getNumResults() != 1)
      return rewriter.notifyMatchFailure(norm, "operand count");
    mlir::Operation *grouped = norm->getOperand(0).getDefiningOp();
    if (!isOnnxOp(grouped, "onnx.Reshape") ||
        !hasSingleUser(grouped->getResult(0), norm))
      return rewriter.notifyMatchFailure(norm, "input is not a 1-use Reshape");
    mlir::Value x = grouped->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType || xType.getRank() < 3 || xType.isDynamicDim(1) ||
        !mlir::isa<mlir::FloatType>(xType.getElementType()))
      return rewriter.notifyMatchFailure(norm, "x needs rank >= 3, static C");
    int64_t channels = xType.getDimSize(1);

    std::optional<int64_t> groupsOrNone = groupsOf(grouped, xType);
    if (!groupsOrNone)
      return rewriter.notifyMatchFailure(norm, "not Reshape(x, [0, G, -1])");
    int64_t groups = *groupsOrNone;
    if (!isFloatSplat(norm->getOperand(1), 1.0) ||
        !isFloatSplat(norm->getOperand(2), 0.0))
      return rewriter.notifyMatchFailure(norm, "scale/bias are not 1 and 0");

    mlir::Operation *restored = soleUser(norm->getResult(0));
    if (!isOnnxOp(restored, "onnx.Reshape") ||
        restored->getOperand(0) != norm->getResult(0) ||
        !isShapedLike(restored->getResult(0), xType) ||
        !restoresShapeOf(restored, x, xType))
      return rewriter.notifyMatchFailure(norm, "not Reshape(_, Shape(x))");

    // Optional per-channel Mul, then Add, then SiLU.
    mlir::Value current = restored->getResult(0);
    mlir::Value gamma, beta;
    llvm::SmallVector<mlir::Operation *> chain{grouped, norm, restored};
    auto takeAffine = [&](llvm::StringRef name, mlir::Value &operand) {
      mlir::Operation *user = soleUser(current);
      if (!isOnnxOp(user, name) || !isShapedLike(user->getResult(0), xType))
        return;
      mlir::Value other = otherOperand(user, current);
      if (!other || other == current || !isPerChannel(other, xType))
        return;
      operand = other;
      chain.push_back(user);
      current = user->getResult(0);
    };
    takeAffine("onnx.Mul", gamma);
    takeAffine("onnx.Add", beta);

    int64_t activation = 0;
    if (!current.use_empty() && llvm::range_size(current.getUsers()) == 2) {
      mlir::Operation *sigmoid = nullptr, *product = nullptr;
      for (mlir::Operation *user : current.getUsers()) {
        if (isOnnxOp(user, "onnx.Sigmoid"))
          sigmoid = user;
        else if (isOnnxOp(user, "onnx.Mul"))
          product = user;
      }
      if (sigmoid && product && sigmoid != product &&
          hasSingleUser(sigmoid->getResult(0), product) &&
          otherOperand(product, current) == sigmoid->getResult(0) &&
          isShapedLike(product->getResult(0), xType)) {
        chain.push_back(sigmoid);
        chain.push_back(product);
        current = product->getResult(0);
        activation = 1;
      }
    }
    mlir::Operation *last = current.getDefiningOp();

    // The ONNX default for InstanceNormalization's optional epsilon.
    float epsilon = 1e-5f;
    if (auto attr = norm->getAttrOfType<mlir::FloatAttr>("epsilon"))
      epsilon = static_cast<float>(attr.getValueAsDouble());

    mlir::Location loc = last->getLoc();
    rewriter.setInsertionPoint(last);
    auto vectorType =
        mlir::RankedTensorType::get({channels}, xType.getElementType());
    auto makeConstant = [&](mlir::DenseElementsAttr value) {
      mlir::OperationState state(loc, "onnx.Constant");
      state.addTypes(value.getType());
      state.addAttribute("value", value);
      return rewriter.create(state)->getResult(0);
    };
    // A per-channel tensor as the 1-D [C] operand GroupNorm takes, or a
    // splat of `fill` when the chain has no such Mul/Add.
    auto asVector = [&](mlir::Value perChannel, double fill) -> mlir::Value {
      if (!perChannel)
        return makeConstant(mlir::DenseElementsAttr::get(
            vectorType, rewriter.getFloatAttr(xType.getElementType(), fill)));
      if (mlir::DenseElementsAttr dense = inlineConstant(perChannel))
        return makeConstant(dense.reshape(vectorType));
      mlir::Value shape = makeConstant(mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({1}, rewriter.getI64Type()),
          llvm::ArrayRef<int64_t>{channels}));
      mlir::OperationState state(loc, "onnx.Reshape");
      state.addOperands({perChannel, shape});
      state.addTypes(vectorType);
      return rewriter.create(state)->getResult(0);
    };
    mlir::Value gammaVector = asVector(gamma, 1.0);
    mlir::Value betaVector = asVector(beta, 0.0);

    auto si64 = mlir::IntegerType::get(rewriter.getContext(), 64,
                                       mlir::IntegerType::Signed);
    mlir::OperationState state(loc, "onnx.Custom");
    state.addOperands({x, gammaVector, betaVector});
    state.addTypes(last->getResult(0).getType());
    state.addAttribute("function_name", rewriter.getStringAttr("GroupNorm"));
    state.addAttribute("domain_name", rewriter.getStringAttr("com.microsoft"));
    state.addAttribute("groups", rewriter.getIntegerAttr(si64, groups));
    state.addAttribute("activation", rewriter.getIntegerAttr(si64, activation));
    state.addAttribute("channels_last", rewriter.getIntegerAttr(si64, 0));
    state.addAttribute("epsilon", rewriter.getF32FloatAttr(epsilon));
    if (auto outputs = last->getAttrOfType<mlir::ArrayAttr>("node.outputs"))
      state.addAttribute("node.outputs", outputs);
    mlir::Operation *fused = rewriter.create(state);

    rewriter.replaceOp(last, fused->getResult(0));
    // Erase the rest of the chain from the end; each now has no users.
    // Shape and constant producers are left to the global onnx.* DCE.
    for (mlir::Operation *op : llvm::reverse(chain))
      if (op != last && op->use_empty())
        rewriter.eraseOp(op);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused G=" << groups << " C=" << channels
               << " silu=" << activation << " at " << loc << "\n");
    ++NumGroupNormFused;
    return mlir::success();
  }
};

} // namespace

void populateGroupNormDecompositionFusionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<GroupNormDecompositionFusion>(ctx);
}

} // namespace hip
} // namespace mlir
