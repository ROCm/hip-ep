/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <iterator>

namespace mlir {
namespace hip {
namespace {

// onnx.Einsum (binary, explicit equation) -> transpose / reshape + hip.matmul.
//
// A binary einsum with one contraction is a batched matmul once each operand
// is permuted so the shared batch letters lead, the free letters form M or N,
// and the contracted letters form K:
//
//   Before:
//     %y = "onnx.Einsum"(%a, %b) {equation = "bhwc,hkc->bhwk"}
//            : (tensor<2x3x4x5xf16>, tensor<3x6x5xf16>) -> tensor<2x3x4x6xf16>
//   After:
//     %at = hip.transpose %a {perm = [1, 0, 2, 3]}  -> tensor<3x2x4x5xf16>
//     %ac = tensor.collapse_shape %at [[0], [1, 2], [3]] -> tensor<3x8x5xf16>
//     %m = hip.matmul ins(%ac, %b) {transB = 1} -> tensor<3x8x6xf16>
//     %e = tensor.expand_shape %m [[0], [1, 2], [3]] -> tensor<3x2x4x6xf16>
//     %y = hip.transpose %e {perm = [1, 0, 2, 3]} -> tensor<2x3x4x6xf16>
//
// Letters that appear in only one operand and not in the output, a repeated
// letter, ellipsis, or an operand count other than two stay unconverted.
// Every extent must be static so a collapsed M/N/K product is a constant.

struct EinsumToMatmul : public mlir::RewritePattern {
  EinsumToMatmul(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Einsum", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

using Letters = llvm::SmallVector<char, 8>;

static bool sameLetters(llvm::ArrayRef<char> a, llvm::ArrayRef<char> b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

static bool hasLetter(llvm::ArrayRef<char> letters, char c) {
  return llvm::is_contained(letters, c);
}

static Letters concatLetters(llvm::ArrayRef<char> a, llvm::ArrayRef<char> b,
                             llvm::ArrayRef<char> c = {}) {
  Letters out;
  out.append(a.begin(), a.end());
  out.append(b.begin(), b.end());
  out.append(c.begin(), c.end());
  return out;
}

static mlir::LogicalResult parseTerm(llvm::StringRef term, Letters &letters,
                                     mlir::Operation *op,
                                     mlir::PatternRewriter &rewriter) {
  letters.clear();
  if (term.empty() || term.find('.') != llvm::StringRef::npos)
    return rewriter.notifyMatchFailure(
        op, "einsum term must be letters, not ellipsis");
  for (char c : term) {
    if (c < 'a' || c > 'z' || hasLetter(letters, c))
      return rewriter.notifyMatchFailure(
          op, "einsum term must be unique lowercase letters");
    letters.push_back(c);
  }
  return mlir::success();
}

mlir::LogicalResult
EinsumToMatmul::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "only binary einsum is lowered to matmul");

  auto equationAttr = op->getAttrOfType<mlir::StringAttr>("equation");
  if (!equationAttr)
    return rewriter.notifyMatchFailure(op, "missing equation");
  std::string equation = equationAttr.getValue().str();
  equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                 equation.end());
  size_t arrow = equation.find("->");
  if (arrow == std::string::npos)
    return rewriter.notifyMatchFailure(
        op, "implicit einsum equations are not lowered");
  llvm::StringRef lhs(equation.data(), arrow);
  llvm::StringRef rhs(equation.data() + arrow + 2, equation.size() - arrow - 2);
  size_t comma = lhs.find(',');
  if (comma == llvm::StringRef::npos ||
      lhs.find(',', comma + 1) != llvm::StringRef::npos)
    return rewriter.notifyMatchFailure(op,
                                       "expected exactly two einsum inputs");

  Letters aLetters, bLetters, outLetters;
  if (mlir::failed(parseTerm(lhs.take_front(comma), aLetters, op, rewriter)) ||
      mlir::failed(
          parseTerm(lhs.drop_front(comma + 1), bLetters, op, rewriter)) ||
      mlir::failed(parseTerm(rhs, outLetters, op, rewriter)))
    return mlir::failure();

  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
  auto bType = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
  auto outType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!aType || !bType || !outType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensors");
  if (aType.getRank() != static_cast<int64_t>(aLetters.size()) ||
      bType.getRank() != static_cast<int64_t>(bLetters.size()) ||
      outType.getRank() != static_cast<int64_t>(outLetters.size()))
    return rewriter.notifyMatchFailure(
        op, "equation rank does not match the tensor");

  mlir::Type elem = aType.getElementType();
  if (elem != bType.getElementType() || elem != outType.getElementType() ||
      !mlir::isa<mlir::FloatType>(elem))
    return rewriter.notifyMatchFailure(
        op, "einsum matmul lowering requires a matching float element type");

  llvm::DenseMap<char, int64_t> extent;
  auto record = [&](llvm::ArrayRef<char> letters,
                    mlir::RankedTensorType type) -> mlir::LogicalResult {
    for (auto [axis, letter] : llvm::enumerate(letters)) {
      int64_t dim = type.getDimSize(axis);
      if (mlir::ShapedType::isDynamic(dim))
        return rewriter.notifyMatchFailure(
            op, "dynamic einsum extents are not lowered");
      auto it = extent.find(letter);
      if (it == extent.end())
        extent[letter] = dim;
      else if (it->second != dim)
        return rewriter.notifyMatchFailure(op,
                                           "einsum letter extents disagree");
    }
    return mlir::success();
  };
  if (mlir::failed(record(aLetters, aType)) ||
      mlir::failed(record(bLetters, bType)) ||
      mlir::failed(record(outLetters, outType)))
    return mlir::failure();

  Letters batch, mDims, nDims, kDims;
  for (char letter : outLetters) {
    bool inA = hasLetter(aLetters, letter);
    bool inB = hasLetter(bLetters, letter);
    if (inA && inB)
      batch.push_back(letter);
    else if (inA)
      mDims.push_back(letter);
    else if (inB)
      nDims.push_back(letter);
    else
      return rewriter.notifyMatchFailure(
          op, "output letter is not present on an input");
  }
  for (char letter : aLetters) {
    if (hasLetter(outLetters, letter))
      continue;
    if (!hasLetter(bLetters, letter))
      return rewriter.notifyMatchFailure(
          op, "einsum reduction of a single operand is not a matmul");
    kDims.push_back(letter);
  }
  for (char letter : bLetters) {
    if (!hasLetter(outLetters, letter) && !hasLetter(aLetters, letter))
      return rewriter.notifyMatchFailure(
          op, "einsum reduction of a single operand is not a matmul");
  }
  if (mDims.empty() || nDims.empty() || kDims.empty())
    return rewriter.notifyMatchFailure(
        op, "einsum is not a batched matmul with M, N, and K");

  auto productOf = [&](llvm::ArrayRef<char> letters,
                       int64_t &product) -> mlir::LogicalResult {
    product = 1;
    for (char letter : letters) {
      int64_t next = 0;
      if (llvm::MulOverflow(product, extent.lookup(letter), next))
        return rewriter.notifyMatchFailure(op,
                                           "collapsed einsum dim overflows");
      product = next;
    }
    return mlir::success();
  };
  int64_t mProd = 0, nProd = 0, kProd = 0;
  if (mlir::failed(productOf(mDims, mProd)) ||
      mlir::failed(productOf(nDims, nProd)) ||
      mlir::failed(productOf(kDims, kProd)))
    return mlir::failure();

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();

  auto shapeOf = [&](llvm::ArrayRef<char> letters) {
    llvm::SmallVector<int64_t> shape;
    for (char letter : letters)
      shape.push_back(extent.lookup(letter));
    return shape;
  };

  auto permuteTo = [&](mlir::Value input, llvm::ArrayRef<char> from,
                       llvm::ArrayRef<char> to) -> mlir::Value {
    if (sameLetters(from, to))
      return input;
    llvm::SmallVector<int64_t> perm;
    for (char letter : to)
      perm.push_back(static_cast<int64_t>(
          std::distance(from.begin(), llvm::find(from, letter))));
    auto resultType = mlir::RankedTensorType::get(shapeOf(to), elem);
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), elem, mlir::ValueRange{});
    return mlir::hip::TransposeOp::create(rewriter, loc, context, input, init,
                                          rewriter.getI64ArrayAttr(perm))
        ->getResult(0);
  };

  // Collapse the two trailing groups. Leading batch letters stay separate so
  // hip.matmul's batch count is their product on both operands.
  auto collapseTail = [&](mlir::Value input, llvm::ArrayRef<char> leading,
                          llvm::ArrayRef<char> group1,
                          llvm::ArrayRef<char> group2, int64_t group1Size,
                          int64_t group2Size) -> mlir::Value {
    if (group1.size() == 1 && group2.size() == 1)
      return input;
    llvm::SmallVector<mlir::ReassociationIndices> reassoc;
    int64_t axis = 0;
    for (size_t i = 0, e = leading.size(); i < e; ++i)
      reassoc.push_back({axis++});
    mlir::ReassociationIndices first, second;
    for (size_t i = 0, e = group1.size(); i < e; ++i)
      first.push_back(axis++);
    for (size_t i = 0, e = group2.size(); i < e; ++i)
      second.push_back(axis++);
    reassoc.push_back(std::move(first));
    reassoc.push_back(std::move(second));
    llvm::SmallVector<int64_t> shape = shapeOf(leading);
    shape.push_back(group1Size);
    shape.push_back(group2Size);
    auto resultType = mlir::RankedTensorType::get(shape, elem);
    return mlir::tensor::CollapseShapeOp::create(rewriter, loc, resultType,
                                                 input, reassoc)
        .getResult();
  };

  Letters aDesired = concatLetters(batch, mDims, kDims);
  mlir::Value aLayout = permuteTo(a, aLetters, aDesired);
  aLayout = collapseTail(aLayout, batch, mDims, kDims, mProd, kProd);

  Letters bAsKN = concatLetters(batch, kDims, nDims);
  Letters bAsNK = concatLetters(batch, nDims, kDims);
  int64_t transB = sameLetters(bLetters, bAsNK) ? 1 : 0;
  Letters bDesired = transB ? bAsNK : bAsKN;
  llvm::ArrayRef<char> bGroup1 =
      transB ? llvm::ArrayRef<char>(nDims) : llvm::ArrayRef<char>(kDims);
  llvm::ArrayRef<char> bGroup2 =
      transB ? llvm::ArrayRef<char>(kDims) : llvm::ArrayRef<char>(nDims);
  int64_t bGroup1Size = transB ? nProd : kProd;
  int64_t bGroup2Size = transB ? kProd : nProd;
  mlir::Value bLayout = permuteTo(b, bLetters, bDesired);
  bLayout =
      collapseTail(bLayout, batch, bGroup1, bGroup2, bGroup1Size, bGroup2Size);

  llvm::SmallVector<int64_t> matmulShape = shapeOf(batch);
  matmulShape.push_back(mProd);
  matmulShape.push_back(nProd);
  auto matmulType = mlir::RankedTensorType::get(matmulShape, elem);
  mlir::Value matmulInit = mlir::tensor::EmptyOp::create(
      rewriter, loc, matmulType.getShape(), elem, mlir::ValueRange{});
  llvm::SmallVector<mlir::NamedAttribute> attrs = {
      rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)),
      rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(transB))};
  llvm::SmallVector<mlir::Value> operands = {context, aLayout, bLayout,
                                             matmulInit};
  mlir::Value matmul =
      mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs).getResult(0);

  mlir::Value expanded = matmul;
  if (mDims.size() != 1 || nDims.size() != 1) {
    llvm::SmallVector<mlir::ReassociationIndices> reassoc;
    int64_t axis = 0;
    for (size_t i = 0, e = batch.size(); i < e; ++i)
      reassoc.push_back({axis++});
    mlir::ReassociationIndices mGroup, nGroup;
    for (size_t i = 0, e = mDims.size(); i < e; ++i)
      mGroup.push_back(axis++);
    for (size_t i = 0, e = nDims.size(); i < e; ++i)
      nGroup.push_back(axis++);
    reassoc.push_back(std::move(mGroup));
    reassoc.push_back(std::move(nGroup));
    Letters expandedLetters = concatLetters(batch, mDims, nDims);
    llvm::SmallVector<mlir::OpFoldResult> outShape;
    for (int64_t dim : shapeOf(expandedLetters))
      outShape.push_back(rewriter.getIndexAttr(dim));
    auto expandedType =
        mlir::RankedTensorType::get(shapeOf(expandedLetters), elem);
    expanded = mlir::tensor::ExpandShapeOp::create(rewriter, loc, expandedType,
                                                   matmul, reassoc, outShape)
                   .getResult();
  }

  Letters resultLetters = concatLetters(batch, mDims, nDims);
  mlir::Value result = permuteTo(expanded, resultLetters, outLetters);
  rewriter.replaceOp(op, result);
  return mlir::success();
}

} // namespace

void populateEinsumConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<EinsumToMatmul>(ctx);
}

} // namespace hip
} // namespace mlir
