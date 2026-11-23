/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MatMulIntegerConversion.cpp - onnx.MatMulInteger -> hip.matmul ----===//
//
// ONNX MatMulInteger: Y = (A - a_zero_point) @ (B - b_zero_point), result i32.
// A and B are int8 or uint8 and multiply like numpy.matmul. a_zero_point is a
// scalar or a length-M vector (one value per row of A). b_zero_point is a
// scalar or a length-N vector (one value per column of B). Either zero point
// may be omitted, which means 0.
//
// hip.matmul of f32 lowers through hipBLASLt and through tosa.matmul. That
// accumulator is an exact int32 dot product only while every partial sum fits
// in the 24-bit mantissa. An 8-bit value minus an 8-bit zero point is at most
// 255 in magnitude, so each product is at most 65025 and K must be at most
// 258. A longer K casts to i32 and
// accumulates there instead. ONNX allows the int32 accumulator to wrap.
//
// A size-1 batch on A is expanded to the result batch: hip.matmul counts
// batches from A and does not broadcast that 1. A rank-2 B is one shared
// matrix and is left that way. Copying it once per batch is a large extra
// buffer, and both hip.matmul (batch stride 0) and tosa.matmul (A's batch
// folded into M) already reuse it. B is expanded only when it has batch
// dimensions of its own.
//
// hip.sub left-pads unequal ranks into the 4D broadcast ABI, so a 1-D zero
// point would land on the last axis. That is correct for B's columns (N)
// and wrong for A's rows (M), which sit on the second-to-last axis. A
// length-M vector is reshaped to [..., M, 1] and a length-N vector to
// [..., 1, N] before the subtract.
//
//   Before:
//     %y = "onnx.MatMulInteger"(%a, %b, %azp, %bzp)
//         : (tensor<MxKxui8>, tensor<KxNxui8>, ...) -> tensor<MxNxi32>
//
//   After:
//     %af = hip.cast %a to f32
//     %bf = hip.cast %b to f32
//     %ad = hip.sub %af, reshape(%azp, [M, 1])
//     %bd = hip.sub %bf, reshape(%bzp, [1, N])
//     %acc = hip.matmul %ad, %bd
//     %y = hip.cast %acc to i32
//
// Rank is 2..4. hip.sub rejects rank > 4, and hip.matmul rejects rank < 2,
// so the numpy rank-1 promotion is left unconverted.
//
// On the rocMLIR pipeline (conv-integer-quant-conv) a fully static
// MatMulInteger in @main_graph's entry block, with a rank-2 B, is lowered to
// an int8 hip.quant_matmul instead. See tryLowerQuantMatmul.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Traits.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {
namespace {

static bool isEightBit(mlir::Type type) { return type.isInteger(8); }

/// True when `255 * 255 * reductionLength` is an exact f32 integer.
static bool eightBitDotIsExactInF32(int64_t reductionLength) {
  constexpr int64_t kMaxAbsProduct = 255 * 255;
  constexpr int64_t kMantissaLimit = int64_t{1} << 24;
  return reductionLength >= 0 &&
         reductionLength <= kMantissaLimit / kMaxAbsProduct;
}

static mlir::Value realOperand(mlir::Operation *op, unsigned index) {
  if (index >= op->getNumOperands())
    return {};
  mlir::Value value = op->getOperand(index);
  if (!value || mlir::isa<mlir::NoneType>(value.getType()))
    return {};
  return value;
}

/// Scalar, or a vector whose length is \p extent (M for A, N for B), or a
/// tensor of \p dataType's rank that already broadcasts onto it.
static bool zeroPointShapeOk(mlir::RankedTensorType zpType,
                             mlir::RankedTensorType dataType, int64_t extent) {
  if (!zpType.hasStaticShape())
    return false;
  if (zpType.getRank() == 0)
    return true;
  if (zpType.getRank() == 1 &&
      (zpType.getDimSize(0) == 1 || zpType.getDimSize(0) == extent))
    return true;
  if (zpType.getRank() != dataType.getRank())
    return false;
  for (int64_t i : llvm::seq<int64_t>(zpType.getRank())) {
    int64_t zpDim = zpType.getDimSize(i);
    int64_t dataDim = dataType.getDimSize(i);
    if (zpDim != 1 && zpDim != dataDim)
      return false;
  }
  return true;
}

static mlir::Value reshapeTo(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value src,
                             mlir::RankedTensorType dstType) {
  if (src.getType() == dstType)
    return src;
  llvm::SmallVector<mlir::Value> dims;
  for (int64_t dim : dstType.getShape())
    dims.push_back(mlir::arith::ConstantIndexOp::create(rewriter, loc, dim));
  auto shapeType =
      mlir::RankedTensorType::get({dstType.getRank()}, rewriter.getIndexType());
  mlir::Value shape =
      mlir::tensor::FromElementsOp::create(rewriter, loc, shapeType, dims);
  return mlir::tensor::ReshapeOp::create(rewriter, loc, dstType, src, shape)
      .getResult();
}

/// Put \p zp on \p axis of \p dataType. A length-M / length-N vector becomes
/// rank(data) with that axis set and every other axis 1. A single element
/// becomes a rank-0 scalar, which the 4D elementwise ABI left-pads to
/// 1x1x1x1.
static mlir::Value alignZeroPoint(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value zp,
                                  mlir::RankedTensorType dataType,
                                  int64_t axis) {
  auto zpType = mlir::cast<mlir::RankedTensorType>(zp.getType());
  if (zpType.getRank() == dataType.getRank())
    return zp;
  if (zpType.getRank() == 1 && zpType.getDimSize(0) > 1) {
    llvm::SmallVector<int64_t> shape(dataType.getRank(), 1);
    shape[axis] = zpType.getDimSize(0);
    auto expanded = mlir::RankedTensorType::get(shape, zpType.getElementType());
    return reshapeTo(rewriter, loc, zp, expanded);
  }
  auto scalar = mlir::RankedTensorType::get({}, zpType.getElementType());
  return reshapeTo(rewriter, loc, zp, scalar);
}

static mlir::Value castTo(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value context, mlir::Value input,
                          mlir::Type elemType, int64_t onnxTo) {
  auto srcType = mlir::cast<mlir::RankedTensorType>(input.getType());
  if (srcType.getElementType() == elemType)
    return input;
  auto dstType = mlir::RankedTensorType::get(srcType.getShape(), elemType);
  mlir::Value init = createEmptyTensor(rewriter, loc, dstType, input);
  auto casted = mlir::hip::CastOp::create(rewriter, loc, context, input, init,
                                          rewriter.getI64IntegerAttr(onnxTo));
  return casted->getResult(0);
}

/// Broadcast \p value to \p dstType. Equal shapes are left alone. hip.expand
/// materializes the copies hip.matmul and tosa.matmul will not.
static mlir::Value expandTo(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value context, mlir::Value value,
                            mlir::RankedTensorType dstType) {
  if (value.getType() == dstType)
    return value;
  auto shapeType =
      mlir::RankedTensorType::get({dstType.getRank()}, rewriter.getI64Type());
  auto shapeAttr = mlir::DenseElementsAttr::get(shapeType, dstType.getShape());
  mlir::Value shape =
      mlir::arith::ConstantOp::create(rewriter, loc, shapeType, shapeAttr);
  mlir::Value init = createEmptyTensor(rewriter, loc, dstType, value);
  return mlir::hip::ExpandOp::create(rewriter, loc, context, value, shape, init)
      ->getResult(0);
}

static mlir::Value subtractZeroPoint(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value context,
                                     mlir::Value data, mlir::Value zp) {
  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, dataType, data);
  return mlir::hip::SubOp::create(rewriter, loc, context, data, zp, init)
      ->getResult(0);
}

/// hip.quant_matmul takes signless i8 for B. uint8 is shifted by 128 into that
/// range and a signed int8 tensor is the same bits under a different type.
static mlir::Value toSigned8(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value context,
                             mlir::Value value) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  mlir::Type elem = type.getElementType();
  if (elem.isSignlessInteger(8))
    return value;
  if (elem.isSignedInteger(8)) {
    auto dstType =
        mlir::RankedTensorType::get(type.getShape(), rewriter.getI8Type());
    return mlir::tensor::BitcastOp::create(rewriter, loc, dstType, value)
        .getResult();
  }
  mlir::Value wide = castTo(rewriter, loc, context, value,
                            rewriter.getI32Type(), /*onnxTo=*/6);
  auto offsetType = mlir::RankedTensorType::get({}, rewriter.getI32Type());
  mlir::Value offset = mlir::hip::ConstantOp::create(
      rewriter, loc, offsetType,
      mlir::DenseElementsAttr::get(offsetType,
                                   rewriter.getI32IntegerAttr(128)));
  mlir::Value shifted = subtractZeroPoint(rewriter, loc, context, wide, offset);
  return castTo(rewriter, loc, context, shifted, rewriter.getI8Type(),
                /*onnxTo=*/3);
}

/// Splat constant of \p type.
static mlir::Value splatConstant(mlir::PatternRewriter &rewriter,
                                 mlir::Location loc,
                                 mlir::RankedTensorType type, int64_t value) {
  return mlir::hip::ConstantOp::create(
      rewriter, loc, type,
      mlir::DenseElementsAttr::get(
          type, rewriter.getIntegerAttr(type.getElementType(), value)));
}

/// Elementwise \p OpT with numpy broadcasting between \p lhs and \p rhs.
template <typename OpT>
static mlir::Value emitBroadcastBinary(mlir::PatternRewriter &rewriter,
                                       mlir::Location loc, mlir::Value context,
                                       mlir::Value lhs, mlir::Value rhs) {
  auto lhsType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(rhs.getType());
  llvm::SmallVector<int64_t> shape;
  if (!mlir::OpTrait::util::getBroadcastedShape(lhsType.getShape(),
                                                rhsType.getShape(), shape))
    return {};
  auto type = mlir::RankedTensorType::get(shape, lhsType.getElementType());
  mlir::Value init = createEmptyTensor(rewriter, loc, type, lhs);
  return OpT::create(rewriter, loc, context, lhs, rhs, init)->getResult(0);
}

/// A zero point as an i32 tensor with the rebase \p offset already removed: a
/// rank-0 scalar, or [1, N] for a per-column vector. Returns null for any
/// other zero point shape.
static mlir::Value rebasedZeroPoint(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value context,
                                    mlir::Value zp, bool allowPerColumn,
                                    int64_t offset) {
  auto zpType = mlir::cast<mlir::RankedTensorType>(zp.getType());
  if (!zpType.hasStaticShape())
    return {};
  mlir::RankedTensorType shaped;
  if (zpType.getNumElements() == 1)
    shaped = mlir::RankedTensorType::get({}, zpType.getElementType());
  else if (allowPerColumn && zpType.getRank() == 1)
    shaped = mlir::RankedTensorType::get({1, zpType.getDimSize(0)},
                                         zpType.getElementType());
  else
    return {};
  mlir::Value value = reshapeTo(rewriter, loc, zp, shaped);
  value = castTo(rewriter, loc, context, value, rewriter.getI32Type(),
                 /*onnxTo=*/6);
  if (offset == 0)
    return value;
  auto valueType = mlir::cast<mlir::RankedTensorType>(value.getType());
  auto offsetType = mlir::RankedTensorType::get({}, rewriter.getI32Type());
  mlir::Value offsetValue = splatConstant(rewriter, loc, offsetType, offset);
  mlir::Value init = createEmptyTensor(rewriter, loc, valueType, value);
  return mlir::hip::SubOp::create(rewriter, loc, context, value, offsetValue,
                                  init)
      ->getResult(0);
}

/// Static MatMulInteger as an int8 hip.quant_matmul. Returns failure when the
/// matmul cannot be formed, and the caller keeps the cast/sub/hip.matmul
/// lowering.
///
/// Let a' = a - offA and b' = b - offB, where the offset is 128 for uint8 and
/// 0 for int8; the quant op applies those offsets. With za = a_zp - offA and
/// zb = b_zp - offB:
///
///   sum_k (a - a_zp)(b - b_zp)
///     = qmm(a, b') - za * colsum(b') - zb * rowsum(a') + za * zb * K
///
/// colsum(b') is qmm(ones[1, K], b') and rowsum(a') is qmm(a, ones[K, 1]). The
/// correction terms are elementwise i32 ops that rocMLIR fuses into the
/// quantized GEMM epilogue. A zero point equal to its offset has no term.
static mlir::LogicalResult
tryLowerQuantMatmul(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                    mlir::Value context, mlir::Location loc,
                    mlir::RankedTensorType aType, mlir::RankedTensorType bType,
                    mlir::RankedTensorType resultType) {
  if (!inMainGraphEntry(op))
    return mlir::failure();
  // hip.quant_matmul is [M, K] x [K, N] with a signless i32 accumulator.
  if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
      !resultType.hasStaticShape() || bType.getRank() != 2 ||
      !resultType.getElementType().isSignlessInteger(32))
    return mlir::failure();

  const int64_t rank = aType.getRank();
  const int64_t k = bType.getDimSize(0);
  const int64_t n = bType.getDimSize(1);
  if (aType.getDimSize(rank - 1) != k)
    return mlir::failure();
  int64_t m = 1;
  for (int64_t i : llvm::seq<int64_t>(rank - 1))
    m *= aType.getDimSize(i);

  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  mlir::Value aZp = realOperand(op, 2);
  mlir::Value bZp = realOperand(op, 3);
  const bool aUnsigned = aType.getElementType().isUnsignedInteger(8);
  const bool bUnsigned = bType.getElementType().isUnsignedInteger(8);
  const int64_t offA = aUnsigned ? 128 : 0;
  const int64_t offB = bUnsigned ? 128 : 0;

  // Reject zero-point shapes the correction terms cannot express before any op
  // is created. a_zero_point must be a single value, because a per-row vector
  // would need its own M-wide term. b_zero_point may also be per column.
  auto singleValue = [](mlir::Value zp) {
    auto type = mlir::cast<mlir::RankedTensorType>(zp.getType());
    return type.hasStaticShape() && type.getNumElements() == 1;
  };
  if (aZp && !singleValue(aZp))
    return mlir::failure();
  if (bZp && !singleValue(bZp)) {
    auto type = mlir::cast<mlir::RankedTensorType>(bZp.getType());
    if (!type.hasStaticShape() || type.getRank() != 1 ||
        type.getDimSize(0) != n)
      return mlir::failure();
  }

  // An omitted uint8 zero point is 0, which is -128 once rebased, so its
  // term is required. An omitted signed zero point is already 0.
  auto zeroPointFor = [&](mlir::Value zp, mlir::RankedTensorType type,
                          bool isUnsigned) -> mlir::Value {
    if (zp || !isUnsigned)
      return zp;
    auto zpType = mlir::RankedTensorType::get({}, type.getElementType());
    return mlir::hip::ConstantOp::create(
        rewriter, loc, zpType,
        mlir::DenseElementsAttr::get(
            zpType, rewriter.getIntegerAttr(type.getElementType(), 0)));
  };
  aZp = zeroPointFor(aZp, aType, aUnsigned);
  bZp = zeroPointFor(bZp, bType, bUnsigned);
  const bool needsA = aZp && !rebasedZeroPointIsZero(aZp);
  const bool needsB = bZp && !rebasedZeroPointIsZero(bZp);

  mlir::Value za, zb;
  if (needsA)
    za = rebasedZeroPoint(rewriter, loc, context, aZp,
                          /*allowPerColumn=*/false, offA);
  if (needsB)
    zb = rebasedZeroPoint(rewriter, loc, context, bZp,
                          /*allowPerColumn=*/true, offB);

  // Fold A's batch dimensions into M.
  if (rank > 2) {
    llvm::SmallVector<mlir::ReassociationIndices> reassoc(2);
    for (int64_t i : llvm::seq<int64_t>(rank - 1))
      reassoc[0].push_back(i);
    reassoc[1].push_back(rank - 1);
    auto flatType = mlir::RankedTensorType::get({m, k}, aType.getElementType());
    a = mlir::tensor::CollapseShapeOp::create(rewriter, loc, flatType, a,
                                              reassoc)
            .getResult();
  }
  // The quant op takes ui8 or signless i8 for A; a signed int8 is a bitcast.
  if (aType.getElementType().isSignedInteger(8))
    a = toSigned8(rewriter, loc, context, a);
  b = toSigned8(rewriter, loc, context, b);

  mlir::Type i32 = rewriter.getI32Type();
  mlir::Type i8 = rewriter.getI8Type();
  auto qmm = [&](mlir::Value lhs, mlir::Value rhs, int64_t rows,
                 int64_t cols) -> mlir::Value {
    auto type = mlir::RankedTensorType::get({rows, cols}, i32);
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, type.getShape(), type.getElementType());
    return mlir::hip::QuantMatmulOp::create(
               rewriter, loc, mlir::TypeRange{type}, context, lhs, rhs, init)
        .getResult(0);
  };

  mlir::Value acc = qmm(a, b, m, n);
  if (za) {
    mlir::Value ones = splatConstant(
        rewriter, loc, mlir::RankedTensorType::get({1, k}, i8), 1);
    mlir::Value colsum = qmm(ones, b, 1, n);
    mlir::Value term = emitBroadcastBinary<mlir::hip::MulOp>(
        rewriter, loc, context, za, colsum);
    acc = emitBroadcastBinary<mlir::hip::SubOp>(rewriter, loc, context, acc,
                                                term);
  }
  if (zb) {
    mlir::Value ones = splatConstant(
        rewriter, loc, mlir::RankedTensorType::get({k, 1}, i8), 1);
    mlir::Value rowsum = qmm(a, ones, m, 1);
    mlir::Value term = emitBroadcastBinary<mlir::hip::MulOp>(
        rewriter, loc, context, rowsum, zb);
    acc = emitBroadcastBinary<mlir::hip::SubOp>(rewriter, loc, context, acc,
                                                term);
  }
  if (za && zb) {
    mlir::Value kConst =
        splatConstant(rewriter, loc, mlir::RankedTensorType::get({}, i32), k);
    mlir::Value zbK = emitBroadcastBinary<mlir::hip::MulOp>(
        rewriter, loc, context, zb, kConst);
    mlir::Value term =
        emitBroadcastBinary<mlir::hip::MulOp>(rewriter, loc, context, za, zbK);
    acc = emitBroadcastBinary<mlir::hip::AddOp>(rewriter, loc, context, acc,
                                                term);
  }

  if (rank > 2) {
    llvm::SmallVector<mlir::ReassociationIndices> reassoc(2);
    for (int64_t i : llvm::seq<int64_t>(rank - 1))
      reassoc[0].push_back(i);
    reassoc[1].push_back(rank - 1);
    acc = mlir::tensor::ExpandShapeOp::create(rewriter, loc, resultType, acc,
                                              reassoc)
              .getResult();
  }
  rewriter.replaceOp(op, acc);
  return mlir::success();
}

/// onnx.MatMulInteger -> hip.cast / hip.sub / hip.matmul / hip.cast, or an
/// int8 hip.quant_matmul when \p quantMatmul is set and the shapes are static.
struct MatMulIntegerToHip : public mlir::RewritePattern {
  MatMulIntegerToHip(mlir::MLIRContext *ctx, bool quantMatmul)
      : RewritePattern("onnx.MatMulInteger", /*benefit=*/1, ctx),
        quantMatmul(quantMatmul) {}

  bool quantMatmul = false;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulIntegerToHip::matchAndRewrite(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
  if (op->getNumResults() != 1 || op->getNumOperands() < 2)
    return rewriter.notifyMatchFailure(
        op, "MatMulInteger expects A, B, and one i32 result");

  auto aType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto bType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!aType || !bType || !resultType)
    return rewriter.notifyMatchFailure(op,
                                       "MatMulInteger requires ranked tensors");
  // hip.sub rejects rank > 4. hip.matmul rejects rank < 2, which is also
  // where numpy would promote a vector by inserting a 1.
  if (aType.getRank() < 2 || bType.getRank() < 2 || aType.getRank() > 4 ||
      bType.getRank() > 4)
    return rewriter.notifyMatchFailure(
        op, "MatMulInteger lowering supports rank 2..4 only");
  if (!isEightBit(aType.getElementType()) ||
      !isEightBit(bType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "MatMulInteger A and B must be int8 or uint8");
  if (!resultType.getElementType().isInteger(32) ||
      resultType.getElementType().isUnsignedInteger(32))
    return rewriter.notifyMatchFailure(op, "MatMulInteger result must be i32");

  // N sizes a per-column zero point and B's K selects the shared matrix.
  // Both stay compile-time values. A's own K may be uninferred; the runtime
  // reads it from A's descriptor and it must equal B's K. A static rank-2 B
  // is one shared matrix, so every other A dimension, including M, may stay
  // dynamic.
  if (bType.isDynamicDim(bType.getRank() - 2) ||
      bType.isDynamicDim(bType.getRank() - 1))
    return rewriter.notifyMatchFailure(op,
                                       "MatMulInteger K and N must be static");
  const bool dynamicActivation =
      !aType.hasStaticShape() || !resultType.hasStaticShape();
  if (dynamicActivation && (bType.getRank() != 2 || !bType.hasStaticShape()))
    return rewriter.notifyMatchFailure(
        op, "MatMulInteger dynamic batch or sequence requires a static rank-2 "
            "B");

  const int64_t m = aType.getDimSize(aType.getRank() - 2);
  const int64_t ka = aType.getDimSize(aType.getRank() - 1);
  const int64_t kb = bType.getDimSize(bType.getRank() - 2);
  const int64_t n = bType.getDimSize(bType.getRank() - 1);
  // An uninferred A K is a wildcard. B's static K is the contraction length
  // used to choose the accumulator; the two must agree at runtime.
  if (!mlir::ShapedType::isDynamic(ka) && ka != kb)
    return rewriter.notifyMatchFailure(
        op, "MatMulInteger contraction dimensions do not match");

  llvm::SmallVector<int64_t> expected;
  if (bType.getRank() == 2) {
    expected.append(aType.getShape().begin(), aType.getShape().end() - 1);
    expected.push_back(n);
  } else {
    if (!mlir::OpTrait::util::getBroadcastedShape(aType.getShape().drop_back(2),
                                                  bType.getShape().drop_back(2),
                                                  expected))
      return rewriter.notifyMatchFailure(
          op, "MatMulInteger batch dimensions do not broadcast");
    expected.push_back(m);
    expected.push_back(n);
  }
  if (resultType.getRank() != static_cast<int64_t>(expected.size()) ||
      !resultType.getShape().equals(expected))
    return rewriter.notifyMatchFailure(
        op, "MatMulInteger result shape does not match A @ B");

  mlir::Value aZp = realOperand(op, 2);
  mlir::Value bZp = realOperand(op, 3);
  if (aZp) {
    auto aZpType = mlir::dyn_cast<mlir::RankedTensorType>(aZp.getType());
    if (aZpType && mlir::ShapedType::isDynamic(m) && aZpType.getRank() == 1 &&
        aZpType.getDimSize(0) > 1)
      return rewriter.notifyMatchFailure(
          op, "MatMulInteger per-row zero point requires a static M");
    if (!aZpType || aZpType.getElementType() != aType.getElementType() ||
        !zeroPointShapeOk(aZpType, aType, m))
      return rewriter.notifyMatchFailure(
          op, "MatMulInteger a_zero_point must be a scalar or a length-M "
              "vector of A's element type");
  }
  if (bZp) {
    auto bZpType = mlir::dyn_cast<mlir::RankedTensorType>(bZp.getType());
    if (!bZpType || bZpType.getElementType() != bType.getElementType() ||
        !zeroPointShapeOk(bZpType, bType, n))
      return rewriter.notifyMatchFailure(
          op, "MatMulInteger b_zero_point must be a scalar or a length-N "
              "vector of B's element type");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();

  if (quantMatmul && mlir::succeeded(tryLowerQuantMatmul(
                         rewriter, op, context, loc, aType, bType, resultType)))
    return mlir::success();

  // ONNX FLOAT is datatype 1 and INT32 is 6. The cast kernel is selected
  // from the buffer types, not from this attribute.
  // A dynamic K has no compile-time bound, so the dot product stays in i32.
  const bool exactF32 =
      !mlir::ShapedType::isDynamic(ka) && eightBitDotIsExactInF32(ka);
  mlir::Type accType = rewriter.getI32Type();
  if (exactF32)
    accType = rewriter.getF32Type();
  const int64_t onnxAcc = exactF32 ? 1 : 6;

  mlir::Value aAcc =
      castTo(rewriter, loc, context, op->getOperand(0), accType, onnxAcc);
  mlir::Value bAcc =
      castTo(rewriter, loc, context, op->getOperand(1), accType, onnxAcc);
  if (aZp) {
    mlir::Value aZpAcc = castTo(rewriter, loc, context, aZp, accType, onnxAcc);
    aZpAcc = alignZeroPoint(rewriter, loc, aZpAcc, aType, aType.getRank() - 2);
    aAcc = subtractZeroPoint(rewriter, loc, context, aAcc, aZpAcc);
  }
  if (bZp) {
    mlir::Value bZpAcc = castTo(rewriter, loc, context, bZp, accType, onnxAcc);
    bZpAcc = alignZeroPoint(rewriter, loc, bZpAcc, bType, bType.getRank() - 1);
    bAcc = subtractZeroPoint(rewriter, loc, context, bAcc, bZpAcc);
  }

  llvm::SmallVector<int64_t> batchShape(expected.begin(), expected.end() - 2);
  llvm::SmallVector<int64_t> aBroadcast = batchShape;
  aBroadcast.push_back(m);
  aBroadcast.push_back(ka);
  aAcc = expandTo(rewriter, loc, context, aAcc,
                  mlir::RankedTensorType::get(aBroadcast, accType));
  // A B whose batch dimensions are all 1 is also one shared matrix. Collapse
  // those dimensions instead of materializing a copy for every result batch.
  if (bType.getRank() > 2 &&
      llvm::all_of(bType.getShape().drop_back(2),
                   [](int64_t dim) { return dim == 1; })) {
    auto sharedBType = mlir::RankedTensorType::get({kb, n}, accType);
    bAcc = reshapeTo(rewriter, loc, bAcc, sharedBType);
  } else if (bType.getRank() > 2) {
    llvm::SmallVector<int64_t> bBroadcast = batchShape;
    bBroadcast.push_back(kb);
    bBroadcast.push_back(n);
    bAcc = expandTo(rewriter, loc, context, bAcc,
                    mlir::RankedTensorType::get(bBroadcast, accType));
  }

  auto matmulType = mlir::RankedTensorType::get(resultType.getShape(), accType);
  mlir::Value matmulInit = createEmptyTensor(rewriter, loc, matmulType, aAcc);
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)));
  attrs.push_back(
      rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(0)));
  llvm::SmallVector<mlir::Value> operands = {context, aAcc, bAcc, matmulInit};
  auto matmul = mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs);
  if (!exactF32) {
    rewriter.replaceOp(op, matmul.getResult(0));
    return mlir::success();
  }

  mlir::Value outInit =
      createEmptyTensor(rewriter, loc, resultType, matmulInit);
  auto casted = mlir::hip::CastOp::create(
      rewriter, loc, context, matmul.getResult(0), outInit,
      rewriter.getI64IntegerAttr(/*onnxTo=*/6));
  rewriter.replaceOp(op, casted->getResult(0));
  return mlir::success();
}

} // namespace

void populateMatMulIntegerConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx,
                                             bool quantMatmul) {
  patterns.add<MatMulIntegerToHip>(ctx, quantMatmul);
}

} // namespace hip
} // namespace mlir
