/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- qdq_chain_fusion.cpp - per-tensor Q/DQ chains to one kernel --------===//
//
// Quantized SwiGLU. Both multiplies commute, and the chain is only exact to
// fuse when each consumer reads an intermediate with the scale and zero point
// its producer wrote it with, so those pairs must agree:
//
//   Before:
//     %qg = hip.quantize_linear(%ctx) ins(%gate, %sg) zero_point(%zg) ...
//     %qs = hip.qsigmoid(%ctx) ins(%qg) outs(..) {input_scale = sg, ...}
//     %qa = hip.qmul(%ctx) ins(%qg, %qs) outs(..) {output_scale = sa, ...}
//     %qu = hip.quantize_linear(%ctx) ins(%up, %su) zero_point(%zu) ...
//     %qo = hip.qmul(%ctx) ins(%qa, %qu) outs(..) {output_scale = so, ...}
//     %y  = hip.dequantize_linear(%ctx) ins(%qo, %sd) zero_point(%zd)
//             outs(%init : tensor<?x?x17920xf32>)
//   After:
//     %y = hip.qswiglu(%ctx) ins(%gate, %up) outs(%init) {gate_scale = sg,
//            sigmoid_scale = .., silu_scale = sa, up_scale = su,
//            mul_scale = so, dq_scale = sd, ...}
//
// Round trip. A per-tensor Q read only by DQs with identical parameters
// (several DQs share the one fused result):
//
//   Before:
//     %q = hip.quantize_linear(%ctx) ins(%x, %s) zero_point(%z)
//            outs(%e : tensor<?x5120xui16>)
//     %y = hip.dequantize_linear(%ctx) ins(%q, %s) zero_point(%z)
//            outs(%init : tensor<?x5120xf32>)
//   After:
//     %y = hip.fake_quantize(%ctx) ins(%x) outs(%init)
//            {scale = s, zero_point = z, qmin = 0, qmax = 65535}
//
// Shape-only tensor.dim reads of a removed intermediate are retargeted to a
// float operand of the same shape.
//===----------------------------------------------------------------------===//

#include "qdq_chain_fusion.hpp"

#include "hip_fusion_transform.hpp"
#include "swiglu_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

struct QParams {
  float scale = 0.0f;
  int64_t zp = 0;
  bool operator==(const QParams &other) const {
    return scale == other.scale && zp == other.zp;
  }
};

bool isF32Tensor(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  return type && type.getElementType().isF32();
}

// The codes the fused kernels store in registers; wider types do not round
// trip through f32 exactly.
std::optional<std::pair<int64_t, int64_t>> quantRange(mlir::IntegerType type) {
  if (!type || (type.getWidth() != 8 && type.getWidth() != 16))
    return std::nullopt;
  int64_t width = type.getWidth();
  if (type.isUnsigned())
    return std::make_pair(int64_t{0}, (int64_t{1} << width) - 1);
  if (type.isSigned() || type.isSignless())
    return std::make_pair(-(int64_t{1} << (width - 1)),
                          (int64_t{1} << (width - 1)) - 1);
  return std::nullopt;
}

// Per-tensor parameters of a Q or DQ that the fused kernels can reproduce.
template <typename OpTy> std::optional<QParams> perTensorParams(OpTy op) {
  if (!op || op.getBlockSize() != 0 || op.getPackedInt4())
    return std::nullopt;
  std::optional<float> scale = tryHipSplatScale(op.getScale());
  std::optional<int64_t> zp = tryHipQdqZeropoint(op, /*absentValue=*/0);
  if (!scale || !zp || *scale == 0.0f)
    return std::nullopt;
  return QParams{*scale, *zp};
}

// A quantize the kernels implement: f32 in, f32 division (the only precision
// hip_quantize_linear accepts), UINT16 out.
std::optional<QParams> u16QuantizeOfF32(mlir::hip::QuantizeLinearOp q) {
  if (!q || !isF32Tensor(q.getInput()))
    return std::nullopt;
  if (q.getPrecision() != 0 && q.getPrecision() != 1)
    return std::nullopt;
  mlir::IntegerType type = getQdqQuantizedElementType(q);
  if (!type || !type.isUnsignedInteger(16))
    return std::nullopt;
  return perTensorParams(q);
}

void eraseIfDead(mlir::PatternRewriter &rewriter, mlir::Operation *def) {
  if (def && mlir::isOpTriviallyDead(def))
    rewriter.eraseOp(def);
}

struct QSwigluFusion
    : public mlir::OpRewritePattern<mlir::hip::DequantizeLinearOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::DequantizeLinearOp dq,
                  mlir::PatternRewriter &rewriter) const override {
    auto outer = dq.getInput().getDefiningOp<mlir::hip::QMulOp>();
    if (!outer)
      return rewriter.notifyMatchFailure(dq, "input is not hip.qmul");
    std::optional<QParams> dqParams = perTensorParams(dq);
    mlir::IntegerType dqType = getQdqQuantizedElementType(dq);
    if (!dqParams || !dqType || !dqType.isUnsignedInteger(16) ||
        !isF32Tensor(dq.getOutput()))
      return rewriter.notifyMatchFailure(dq, "DQ is not per-tensor u16->f32");

    // Operand slot of the outer qmul holding the SiLU product.
    for (bool swapOuter : {false, true}) {
      mlir::Value siluCode = swapOuter ? outer.getRhs() : outer.getLhs();
      mlir::Value upCode = swapOuter ? outer.getLhs() : outer.getRhs();
      QParams siluRead{swapOuter ? outer.getRhsScale().convertToFloat()
                                 : outer.getLhsScale().convertToFloat(),
                       static_cast<int64_t>(swapOuter ? outer.getRhsZp()
                                                      : outer.getLhsZp())};
      QParams upRead{swapOuter ? outer.getLhsScale().convertToFloat()
                               : outer.getRhsScale().convertToFloat(),
                     static_cast<int64_t>(swapOuter ? outer.getLhsZp()
                                                    : outer.getRhsZp())};

      auto inner = siluCode.getDefiningOp<mlir::hip::QMulOp>();
      auto upQ = upCode.getDefiningOp<mlir::hip::QuantizeLinearOp>();
      if (!inner || !upQ)
        continue;
      std::optional<QParams> upParams = u16QuantizeOfF32(upQ);
      if (!upParams || !(*upParams == upRead))
        continue;
      QParams siluWrite{inner.getOutputScale().convertToFloat(),
                        static_cast<int64_t>(inner.getOutputZp())};
      if (!(siluWrite == siluRead))
        continue;

      for (bool swapInner : {false, true}) {
        mlir::Value gateCode = swapInner ? inner.getRhs() : inner.getLhs();
        mlir::Value sigCode = swapInner ? inner.getLhs() : inner.getRhs();
        QParams gateRead{swapInner ? inner.getRhsScale().convertToFloat()
                                   : inner.getLhsScale().convertToFloat(),
                         static_cast<int64_t>(swapInner ? inner.getRhsZp()
                                                        : inner.getLhsZp())};
        QParams sigRead{swapInner ? inner.getLhsScale().convertToFloat()
                                  : inner.getRhsScale().convertToFloat(),
                        static_cast<int64_t>(swapInner ? inner.getLhsZp()
                                                       : inner.getRhsZp())};

        auto sigmoid = sigCode.getDefiningOp<mlir::hip::QSigmoidOp>();
        if (!sigmoid || sigmoid.getInput() != gateCode)
          continue;
        auto gateQ = gateCode.getDefiningOp<mlir::hip::QuantizeLinearOp>();
        std::optional<QParams> gateParams = u16QuantizeOfF32(gateQ);
        if (!gateParams || !(*gateParams == gateRead))
          continue;
        QParams sigIn{sigmoid.getInputScale().convertToFloat(),
                      static_cast<int64_t>(sigmoid.getInputZp())};
        QParams sigOut{sigmoid.getOutputScale().convertToFloat(),
                       static_cast<int64_t>(sigmoid.getOutputZp())};
        if (!(sigIn == *gateParams) || !(sigOut == sigRead))
          continue;

        mlir::Value gate = gateQ.getInput();
        mlir::Value up = upQ.getInput();
        return rewrite(rewriter, dq, outer, inner, sigmoid, gateQ, upQ, gate,
                       up, *gateParams, sigOut, siluWrite, *upParams,
                       *dqParams);
      }
    }
    return rewriter.notifyMatchFailure(dq, "not a quantized SwiGLU chain");
  }

private:
  mlir::LogicalResult
  rewrite(mlir::PatternRewriter &rewriter, mlir::hip::DequantizeLinearOp dq,
          mlir::hip::QMulOp outer, mlir::hip::QMulOp inner,
          mlir::hip::QSigmoidOp sigmoid, mlir::hip::QuantizeLinearOp gateQ,
          mlir::hip::QuantizeLinearOp upQ, mlir::Value gate, mlir::Value up,
          QParams gateParams, QParams sigParams, QParams siluParams,
          QParams upParams, QParams dqParams) const {
    mlir::Value ctx = dq.getCtx();
    if (outer.getCtx() != ctx || inner.getCtx() != ctx ||
        sigmoid.getCtx() != ctx || gateQ.getCtx() != ctx || upQ.getCtx() != ctx)
      return rewriter.notifyMatchFailure(dq, "chain must share a context");

    auto gateType = mlir::dyn_cast<mlir::RankedTensorType>(gate.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(dq->getResult(0).getType());
    if (!gateType || gateType != up.getType() || gateType != resultType)
      return rewriter.notifyMatchFailure(dq, "gate, up and output differ");
    // hip.qmul broadcasts; the fused kernel is flat, so every code tensor
    // must have the float tensors' shape.
    for (mlir::Value code :
         {gateQ->getResult(0), upQ->getResult(0), sigmoid->getResult(0),
          inner->getResult(0), outer->getResult(0)}) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(code.getType());
      if (!type || type.getShape() != gateType.getShape())
        return rewriter.notifyMatchFailure(dq, "chain broadcasts");
    }
    if (!sameRuntimeShape(gate, up))
      return rewriter.notifyMatchFailure(
          dq, "dynamic dimensions are not known to match");

    mlir::Value gateCode = gateQ->getResult(0);
    unsigned gateUses = 0;
    for (mlir::OpOperand &use : gateCode.getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (mlir::isa<mlir::tensor::DimOp>(owner))
        continue;
      if (owner != sigmoid.getOperation() && owner != inner.getOperation())
        return rewriter.notifyMatchFailure(dq, "gate code escapes the chain");
      ++gateUses;
    }
    if (gateUses != 2 || !hasSingleValueUse(sigmoid->getResult(0), inner) ||
        !hasSingleValueUse(inner->getResult(0), outer) ||
        !hasSingleValueUse(upQ->getResult(0), outer) ||
        !hasSingleValueUse(outer->getResult(0), dq))
      return rewriter.notifyMatchFailure(dq, "intermediate has another use");

    for (mlir::Operation *op :
         {gateQ.getOperation(), sigmoid.getOperation(), inner.getOperation(),
          upQ.getOperation(), outer.getOperation()})
      retargetShapeQueries(rewriter, op->getResult(0), gate);

    auto f32 = [&](float v) { return rewriter.getF32FloatAttr(v); };
    auto i64 = [&](int64_t v) { return rewriter.getI64IntegerAttr(v); };
    auto fused = mlir::hip::QSwigluOp::create(
        rewriter, dq.getLoc(), resultType, ctx, gate, up, dq.getOutput(),
        f32(gateParams.scale), i64(gateParams.zp), f32(sigParams.scale),
        i64(sigParams.zp), f32(siluParams.scale), i64(siluParams.zp),
        f32(upParams.scale), i64(upParams.zp),
        f32(outer.getOutputScale().convertToFloat()),
        i64(static_cast<int64_t>(outer.getOutputZp())), f32(dqParams.scale),
        i64(dqParams.zp));

    mlir::Operation *inits[] = {
        outer.getOutput().getDefiningOp(), inner.getOutput().getDefiningOp(),
        sigmoid.getOutput().getDefiningOp(), upQ.getOutput().getDefiningOp(),
        gateQ.getOutput().getDefiningOp()};
    rewriter.replaceOp(dq, fused->getResults());
    rewriter.eraseOp(outer);
    rewriter.eraseOp(inner);
    rewriter.eraseOp(upQ);
    rewriter.eraseOp(sigmoid);
    rewriter.eraseOp(gateQ);
    for (mlir::Operation *def : inits)
      eraseIfDead(rewriter, def);
    return mlir::success();
  }
};

// A Q whose code feeds several DQs with identical parameters (one per
// projection that reads the activation) becomes one hip.fake_quantize
// shared by all of them.
struct FakeQuantizeFusion
    : public mlir::OpRewritePattern<mlir::hip::DequantizeLinearOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::DequantizeLinearOp dq,
                  mlir::PatternRewriter &rewriter) const override {
    auto q = dq.getInput().getDefiningOp<mlir::hip::QuantizeLinearOp>();
    if (!q || q.getCtx() != dq.getCtx())
      return rewriter.notifyMatchFailure(dq,
                                         "input is not hip.quantize_linear");
    if (q.getPrecision() != 0 && q.getPrecision() != 1)
      return rewriter.notifyMatchFailure(dq, "unsupported Q precision");
    std::optional<QParams> qParams = perTensorParams(q);
    if (!qParams)
      return rewriter.notifyMatchFailure(dq, "Q is not per-tensor");
    auto range = quantRange(getQdqQuantizedElementType(q));
    if (!range)
      return rewriter.notifyMatchFailure(dq, "unsupported storage type");

    mlir::Value input = q.getInput();
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(dq->getResult(0).getType());
    if (!isF32Tensor(input) || input.getType() != resultType)
      return rewriter.notifyMatchFailure(dq, "expected f32 -> f32");

    mlir::Value code = q->getResult(0);
    llvm::SmallVector<mlir::hip::DequantizeLinearOp> readers;
    for (mlir::Operation *user : code.getUsers()) {
      if (mlir::isa<mlir::tensor::DimOp>(user))
        continue;
      auto reader = mlir::dyn_cast<mlir::hip::DequantizeLinearOp>(user);
      if (!reader || reader.getInput() != code ||
          reader.getCtx() != dq.getCtx() ||
          reader->getBlock() != dq->getBlock() ||
          reader->getResult(0).getType() != resultType)
        return rewriter.notifyMatchFailure(dq, "Q result has another use");
      std::optional<QParams> readParams = perTensorParams(reader);
      if (!readParams || !(*readParams == *qParams))
        return rewriter.notifyMatchFailure(dq, "Q and DQ parameters differ");
      if (!llvm::is_contained(readers, reader))
        readers.push_back(reader);
    }

    // The first reader's position dominates every other reader, and its
    // init is already defined there.
    mlir::hip::DequantizeLinearOp first = readers.front();
    for (mlir::hip::DequantizeLinearOp reader : readers)
      if (reader->isBeforeInBlock(first))
        first = reader;

    retargetShapeQueries(rewriter, code, input);
    llvm::SmallVector<mlir::Operation *> inits = {
        q.getOutput().getDefiningOp()};
    for (mlir::hip::DequantizeLinearOp reader : readers)
      if (reader != first)
        inits.push_back(reader.getOutput().getDefiningOp());

    rewriter.setInsertionPoint(first);
    auto fused = mlir::hip::FakeQuantizeOp::create(
        rewriter, first.getLoc(), resultType, first.getCtx(), input,
        first.getOutput(), rewriter.getF32FloatAttr(qParams->scale),
        rewriter.getI64IntegerAttr(qParams->zp),
        rewriter.getI64IntegerAttr(range->first),
        rewriter.getI64IntegerAttr(range->second));
    for (mlir::hip::DequantizeLinearOp reader : readers)
      rewriter.replaceOp(reader, fused->getResults());
    rewriter.eraseOp(q);
    for (mlir::Operation *def : inits)
      eraseIfDead(rewriter, def);
    return mlir::success();
  }
};

} // namespace

void populateQdqChainFusionPatterns(mlir::RewritePatternSet &patterns) {
  // Both root at a DQ but need distinct producers (qmul, quantize), so they
  // never compete for one root.
  patterns.add<QSwigluFusion, FakeQuantizeFusion>(patterns.getContext());
}

} // namespace fusion_transform
} // namespace hip
