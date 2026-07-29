/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX MatMulNBits -> HIP MatMulNBits (com.microsoft custom op)
//===----------------------------------------------------------------------===//

struct MatMulNBitsToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulNBitsToHip)
  MatMulNBitsToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulNBitsToHip::matchAndRewrite(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "MatMulNBits") {
    return rewriter.notifyMatchFailure(op, "not a MatMulNBits custom op");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  }

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() < 3) {
    return rewriter.notifyMatchFailure(
        op, "expected at least 3 inputs for MatMulNBits");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for MatMulNBits");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  mlir::Value context = *ctxOrFailure;

  mlir::Value A = op->getOperand(0);
  mlir::Value B = op->getOperand(1);
  mlir::Value scales = op->getOperand(2);

  auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
    if (idx >= op->getNumOperands()) {
      return mlir::Value{};
    }
    mlir::Value v = op->getOperand(idx);
    if (!v || mlir::isa<mlir::NoneType>(v.getType())) {
      return mlir::Value{};
    }
    return v;
  };
  mlir::Value zeroPoints = getOptionalInput(3);
  mlir::Value gIdx = getOptionalInput(4);
  mlir::Value bias = getOptionalInput(5);

  auto KAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("K").getSInt());
  auto NAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("N").getSInt());

  auto bitsIntAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  auto bitsAttr =
      rewriter.getI64IntegerAttr(bitsIntAttr ? bitsIntAttr.getSInt() : 4);

  auto blockSizeAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("block_size").getSInt());

  auto accuracyIntAttr = op->getAttrOfType<mlir::IntegerAttr>("accuracy_level");
  auto accuracyLevelAttr = rewriter.getI64IntegerAttr(
      accuracyIntAttr ? accuracyIntAttr.getSInt() : 0);

  // Validate and detect zero_points element type.
  // Supported formats:
  //   - uint8 (i8): packed uint4 nibbles, two zero_points per byte
  //   - fp16  (f16): one zero_point per element
  // Reject anything else at compile time to avoid silent precision bugs.
  int64_t zpElemSize = 2; // default: fp16 (when zeroPoints is absent)
  if (zeroPoints) {
    auto zpTy = mlir::cast<mlir::ShapedType>(zeroPoints.getType());
    mlir::Type elemTy = zpTy.getElementType();
    if (elemTy.isInteger(8)) {
      zpElemSize = 1; // uint8 container holding packed uint4 nibbles
    } else if (elemTy.isF16()) {
      zpElemSize = 2;
    } else {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "MatMulNBits: unsupported zero_points element type: ";
      elemTy.print(os);
      os << ". Expected i8 (packed uint4) or f16";
      return rewriter.notifyMatchFailure(op, msg);
    }
  }
  auto zpElemSizeAttr = rewriter.getI64IntegerAttr(zpElemSize);

  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, A);

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp = mlir::hip::MatMulNBitsOp::create(
      rewriter, loc, context, A, B, scales, zeroPoints, gIdx, bias, init, KAttr,
      NAttr, bitsAttr, blockSizeAttr, accuracyLevelAttr, zpElemSizeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateMatMulNBitsConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<MatMulNBitsToHip>(ctx);
}

} // namespace hip
} // namespace mlir
