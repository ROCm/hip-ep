//===- MatMulNBitsConversion.cpp - ONNX-to-HIP MatMulNBits conversion - *- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX MatMulNBits -> HIP MatMulNBits (com.microsoft custom op)
//===----------------------------------------------------------------------===//

struct MatMulNBitsToHip : public RewritePattern {
  MatMulNBitsToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult
MatMulNBitsToHip::matchAndRewrite(Operation* op,
                                  PatternRewriter& rewriter) const {
  auto funcNameAttr = op->getAttrOfType<StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "MatMulNBits") {
    return rewriter.notifyMatchFailure(op, "not a MatMulNBits custom op");
  }
  auto domainAttr = op->getAttrOfType<StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  }

  Location loc = op->getLoc();

  if (op->getNumOperands() < 3) {
    return rewriter.notifyMatchFailure(
        op, "expected at least 3 inputs for MatMulNBits");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for MatMulNBits");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  Value context = *ctxOrFailure;

  Value A = op->getOperand(0);
  Value B = op->getOperand(1);
  Value scales = op->getOperand(2);

  auto getOptionalInput = [&](unsigned idx) -> Value {
    if (idx >= op->getNumOperands()) {
      return Value{};
    }
    Value v = op->getOperand(idx);
    if (!v || isa<NoneType>(v.getType())) {
      return Value{};
    }
    return v;
  };
  Value zeroPoints = getOptionalInput(3);
  Value gIdx = getOptionalInput(4);
  Value bias = getOptionalInput(5);

  auto KAttr =
      rewriter.getI64IntegerAttr(op->getAttrOfType<IntegerAttr>("K").getSInt());
  auto NAttr =
      rewriter.getI64IntegerAttr(op->getAttrOfType<IntegerAttr>("N").getSInt());

  auto bitsIntAttr = op->getAttrOfType<IntegerAttr>("bits");
  auto bitsAttr =
      rewriter.getI64IntegerAttr(bitsIntAttr ? bitsIntAttr.getSInt() : 4);

  auto blockSizeAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<IntegerAttr>("block_size").getSInt());

  auto accuracyIntAttr = op->getAttrOfType<IntegerAttr>("accuracy_level");
  auto accuracyLevelAttr = rewriter.getI64IntegerAttr(
      accuracyIntAttr ? accuracyIntAttr.getSInt() : 0);

  // Validate and detect zero_points element type.
  // Supported formats:
  //   - uint8 (i8): packed uint4 nibbles, two zero_points per byte
  //   - fp16  (f16): one zero_point per element
  // Reject anything else at compile time to avoid silent precision bugs.
  int64_t zpElemSize = 2; // default: fp16 (when zeroPoints is absent)
  if (zeroPoints) {
    auto zpTy = cast<ShapedType>(zeroPoints.getType());
    Type elemTy = zpTy.getElementType();
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

  auto rt = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, rt, A);

  auto hipOp = mlir::hip::MatMulNBitsOp::create(
      rewriter, loc, TypeRange{rt}, context, A, B, scales, zeroPoints, gIdx,
      bias, init, KAttr, NAttr, bitsAttr, blockSizeAttr, accuracyLevelAttr,
      zpElemSizeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return success();
}

} // namespace

void mlir::hip::populateMatMulNBitsConversionPatterns(
    RewritePatternSet& patterns, MLIRContext* ctx) {
  patterns.add<MatMulNBitsToHip>(ctx);
}

} // namespace hip
} // namespace mlir
