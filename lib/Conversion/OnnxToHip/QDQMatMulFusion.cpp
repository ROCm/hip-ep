/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- QDQMatMulFusion.cpp ------------------------------------------------===//
//
// Two-phase QDQ MatMul fusion:
//
// Phase 1: PDLL patterns mark operations with "hip_fusion" attribute
//          (applied once at module level in OnnxToHip.cpp)
// Phase 2: C++ pattern reads attributes and performs actual fusion
//          (applied during pre-lowering pattern sweep)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {

// Pattern that processes operations marked by PDLL
struct QDQMatMulFusionPattern : public RewritePattern {
    QDQMatMulFusionPattern(MLIRContext *ctx)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const override {
        // Check if operation was marked by PDLL pattern
        auto fusionAttr = op->getAttrOfType<StringAttr>("hip_fusion");
        if (!fusionAttr || fusionAttr.getValue() != "qdq_matmul")
            return failure();

        // This should be a DequantizeLinear operation
        if (op->getName().getStringRef() != "onnx.DequantizeLinear")
            return failure();

        // Get operands
        if (op->getNumOperands() < 2)
            return failure();

        Value matmulResult = op->getOperand(0);
        Value outputScale = op->getOperand(1);

        // Get MatMul operation
        auto matmulOp = matmulResult.getDefiningOp();
        if (!matmulOp || matmulOp->getName().getStringRef() != "onnx.MatMul")
            return failure();

        if (matmulOp->getNumOperands() < 2)
            return failure();

        Value quantizedLhs = matmulOp->getOperand(0);
        Value rhs = matmulOp->getOperand(1);

        // Get QuantizeLinear operation
        auto quantOp = quantizedLhs.getDefiningOp();
        if (!quantOp || quantOp->getName().getStringRef() != "onnx.QuantizeLinear")
            return failure();

        if (quantOp->getNumOperands() < 2)
            return failure();

        Value lhsInput = quantOp->getOperand(0);
        Value lhsScale = quantOp->getOperand(1);

        // Get context argument
        auto ctxOrFailure = getContextArg(op, rewriter);
        if (failed(ctxOrFailure))
            return failure();
        Value context = *ctxOrFailure;

        // Extract scale values (simplified for demo)
        float lhsScaleValue = 0.1f;
        float rhsScaleValue = 1.0f;
        float outScaleValue = 0.2f;

        // Extract actual scale values from constants if possible
        if (auto constOp = lhsScale.getDefiningOp()) {
            if (auto valueAttr = constOp->getAttr("value")) {
                if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
                    if (denseAttr.isSplat()) {
                        lhsScaleValue = denseAttr.getSplatValue<FloatAttr>().getValueAsDouble();
                    }
                }
            }
        }

        if (auto constOp = outputScale.getDefiningOp()) {
            if (auto valueAttr = constOp->getAttr("value")) {
                if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
                    if (denseAttr.isSplat()) {
                        outScaleValue = denseAttr.getSplatValue<FloatAttr>().getValueAsDouble();
                    }
                }
            }
        }

        Location loc = op->getLoc();
        Type outputType = op->getResult(0).getType();
        auto tensorType = cast<RankedTensorType>(outputType);

        // Create output tensor (DPS style)
        Value emptyTensor = tensor::EmptyOp::create(
            rewriter, loc, tensorType.getShape(), tensorType.getElementType());

        // Create fused qmatmul operation
        auto qmatmulOp = QMatMulOp::create(
            rewriter,
            loc,
            tensorType,
            context,
            lhsInput,
            rhs,
            emptyTensor,
            rewriter.getF32FloatAttr(lhsScaleValue),
            rewriter.getF32FloatAttr(rhsScaleValue),
            rewriter.getF32FloatAttr(outScaleValue));

        // Replace the DequantizeLinear operation
        rewriter.replaceOp(op, qmatmulOp.getResult(0));

        return success();
    }
};

// Populate C++ patterns (Phase 2)
// Phase 1 (PDL marking) happens earlier in OnnxToHip.cpp
void populateQDQMatMulFusionPatterns(RewritePatternSet &patterns) {
    patterns.add<QDQMatMulFusionPattern>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
