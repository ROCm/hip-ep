/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== Convolution ops ================================

// hip.conv(%ctx, %input, %weights, %bias, %output) -> wrap_conv
//
// Before:
//   %out = hip.conv(%ctx) ins(%in, %w, %b :
//                              memref<1x3x896x896xf16, 1>,
//                              memref<1152x3x14x14xf16, 1>,
//                              memref<1152xf16, 1>)
//                          outs(%o : memref<1x1152x64x64xf16, 1>)
//                          {kernel_shape=[14,14], strides=[14,14], ...}
// After:
//   llvm.call @wrap_conv(%ctx, slot, %in, %w, %b, %o,
//                        /*data_type=*/1 /* f16 */, /*spatial_rank=*/2,
//                        1, 3, 1152,             // N, Cin, Cout
//                        896, 896, 1,            // in extents  (slot 2 unused)
//                        64, 64, 1,              // out extents
//                        14, 14, 1,              // kernel
//                        14, 14, 1,              // strides
//                        0, 0, 0,                // pad_begin
//                        1, 1, 1,                // dilations
//                        1)                      // group
//
// The per-axis slots follow the hip_pool convention: `spatial_rank` says how
// many are live, and the trailing unused ones are 1 (extent / kernel / stride /
// dilation) or 0 (pad), which makes the kernel's loops collapse to a single
// iteration over index 0 rather than needing a rank-specialised path.
//
// Only pad_begin is passed. Pad positions are never read by the kernel, so the
// trailing pad affects nothing beyond how many output positions exist, and that
// is already carried by the output extents.
//
// `data_type` comes from the OUTPUT memref's element type and applies to all
// four buffers. That is guaranteed by the host-side typing rule in
// OnnxToHip::ConvConversion, which allocates the output with
// `resultType.getElementType()` and requires the operands to match.
struct ConvOpLowering : public ConvertOpToLLVMPattern<ConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getCtx(); // RuntimeState* (opaque)
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value weightsPtr =
        extractContiguousMemRefPtr(adaptor.getWeights(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value biasPtr;
    if (adaptor.getBias())
      biasPtr = extractContiguousMemRefPtr(adaptor.getBias(), rewriter, loc);
    else
      biasPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // [N, C, D_1..D_k] for k in 1..3.
    const int64_t rank = inputType.getRank();
    if (rank < 3 || rank > 5)
      return op.emitError("hip.conv: input must be rank 3, 4 or 5, got ")
             << rank;
    if (weightsType.getRank() != rank || outputType.getRank() != rank)
      return op.emitError(
                 "hip.conv: input, weights and output ranks must match (")
             << rank << ", " << weightsType.getRank() << ", "
             << outputType.getRank() << ")";
    const int64_t spatialRank = rank - 2;

    Value inputN =
        getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value inputC =
        getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
    // Output channels are the weight tensor's first dim.
    Value weightsK =
        getMemRefDimSize(weightsType, 0, adaptor.getWeights(), rewriter, loc);

    // Per-axis extents, padded out to three slots with 1.
    SmallVector<Value, 3> inDims, outDims;
    for (int64_t i : llvm::seq<int64_t>(0, 3)) {
      if (i < spatialRank) {
        inDims.push_back(getMemRefDimSize(inputType, 2 + i, adaptor.getInput(),
                                          rewriter, loc));
        outDims.push_back(getMemRefDimSize(outputType, 2 + i,
                                           adaptor.getOutput(), rewriter, loc));
      } else {
        inDims.push_back(createI64Const(1));
        outDims.push_back(createI64Const(1));
      }
    }

    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    auto dilations = op.getDilations();
    auto group = op.getGroup();

    if (static_cast<int64_t>(kernelShape.size()) != spatialRank ||
        static_cast<int64_t>(strides.size()) != spatialRank ||
        static_cast<int64_t>(dilations.size()) != spatialRank ||
        static_cast<int64_t>(pads.size()) != 2 * spatialRank)
      return op.emitError("hip.conv: attribute arity does not match spatial "
                          "rank ")
             << spatialRank;

    auto getI64 = [](Attribute attr) -> int64_t {
      return cast<IntegerAttr>(attr).getInt();
    };
    // Per-axis attribute slot: the real value for live axes, then the identity
    // the kernel needs for the dead ones.
    auto slot = [&](ArrayAttr attr, int64_t i, int64_t dead) -> Value {
      return createI64Const(i < spatialRank ? getI64(attr[i]) : dead);
    };

    SmallVector<Value, 3> kVals, sVals, pVals, dVals;
    for (int64_t i : llvm::seq<int64_t>(0, 3)) {
      kVals.push_back(slot(kernelShape, i, 1));
      sVals.push_back(slot(strides, i, 1));
      // ONNX pads are [begin_0..begin_{k-1}, end_0..end_{k-1}]; only the begins
      // are needed.
      pVals.push_back(slot(pads, i, 0));
      dVals.push_back(slot(dilations, i, 1));
    }

    int64_t dataTypeEnum = getHipdnnDataType(outputType.getElementType());
    if (dataTypeEnum < 0)
      return op.emitError("hip.conv: unsupported output element type ")
             << outputType.getElementType();

    SmallVector<Type, 32> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // input
        ptrType, // weights
        ptrType, // bias (nullable)
        ptrType, // output
        i64Type, // data_type
        i64Type, // spatial_rank
        i64Type, // N
        i64Type, // Cin
        i64Type, // Cout
    };
    // in[3], out[3], k[3], s[3], pad_begin[3], dil[3], group
    paramTypes.append(19, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapConv, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 32> args = {statePtr,
                                   getOpStateSlotValue(op, rewriter, loc),
                                   inputPtr,
                                   weightsPtr,
                                   biasPtr,
                                   outputPtr,
                                   createI64Const(dataTypeEnum),
                                   createI64Const(spatialRank),
                                   inputN,
                                   inputC,
                                   weightsK};
    args.append(inDims.begin(), inDims.end());
    args.append(outDims.begin(), outDims.end());
    args.append(kVals.begin(), kVals.end());
    args.append(sVals.begin(), sVals.end());
    args.append(pVals.begin(), pVals.end());
    args.append(dVals.begin(), dVals.end());
    args.push_back(createI64Const(group));

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // hip.conv writes into its `outs` operand and has no lowered result.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateConvLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<ConvOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
