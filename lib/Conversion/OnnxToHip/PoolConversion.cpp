/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "hip/Dialect/IR/HipShapeUtilsConvPool.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// onnx.MaxPool / onnx.AveragePool / onnx.LpPool -> hip.pool
//===----------------------------------------------------------------------===//
//
// General (fallback) lowering for window pooling. MaxPool and LpPool always
// land here. AveragePool is tried first by `AveragePoolToReshapeMean` in
// `ProjectorOpsRewrites.cpp` (pre-lowering, benefit = 2) when NCHW,
// kernel == stride, no pad, and static divisible spatial dims — that fast
// path decomposes into Reshape + Transpose + ReduceMean. Any AveragePool
// outside those constraints (overlapping windows, padding, 1D/3D, etc.)
// survives pre-lowering and is converted here via `hip.pool` (benefit = 1).
//
// One conversion pattern services all three ONNX window-pooling ops; the
// reduction kind is injected per registered op name via `poolMode`
// (0 = AVERAGE, 1 = MAX, 2 = LP — mirrors HIPDNN_EP_POOL_*).  All three
// share the same window geometry (kernel_shape / strides / pads / dilations
// / auto_pad / ceil_mode), so only a few attributes and the optional second
// output differ between modes:
//   * MAX     : may carry a second i64 `Indices` output and `storage_order`.
//   * AVERAGE : carries `count_include_pad`; single output.
//   * LP      : carries `p` (>= 1); single output.
//
// Compile-time work done here (so the runtime kernel has nothing to figure
// out about layout / padding):
//   * fill defaults for `strides`, `dilations`, `pads` when omitted
//   * resolve `auto_pad` to explicit `pads` (requires static spatial dims;
//     SAME_UPPER / SAME_LOWER need to know the input extent + stride to
//     split the pad budget between begin/end).  After this rewrite the
//     `hip.pool` op carries pure explicit-pads form — `auto_pad` is never
//     visible past this pass.
//   * (MAX only) reject `storage_order = 1` (column-major Indices) — our
//     runtime emits row-major flat indices only.
//
// Before:
//   %y = "onnx.AveragePool"(%x) {kernel_shape = [3, 3], auto_pad =
//   "SAME_UPPER",
//                                strides = [2, 2], count_include_pad = 1} :
//          (tensor<1x3x32x32xf16>) -> tensor<1x3x16x16xf16>
//
// After (single-output form, dynamic N example):
//   %dim0 = tensor.dim %x, %c0 : tensor<?x3x32x32xf16>
//   %init = tensor.empty(%dim0) : tensor<?x3x16x16xf16>
//   %y = hip.pool(%ctx) ins(%x : tensor<?x3x32x32xf16>)
//                       outs(%init : tensor<?x3x16x16xf16>)
//                       {pool_mode = 0, kernel_shape = [3, 3], strides = [2,
//                       2],
//                        pads = [0, 0, 1, 1], dilations = [1, 1],
//                        ceil_mode = 0, storage_order = 0,
//                        count_include_pad = 1, p = 2}

// Reduction-mode tags.  Values mirror HIPDNN_EP_POOL_* in
// lib/Runtime/hipdnn_ep_runtime.h and kPool* in HipToLLVMUtils.h.
constexpr int64_t kPoolAverage = 0;
constexpr int64_t kPoolMax = 1;
constexpr int64_t kPoolLp = 2;

struct PoolToHip : public mlir::RewritePattern {
  int64_t poolMode;
  PoolToHip(mlir::MLIRContext *ctx, llvm::StringRef onnxName, int64_t mode)
      : RewritePattern(onnxName, /*benefit=*/1, ctx), poolMode(mode) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input");
    size_t numResults = op->getNumResults();
    // Only MAX exposes the optional Indices output; AVERAGE / LP are single.
    size_t maxResults = (poolMode == kPoolMax) ? 2 : 1;
    if (numResults < 1 || numResults > maxResults)
      return rewriter.notifyMatchFailure(
          op, "unexpected result count for this pool mode");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

    int64_t rank = inputType.getRank();
    if (rank < 3 || rank != outputType.getRank())
      return rewriter.notifyMatchFailure(
          op, "pool needs rank >= 3 and matching in/out ranks");
    int64_t spatialRank = rank - 2;
    if (spatialRank < 1 || spatialRank > 3)
      return rewriter.notifyMatchFailure(
          op, "only 1D / 2D / 3D pool supported (spatial_rank in {1,2,3})");

    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
        inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "pool runtime supports only float types and matching in/out");

    // Optional Indices output (MAX only) must be i64.
    if (numResults == 2) {
      auto idxType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
      if (!idxType || idxType.getRank() != rank ||
          !idxType.getElementType().isInteger(64))
        return rewriter.notifyMatchFailure(
            op, "MaxPool Indices output must be i64 with same rank");
    }

    // ===== Attribute extraction (apply defaults inline) =====================

    auto getI64Array = [&](mlir::StringRef name,
                           llvm::SmallVectorImpl<int64_t> &out) {
      if (auto attr = op->getAttrOfType<mlir::ArrayAttr>(name))
        for (auto a : attr)
          out.push_back(
              mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
    };

    llvm::SmallVector<int64_t> kernelShape, strides, pads, dilations;
    getI64Array("kernel_shape", kernelShape);
    getI64Array("strides", strides);
    getI64Array("pads", pads);
    getI64Array("dilations", dilations);

    if ((int64_t)kernelShape.size() != spatialRank)
      return rewriter.notifyMatchFailure(
          op, "kernel_shape length must equal spatial rank");
    if (strides.empty())
      strides.assign(spatialRank, 1);
    if (dilations.empty())
      dilations.assign(spatialRank, 1);

    int64_t ceilMode = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("ceil_mode"))
      ceilMode = attr.getValue().getSExtValue();

    // storage_order is a MaxPool-only attribute; col-major Indices (=1) is
    // rejected because the runtime only emits row-major flat indices.
    int64_t storageOrder = 0;
    if (poolMode == kPoolMax) {
      if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("storage_order"))
        storageOrder = attr.getValue().getSExtValue();
      if (storageOrder != 0)
        return rewriter.notifyMatchFailure(
            op, "MaxPool storage_order=1 (col-major Indices) not supported");
    }

    // count_include_pad is an AveragePool-only divisor selector.
    int64_t countIncludePad = 0;
    if (poolMode == kPoolAverage) {
      if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("count_include_pad"))
        countIncludePad = attr.getValue().getSExtValue();
    }

    // p is an LpPool-only norm exponent; ONNX requires p >= 1.
    int64_t p = 2;
    if (poolMode == kPoolLp) {
      if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("p"))
        p = attr.getValue().getSExtValue();
      if (p < 1)
        return rewriter.notifyMatchFailure(op, "LpPool requires p >= 1");
    }

    // Resolve auto_pad to explicit pads.  Only NOTSET keeps the user-supplied
    // `pads`; the four other modes derive pads from the static input/output
    // spatial extents.  `auto_pad` is ONNX-deprecated but still produced by
    // some exporters. The SAME_UPPER / SAME_LOWER pad-budget split below (split
    // pad_total in half, give the odd pad to the end for SAME_UPPER / to the
    // begin for SAME_LOWER) follows onnx-mlir's `customComputeShape`
    // SAME_UPPER/SAME_LOWER handling
    // (src/Dialect/ONNX/ONNXOps/NN/NNHelper.cpp.inc).
    std::string autoPad = "NOTSET";
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("auto_pad"))
      autoPad = attr.getValue().str();

    if (autoPad == "VALID") {
      pads.assign(2 * spatialRank, 0);
    } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      // Need static input + output spatial extents to split the pad budget.
      for (int64_t i : llvm::seq<int64_t>(spatialRank)) {
        if (inputType.isDynamicDim(2 + i) || outputType.isDynamicDim(2 + i))
          return rewriter.notifyMatchFailure(
              op, "auto_pad=SAME_* requires static spatial dims");
      }
      pads.assign(2 * spatialRank, 0);
      for (int64_t i : llvm::seq<int64_t>(spatialRank)) {
        int64_t in = inputType.getDimSize(2 + i);
        int64_t out = outputType.getDimSize(2 + i);
        int64_t k = kernelShape[i];
        int64_t s = strides[i];
        int64_t d = dilations[i];
        int64_t effK = (k - 1) * d + 1;
        // pad_total = (out - 1) * s + effK - in   (clamped >= 0)
        int64_t padTotal = (out - 1) * s + effK - in;
        if (padTotal < 0)
          padTotal = 0;
        int64_t padBegin, padEnd;
        if (autoPad == "SAME_UPPER") {
          padBegin = padTotal / 2;
          padEnd = padTotal - padBegin;
        } else { // SAME_LOWER
          padEnd = padTotal / 2;
          padBegin = padTotal - padEnd;
        }
        pads[i] = padBegin;
        pads[spatialRank + i] = padEnd;
      }
    } else if (autoPad != "NOTSET") {
      return rewriter.notifyMatchFailure(op, "unknown auto_pad value");
    }

    if (pads.empty())
      pads.assign(2 * spatialRank, 0);
    if ((int64_t)pads.size() != 2 * spatialRank)
      return rewriter.notifyMatchFailure(
          op, "pads length must equal 2 * spatial_rank");
    if ((int64_t)strides.size() != spatialRank ||
        (int64_t)dilations.size() != spatialRank)
      return rewriter.notifyMatchFailure(
          op, "strides / dilations length must equal spatial rank");

    // ===== DPS init tensors =================================================
    //
    // Output shape leading dims (N, C) are passed through from the input.
    // A dynamic spatial output dim is materialized at runtime by emitting
    // the ONNX pooling output-size formula in arith from the (dynamic)
    // input spatial extent. Every window parameter (kernel / stride /
    // dilation / pad) is a compile-time constant by this point (auto_pad
    // already resolved to explicit `pads` above), so the only runtime input
    // is the spatial extent read via `tensor.dim`:
    //
    //   kdTerm = (k - 1) * dilation + 1
    //   t      = in + pad_begin + pad_end - kdTerm
    //   out    = floor(t / stride) + 1        (ceil_mode = 0)
    //          = ceil (t / stride) + 1        (ceil_mode = 1)
    //
    // Mirrors the NOTSET branch of onnx-mlir's
    // `ONNXGenericPoolOpShapeHelper<>::customComputeShape`
    // (src/Dialect/ONNX/ONNXOps/NN/NNHelper.cpp.inc), whose published formula
    // is `O[i] = floor((I[i] + P[i] - ((K[i]-1)*d[i]+1)) / s[i]) + 1`.
    //
    // Before (dynamic spatial AveragePool, kernel=stride=4, no pad):
    //   %y = "onnx.AveragePool"(%x) {kernel_shape=[4,4], strides=[4,4], ...}
    //          : (tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
    // After:
    //   %n   = tensor.dim %x, %c0
    //   %c   = tensor.dim %x, %c1
    //   %h   = tensor.dim %x, %c2
    //   %ho  = arith.addi (arith.floordivsi (arith.addi %h, -4), 4-as-stride),
    //   1
    //   ... (same for W) ...
    //   %init = tensor.empty(%n, %c, %ho, %wo) : tensor<?x?x?x?xf16>
    //   %y    = hip.pool(%ctx) ins(%x) outs(%init) {pool_mode=0, ...}
    auto buildInit = [&](mlir::RankedTensorType resultType) -> mlir::Value {
      llvm::SmallVector<mlir::Value> dynSizes;
      for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
        if (!resultType.isDynamicDim(dimIdx))
          continue;
        if (dimIdx < 2) {
          // N or C — passthrough from input.
          dynSizes.push_back(
              mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx));
          continue;
        }
        // Dynamic spatial dim: materialize out = f(in) in arith. The window
        // params are all compile-time constants, so `in` is the only runtime
        // value; `kdTerm`/`pad`/`stride` fold into two index constants.
        int64_t s = dimIdx - 2; // spatial index
        int64_t kdTerm = (kernelShape[s] - 1) * dilations[s] + 1;
        int64_t addConst = pads[s] + pads[spatialRank + s] - kdTerm;
        mlir::Value inDim =
            mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx);
        mlir::Value t = mlir::arith::AddIOp::create(
            rewriter, loc, inDim,
            mlir::arith::ConstantIndexOp::create(rewriter, loc, addConst));
        mlir::Value strideV =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, strides[s]);
        mlir::Value div =
            ceilMode
                ? mlir::arith::CeilDivSIOp::create(rewriter, loc, t, strideV)
                      .getResult()
                : mlir::arith::FloorDivSIOp::create(rewriter, loc, t, strideV)
                      .getResult();
        mlir::Value outDim = mlir::arith::AddIOp::create(
            rewriter, loc, div,
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
        dynSizes.push_back(outDim);
      }
      return mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                           resultType.getElementType(),
                                           dynSizes);
    };

    mlir::Value yInit = buildInit(outputType);
    llvm::SmallVector<mlir::Value> outputs = {yInit};
    llvm::SmallVector<mlir::Type> resultTypes = {outputType};
    if (numResults == 2) {
      auto idxType =
          mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
      mlir::Value idxInit = buildInit(idxType);
      outputs.push_back(idxInit);
      resultTypes.push_back(idxType);
    }

    // ===== Build hip.pool ===================================================

    auto poolModeAttr = rewriter.getI64IntegerAttr(poolMode);
    auto kernelShapeAttr = rewriter.getI64ArrayAttr(kernelShape);
    auto stridesAttr = rewriter.getI64ArrayAttr(strides);
    auto padsAttr = rewriter.getI64ArrayAttr(pads);
    auto dilationsAttr = rewriter.getI64ArrayAttr(dilations);
    auto ceilModeAttr = rewriter.getI64IntegerAttr(ceilMode);
    auto storageOrderAttr = rewriter.getI64IntegerAttr(storageOrder);
    auto countIncludePadAttr = rewriter.getI64IntegerAttr(countIncludePad);
    auto pAttr = rewriter.getI64IntegerAttr(p);

    llvm::SmallVector<mlir::Value> operands = {context, input};
    operands.append(outputs.begin(), outputs.end());

    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("pool_mode", poolModeAttr));
    attrs.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr));
    attrs.push_back(rewriter.getNamedAttr("strides", stridesAttr));
    attrs.push_back(rewriter.getNamedAttr("pads", padsAttr));
    attrs.push_back(rewriter.getNamedAttr("dilations", dilationsAttr));
    attrs.push_back(rewriter.getNamedAttr("ceil_mode", ceilModeAttr));
    attrs.push_back(rewriter.getNamedAttr("storage_order", storageOrderAttr));
    attrs.push_back(
        rewriter.getNamedAttr("count_include_pad", countIncludePadAttr));
    attrs.push_back(rewriter.getNamedAttr("p", pAttr));

    auto hipOp = mlir::hip::PoolOp::create(
        rewriter, loc, mlir::TypeRange(resultTypes), operands, attrs);
    rewriter.replaceOp(op, hipOp.getResults());
    return mlir::success();
  }
};

} // namespace

void populatePoolConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<PoolToHip>(ctx, "onnx.MaxPool", kPoolMax);
  patterns.add<PoolToHip>(ctx, "onnx.AveragePool", kPoolAverage);
  patterns.add<PoolToHip>(ctx, "onnx.LpPool", kPoolLp);
}

} // namespace hip
} // namespace mlir
