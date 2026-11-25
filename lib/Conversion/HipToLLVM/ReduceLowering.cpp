/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "mlir/IR/SymbolTable.h"

#include <optional>

namespace mlir {
namespace hip {
namespace {

// The single axis stored in a constant axes memref, or failure when that
// memref is not a one-element constant. keepdims=0 deletes the axis, so
// matching input and output shapes from the trailing end is ambiguous when
// the deleted extent equals a surviving neighbor. By the time this pattern
// runs, the constant may still be a memref.global or already an
// llvm.mlir.global.
static FailureOr<int64_t> constantReducedAxis(Value axes, int64_t rank) {
  Value root = axes;
  while (auto castOp = root.getDefiningOp<memref::CastOp>())
    root = castOp.getSource();
  auto getGlobal = root.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return failure();
  auto module = getGlobal->getParentOfType<ModuleOp>();
  if (!module)
    return failure();
  Operation *sym = SymbolTable::lookupSymbolIn(module, getGlobal.getNameAttr());
  if (!sym)
    return failure();
  Attribute init;
  if (auto global = dyn_cast<memref::GlobalOp>(sym))
    init = global.getInitialValueAttr();
  else if (auto global = dyn_cast<LLVM::GlobalOp>(sym)) {
    if (std::optional<Attribute> value = global.getValue())
      init = *value;
  }
  auto dense = dyn_cast_or_null<DenseIntElementsAttr>(init);
  if (!dense || dense.getNumElements() != 1)
    return failure();
  int64_t axis = (*dense.getValues<APInt>().begin()).getSExtValue();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return failure();
  return axis;
}

// Generic template for reduction-style ops with the shared
//   (data, axes, output, keepdims, noop_with_empty_axes)
// signature, lowered to wrap_{op}.
//
// Handles: hip.reduce_sum, hip.reduce_mean, hip.reduce_max, hip.reduce_min,
// hip.reduce_prod.
//
// All three runtime functions share the exact same calling convention:
//   int wrap_reduce_*(RuntimeState* state, void* data, void* axes,
//                     void* output, int64_t data_num_elements,
//                     int64_t output_num_elements,
//                     int64_t axes_num_elements, int64_t data_type,
//                     int64_t keepdims, int64_t noop_with_empty_axes)
//
// Supports both static and dynamic shapes (computes num_elements at runtime).

// The single axis whose removal (keepdims=0) or collapse to 1 (keepdims=1)
// turns `in` into `out`, or -1 if none or more than one axis fits. A dynamic
// dim matches only a dynamic dim.
static int64_t inferSingleReducedAxis(ArrayRef<int64_t> in,
                                      ArrayRef<int64_t> out, bool keepdims) {
  const int64_t rank = static_cast<int64_t>(in.size());
  if (static_cast<int64_t>(out.size()) != (keepdims ? rank : rank - 1)) {
    return -1;
  }
  auto same = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) ? ShapedType::isDynamic(b) : a == b;
  };
  int64_t found = -1;
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (in[axis] == 1) {
      continue;
    }
    bool fits = true;
    for (int64_t i = 0, o = 0; i < rank && fits; ++i) {
      if (i == axis) {
        if (keepdims) {
          fits = out[o++] == 1;
        }
        continue;
      }
      fits = same(in[i], out[o++]);
    }
    if (fits) {
      if (found >= 0) {
        return -1;
      }
      found = axis;
    }
  }
  return found;
}

template <typename OpTy>
struct ReduceOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  const char *funcName;
  const char *opName;

  ReduceOpLowering(const LLVMTypeConverter &converter, const char *func,
                   const char *op)
      : ConvertOpToLLVMPattern<OpTy>(converter), funcName(func), opName(op) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value axesPtr =
        extractContiguousMemRefPtr(adaptor.getAxes(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto axesType = cast<MemRefType>(op.getAxes().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataTypeEnum = getHipdnnDataType(dataType.getElementType());
    if (dataTypeEnum < 0) {
      std::string msg = "unsupported element type for hip.";
      msg += opName;
      return rewriter.notifyMatchFailure(op, msg);
    }

    Value dataNumElements =
        computeNumElements(dataType, adaptor.getData(), rewriter, loc);
    Value outputNumElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);
    Value axesNumElements =
        computeNumElements(axesType, adaptor.getAxes(), rewriter, loc);

    Value dataTypeVal = createI64Const(dataTypeEnum);
    Value keepdimsVal = createI64Const(op.getKeepdims());
    Value noopWithEmptyAxesVal = createI64Const(op.getNoopWithEmptyAxes());

    // inner_size = product of the input dims after the last reduced axis. The
    // runtime kernel strides by it, so a non-trailing reduce (e.g. NCHW
    // channel-axis LayerNorm2d, axes=[1]) reads the right elements; it handles
    // one contiguous run of reduced axes.
    //
    // `reduced_axes` gives the axes directly. Without that attribute, a
    // one-element constant axes memref still names the axis. Shape inference
    // is only the fallback: the first trailing mismatch for static shapes,
    // inferSingleReducedAxis otherwise. Shapes alone cannot always identify
    // the axis (e.g. [1,6,5,5] -> [1,6,5] or [4,4,4] -> [4,4]).
    auto innerSizeAfter = [&](int64_t axis) -> Value {
      int64_t staticInner = 1;
      SmallVector<int64_t> dynamicDims;
      for (int64_t i = axis + 1; i < dataType.getRank(); ++i) {
        if (dataType.isDynamicDim(i)) {
          dynamicDims.push_back(i);
        } else {
          staticInner *= dataType.getDimSize(i);
        }
      }
      Value inner = createI64Const(staticInner);
      MemRefDescriptor desc(adaptor.getData());
      for (int64_t i : dynamicDims) {
        inner = LLVM::MulOp::create(rewriter, loc, inner,
                                    desc.size(rewriter, loc, i));
      }
      return inner;
    };

    Value innerSizeVal;
    if (std::optional<ArrayRef<int64_t>> axes = op.getReducedAxes()) {
      if (axes->empty()) {
        innerSizeVal = createI64Const(1);
      } else {
        if (axes->back() >= dataType.getRank()) {
          return rewriter.notifyMatchFailure(op, "reduced_axes out of range");
        }
        for (int64_t i = axes->front(); i < axes->back(); ++i) {
          if (!llvm::is_contained(*axes, i) && dataType.getDimSize(i) != 1) {
            return rewriter.notifyMatchFailure(
                op, "reduced_axes must be contiguous");
          }
        }
        innerSizeVal = innerSizeAfter(axes->back());
      }
    } else if (FailureOr<int64_t> axis =
                   constantReducedAxis(op.getAxes(), dataType.getRank());
               succeeded(axis)) {
      innerSizeVal = innerSizeAfter(*axis);
    } else if (dataType.hasStaticShape() && outputType.hasStaticShape()) {
      int64_t innerSize = 1;
      llvm::ArrayRef<int64_t> inShape = dataType.getShape();
      llvm::ArrayRef<int64_t> outShape = outputType.getShape();
      int64_t i = static_cast<int64_t>(inShape.size()) - 1;
      int64_t o = static_cast<int64_t>(outShape.size()) - 1;
      while (i >= 0 && o >= 0 && inShape[i] == outShape[o]) {
        innerSize *= inShape[i];
        --i;
        --o;
      }
      int64_t outNum = 1;
      for (int64_t d : outShape)
        outNum *= d;
      if (innerSize <= 0 || (outNum % innerSize) != 0)
        innerSize = 1;
      innerSizeVal = createI64Const(innerSize);
    } else {
      int64_t axis = inferSingleReducedAxis(
          dataType.getShape(), outputType.getShape(), op.getKeepdims() != 0);
      innerSizeVal = axis >= 0 ? innerSizeAfter(axis) : createI64Const(1);
    }

    SmallVector<Type, 11> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, funcName, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 11> args = {statePtr,        dataPtr,
                                   axesPtr,         outputPtr,
                                   dataNumElements, outputNumElements,
                                   axesNumElements, dataTypeVal,
                                   keepdimsVal,     noopWithEmptyAxesVal,
                                   innerSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateReduceLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.insert<ReduceOpLowering<ReduceSumOp>>(converter, kWrapReduceSum,
                                                 "reduce_sum");
  patterns.insert<ReduceOpLowering<ReduceMeanOp>>(converter, kWrapReduceMean,
                                                  "reduce_mean");
  patterns.insert<ReduceOpLowering<ReduceL2Op>>(converter, kWrapReduceL2,
                                                "reduce_l2");
  patterns.insert<ReduceOpLowering<ReduceMaxOp>>(converter, kWrapReduceMax,
                                                 "reduce_max");
  patterns.insert<ReduceOpLowering<ReduceMinOp>>(converter, kWrapReduceMin,
                                                 "reduce_min");
  patterns.insert<ReduceOpLowering<ReduceProdOp>>(converter, kWrapReduceProd,
                                                  "reduce_prod");
}

} // namespace hip
} // namespace mlir
