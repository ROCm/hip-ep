/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"
#include "hip/Dialect/Transforms/BufferUtils.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace hip {
namespace {

Value i64Const(ConversionPatternRewriter &rewriter, Location loc,
               int64_t value) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  rewriter.getI64IntegerAttr(value));
}

/// Stack array of \p values (i64) for passing host metadata to the runtime.
Value buildI64Array(ConversionPatternRewriter &rewriter, Location loc,
                    ArrayRef<Value> values) {
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Type i64Type = rewriter.getI64Type();
  auto arrayType = LLVM::LLVMArrayType::get(i64Type, values.size());
  Value array = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType,
                                       i64Const(rewriter, loc, 1),
                                       /*alignment=*/8);
  for (auto [i, v] : llvm::enumerate(values)) {
    Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, array,
                                    ArrayRef<LLVM::GEPArg>{(int32_t)i});
    LLVM::StoreOp::create(rewriter, loc, v, gep);
  }
  return array;
}

// Bool (i1) buffers hold one byte per element, as hip.alloc sizes them.
std::optional<int64_t> elementBytes(Type elemType) {
  if (elemType.isInteger(1))
    return 1;
  if (!elemType.isIntOrFloat() || elemType.getIntOrFloatBitWidth() % 8 != 0)
    return std::nullopt;
  return elemType.getIntOrFloatBitWidth() / 8;
}

// Identity layout, or a static view whose elements form one flat run and so
// differs from identity only by its descriptor offset.
bool isContiguousView(MemRefType type) {
  if (type.getLayout().isIdentity())
    return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.hasStaticShape() &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         isContiguousRowMajor(type.getShape(), strides);
}

// hip.reverse(%ctx) ins(%in) outs(%out) {axes = [...]}
//   -> wrap_strided_copy_nd(state, out, in_last, elem_bytes, rank, sizes,
//                           src_strides, dst_strides)
//
// The input is read through its descriptor, so a strided view (the sliced
// tensor of a negative-step Slice) needs no identity-layout copy. On a
// reversed axis d the source starts at index size[d] - 1 and walks with stride
// -stride[d]:
//
//   in  : memref<4x49xf32, strided<[51, 1], offset: 1>>, axes = [1]
//   src = aligned + (1 + 48 * 1) elements, sizes = [4, 49],
//   src_strides = [51, -1], dst_strides = [49, 1]
struct ReverseOpLowering : public ConvertOpToLLVMPattern<ReverseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected memref operands");
    std::optional<int64_t> elemBytes = elementBytes(inputType.getElementType());
    if (!elemBytes)
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    Type llvmElemType =
        getTypeConverter()->convertType(inputType.getElementType());
    if (!llvmElemType)
      return rewriter.notifyMatchFailure(op, "unconvertible element type");

    int64_t rank = inputType.getRank();
    llvm::SmallDenseSet<int64_t> reversed;
    for (Attribute a : op.getAxes())
      reversed.insert(cast<IntegerAttr>(a).getInt());

    MemRefDescriptor in(adaptor.getInput());
    MemRefDescriptor out(adaptor.getOutput());
    Value offset = in.offset(rewriter, loc);
    Value one = i64Const(rewriter, loc, 1);
    SmallVector<Value> sizes, srcStrides, dstStrides;
    for (int64_t d : llvm::seq<int64_t>(rank)) {
      Value size = in.size(rewriter, loc, d);
      Value stride = in.stride(rewriter, loc, d);
      if (reversed.contains(d)) {
        Value last = LLVM::SubOp::create(rewriter, loc, size, one);
        offset = LLVM::AddOp::create(
            rewriter, loc, offset,
            LLVM::MulOp::create(rewriter, loc, last, stride));
        stride = LLVM::SubOp::create(rewriter, loc, i64Const(rewriter, loc, 0),
                                     stride);
      }
      sizes.push_back(size);
      srcStrides.push_back(stride);
      dstStrides.push_back(out.stride(rewriter, loc, d));
    }

    Value srcBase = in.alignedPtr(rewriter, loc);
    Value srcPtr =
        LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), llvmElemType,
                            srcBase, ValueRange{offset});
    if (cast<LLVM::LLVMPointerType>(srcPtr.getType()).getAddressSpace() != 0)
      srcPtr =
          LLVM::AddrSpaceCastOp::create(rewriter, loc, getPtrType(), srcPtr);
    Value dstPtr = extractMemRefDataPtr(adaptor.getOutput(), outputType,
                                        getTypeConverter(), rewriter, loc);
    if (!dstPtr)
      return rewriter.notifyMatchFailure(op, "cannot form output pointer");

    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    // int wrap_strided_copy_nd(RuntimeState*, void* dst, const void* src,
    //                          int64_t elem_bytes, int64_t rank,
    //                          const int64_t* sizes,
    //                          const int64_t* src_strides,
    //                          const int64_t* dst_strides)
    FailureOr<LLVM::LLVMFuncOp> fn = LLVM::lookupOrCreateFn(
        rewriter, op->getParentOfType<ModuleOp>(), kWrapStridedCopyNd,
        {ptrType, ptrType, ptrType, i64Type, i64Type, ptrType, ptrType,
         ptrType},
        rewriter.getI32Type());
    if (failed(fn))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *fn,
                         ValueRange{adaptor.getCtx(), dstPtr, srcPtr,
                                    i64Const(rewriter, loc, *elemBytes),
                                    i64Const(rewriter, loc, rank),
                                    buildI64Array(rewriter, loc, sizes),
                                    buildI64Array(rewriter, loc, srcStrides),
                                    buildI64Array(rewriter, loc, dstStrides)});
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.concat(%ctx) ins(%a, %b, ...) outs(%out) {axis = k}
//   -> wrap_concat(state, out, num_inputs, src_ptrs, src_widths, outer,
//                  elem_bytes)
//
// The kernel reads each input and writes the output as packed rows from one
// pointer. An input may be a contiguous view (hip-promote-strided-operands
// keeps those), so its pointer includes the descriptor offset. The output
// must have identity layout. outer is the product of the output dims before
// `axis`; an input's width is the product of its dims from `axis` on:
//
//   ins(memref<8x2x3xf32>, memref<8x2x5xf32>) outs(memref<8x2x8xf32>), axis 2
//   -> outer = 16, src_widths = [3, 5]
struct ConcatOpLowering : public ConvertOpToLLVMPattern<ConcatOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(op, "expected memref output");
    if (!outputType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(op,
                                         "output must have identity layout");
    std::optional<int64_t> elemBytes =
        elementBytes(outputType.getElementType());
    if (!elemBytes)
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    if (!getTypeConverter()->convertType(outputType.getElementType()))
      return rewriter.notifyMatchFailure(op, "unconvertible element type");
    for (Value input : op.getInputs())
      if (!isContiguousView(cast<MemRefType>(input.getType())))
        return rewriter.notifyMatchFailure(op, "input must be contiguous");
    int64_t axis = op.getAxis();

    Value outer = i64Const(rewriter, loc, 1);
    for (int64_t d : llvm::seq<int64_t>(axis))
      outer = LLVM::MulOp::create(
          rewriter, loc, outer,
          getMemRefDimSize(outputType, d, adaptor.getOutput(), rewriter, loc));

    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    SmallVector<Value> widths, srcPtrs;
    for (auto [input, converted] :
         llvm::zip_equal(op.getInputs(), adaptor.getInputs())) {
      auto inputType = cast<MemRefType>(input.getType());
      Value width = i64Const(rewriter, loc, 1);
      for (int64_t d : llvm::seq<int64_t>(axis, inputType.getRank()))
        width = LLVM::MulOp::create(
            rewriter, loc, width,
            getMemRefDimSize(inputType, d, converted, rewriter, loc));
      widths.push_back(width);
      srcPtrs.push_back(extractMemRefDataPtr(
          converted, inputType, getTypeConverter(), rewriter, loc));
    }

    auto ptrArrayType = LLVM::LLVMArrayType::get(ptrType, srcPtrs.size());
    Value ptrArray = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, ptrArrayType, i64Const(rewriter, loc, 1),
        /*alignment=*/8);
    for (auto [i, p] : llvm::enumerate(srcPtrs)) {
      Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType, ptrArray,
                                      ArrayRef<LLVM::GEPArg>{(int32_t)i});
      LLVM::StoreOp::create(rewriter, loc, p, gep);
    }

    Value dstPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    // int wrap_concat(RuntimeState*, void* dst, int64_t num_inputs,
    //                 void* const* src_ptrs, const int64_t* src_widths,
    //                 int64_t outer, int64_t elem_bytes)
    FailureOr<LLVM::LLVMFuncOp> fn = LLVM::lookupOrCreateFn(
        rewriter, op->getParentOfType<ModuleOp>(), kWrapConcat,
        {ptrType, ptrType, i64Type, ptrType, ptrType, i64Type, i64Type},
        rewriter.getI32Type());
    if (failed(fn))
      return failure();
    LLVM::CallOp::create(
        rewriter, loc, *fn,
        ValueRange{adaptor.getCtx(), dstPtr,
                   i64Const(rewriter, loc, (int64_t)srcPtrs.size()), ptrArray,
                   buildI64Array(rewriter, loc, widths), outer,
                   i64Const(rewriter, loc, *elemBytes)});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateCopyOpsLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.add<ReverseOpLowering, ConcatOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
