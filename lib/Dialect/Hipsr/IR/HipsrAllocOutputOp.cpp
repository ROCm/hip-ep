/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace mlir {
namespace hipsr {

void AllocOutputOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Allocate on the result so --buffer-deallocation-pipeline treats a returned
  // alloc_output as owned and does not clone it:
  //
  //   %out = hipsr.alloc_output(%ctx) {out_idx = 0}
  //   hipsr.cast(%ctx) ins(%in) outs(%out)
  //   return %out
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0),
                       SideEffects::DefaultResource::get());
}

LogicalResult AllocOutputOp::verify() {
  auto memrefTy = cast<MemRefType>(getMemref().getType());
  if (static_cast<int64_t>(getDynamicSizes().size()) !=
      memrefTy.getNumDynamicDims())
    return emitOpError("expected ")
           << memrefTy.getNumDynamicDims() << " dynamic size operand(s), got "
           << getDynamicSizes().size();
  return success();
}

} // namespace hipsr
} // namespace mlir

namespace {

constexpr const char *kAllocOutput = "hipdnn_ep_alloc_output";

struct AllocOutputLowering : public ConvertOpToLLVMPattern<AllocOutputOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AllocOutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MemRefType memRefType = cast<MemRefType>(op.getMemref().getType());

    if (!isConvertibleAndHasIdentityMaps(memRefType)) {
      return rewriter.notifyMatchFailure(op, "incompatible memref type");
    }

    Type elemType = memRefType.getElementType();
    if (!elemType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }
    // Round up to whole bytes so a sub-byte type reports at least one: an i1
    // output occupies a byte per element, not an eighth of one.
    int64_t elemSizeBytes = (elemType.getIntOrFloatBitWidth() + 7) / 8;

    int64_t rank = memRefType.getRank();

    // Interleaves the static extents from the type with the dynamic-size
    // operands, in type order, and derives row-major strides.
    llvm::SmallVector<Value> sizes;
    llvm::SmallVector<Value> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, sizeBytes,
                             /*sizeInBytes=*/true);

    // The callback takes the shape as an i64 array. Unlike the hip dialect's
    // alloc_output, there is no ABI-rank fixup to apply: the hipsr output
    // allocator hands back a buffer of the op's own type, so the shape the
    // runtime sees is the memref's own.
    Value shapeArray = emitHostI64Array(sizes, rewriter, loc);

    // void *hipdnn_ep_alloc_output(void *state, int64_t out_idx,
    //                              const int64_t *shape, int64_t rank,
    //                              int64_t elem_size)
    using AllocOutputCall =
        RuntimeFunc<hostPtr, hostPtr, i64, hostPtr, i64, i64>;
    auto allocFunc =
        AllocOutputCall::lookupOrCreateFn(rewriter, loc, module, kAllocOutput);
    if (failed(allocFunc)) {
      return failure();
    }
    FailureOr<Value> rawPtr = allocFunc->call(
        adaptor.getCtx(), static_cast<int64_t>(op.getOutIdx()), shapeArray,
        rank, elemSizeBytes);
    if (failed(rawPtr)) {
      return failure();
    }

    // The callback returns a generic (address space 0) pointer, so cast it
    // into the memref's space when they differ.
    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace)) {
      return failure();
    }

    Value dataPtr = *rawPtr;
    if (cast<LLVM::LLVMPointerType>(dataPtr.getType()).getAddressSpace() !=
        *addrSpace)
      dataPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          dataPtr);

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, dataPtr, dataPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrAllocOutputLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<AllocOutputLowering>(converter);
}
