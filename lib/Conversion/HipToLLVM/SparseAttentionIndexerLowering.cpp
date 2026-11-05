/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"

#include <string>

namespace mlir {
namespace hip {
namespace {

// Returns a pointer to an internal byte global holding `sections`, one byte per
// frequency. Indexers with the same map share one global.
Value sectionMapPtr(ModuleOp module, RewriterBase &rewriter, Location loc,
                    ArrayRef<int32_t> sections) {
  std::string bytes;
  for (int32_t s : sections) {
    bytes.push_back(static_cast<char>(s));
  }
  std::string name =
      "__hip_sai_rotary_sections_" + llvm::toHex(bytes, /*LowerCase=*/true);
  if (auto global = module.lookupSymbol<LLVM::GlobalOp>(name)) {
    return LLVM::AddressOfOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
        global.getSymNameAttr());
  }
  return LLVM::createGlobalString(loc, rewriter, name, bytes,
                                  LLVM::Linkage::Internal);
}

struct SparseAttentionIndexerOpLowering
    : public ConvertOpToLLVMPattern<SparseAttentionIndexerOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SparseAttentionIndexerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto cosType = cast<MemRefType>(op.getCosCache().getType());
    auto pastType = cast<MemRefType>(op.getPastKey().getType());
    const int64_t cosRank = cosType.getRank();
    const int64_t headSize = pastType.getDimSize(2);
    const int64_t numHeads = queryType.getDimSize(2) / headSize - 1;
    const bool gathered = static_cast<bool>(op.getRotaryPositionIds());
    const int64_t rotary = cosType.getDimSize(cosRank - 1) * (gathered ? 2 : 1);

    auto presentType = cast<MemRefType>(op.getPresentKey().getType());
    auto optionalPtr = [&](Value v) -> Value {
      return v ? extractContiguousMemRefPtr(v, rewriter, loc)
               : LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();
    };
    Value maskPtr = optionalPtr(adaptor.getMask());
    Value pastSeqLenPtr = optionalPtr(adaptor.getPastSequenceLength());

    SmallVector<Value, 31> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getQueryNormWeight(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getKeyNormWeight(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getCosCache(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getSinCache(), rewriter, loc),
        maskPtr,
        pastSeqLenPtr,
        extractContiguousMemRefPtr(adaptor.getPastKey(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getSelectedIndices(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPresentKey(), rewriter, loc),
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc),
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc),
        getMemRefDimSize(pastType, 1, adaptor.getPastKey(), rewriter, loc),
        getMemRefDimSize(presentType, 1, adaptor.getPresentKey(), rewriter,
                         loc),
        createI64Const(numHeads),
        createI64Const(headSize),
        createI64Const(rotary),
        getMemRefDimSize(cosType, cosRank - 2, adaptor.getCosCache(), rewriter,
                         loc),
        createI64Const(cosRank == 3 ? 1 : 0),
        createI64Const(op.getCompressRatio()),
        createI64Const(op.getTokenBudget()),
        createI64Const(queryType.getElementTypeBitWidth() / 8),
        getMemRefDimSize(pastType, 0, adaptor.getPastKey(), rewriter, loc)};
    if (adaptor.getMask()) {
      auto maskType = cast<MemRefType>(op.getMask().getType());
      args.push_back(
          getMemRefDimSize(maskType, 0, adaptor.getMask(), rewriter, loc));
      args.push_back(
          getMemRefDimSize(maskType, 1, adaptor.getMask(), rewriter, loc));
    } else {
      args.push_back(createI64Const(0));
      args.push_back(createI64Const(0));
    }
    auto sinType = cast<MemRefType>(op.getSinCache().getType());
    if (cosRank == 3) {
      args.push_back(
          getMemRefDimSize(cosType, 0, adaptor.getCosCache(), rewriter, loc));
      args.push_back(
          getMemRefDimSize(sinType, 0, adaptor.getSinCache(), rewriter, loc));
    } else {
      args.push_back(createI64Const(0));
      args.push_back(createI64Const(0));
    }
    args.push_back(getMemRefDimSize(sinType, cosRank - 2, adaptor.getSinCache(),
                                    rewriter, loc));
    args.push_back(createF32Const(op.getEpsilon().convertToFloat()));
    args.push_back(createF32Const(op.getScale().convertToFloat()));
    args.push_back(optionalPtr(adaptor.getRotaryPositionIds()));
    if (gathered) {
      auto idsType = cast<MemRefType>(op.getRotaryPositionIds().getType());
      Value ids = adaptor.getRotaryPositionIds();
      args.push_back(
          sectionMapPtr(module, rewriter, loc, *op.getRotarySections()));
      for (int64_t dim : llvm::seq<int64_t>(0, 3)) {
        args.push_back(getMemRefDimSize(idsType, dim, ids, rewriter, loc));
      }
    } else {
      args.push_back(LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult());
      args.append(3, createI64Const(0));
    }

    SmallVector<Type, 36> paramTypes(11, ptrType);
    paramTypes.append(18, i64Type);
    paramTypes.append(2, f32Type);
    paramTypes.append(2, ptrType);
    paramTypes.append(3, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSparseAttentionIndexer, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateSparseAttentionIndexerLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<SparseAttentionIndexerOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
