/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

static constexpr const char *kHipDumpTensor = "hipdnn_ep_dump_tensor";

/// Get or create a global NUL-terminated string constant.
/// Returns a pointer (address-space 0) to the first byte.
static Value getOrCreateGlobalString(ConversionPatternRewriter &rewriter,
                                     Location loc, ModuleOp module,
                                     StringRef symName, StringRef value) {
  Type i8Type = rewriter.getI8Type();
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);

  if (!module.lookupSymbol<LLVM::GlobalOp>(symName)) {
    auto arrayType = LLVM::LLVMArrayType::get(i8Type, value.size() + 1);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&module.getBody()->front());
    LLVM::GlobalOp::create(rewriter, loc, arrayType, /*isConstant=*/true,
                           LLVM::Linkage::Internal, symName,
                           rewriter.getStringAttr(value.str() + '\0'));
  }

  Value globalAddr =
      LLVM::AddressOfOp::create(rewriter, loc, ptrType, symName);
  return globalAddr;
}

struct DumpTensorOpLowering
    : public ConvertOpToLLVMPattern<hip::DumpTensorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DumpTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i64Type = rewriter.getI64Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    auto memrefType = cast<MemRefType>(op.getInput().getType());
    int64_t rank = memrefType.getRank();
    int64_t dataType = getHipdnnDataType(memrefType.getElementType());

    Value statePtr = adaptor.getCtx();
    Value gpuPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);

    // Build shape array on stack: alloca rank x i64, store each dim.
    Value rankVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(rank));
    Value shapeArray = LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type,
                                              rankVal, /*alignment=*/0);
    for (int64_t d = 0; d < rank; ++d) {
      Value dimVal =
          getMemRefDimSize(memrefType, d, adaptor.getInput(), rewriter, loc);
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                           rewriter.getI64IntegerAttr(d));
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                       shapeArray, ArrayRef<LLVM::GEPArg>{idx});
      LLVM::StoreOp::create(rewriter, loc, dimVal, slot);
    }

    Value dataTypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(dataType));

    // Global string constants for name and dump_tensors_dir.
    std::string nameStr = op.getName().str();
    std::string dirStr = op.getDumpTensorsDir().str();
    // Use a sanitized version for the global symbol name.
    std::string nameSym = "__dump_name_" + nameStr;
    std::replace(nameSym.begin(), nameSym.end(), '/', '_');
    std::replace(nameSym.begin(), nameSym.end(), '\\', '_');
    std::replace(nameSym.begin(), nameSym.end(), '.', '_');
    std::string dirSym = "__dump_tensors_dir";

    Value namePtr = getOrCreateGlobalString(rewriter, loc, module, nameSym,
                                            nameStr);
    Value dirPtr =
        getOrCreateGlobalString(rewriter, loc, module, dirSym, dirStr);

    // Declare and call hipdnn_ep_dump_tensor.
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType,
                                    i64Type, i64Type, ptrType, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipDumpTensor, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,    gpuPtr,      shapeArray,
                               rankVal,     dataTypeVal, namePtr,
                               dirPtr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateDumpTensorLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns.add<DumpTensorOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
