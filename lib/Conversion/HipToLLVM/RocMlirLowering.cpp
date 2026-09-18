/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace hip {
namespace {

// hip.rocmlir(ctx) @callee ins(%in...) outs(%out)
//   {kernel_binary, grid_size, block_size}
//     ->
//   %kernargs = alloca [N x ptr]            (N = #inputs + 1 output)
//   store in0_ptr,  kernargs[0]
//   ...
//   store out_ptr,  kernargs[N-1]
//   wrap_rocmlir(state, kernel_binary_str, func_name_str,
//                block_size, grid_size, kernargs, N * sizeof(void*))
//
// The compiled GPU kernel (embedded in `kernel_binary`) takes the operand data
// pointers positionally: inputs in order, then the output. We stage those bare
// pointers into a contiguous kernargs buffer and hand it, together with the
// launch geometry and the module/function names, to the runtime wrapper.
struct RocMlirOpLowering : public ConvertOpToLLVMPattern<RocMlirOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Create (or reuse) a module-level global holding `value` and return a
  // pointer to its first byte. `baseName` is uniqued against the module symbol
  // table so distinct kernels never collide.
  static Value globalString(ModuleOp module, RewriterBase &rewriter,
                            Location loc, StringRef baseName, StringRef value) {
    std::string name = baseName.str();
    unsigned suffix = 0;
    while (module.lookupSymbol(name))
      name = (baseName + "_" + Twine(suffix++)).str();
    return LLVM::createGlobalString(loc, rewriter, name, value,
                                    LLVM::Linkage::Internal);
  }

  LogicalResult
  matchAndRewrite(RocMlirOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto i64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // Kernel ELF blob and callee function name as module-level strings. The
    // ELF carries embedded NULs, so keep the exact byte length (createGlobal
    // string stores the StringRef's full range, no NUL-termination scan).
    StringRef callee = op.getKernel();
    Value kernelBinary =
        globalString(module, rewriter, loc, (callee + "_binary").str(),
                     op.getKernelBinary());
    // NUL-terminate the function name so a C `char*` consumer can read it.
    std::string funcNameStr = callee.str();
    funcNameStr.push_back('\0');
    Value funcName = globalString(module, rewriter, loc,
                                  (callee + "_name").str(), funcNameStr);

    // Operand data pointers, in kernel-arg order: inputs first, then output.
    SmallVector<Value> argPtrs;
    for (Value in : adaptor.getInputs())
      argPtrs.push_back(extractContiguousMemRefPtr(in, rewriter, loc));
    argPtrs.push_back(
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc));

    // alloc(kernargs, size): stack buffer of N pointer slots.
    int64_t numArgs = static_cast<int64_t>(argPtrs.size());
    auto arrayTy = LLVM::LLVMArrayType::get(ptrType, numArgs);
    Value kernargs = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayTy,
                                            i64Const(1), /*alignment=*/8);

    // set_value(kernargs, ptr): store each operand pointer into its slot.
    for (int64_t i : llvm::seq<int64_t>(0, numArgs)) {
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                           rewriter.getI32IntegerAttr(i));
      Value slot =
          LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType, kernargs, idx);
      LLVM::StoreOp::create(rewriter, loc, argPtrs[i], slot);
    }

    // Byte size of the kernargs buffer (N * sizeof(void*)).
    Value kernargsSize = i64Const(numArgs * 8);

    // wrap_rocmlir(RuntimeState* state, const char* kernel_binary,
    //              char* func_name, int64_t block_size, int64_t grid_size,
    //              void* kernargs, size_t size)
    Value statePtr = adaptor.getCtx();
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, ptrType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRocMlir, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,
                                  kernelBinary,
                                  funcName,
                                  i64Const(op.getBlockSize()),
                                  i64Const(op.getGridSize()),
                                  kernargs,
                                  kernargsSize};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateRocMlirLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.add<RocMlirOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
