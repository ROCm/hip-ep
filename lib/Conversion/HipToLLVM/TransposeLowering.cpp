/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== Custom HIP kernel ops ================================================

// hip.transpose(%handle, %dim0, %dim1) ins(%input) outs(%output)
//   -> hip_transpose(handle, input, output, rank, dim0, dim1, s0, s1, s2)
// Swaps the two specified dimensions. Pads shape to 3 dims (trailing 1s).
/// hip.transpose lowers to either:
///
///   - the legacy `hip_transpose(handle, in, out, rank, dim0, dim1, s0, s1, s2)`
///     kernel for rank <= 3 (kept for back-compat with existing test_transpose
///     LIT tests); OR
///   - the new generic `hip_transpose_nd(stream, in, out, rank, in_shape,
///     dim0, dim1, hip_dtype)` kernel (in `transpose_kernel.hip`) for
///     rank >= 4 -- needed for Kokoro Conv1D-via-NC1W and any 4-D
///     Transpose with perm = [0, 2, 1, 3].
struct TransposeOpLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inMemRef = cast<MemRefType>(op.getInput().getType());
    int rank = inMemRef.getRank();
    MemRefDescriptor inputDesc(adaptor.getInput());

    int64_t dataType = getHipdnnDataType(inMemRef.getElementType());
    if (dataType < 0)
      return op.emitOpError("hip.transpose unsupported element type");
    int hipDtype = -1;
    switch (dataType) {
    case 0: hipDtype = 0; break; // FLOAT -> FLOAT32
    case 1: hipDtype = 1; break; // HALF  -> FLOAT16
    case 2: hipDtype = 5; break; // BFLOAT16
    case 3: hipDtype = 3; break; // INT32
    case 4: hipDtype = 2; break; // INT64
    default:
      return op.emitOpError("hip.transpose unsupported runtime element type");
    }

    // Stack-allocate in_shape[rank] and fill from memref descriptor.
    auto arrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                            rewriter.getI64IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType,
                                           oneI64, /*alignment=*/8);
    for (int i = 0; i < rank; ++i) {
      Value sz = inputDesc.size(rewriter, loc, i);
      if (sz.getType() != i64Type)
        sz = LLVM::ZExtOp::create(rewriter, loc, i64Type, sz);
      SmallVector<LLVM::GEPArg, 2> idx = {0, static_cast<int32_t>(i)};
      Value elemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, arrayType, alloca, idx);
      LLVM::StoreOp::create(rewriter, loc, sz, elemPtr);
    }

    // Call wrap_transpose(state, input, output, rank, in_shape, dim0,
    //                     dim1, hip_dtype) -- goes through runtime
    // bitcode so the GPU kernel launches from the EP DLL context where
    // __hipRegisterFunction has already run.
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                     ptrType, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "wrap_transpose", paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    Value rankI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                              rewriter.getI64IntegerAttr(rank));
    Value dim0I64 = adaptor.getDim0();
    Value dim1I64 = adaptor.getDim1();
    if (dim0I64.getType() != i64Type)
      dim0I64 = LLVM::ZExtOp::create(rewriter, loc, i64Type, dim0I64);
    if (dim1I64.getType() != i64Type)
      dim1I64 = LLVM::ZExtOp::create(rewriter, loc, i64Type, dim1I64);
    Value dtypeI64 =
        LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                 rewriter.getI64IntegerAttr(hipDtype));

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        rankI64,
        alloca,
        dim0I64,
        dim1I64,
        dtypeI64};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateTransposeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TransposeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
