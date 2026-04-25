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

    if (rank <= 3) {
      // Legacy fast-path matching the existing hip_transpose runtime.
      SmallVector<Type> paramTypes = {ptrType,   ptrType,   ptrType,
                                      indexType, indexType, indexType,
                                      indexType, indexType, indexType};
      FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
          rewriter, module, kHipTranspose, paramTypes, voidType);
      if (failed(funcOp))
        return failure();
      Value rankVal = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                               rewriter.getIndexAttr(rank));
      Value one = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                           rewriter.getIndexAttr(1));
      SmallVector<Value, 3> shape;
      for (int dimIdx : llvm::seq<int>(3))
        shape.push_back(dimIdx < rank ? inputDesc.size(rewriter, loc, dimIdx)
                                      : one);
      SmallVector<Value> args = {
          adaptor.getCtx(),
          extractMemRefPtr(adaptor.getInput(), rewriter, loc),
          extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
          rankVal,
          adaptor.getDim0(),
          adaptor.getDim1(),
          shape[0],
          shape[1],
          shape[2]};
      LLVM::CallOp::create(rewriter, loc, *funcOp, args);
      rewriter.eraseOp(op);
      return success();
    }

    // Rank >= 4: use the generic hip_transpose_nd kernel.  Stride
    // computation is done inside the kernel from the input shape, so
    // we pass an i64* of in_shape on the stack.
    int64_t dataType = getHipdnnDataType(inMemRef.getElementType());
    if (dataType < 0)
      return op.emitOpError("hip.transpose unsupported element type");
    // Map HIPDNN_EP_DATATYPE_* to HIP_DTYPE_* (same mapping the
    // runtime wrappers use).  See lib/Runtime/real/cumsum.cpp.
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

    auto arrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                            rewriter.getI64IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType,
                                           oneI64, /*alignment=*/8);
    for (int i = 0; i < rank; ++i) {
      Value sz = inputDesc.size(rewriter, loc, i);
      // Cast index -> i64 if needed.
      if (sz.getType() != i64Type)
        sz = LLVM::ZExtOp::create(rewriter, loc, i64Type, sz);
      SmallVector<LLVM::GEPArg, 2> idx = {0, static_cast<int32_t>(i)};
      Value elemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, arrayType, alloca, idx);
      LLVM::StoreOp::create(rewriter, loc, sz, elemPtr);
    }

    SmallVector<Type> paramTypesNd = {ptrType, ptrType, ptrType, i32Type,
                                       ptrType, i32Type, i32Type, i32Type};
    FailureOr<LLVM::LLVMFuncOp> funcOpNd = LLVM::lookupOrCreateFn(
        rewriter, module, "hip_transpose_nd", paramTypesNd, i32Type);
    if (failed(funcOpNd))
      return failure();
    Value rankI32 = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                              rewriter.getI32IntegerAttr(rank));
    // dim0 / dim1 come in as `index`; truncate to i32.
    Value dim0I32 = LLVM::TruncOp::create(rewriter, loc, i32Type,
                                           adaptor.getDim0());
    Value dim1I32 = LLVM::TruncOp::create(rewriter, loc, i32Type,
                                           adaptor.getDim1());
    Value dtypeI32 =
        LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                 rewriter.getI32IntegerAttr(hipDtype));
    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        rankI32,
        alloca,
        dim0I32,
        dim1I32,
        dtypeI32};
    LLVM::CallOp::create(rewriter, loc, *funcOpNd, args);
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
