/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.where(ctx, condition, x, y, output)
//   -> wrap_where(state, cond_ptr, x_ptr, y_ptr, out_ptr,
//                 cond_shape_ptr, cond_rank,
//                 x_shape_ptr,    x_rank,
//                 y_shape_ptr,    y_rank,
//                 out_shape_ptr,  out_rank,
//                 data_type)
//
// Where has no fixed layout: each operand may be any rank and ONNX
// multidirectional (NumPy-style) broadcasting applies. For each operand we
// alloca a stack array of its real dim sizes (compile-time constants for
// static dims, descriptor.size() for dynamic dims), pass the pointer plus
// rank to the runtime, and let the runtime left-pad each operand shape to
// the output rank before launching the HIP kernel.
//
// data_type identifies the X/Y/output element type (HIPDNN_EP_DATATYPE_*);
// the condition is always 1-byte bool.
//
// CONDITION BUFFER LAYOUT ASSUMPTION (1 byte / element):
// The condition memref carries an ONNX `tensor(bool)`. Two element-type
// encodings reach this lowering in practice:
//   * MLIR-native `i1`     -- emitted by hand-written IR (e.g. lit tests).
//   * 8-bit integer (`i8`/
//     `ui8`/`si8`)         -- emitted by the ORT/morphizen ONNX -> MLIR
//                              frontend, which mirrors the on-disk
//                              `TensorProto` BOOL layout (1 byte / element).
//
// Both encodings occupy exactly 1 byte per element under the default
// LLVM/MLIR memref-to-LLVM lowering and are ABI-compatible with C/C++
// `bool`, so we can safely reinterpret the device pointer as
// `const bool *` in the runtime kernel. This lowering therefore passes
// the raw `alignedPtr` of the condition memref straight to `wrap_where`,
// which forwards it to the HIP kernel as `const bool *`. We do *not*
// re-cast the buffer (e.g. to i8) in IR -- the contract is implicit.
//
// If a future MLIR upgrade switches to bit-packed `i1` memrefs (multiple
// bools per byte), both this lowering and
// `lib/Runtime/Kernels/hip/elementwise_where_kernel.hip` must be
// updated together; otherwise the kernel will read wrong offsets
// silently. See the matching comment block at the top of that .hip file.
struct WhereOpLowering : public ConvertOpToLLVMPattern<WhereOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto condType = cast<MemRefType>(op.getCondition().getType());
    auto xType = cast<MemRefType>(op.getX().getType());
    auto yType = cast<MemRefType>(op.getY().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Defensive check on the implicit 1-byte-per-element layout contract
    // (see header comment). ONNX requires condition to be bool, but the
    // ORT/morphizen frontend encodes that as a 1-byte integer (`ui8` /
    // `i8`), while hand-written IR (lit tests) typically uses MLIR-native
    // `i1`. Both encodings are 1 byte/element under the default memref-
    // to-LLVM lowering and ABI-compatible with C/C++ `bool`, so the kernel
    // can read either as `const bool *`. We accept i1 *or* any 8-bit
    // integer here and reject anything wider, so violations of the contract
    // (e.g. i32 condition from a buggy upstream pass) fail the pattern
    // explicitly instead of silently miscomputing.
    Type condElemTy = condType.getElementType();
    if (!condElemTy.isInteger(1) && !condElemTy.isInteger(8))
      return rewriter.notifyMatchFailure(
          op, "hip.where: condition must be a 1-byte integer "
              "(i1 / i8 / ui8 / si8) per the implicit "
              "1-byte-per-element layout contract");

    // X / Y / output must share the same element type per the ONNX `Where`
    // type constraint T. Without this guard, mismatched operands (e.g. due
    // to a shape-inference bug or hand-written IR) would be reinterpreted
    // by the kernel using the *output* dtype's element size against the
    // operand buffers, silently producing wrong results.
    Type xElemTy = xType.getElementType();
    Type yElemTy = yType.getElementType();
    Type outElemTy = outputType.getElementType();
    if (xElemTy != yElemTy || xElemTy != outElemTy)
      return rewriter.notifyMatchFailure(
          op, "hip.where: x/y/output element types must match");

    // Restrict to the dtypes actually dispatched by `wrap_where` /
    // `hip_elementwise_where`. We deliberately do *not* fall back to the
    // full `getHipdnnDataType` set here: types like i8 / ui8 / f64 would
    // map to a HIPDNN_EP_DATATYPE_* value that the runtime's
    // `hipdnn_to_hip_dtype_where` rejects, surfacing the failure only at
    // execution time. By gating in the lowering, an unsupported Where node
    // fails the conversion pattern and the EP can fall back to CPU.
    //
    // Keep this list in sync with `hipdnn_to_hip_dtype_where` in
    // lib/Runtime/real/where.cpp and the dtype switch in
    // lib/Runtime/Kernels/hip/elementwise_where_kernel.hip. We do NOT
    // extend the runtime/kernel dtype set as part of this lowering; if a
    // new dtype is needed, add it in all three places together.
    bool dtypeSupported = outElemTy.isF32() || outElemTy.isF16() ||
                          outElemTy.isBF16() || outElemTy.isInteger(32) ||
                          outElemTy.isInteger(64);
    if (!dtypeSupported)
      return rewriter.notifyMatchFailure(
          op, "hip.where: x/y/output dtype not supported by runtime kernel "
              "(allowed: f32, f16, bf16, i32, i64)");

    int64_t dataType = getHipdnnDataType(outElemTy);
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.where: unsupported output element type");

    // alloca a stack [max(rank, 1) x i64] array, store each dim size, return
    // the base pointer. Uses max(rank, 1) so rank-0 (scalar) operands still
    // have a valid allocation; the runtime ignores the buffer when rank == 0.
    Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                         rewriter.getI64IntegerAttr(1));
    auto emitShapeArray = [&](MemRefType type, Value descriptor) -> Value {
      int rank = type.getRank();
      int arrLen = std::max(rank, 1);
      auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (int i = 0; i < rank; ++i) {
        Value dim = getMemRefDimSize(type, i, descriptor, rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value condShape = emitShapeArray(condType, adaptor.getCondition());
    Value xShape = emitShapeArray(xType, adaptor.getX());
    Value yShape = emitShapeArray(yType, adaptor.getY());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value condRank = createI64Const(condType.getRank());
    Value xRank = createI64Const(xType.getRank());
    Value yRank = createI64Const(yType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value dataTypeVal = createI64Const(dataType);

    // Signature: state, 4 data ptrs, 4 (shape_ptr, rank) pairs, data_type.
    //            = 5 ptrs + 4 ptrs + 4 i64 + 1 i64 = 14 params
    SmallVector<Type, 14> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, // state + 4 data ptrs
        ptrType, i64Type,                            // cond_shape, cond_rank
        ptrType, i64Type,                            // x_shape,    x_rank
        ptrType, i64Type,                            // y_shape,    y_rank
        ptrType, i64Type,                            // out_shape,  out_rank
        i64Type};                                    // data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapWhere, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 14> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getCondition(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        condShape,
        condRank,
        xShape,
        xRank,
        yShape,
        yRank,
        outShape,
        outRank,
        dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateWhereLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<WhereOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
