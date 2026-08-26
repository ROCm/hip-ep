/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPSR_TO_LLVM_H
#define HIP_CONVERSION_HIPSR_TO_LLVM_H

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace hipsr {

namespace detail {

struct I32Tag {};
struct I64Tag {};
struct HostPtrTag {};
struct DevicePtrTag {};
struct SlotIndexTag {};

template <typename Tag> struct TypeMaterializer;

template <> struct TypeMaterializer<I32Tag> {
  static Type get(MLIRContext *, OpBuilder &builder) {
    return builder.getI32Type();
  }
};

template <> struct TypeMaterializer<I64Tag> {
  static Type get(MLIRContext *, OpBuilder &builder) {
    return builder.getI64Type();
  }
};

template <> struct TypeMaterializer<HostPtrTag> {
  static Type get(MLIRContext *ctx, OpBuilder &) {
    return LLVM::LLVMPointerType::get(ctx, 0);
  }
};

template <> struct TypeMaterializer<DevicePtrTag> {
  static Type get(MLIRContext *ctx, OpBuilder &) {
    return LLVM::LLVMPointerType::get(ctx, 1);
  }
};

template <> struct TypeMaterializer<SlotIndexTag> {
  static Type get(MLIRContext *, OpBuilder &builder) {
    return builder.getI32Type();
  }
};

template <typename Tag>
Type materializeType(MLIRContext *ctx, OpBuilder &builder) {
  return TypeMaterializer<Tag>::get(ctx, builder);
}

} // namespace detail

using i32 = detail::I32Tag;
using i64 = detail::I64Tag;
using hostPtr = detail::HostPtrTag;
using devicePtr = detail::DevicePtrTag;
using slotIndex = detail::SlotIndexTag;

struct SlotIndex {
  Operation *op;
};

namespace detail {

template <typename Tag, typename ArgType> struct ArgConverter;

template <> struct ArgConverter<DevicePtrTag, Value> {
  static Value convert(ConversionPatternRewriter &rewriter, Location loc,
                       Value memref) {
    // An operand the op does not have goes over as a null pointer.
    if (!memref) {
      return LLVM::ZeroOp::create(
          rewriter, loc,
          materializeType<DevicePtrTag>(rewriter.getContext(), rewriter));
    }
    return extractContiguousMemRefPtr(memref, rewriter, loc);
  }
};

template <> struct ArgConverter<HostPtrTag, Value> {
  static Value convert(ConversionPatternRewriter &, Location, Value value) {
    return value;
  }
};

template <> struct ArgConverter<I32Tag, Value> {
  static Value convert(ConversionPatternRewriter &, Location, Value value) {
    return value;
  }
};

template <> struct ArgConverter<I64Tag, Value> {
  static Value convert(ConversionPatternRewriter &, Location, Value value) {
    return value;
  }
};

template <> struct ArgConverter<SlotIndexTag, Value> {
  static Value convert(ConversionPatternRewriter &, Location, Value value) {
    return value;
  }
};

template <> struct ArgConverter<SlotIndexTag, SlotIndex> {
  static Value convert(ConversionPatternRewriter &rewriter, Location loc,
                       SlotIndex slot) {
    int32_t slotValue = -1;
    if (auto attr = slot.op->getAttrOfType<IntegerAttr>("hip.op_state_slot")) {
      slotValue = static_cast<int32_t>(attr.getInt());
    }
    return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                    rewriter.getI32IntegerAttr(slotValue));
  }
};

template <> struct ArgConverter<I32Tag, int32_t> {
  static Value convert(ConversionPatternRewriter &rewriter, Location loc,
                       int32_t value) {
    return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                    rewriter.getI32IntegerAttr(value));
  }
};

template <> struct ArgConverter<I64Tag, int64_t> {
  static Value convert(ConversionPatternRewriter &rewriter, Location loc,
                       int64_t value) {
    return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    rewriter.getI64IntegerAttr(value));
  }
};

} // namespace detail

template <typename Ret, typename... Params> class RuntimeFunc {
public:
  static FailureOr<RuntimeFunc>
  lookupOrCreateFn(ConversionPatternRewriter &rewriter, Location loc,
                   ModuleOp module, llvm::StringRef name) {
    MLIRContext *ctx = rewriter.getContext();
    llvm::SmallVector<Type, sizeof...(Params)> paramTypes{
        detail::materializeType<Params>(ctx, rewriter)...};
    Type resultType = detail::materializeType<Ret>(ctx, rewriter);

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, name, paramTypes, resultType);
    if (failed(funcOp)) {
      return failure();
    }
    return RuntimeFunc(*funcOp, rewriter, loc);
  }

  template <typename... Args> FailureOr<Value> call(Args &&...args) {
    static_assert(sizeof...(Params) == sizeof...(Args),
                  "argument count must match parameter count");

    llvm::SmallVector<Value, sizeof...(Params)> convertedArgs{
        detail::ArgConverter<Params, std::decay_t<Args>>::convert(
            rewriter, loc, std::forward<Args>(args))...};
    Value result =
        LLVM::CallOp::create(rewriter, loc, funcOp, convertedArgs).getResult();
    return result;
  }

private:
  RuntimeFunc(LLVM::LLVMFuncOp funcOp, ConversionPatternRewriter &rewriter,
              Location loc)
      : funcOp(funcOp), rewriter(rewriter), loc(loc) {}

  LLVM::LLVMFuncOp funcOp;
  ConversionPatternRewriter &rewriter;
  Location loc;
};

void populateHipsrAddLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);
void populateHipsrMulLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);
void populateHipsrEqualLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);
void populateHipsrTransposeLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);
void populateHipsrGatherLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
void populateHipsrSliceLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);
void populateHipsrScatterNDLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);
void populateHipsrConstantLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
void populateHipsrGetPoolLoweringPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);
void populateHipsrCastLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns);
void populateHipsrCopyD2HLoweringPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);
void populateHipsrMatMulLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
void populateHipsrExpandLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_HIPSR_TO_LLVM_H
