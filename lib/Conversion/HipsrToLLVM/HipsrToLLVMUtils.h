/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPSRTOLLVM_UTILS_H
#define HIP_CONVERSION_HIPSRTOLLVM_UTILS_H

#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hipsr {

// Runtime symbol returning a constant's device pointer by its byte offset into
// the constants blob (wrap_get_global(state, offset, size) in
// hipdnn_ep_runtime_state.cpp).
inline constexpr const char *kHipsrWrapGetGlobal = "wrap_get_global";

// Registers the hipsr.constant -> LLVM pattern. One populate per op/category
// (mirrors HipToLLVMUtils.h's populate*LoweringPatterns). The pattern is
// stateless (ctx is sourced inside it), so no caller state is needed.
void populateHipsrConstantLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_HIPSRTOLLVM_UTILS_H
