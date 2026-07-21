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

// Runtime symbol fetching a constant's device pointer by index. Same name the
// hip.get_constant lowering uses (kHipGetConstant in HipToLLVMUtils.h) so both
// pipelines bind to the identical RuntimeState callback.
inline constexpr const char *kHipsrGetConstant = "hipdnn_ep_constant_get";

// Lowers hipsr.constant to LLVM. \p indexMap gives each externalized constant
// its module-walk-order index (the second @hipdnn_ep_constant_get argument);
// \p ctxMap gives each externalized constant the (pre-conversion) !hip.context
// block argument of its enclosing function. Both are owned by the caller and
// must outlive the pattern set.
void populateHipsrConstantLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    const llvm::DenseMap<Operation *, int64_t> &indexMap,
    const llvm::DenseMap<Operation *, Value> &ctxMap);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_HIPSRTOLLVM_UTILS_H
