/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


#define DEBUG_TYPE "scheme-hipsr-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

extern "C" {

uint64_t mlir_get_hipsr_context_arg(uint64_t op_ptr) {
  if (!op_ptr) return 0;

  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();

  if (!funcOp || funcOp.getBody().empty()) {
    mlir_log_debug("mlir_get_hipsr_context_arg: not inside a function body");
    return 0;
  }

  mlir::Block &entry = funcOp.getBody().front();
  if (entry.getNumArguments() == 0) {
    mlir_log_debug("mlir_get_hipsr_context_arg: function has no arguments");
    return 0;
  }

  mlir::Value ctx = entry.getArgument(0);
  if (!mlir::isa<mlir::hipsr::ContextType>(ctx.getType())) {
    mlir_log_debug("mlir_get_hipsr_context_arg: arg 0 is not !hipsr.context");
    return 0;
  }

  MlirValue cVal = wrap(ctx);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

// Check if a type is RankedTensorType
// Returns 1 if true, 0 if false
void mlir_populate_cast_conversion_patterns(
    uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t ctx_ptr) {
  if (!converter_ptr || !patterns_ptr || !ctx_ptr) return;

  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  auto* patterns = reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);

  mlir::hipsr::populateCastConversionPatterns(*converter, *patterns, ctx);
}

// Helper: Populate Return conversion patterns
void mlir_erase_dead_novalue_ops(uint64_t module_ptr) {
  if (!module_ptr) return;

  auto module = mlir::dyn_cast<mlir::ModuleOp>(reinterpret_cast<mlir::Operation*>(module_ptr));
  if (!module) return;

  llvm::SmallVector<mlir::onnx::NoValueOp> dead;
  module.walk([&](mlir::onnx::NoValueOp op) {
    if (op->use_empty()) {
      dead.push_back(op);
    }
  });

  for (auto op : dead) {
    op.erase();
  }
}

// Helper: Rewire placeholder inputs to follow shape graph
void mlir_rewire_placeholder_inputs(uint64_t module_ptr) {
  if (!module_ptr) return;

  auto module = mlir::dyn_cast<mlir::ModuleOp>(reinterpret_cast<mlir::Operation*>(module_ptr));
  if (!module) return;

  module.walk([](mlir::hipsr::PlaceholderOp placeholder) {
    llvm::SmallVector<mlir::Value> resolvedInputs;
    for (mlir::Value input : placeholder.getInputs()) {
      resolvedInputs.push_back(mlir::hipsr::getShapeGraphCounterpart(input));
    }
    placeholder.getInputsMutable().assign(resolvedInputs);
  });
}

//===----------------------------------------------------------------------===//
// MLIR Dialect Conversion Framework Primitives
//===----------------------------------------------------------------------===//

// Create a TypeConverter object
// Returns TypeConverter* as uint64_t (opaque handle for Scheme)
void mlir_conversion_target_add_legal_hipsr(uint64_t target_ptr) {
  if (!target_ptr) return;
  auto* target = reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
  target->addLegalDialect<mlir::hipsr::HipsrDialect>();
}

// Mark common operations legal (ModuleOp, arith.constant)
void mlir_conversion_target_mark_unknown_ops_nested_legal(uint64_t target_ptr) {
  if (!target_ptr) return;
  auto* target = reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
  target->markUnknownOpDynamicallyLegal([](mlir::Operation *op) {
    return op->getParentOfType<mlir::hipsr::ComputeOp>() != nullptr ||
           op->getParentOfType<mlir::hipsr::PlaceholderOp>() != nullptr;
  });
}

// Create a RewritePatternSet
// Returns RewritePatternSet* as uint64_t (opaque handle for Scheme)
void mlir_placeholder_set_barrier_type(uint64_t op_ptr) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  op->setAttr("placeholder_type",
      mlir::hipsr::PlaceholderTypeAttr::get(op->getContext(),
                                             mlir::hipsr::PlaceholderType::Barrier));
}

// Copy a named attribute from src_op to dst_op. No-op if attr is absent on src.

} // extern "C"

namespace mlir {
namespace hipsr {

void registerHipsrBindings() {
  Sregister_symbol("mlir_get_hipsr_context_arg", (void*)::mlir_get_hipsr_context_arg);
  Sregister_symbol("mlir_populate_cast_conversion_patterns", (void*)::mlir_populate_cast_conversion_patterns);
  Sregister_symbol("mlir_erase_dead_novalue_ops", (void*)::mlir_erase_dead_novalue_ops);
  Sregister_symbol("mlir_rewire_placeholder_inputs", (void*)::mlir_rewire_placeholder_inputs);
  Sregister_symbol("mlir_conversion_target_add_legal_hipsr", (void*)::mlir_conversion_target_add_legal_hipsr);
  Sregister_symbol("mlir_conversion_target_mark_unknown_ops_nested_legal", (void*)::mlir_conversion_target_mark_unknown_ops_nested_legal);
  Sregister_symbol("mlir_placeholder_set_barrier_type", (void*)::mlir_placeholder_set_barrier_type);
}

} // namespace hipsr
} // namespace mlir
