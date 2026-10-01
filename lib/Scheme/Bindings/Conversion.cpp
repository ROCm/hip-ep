/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "hip/Scheme/Bindings/LockedSchemeObject.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"


#define DEBUG_TYPE "scheme-conversion-bindings"

// Forward declarations from Logging.cpp
extern "C" {

namespace {

// Wrapper class that implements ConversionPattern by calling a Scheme callback
class SchemeConversionPattern : public mlir::ConversionPattern {
public:
  SchemeConversionPattern(mlir::TypeConverter *typeConverter, mlir::MLIRContext *ctx,
                          ptr schemeCallback, llvm::StringRef opName, int benefit = 1)
      : ConversionPattern(*typeConverter, opName, benefit, ctx),
        callback_(schemeCallback),  // RAII lock
        targetOpName(opName.str()) {}

  // Destructor automatically unlocks via LockedSchemeObject

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if this is the target operation
    if (op->getName().getStringRef() != targetOpName) {
      return mlir::failure();
    }

    // Call Scheme callback: (callback op operands-ref rewriter type-converter)
    // rewriter and op are passed explicitly — no implicit global state
    ptr opPtr            = Sunsigned64(reinterpret_cast<uint64_t>(op));
    ptr operandsRefPtr   = Sunsigned64(reinterpret_cast<uint64_t>(&operands));
    ptr rewriterPtr      = Sunsigned64(reinterpret_cast<uint64_t>(&rewriter));
    ptr typeConverterPtr = Sunsigned64(reinterpret_cast<uint64_t>(getTypeConverter()));

    ptr args_list = Scons(opPtr, Scons(operandsRefPtr, Scons(rewriterPtr, Scons(typeConverterPtr, Snil))));
    ptr apply_proc = Stop_level_value(Sstring_to_symbol("apply"));
    ptr result = Scall2(apply_proc, callback_.get(), args_list);

    // Check result: #t = success, #f = failure
    if (result == Strue) {
      return mlir::success();
    } else {
      return mlir::failure();
    }
  }

private:
  mlir::hipsr::LockedSchemeObject callback_;  // RAII-locked Scheme procedure
  std::string targetOpName;      // Target operation name
};

// Wrapper that implements RewritePattern by calling a Scheme callback.
// Callback signature: (lambda (op rewriter) → #t/#f)
// No TypeConverter or converted-operands adaptor — for local rewrites.
class SchemeRewritePattern : public mlir::RewritePattern {
public:
  SchemeRewritePattern(mlir::MLIRContext *ctx,
                       ptr schemeCallback, llvm::StringRef opName,
                       int benefit = 1)
      : RewritePattern(opName, benefit, ctx),
        callback_(schemeCallback),
        targetOpName(opName.str()) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != targetOpName)
      return mlir::failure();

    ptr opPtr       = Sunsigned64(reinterpret_cast<uint64_t>(op));
    ptr rewriterPtr = Sunsigned64(reinterpret_cast<uint64_t>(&rewriter));
    ptr args_list   = Scons(opPtr, Scons(rewriterPtr, Snil));
    ptr apply_proc  = Stop_level_value(Sstring_to_symbol("apply"));
    ptr result      = Scall2(apply_proc, callback_.get(), args_list);

    return result == Strue ? mlir::success() : mlir::failure();
  }

private:
  mlir::hipsr::LockedSchemeObject callback_;
  std::string targetOpName;
};

} // anonymous namespace


// Register a Scheme-defined conversion pattern for a named MLIR op.
// The callback is called as (callback op operands-ref rewriter type-converter)
// and must return #t on success or #f on failure (pattern does not apply).
// The SchemeConversionPattern wrapper handles GC locking via LockedSchemeObject.
// patterns_ptr:       RewritePatternSet* as ptr (scheme-object)
// op_name:            MLIR op name string, e.g. "onnx.Cast"
// callback:           Scheme procedure ptr (GC-locked for the pattern's lifetime)
// type_converter_ptr: TypeConverter* as ptr
void mlir_register_conversion_pattern(ptr patterns_ptr,
                                      const char* op_name,
                                      ptr callback,
                                      ptr type_converter_ptr,
                                      int benefit) {
  auto *patterns = reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);
  auto *typeConverter = reinterpret_cast<mlir::TypeConverter*>(type_converter_ptr);
  ptr schemeCallback = static_cast<ptr>(callback);

  mlir_log_info((std::string("Registering Scheme pattern for ") + op_name).c_str());

  patterns->add<SchemeConversionPattern>(
      typeConverter, patterns->getContext(), schemeCallback, llvm::StringRef(op_name), benefit);
}

// Register a Scheme-defined rewrite pattern for a named MLIR op.
// The callback is called as (callback op rewriter) and must return #t/#f.
// patterns_ptr: RewritePatternSet* as ptr
// op_name:      MLIR op name string, e.g. "some.Op"
// callback:     Scheme procedure ptr (GC-locked for the pattern's lifetime)
void mlir_register_rewrite_pattern(ptr patterns_ptr,
                                    const char *op_name,
                                    ptr callback,
                                    int benefit) {
  auto *patterns = reinterpret_cast<mlir::RewritePatternSet *>(patterns_ptr);
  mlir_log_info((std::string("Registering Scheme rewrite pattern for ") + op_name).c_str());
  patterns->add<SchemeRewritePattern>(
      patterns->getContext(), callback, llvm::StringRef(op_name), benefit);
}

//===----------------------------------------------------------------------===//
// Additional utility FFI functions
//===----------------------------------------------------------------------===//

// Allocate a new mlir::TypeConverter on the heap.
// Returns: TypeConverter* as uptr — caller must destroy via mlir_destroy_type_converter
uint64_t mlir_create_type_converter() {
  mlir::TypeConverter* tc = new mlir::TypeConverter();
  uint64_t result = reinterpret_cast<uint64_t>(tc);
  mlir_log_info((std::string("mlir_create_type_converter: created TypeConverter at ") +
                  std::to_string(result) + " (ptr=" +
                  std::to_string(reinterpret_cast<uintptr_t>(tc)) + ")").c_str());
  return result;
}

// Destroy a TypeConverter object
void mlir_destroy_type_converter(uint64_t converter_ptr) {
  if (!converter_ptr) return;
  mlir_log_info((std::string("mlir_destroy_type_converter: destroying TypeConverter at ") +
                  std::to_string(converter_ptr)).c_str());
  delete reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
}

// Create a ConversionTarget object
// Returns ConversionTarget* as uint64_t (opaque handle for Scheme)
uint64_t mlir_create_conversion_target(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(new mlir::ConversionTarget(*ctx));
}

// Destroy a ConversionTarget object
void mlir_destroy_conversion_target(uint64_t target_ptr) {
  if (!target_ptr) return;
  delete reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
}

// Generic: mark a named dialect illegal in the conversion target
void mlir_conversion_target_add_illegal_dialect(uint64_t target_ptr, const char* dialect_name) {
  if (!target_ptr || !dialect_name) return;
  reinterpret_cast<mlir::ConversionTarget*>(target_ptr)->addIllegalDialect(dialect_name);
}

// Generic: mark a named dialect legal in the conversion target
void mlir_conversion_target_add_legal_dialect(uint64_t target_ptr, const char* dialect_name) {
  if (!target_ptr || !dialect_name) return;
  reinterpret_cast<mlir::ConversionTarget*>(target_ptr)->addLegalDialect(dialect_name);
}

// Generic: mark a named op legal in the conversion target
void mlir_conversion_target_add_legal_op(uint64_t target_ptr, uint64_t ctx_ptr, const char* op_name) {
  if (!target_ptr || !ctx_ptr || !op_name) return;
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  reinterpret_cast<mlir::ConversionTarget*>(target_ptr)
      ->addLegalOp(mlir::OperationName(op_name, ctx));
}

// Generic: mark a named op dynamically legal with a Scheme callback (op → bool)
void mlir_conversion_target_add_dynamically_legal_op(
    uint64_t target_ptr, uint64_t ctx_ptr, const char* op_name, ptr callback) {
  if (!target_ptr || !ctx_ptr || !op_name) return;
  auto* target = reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  // shared_ptr needed: std::function requires a copyable callable; LockedSchemeObject is non-copyable.
  auto locked = std::make_shared<mlir::hipsr::LockedSchemeObject>(callback);
  target->addDynamicallyLegalOp(
      mlir::OperationName(op_name, ctx),
      [locked](mlir::Operation* op) -> bool {
        ptr op_arg = Sunsigned64(reinterpret_cast<uint64_t>(op));
        ptr result = Scall1(locked->get(), op_arg);
        return result != Sfalse && result != Sfixnum(0);
      });
}

// Generic: mark unknown ops dynamically legal with a Scheme callback (op → bool)
void mlir_conversion_target_mark_unknown_ops_dynamically_legal(uint64_t target_ptr, ptr callback) {
  if (!target_ptr) return;
  auto locked = std::make_shared<mlir::hipsr::LockedSchemeObject>(callback);
  reinterpret_cast<mlir::ConversionTarget*>(target_ptr)
      ->markUnknownOpDynamicallyLegal([locked](mlir::Operation* op) -> bool {
        ptr op_arg = Sunsigned64(reinterpret_cast<uint64_t>(op));
        ptr result = Scall1(locked->get(), op_arg);
        return result != Sfalse && result != Sfixnum(0);
      });
}

// Generic: add a Scheme type-conversion callback to a TypeConverter.
// callback: (lambda (type-uptr) -> type-uptr-or-#f)
// Returns #f or 0 from callback means "not handled by this conversion".
void mlir_type_converter_add_conversion(uint64_t converter_ptr, ptr callback) {
  if (!converter_ptr) return;
  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  auto locked = std::make_shared<mlir::hipsr::LockedSchemeObject>(callback);
  converter->addConversion([locked](mlir::Type type) -> std::optional<mlir::Type> {
    ptr type_arg = Sunsigned64(reinterpret_cast<uint64_t>(type.getAsOpaquePointer()));
    ptr result = Scall1(locked->get(), type_arg);
    if (result == Sfalse) return std::nullopt;
    uint64_t result_val = Sunsigned64_value(result);
    if (result_val == 0) return std::nullopt;
    return mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(result_val));
  });
}

// Generic: check if a type is legal according to a TypeConverter
int mlir_type_converter_is_legal_type(uint64_t converter_ptr, uint64_t type_ptr) {
  if (!converter_ptr || !type_ptr) return 0;
  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(type_ptr));
  return converter->isLegal(type) ? 1 : 0;
}

// Generic: check if all operand/result types of an op are legal
int mlir_type_converter_is_legal(uint64_t converter_ptr, uint64_t op_ptr) {
  if (!converter_ptr || !op_ptr) return 0;
  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  return converter->isLegal(op) ? 1 : 0;
}

// Generic: check if a func op's signature is legal according to a TypeConverter
int mlir_type_converter_is_signature_legal(uint64_t converter_ptr, uint64_t func_op_ptr) {
  if (!converter_ptr || !func_op_ptr) return 0;
  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(
      reinterpret_cast<mlir::Operation*>(func_op_ptr));
  if (!func_op) return 0;
  return converter->isSignatureLegal(func_op.getFunctionType()) ? 1 : 0;
}

// Mark ModuleOp and arith.constant legal in the conversion target.
// These ops appear in every module and are not lowered by the ONNX→HipSR pass.
// target_ptr: ConversionTarget* as uptr
void mlir_conversion_target_add_legal_common_ops(uint64_t target_ptr) {
  if (!target_ptr) return;
  auto* target = reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
  target->addLegalOp<mlir::ModuleOp>();
  target->addLegalOp<mlir::arith::ConstantOp>();
}

// Mark func.func and func.return dynamically legal based on TypeConverter
void mlir_conversion_target_add_dynamically_legal_func(
    uint64_t target_ptr, uint64_t converter_ptr) {
  if (!target_ptr || !converter_ptr) return;
  auto* target = reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);

  mlir_log_info((std::string("mlir_conversion_target_add_dynamically_legal_func: IN converter_ptr=") +
                  std::to_string(converter_ptr) + " (as ptr=" +
                  std::to_string(reinterpret_cast<uintptr_t>(converter)) + ")").c_str());

  target->addDynamicallyLegalOp<mlir::func::FuncOp>([converter](mlir::func::FuncOp op) {
    mlir_log_info((std::string("FuncOp lambda: converter=") +
                    std::to_string(reinterpret_cast<uint64_t>(converter))).c_str());
    return converter->isSignatureLegal(op.getFunctionType());
  });
  target->addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [converter](mlir::func::ReturnOp op) {
        mlir_log_info((std::string("ReturnOp lambda: IN converter=") +
                        std::to_string(reinterpret_cast<uint64_t>(converter))).c_str());
        bool result = converter->isLegal(op);
        mlir_log_info((std::string("ReturnOp lambda: OUT result=") +
                        std::to_string(result)).c_str());
        return result;
      });

  mlir_log_info("mlir_conversion_target_add_dynamically_legal_func: lambdas registered");
}

// Allocate a new mlir::RewritePatternSet on the heap for the given MLIRContext.
// ctx_ptr: MLIRContext* as uptr; returns 0 if null
// Returns: RewritePatternSet* as uptr — caller must destroy via mlir_destroy_rewrite_pattern_set
uint64_t mlir_create_rewrite_pattern_set(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(new mlir::RewritePatternSet(ctx));
}

// Destroy a RewritePatternSet object
void mlir_destroy_rewrite_pattern_set(uint64_t patterns_ptr) {
  if (!patterns_ptr) return;
  delete reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);
}

// Apply full conversion
// Returns 1 on success, 0 on failure
// NOTE: This takes ownership of the patterns (moves them)
int mlir_apply_full_conversion(uint64_t module_ptr, uint64_t target_ptr, uint64_t patterns_ptr) {
  mlir_log_info((std::string("mlir_apply_full_conversion: IN module=") +
                  std::to_string(module_ptr) + " target=" + std::to_string(target_ptr) +
                  " patterns=" + std::to_string(patterns_ptr)).c_str());

  if (!module_ptr || !target_ptr || !patterns_ptr) return 0;

  auto module = mlir::dyn_cast<mlir::ModuleOp>(reinterpret_cast<mlir::Operation*>(module_ptr));
  if (!module) return 0;

  auto* target = reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
  auto* patterns = reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);

  mlir_log_info("mlir_apply_full_conversion: calling applyFullConversion");
  if (mlir::failed(mlir::applyFullConversion(module, *target, std::move(*patterns)))) {
    mlir_log_info("mlir_apply_full_conversion: FAILED");
    return 0;
  }

  mlir_log_info("mlir_apply_full_conversion: SUCCESS");
  return 1;
}

// Populate the FuncOp type-conversion pattern that rewrites func.func signatures
// according to the TypeConverter. Required when converting function argument types.
// patterns_ptr:   RewritePatternSet* as uptr
// converter_ptr:  TypeConverter* as uptr
void mlir_populate_func_type_conversion_pattern(
    uint64_t patterns_ptr, uint64_t converter_ptr) {
  if (!patterns_ptr || !converter_ptr) return;
  auto* patterns = reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);
  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      *patterns, *converter);
}

// Add a source materialization to a TypeConverter that resolves unresolved
// conversion casts between ranked tensor types by inserting tensor.cast.
//
// This handles cases where a conversion pattern replaces an op with a value
// whose type is more specific than what the type converter produces for the
// declared result type (e.g., tensor<?x32xf16, device> replacing a use that
// expects tensor<?x?xf16, device>). Without this materialization, the
// unrealized_conversion_cast left behind cannot be resolved and the full
// conversion fails.
//
// Only inserts the cast when tensor::CastOp::areCastCompatible confirms the
// types are structurally compatible (same rank and element type; dims may
// differ in specificity).
void mlir_type_converter_add_tensor_widening_materialization(
    uint64_t converter_ptr) {
  if (!converter_ptr)
    return;
  auto *converter = reinterpret_cast<mlir::TypeConverter *>(converter_ptr);
  // Source materialization: when a conversion pattern replaces an op with a
  // value whose type is more specific (e.g. tensor<?x32xfloat, device>) than
  // what the type converter derives from the declared result type
  // (e.g. tensor<?x?xfloat, device>), MLIR creates an unrealized_conversion_cast
  // between them. This materialization resolves it by inserting a tensor.cast.
  // Source materialization: widen a more specific tensor type back to a
  // more general one (e.g. tensor<?x32xf16,device> → tensor<?x?xf16,device>).
  converter->addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type resultType,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return nullptr;
        mlir::Value input = inputs[0];
        auto inputType =
            mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
        auto outType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
        if (!inputType || !outType)
          return nullptr;
        if (!mlir::tensor::CastOp::areCastCompatible(inputType, outType))
          return nullptr;
        return mlir::tensor::CastOp::create(builder, loc, resultType, input);
      });
  // Target materialization: same direction for target-kind unrealized casts.
  converter->addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type resultType,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return nullptr;
        mlir::Value input = inputs[0];
        auto inputType =
            mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
        auto outType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
        if (!inputType || !outType)
          return nullptr;
        if (!mlir::tensor::CastOp::areCastCompatible(inputType, outType))
          return nullptr;
        return mlir::tensor::CastOp::create(builder, loc, resultType, input);
      });
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerConversionBindings() {
  Sregister_symbol("mlir_register_conversion_pattern", (void*)::mlir_register_conversion_pattern);
  Sregister_symbol("mlir_register_rewrite_pattern",    (void*)::mlir_register_rewrite_pattern);
  Sregister_symbol("mlir_create_type_converter", (void*)::mlir_create_type_converter);
  Sregister_symbol("mlir_destroy_type_converter", (void*)::mlir_destroy_type_converter);
  Sregister_symbol("mlir_create_conversion_target", (void*)::mlir_create_conversion_target);
  Sregister_symbol("mlir_destroy_conversion_target", (void*)::mlir_destroy_conversion_target);
  Sregister_symbol("mlir_conversion_target_add_illegal_dialect", (void*)::mlir_conversion_target_add_illegal_dialect);
  Sregister_symbol("mlir_conversion_target_add_legal_dialect", (void*)::mlir_conversion_target_add_legal_dialect);
  Sregister_symbol("mlir_conversion_target_add_legal_op", (void*)::mlir_conversion_target_add_legal_op);
  Sregister_symbol("mlir_conversion_target_add_dynamically_legal_op", (void*)::mlir_conversion_target_add_dynamically_legal_op);
  Sregister_symbol("mlir_conversion_target_mark_unknown_ops_dynamically_legal", (void*)::mlir_conversion_target_mark_unknown_ops_dynamically_legal);
  Sregister_symbol("mlir_type_converter_add_conversion", (void*)::mlir_type_converter_add_conversion);
  Sregister_symbol("mlir_type_converter_is_legal_type", (void*)::mlir_type_converter_is_legal_type);
  Sregister_symbol("mlir_type_converter_is_legal", (void*)::mlir_type_converter_is_legal);
  Sregister_symbol("mlir_type_converter_is_signature_legal", (void*)::mlir_type_converter_is_signature_legal);
  Sregister_symbol("mlir_create_rewrite_pattern_set", (void*)::mlir_create_rewrite_pattern_set);
  Sregister_symbol("mlir_destroy_rewrite_pattern_set", (void*)::mlir_destroy_rewrite_pattern_set);
  Sregister_symbol("mlir_apply_full_conversion", (void*)::mlir_apply_full_conversion);
  Sregister_symbol("mlir_populate_func_type_conversion_pattern",
                   (void*)::mlir_populate_func_type_conversion_pattern);
  Sregister_symbol("mlir_type_converter_add_tensor_widening_materialization",
                   (void*)::mlir_type_converter_add_tensor_widening_materialization);
}
} // namespace hipsr
} // namespace mlir
