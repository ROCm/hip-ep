/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "SchemeBindings.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include <cstddef>
#include <cstring>
#include <fstream>

// Include Chez Scheme C API header - use the ta6le machine-specific version
// where ptr is defined as void*, not the portable boot (pb) version
extern "C" {
#include "boot/ta6le/scheme.h"
}

#include "ChezBootPetite.h"
#include "ChezBootScheme.h"
#include "SchemeBindingsScm.h"

namespace {
const size_t petite_boot_size = sizeof(petite_boot_data) - 1;
const size_t scheme_boot_size = sizeof(scheme_boot_data) - 1;
const size_t scheme_bindings_scm_size = sizeof(scheme_bindings_scm_data) - 1;
}

namespace {
static bool scheme_initialized = false;
}

namespace mlir {
namespace hipsr {

bool initializeSchemeRuntime() {
  if (scheme_initialized)
    return true;

  llvm::errs() << "Initializing Chez Scheme runtime...\n";
  llvm::errs() << "  Version: " << Skernel_version() << "\n";
  llvm::errs() << "  Petite boot: " << petite_boot_size << " bytes\n";
  llvm::errs() << "  Scheme boot: " << scheme_boot_size << " bytes\n";

  Sscheme_init(nullptr);
  Sregister_boot_file_bytes("petite.boot", const_cast<void*>(static_cast<const void*>(petite_boot_data)), petite_boot_size);
  Sregister_boot_file_bytes("scheme.boot", const_cast<void*>(static_cast<const void*>(scheme_boot_data)), scheme_boot_size);
  Sbuild_heap("hip-mlir-opt", nullptr);

  // Register MLIR foreign functions AFTER Scheme is initialized
  // but BEFORE loading user Scheme code
  registerMlirForeignFunctions();

  ptr multiply = Stop_level_value(Sstring_to_symbol("*"));
  ptr result = Scall2(multiply, Sinteger(6), Sinteger(7));
  iptr answer = Sinteger_value(result);

  llvm::errs() << "Scheme test: (* 6 7) = " << answer << "\n";

  std::string scm_code(reinterpret_cast<const char*>(scheme_bindings_scm_data),
                       scheme_bindings_scm_size);
  ptr eval_sym = Stop_level_value(Sstring_to_symbol("eval"));
  ptr read_sym = Stop_level_value(Sstring_to_symbol("read"));
  ptr open_string_input_port_sym = Stop_level_value(Sstring_to_symbol("open-string-input-port"));
  ptr eof_object_p = Stop_level_value(Sstring_to_symbol("eof-object?"));

  ptr port = Scall1(open_string_input_port_sym, Sstring(scm_code.c_str()));

  while (true) {
    ptr expr = Scall1(read_sym, port);
    // Check if we hit EOF using eof-object? predicate
    if (Scall1(eof_object_p, expr) != Sfalse)
      break;
    Scall1(eval_sym, expr);
  }

  llvm::errs() << "Chez Scheme runtime initialized successfully!\n";

  scheme_initialized = true;
  return true;
}

std::string callSchemeFunction(const char* functionName,
                                const std::vector<void*>& args) {
  if (!scheme_initialized)
    return "";

  ptr func = Stop_level_value(Sstring_to_symbol(functionName));
  if (func == Sfalse)
    return "";

  ptr args_list = Snil;
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    args_list = Scons(*it, args_list);
  }

  ptr apply_proc = Stop_level_value(Sstring_to_symbol("apply"));
  ptr result = Scall2(apply_proc, func, args_list);

  ptr string_p = Stop_level_value(Sstring_to_symbol("string?"));
  if (Scall1(string_p, result) != Sfalse) {
    // Extract string using macros - Chez strings are 32-bit chars, convert to C string
    iptr len = Sstring_length(result);
    std::string str;
    str.reserve(len);
    for (iptr i = 0; i < len; i++) {
      str.push_back(static_cast<char>(Sstring_ref(result, i)));
    }
    return str;
  }

  return "";
}

void* makeSchemeString(const char* str) {
  return Sstring(str);
}

void* makeSchemeInteger(long value) {
  return Sinteger(value);
}

// MLIR C++ to Scheme conversions - wrap as foreign pointers
SchemeValue makeSchemeOperation(mlir::Operation* op) {
  // Convert pointer to Scheme unsigned-64
  return Sunsigned64(reinterpret_cast<uint64_t>(op));
}

SchemeValue makeSchemeValue(mlir::Value val) {
  MlirValue cVal = wrap(val);
  // Cast away const - Scheme needs non-const pointer
  return const_cast<void*>(cVal.ptr);
}

SchemeValue makeSchemeType(mlir::Type type) {
  MlirType cType = wrap(type);
  return const_cast<void*>(cType.ptr);
}

SchemeValue makeSchemeAttribute(mlir::Attribute attr) {
  MlirAttribute cAttr = wrap(attr);
  return const_cast<void*>(cAttr.ptr);
}

// Load and evaluate a Scheme script file
bool loadSchemeScript(const char* scriptPath) {
  if (!scheme_initialized)
    return false;

  // Read the file
  std::ifstream file(scriptPath);
  if (!file.is_open()) {
    llvm::errs() << "Error: Cannot open Scheme script: " << scriptPath << "\n";
    return false;
  }

  std::string scm_code((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  file.close();

  llvm::errs() << "Loading Scheme script: " << scriptPath << "\n";

  // Evaluate the script
  ptr eval_sym = Stop_level_value(Sstring_to_symbol("eval"));
  ptr read_sym = Stop_level_value(Sstring_to_symbol("read"));
  ptr open_string_input_port_sym = Stop_level_value(Sstring_to_symbol("open-string-input-port"));
  ptr eof_object_p = Stop_level_value(Sstring_to_symbol("eof-object?"));

  ptr port = Scall1(open_string_input_port_sym, Sstring(scm_code.c_str()));

  while (true) {
    ptr expr = Scall1(read_sym, port);
    if (Scall1(eof_object_p, expr) != Sfalse)
      break;
    Scall1(eval_sym, expr);
  }

  llvm::errs() << "Loaded Scheme script: " << scriptPath << "\n";
  return true;
}

// Call a Scheme function with a single MLIR operation argument
void callSchemePassFunction(const char* functionName, mlir::Operation* op) {
  if (!scheme_initialized)
    return;

  ptr func = Stop_level_value(Sstring_to_symbol(functionName));
  if (func == Sfalse) {
    llvm::errs() << "Warning: Scheme function '" << functionName << "' not found\n";
    return;
  }

  ptr schemeOp = makeSchemeOperation(op);
  Scall1(func, schemeOp);
}

// C functions callable from Scheme via FFI
extern "C" {

// Get operation name - takes unsigned-64 (pointer as uint64_t)
static const char* mlir_operation_get_name(uint64_t op) {
  if (!op) return "";
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getName().getStringRef().data();
}

// Get number of operands
static int64_t mlir_operation_num_operands(uint64_t op) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getNumOperands();
}

// Get number of results
static int64_t mlir_operation_num_results(uint64_t op) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getNumResults();
}

// Get operand at index
static uint64_t mlir_operation_get_operand(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumOperands()) return 0;
  mlir::Value val = cppOp->getOperand(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

// Get result at index
static uint64_t mlir_operation_get_result(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumResults()) return 0;
  mlir::Value val = cppOp->getResult(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

// Walk operation tree and call Scheme callback for each operation
// callback: Scheme procedure (lambda (op) ...)
static void mlir_operation_walk(uint64_t op, ptr callback) {
  if (!op) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);

  cppOp->walk([callback](mlir::Operation* walkOp) {
    ptr schemeOp = Sunsigned64(reinterpret_cast<uint64_t>(walkOp));
    Scall1(callback, schemeOp);
  });
}

} // extern "C"

// Register all MLIR foreign functions in Scheme
void registerMlirForeignFunctions() {
  // Register C functions so Scheme can call them via foreign-procedure
  Sregister_symbol("mlir_operation_get_name", (void*)mlir_operation_get_name);
  Sregister_symbol("mlir_operation_num_operands", (void*)mlir_operation_num_operands);
  Sregister_symbol("mlir_operation_num_results", (void*)mlir_operation_num_results);
  Sregister_symbol("mlir_operation_get_operand", (void*)mlir_operation_get_operand);
  Sregister_symbol("mlir_operation_get_result", (void*)mlir_operation_get_result);
  Sregister_symbol("mlir_operation_walk", (void*)mlir_operation_walk);

  llvm::errs() << "Registered MLIR foreign functions for Scheme\n";
}

} // namespace hipsr
} // namespace mlir
