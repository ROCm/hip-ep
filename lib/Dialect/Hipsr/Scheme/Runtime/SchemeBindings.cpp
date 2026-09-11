/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "SchemeBindings.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstddef>
#include <cstring>
#include <fstream>

#define DEBUG_TYPE "scheme-bindings"

// Include Chez Scheme C API header - use the ta6le machine-specific version
// where ptr is defined as void*, not the portable boot (pb) version
extern "C" {
#include "boot/ta6le/scheme.h"
}

#include "ChezBootPetite.h"
#include "ChezBootScheme.h"
#include "SchemeBindingsScm.h"
#include "PatternDSLScm.h"

namespace {
const size_t petite_boot_size = sizeof(petite_boot_data) - 1;
const size_t scheme_boot_size = sizeof(scheme_boot_data) - 1;
const size_t scheme_bindings_scm_size = sizeof(scheme_bindings_scm_data) - 1;
const size_t pattern_dsl_scm_size = sizeof(pattern_dsl_scm_data) - 1;
}

namespace {
static bool scheme_initialized = false;
// Cached Scheme symbols for script loading
static ptr cached_eval_sym = nullptr;
static ptr cached_read_sym = nullptr;
static ptr cached_open_string_input_port_sym = nullptr;
static ptr cached_eof_object_p = nullptr;
}

namespace mlir {
namespace hipsr {

// Current log level - used by FFI logging functions
static SchemeLogLevel current_log_level = SchemeLogLevel::Warning;

// Custom init called by Sbuild_heap before loading boot files
static void custom_init() {
  // Register all MLIR foreign functions
  registerMlirForeignFunctions();

  if (current_log_level <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] custom_init: Registered foreign functions\n";
  }
}

SchemeLogLevel parseLogLevel(const std::string& level) {
  if (level == "trace") return SchemeLogLevel::Trace;
  if (level == "debug") return SchemeLogLevel::Debug;
  if (level == "info") return SchemeLogLevel::Info;
  if (level == "warning") return SchemeLogLevel::Warning;
  if (level == "error") return SchemeLogLevel::Error;
  if (level == "fatal") return SchemeLogLevel::Fatal;

  llvm::errs() << "Warning: unknown log level '" << level
               << "', defaulting to 'warning'\n";
  return SchemeLogLevel::Warning;
}

bool initializeSchemeRuntime(SchemeLogLevel logLevel) {
  if (scheme_initialized)
    return true;

  current_log_level = logLevel;

  if (logLevel <= SchemeLogLevel::Info) {
    llvm::errs() << "[info] Initializing Chez Scheme runtime "
                 << Skernel_version() << "\n";
  }
  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Petite boot: " << petite_boot_size << " bytes\n";
    llvm::errs() << "[debug] Scheme boot: " << scheme_boot_size << " bytes\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "Initializing Chez Scheme runtime "
                          << Skernel_version() << "\n");

  // Initialize Scheme system (must be called first)
  Sscheme_init(nullptr);

  // Register embedded boot files
  Sregister_boot_file_bytes("petite.boot", const_cast<void*>(static_cast<const void*>(petite_boot_data)), petite_boot_size);
  Sregister_boot_file_bytes("scheme.boot", const_cast<void*>(static_cast<const void*>(scheme_boot_data)), scheme_boot_size);

  // Build heap and call custom_init (which registers foreign functions)
  // custom_init is called BEFORE boot files are loaded
  Sbuild_heap(nullptr, custom_init);

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Heap built, caching Scheme symbols\n";
  }

  // Cache Scheme symbols we'll use
  #define CALL0(who) Scall0(Stop_level_value(Sstring_to_symbol(who)))
  #define CALL1(who, arg) Scall1(Stop_level_value(Sstring_to_symbol(who)), arg)

  cached_eval_sym = Stop_level_value(Sstring_to_symbol("eval"));
  cached_read_sym = Stop_level_value(Sstring_to_symbol("read"));
  cached_open_string_input_port_sym = Stop_level_value(Sstring_to_symbol("open-string-input-port"));
  cached_eof_object_p = Stop_level_value(Sstring_to_symbol("eof-object?"));

  ptr eval_sym = cached_eval_sym;
  ptr read_sym = cached_read_sym;
  ptr open_string_input_port_sym = cached_open_string_input_port_sym;
  ptr eof_object_p = cached_eof_object_p;

  // Set up library path to find rime libraries
  // Find lib/scheme directory relative to the library module path
  // Chez Scheme needs the parent directory of rime/ to resolve (rime) libraries
  std::string modulePath = llvm::sys::fs::getMainExecutable(nullptr, (void*)&initializeSchemeRuntime);
  llvm::SmallString<256> schemePath(modulePath);
  llvm::sys::path::remove_filename(schemePath);  // Remove binary name
  if (llvm::sys::path::filename(schemePath) == "bin")
    llvm::sys::path::remove_filename(schemePath);  // Remove bin/
  llvm::sys::path::append(schemePath, "lib", "scheme");

  // Add lib/scheme to library-directories so Chez can find (rime) as rime/*.sls
  std::string setup_code = "(library-directories (cons \"" + std::string(schemePath.c_str()) + "\" (library-directories)))";
  ptr setup_port = Scall1(open_string_input_port_sym, Sstring(setup_code.c_str()));
  ptr setup_expr = Scall1(read_sym, setup_port);
  Scall1(eval_sym, setup_expr);

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Added Scheme library path: " << schemePath.c_str() << "\n";
  }

  // NOTE: Foreign functions are registered in custom_init(), which was called
  // by Sbuild_heap before loading boot files

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] About to load SchemeBindings.scm\n";
    llvm::errs() << "[debug] open_string_input_port_sym = " << open_string_input_port_sym << "\n";
    llvm::errs() << "[debug] read_sym = " << read_sym << "\n";
  }

  // Load the Scheme bindings library
  std::string scm_code(reinterpret_cast<const char*>(scheme_bindings_scm_data),
                       scheme_bindings_scm_size);

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Loading SchemeBindings.scm (" << scheme_bindings_scm_size << " bytes)\n";
    llvm::errs() << "[debug] First 50 chars: " << scm_code.substr(0, 50) << "\n";
  }

  ptr port = Scall1(open_string_input_port_sym, Sstring(scm_code.c_str()));

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Created port = " << port << "\n";
  }

  while (true) {
    ptr expr = Scall1(read_sym, port);
    if (Scall1(eof_object_p, expr) != Sfalse)
      break;
    Scall1(eval_sym, expr);
  }

  // Load PatternDSL.scm (embedded)
  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Loading PatternDSL.scm (" << pattern_dsl_scm_size << " bytes)\n";
  }

  std::string pattern_dsl_code(reinterpret_cast<const char*>(pattern_dsl_scm_data),
                               pattern_dsl_scm_size);
  ptr pattern_dsl_port = Scall1(open_string_input_port_sym, Sstring(pattern_dsl_code.c_str()));

  while (true) {
    ptr expr = Scall1(read_sym, pattern_dsl_port);
    if (Scall1(eof_object_p, expr) != Sfalse)
      break;
    Scall1(eval_sym, expr);
  }

  if (logLevel <= SchemeLogLevel::Info)
    llvm::errs() << "[info] Scheme runtime initialized\n";
  LLVM_DEBUG(llvm::dbgs() << "Scheme runtime initialized\n");

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

  std::ifstream file(scriptPath);
  if (!file.is_open()) {
    llvm::errs() << "error: cannot open Scheme script: " << scriptPath << "\n";
    return false;
  }

  std::string scm_code((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  file.close();

  if (current_log_level <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Loading " << scriptPath << " (" << scm_code.size() << " bytes)\n";
    llvm::errs() << "[debug] First 100 chars: " << scm_code.substr(0, 100) << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "Loading Scheme script: " << scriptPath << "\n");

  // Evaluate the script content using cached symbols from initialization
  if (current_log_level <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Creating string input port for " << scm_code.size() << " bytes\n";
    llvm::errs() << "[debug] cached_open_string_input_port_sym: " << cached_open_string_input_port_sym << "\n";
    llvm::errs() << "[debug] cached_read_sym: " << cached_read_sym << "\n";
  }

  ptr scheme_string = Sstring(scm_code.c_str());
  if (current_log_level <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Created Scheme string: " << scheme_string << "\n";
  }

  ptr port = Scall1(cached_open_string_input_port_sym, scheme_string);
  if (current_log_level <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Created port: " << port << "\n";
  }

  while (true) {
    ptr expr = Scall1(cached_read_sym, port);
    if (Scall1(cached_eof_object_p, expr) != Sfalse)
      break;
    Scall1(cached_eval_sym, expr);
  }

  if (current_log_level <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] Loaded " << scriptPath << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "Loaded Scheme script: " << scriptPath << "\n");
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

} // namespace hipsr
} // namespace mlir

//===----------------------------------------------------------------------===//
// C functions callable from Scheme via FFI (global scope, C linkage)
//===----------------------------------------------------------------------===//

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

// Logging functions callable from Scheme
static void mlir_log_trace(const char* msg) {
  if (mlir::hipsr::current_log_level <= mlir::hipsr::SchemeLogLevel::Trace)
    llvm::errs() << "[trace] " << msg << "\n";
}

static void mlir_log_debug(const char* msg) {
  if (mlir::hipsr::current_log_level <= mlir::hipsr::SchemeLogLevel::Debug)
    llvm::errs() << "[debug] " << msg << "\n";
}

static void mlir_log_info(const char* msg) {
  if (mlir::hipsr::current_log_level <= mlir::hipsr::SchemeLogLevel::Info)
    llvm::errs() << "[info] " << msg << "\n";
}

static void mlir_log_warning(const char* msg) {
  if (mlir::hipsr::current_log_level <= mlir::hipsr::SchemeLogLevel::Warning)
    llvm::errs() << "[warning] " << msg << "\n";
}

static void mlir_log_error(const char* msg) {
  if (mlir::hipsr::current_log_level <= mlir::hipsr::SchemeLogLevel::Error)
    llvm::errs() << "[error] " << msg << "\n";
}

static void mlir_log_fatal(const char* msg) {
  if (mlir::hipsr::current_log_level <= mlir::hipsr::SchemeLogLevel::Fatal)
    llvm::errs() << "[fatal] " << msg << "\n";
}

//===----------------------------------------------------------------------===//
// Phase 1: Type System FFI
//===----------------------------------------------------------------------===//

int mlir_type_is_ranked_tensor(SchemeValue type_ptr) {
  if (!type_ptr) return 0;
  // SchemeValue is void*, representing Type* from MLIR C API
  // We stored it via wrap(Type).ptr, so retrieve it the same way
  mlir::Type type = mlir::Type::getFromOpaquePointer(type_ptr);
  return llvm::isa<mlir::RankedTensorType>(type) ? 1 : 0;
}

SchemeValue mlir_type_get_element_type(SchemeValue type_ptr) {
  if (!type_ptr) return nullptr;
  mlir::Type type = mlir::Type::getFromOpaquePointer(type_ptr);
  if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    return const_cast<void*>(tensorType.getElementType().getAsOpaquePointer());
  }
  return nullptr;
}

SchemeValue mlir_type_get_shape(SchemeValue type_ptr) {
  if (!type_ptr) return Snil;
  mlir::Type type = mlir::Type::getFromOpaquePointer(type_ptr);
  if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    llvm::ArrayRef<int64_t> shape = tensorType.getShape();
    // Convert to Scheme list
    ptr list = Snil;
    for (int i = shape.size() - 1; i >= 0; --i) {
      list = Scons(Sinteger(shape[i]), list);
    }
    return list;
  }
  return Snil;
}

int mlir_type_get_rank(SchemeValue type_ptr) {
  if (!type_ptr) return -1;
  mlir::Type type = mlir::Type::getFromOpaquePointer(type_ptr);
  if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    return tensorType.getRank();
  }
  return -1;
}

SchemeValue mlir_value_get_type(SchemeValue value_ptr) {
  if (!value_ptr) return nullptr;
  mlir::Value value = mlir::Value::getFromOpaquePointer(value_ptr);
  return const_cast<void*>(value.getType().getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// Phase 2: Operation/Value Navigation FFI
//===----------------------------------------------------------------------===//

SchemeValue mlir_operation_get_parent(SchemeValue op_ptr) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  mlir::Operation* parent = op->getParentOp();
  return parent;
}

SchemeValue mlir_operation_get_operand_value(SchemeValue op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumOperands())
    return nullptr;
  mlir::Value operand = op->getOperand(index);
  return const_cast<void*>(operand.getAsOpaquePointer());
}

SchemeValue mlir_operation_get_result_value(SchemeValue op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumResults())
    return nullptr;
  mlir::Value result = op->getResult(index);
  return const_cast<void*>(result.getAsOpaquePointer());
}

SchemeValue mlir_operation_get_loc(SchemeValue op_ptr) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  return const_cast<void*>(op->getLoc().getAsOpaquePointer());
}

SchemeValue mlir_operation_get_block_argument(SchemeValue op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  // Walk up to parent function
  while (op && !llvm::isa<mlir::func::FuncOp>(op)) {
    op = op->getParentOp();
  }
  if (!op)
    return nullptr;

  auto funcOp = llvm::cast<mlir::func::FuncOp>(op);
  if (index < 0 || index >= (int)funcOp.getNumArguments())
    return nullptr;

  mlir::Value arg = funcOp.getArgument(index);
  return const_cast<void*>(arg.getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// Phase 3: IR Construction FFI (OpBuilder)
//===----------------------------------------------------------------------===//

// TODO: These require OpBuilder/PatternRewriter context integration
// For now, stubs that log error

SchemeValue mlir_create_placeholder_op(SchemeValue ctx_value, SchemeValue input_value,
                                       SchemeValue result_type, int placeholder_type_int) {
  mlir_log_error("mlir_create_placeholder_op: Not yet implemented - requires PatternRewriter context");
  return nullptr;
}

SchemeValue mlir_create_cast_op(SchemeValue ctx_value, SchemeValue input_value,
                                SchemeValue output_value, SchemeValue result_type) {
  mlir_log_error("mlir_create_cast_op: Not yet implemented - requires PatternRewriter context");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Phase 4: Pattern Rewriter FFI
//===----------------------------------------------------------------------===//

int mlir_replace_op(SchemeValue old_op, SchemeValue new_value) {
  mlir_log_error("mlir_replace_op: Not yet implemented - requires PatternRewriter context");
  return 0;
}

int mlir_erase_op(SchemeValue op) {
  mlir_log_error("mlir_erase_op: Not yet implemented - requires PatternRewriter context");
  return 0;
}

void mlir_notify_match_failure(SchemeValue op, const char* reason) {
  mlir_log_debug((std::string("Pattern match failure: ") + reason).c_str());
}

} // extern "C"

namespace mlir {
namespace hipsr {

// Register all MLIR foreign functions in Scheme
void registerMlirForeignFunctions() {
  // Register C functions so Scheme can call them via foreign-procedure
  Sregister_symbol("mlir_operation_get_name", (void*)mlir_operation_get_name);
  Sregister_symbol("mlir_operation_num_operands", (void*)mlir_operation_num_operands);
  Sregister_symbol("mlir_operation_num_results", (void*)mlir_operation_num_results);
  Sregister_symbol("mlir_operation_get_operand", (void*)mlir_operation_get_operand);
  Sregister_symbol("mlir_operation_get_result", (void*)mlir_operation_get_result);
  Sregister_symbol("mlir_operation_walk", (void*)mlir_operation_walk);

  // Register logging functions
  Sregister_symbol("mlir_log_trace", (void*)mlir_log_trace);
  Sregister_symbol("mlir_log_debug", (void*)mlir_log_debug);
  Sregister_symbol("mlir_log_info", (void*)mlir_log_info);
  Sregister_symbol("mlir_log_warning", (void*)mlir_log_warning);
  Sregister_symbol("mlir_log_error", (void*)mlir_log_error);
  Sregister_symbol("mlir_log_fatal", (void*)mlir_log_fatal);

  // Phase 1: Type System FFI
  Sregister_symbol("mlir_type_is_ranked_tensor", (void*)::mlir_type_is_ranked_tensor);
  Sregister_symbol("mlir_type_get_element_type", (void*)::mlir_type_get_element_type);
  Sregister_symbol("mlir_type_get_shape", (void*)::mlir_type_get_shape);
  Sregister_symbol("mlir_type_get_rank", (void*)::mlir_type_get_rank);
  Sregister_symbol("mlir_value_get_type", (void*)::mlir_value_get_type);

  // Phase 2: Operation/Value Navigation FFI
  Sregister_symbol("mlir_operation_get_parent", (void*)::mlir_operation_get_parent);
  Sregister_symbol("mlir_operation_get_operand_value", (void*)::mlir_operation_get_operand_value);
  Sregister_symbol("mlir_operation_get_result_value", (void*)::mlir_operation_get_result_value);
  Sregister_symbol("mlir_operation_get_loc", (void*)::mlir_operation_get_loc);
  Sregister_symbol("mlir_operation_get_block_argument", (void*)::mlir_operation_get_block_argument);

  // Phase 3: IR Construction FFI (OpBuilder) - TODO: needs PatternRewriter integration
  Sregister_symbol("mlir_create_placeholder_op", (void*)::mlir_create_placeholder_op);
  Sregister_symbol("mlir_create_cast_op", (void*)::mlir_create_cast_op);

  // Phase 4: Pattern Rewriter FFI - TODO: needs PatternRewriter integration
  Sregister_symbol("mlir_replace_op", (void*)::mlir_replace_op);
  Sregister_symbol("mlir_erase_op", (void*)::mlir_erase_op);
  Sregister_symbol("mlir_notify_match_failure", (void*)::mlir_notify_match_failure);

  LLVM_DEBUG(llvm::dbgs() << "Registered " << 27 << " MLIR FFI functions\n");
}

} // namespace hipsr
} // namespace mlir
