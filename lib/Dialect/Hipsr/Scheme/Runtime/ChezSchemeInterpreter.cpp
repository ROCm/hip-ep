/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Scheme/Runtime/ChezSchemeInterpreter.h"
#include "hip/Dialect/Hipsr/Scheme/Runtime/SchemeMlirBindings.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"

// Note: scheme.h already included via ChezSchemeInterpreter.h
// Do NOT include it again here to avoid redefinition errors with static inline functions

#include "ChezBootPetite.h"
#include "ChezBootScheme.h"

// Forward declaration from SchemeMlirBindings.cpp
namespace mlir {
namespace hipsr {
void setSchemeLogLevel(SchemeLogLevel level);
}
}

namespace {
const size_t petite_boot_size = sizeof(petite_boot_data) - 1;
const size_t scheme_boot_size = sizeof(scheme_boot_data) - 1;

// Custom init called by Sbuild_heap before loading boot files
static void custom_init() {
  // Register all MLIR foreign functions
  mlir::hipsr::registerMlirForeignFunctions();
}

// Global log level
static mlir::hipsr::SchemeLogLevel current_log_level = mlir::hipsr::SchemeLogLevel::Warning;

} // anonymous namespace

namespace mlir {
namespace hipsr {

// Static member initialization
bool ChezSchemeInterpreter::initialized = false;
SchemeLogLevel ChezSchemeInterpreter::logLevel = SchemeLogLevel::Warning;

// Parse log level from string
SchemeLogLevel ChezSchemeInterpreter::parseLogLevel(const std::string& level) {
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

// Set global log level
void ChezSchemeInterpreter::setLogLevel(SchemeLogLevel level) {
  current_log_level = level;
  logLevel = level;
  // Also update the log level in SchemeMlirBindings.cpp
  mlir::hipsr::setSchemeLogLevel(level);
}

void ChezSchemeInterpreter::initialize(SchemeLogLevel level) {
  if (initialized) {
    return;  // Already initialized
  }

  logLevel = level;
  current_log_level = level;

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] ChezSchemeInterpreter1: Initializing Chez Scheme runtime\n";
  }

  // Initialize Scheme runtime
  Sscheme_init(nullptr);

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] ChezSchemeInterpreter: Registering embedded boot files\n";
  }

  // Register embedded boot files
  Sregister_boot_file_bytes("petite.boot",
      const_cast<void*>(static_cast<const void*>(petite_boot_data)),
      petite_boot_size);
  Sregister_boot_file_bytes("scheme.boot",
      const_cast<void*>(static_cast<const void*>(scheme_boot_data)),
      scheme_boot_size);

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] ChezSchemeInterpreter: Building heap from embedded boot files\n";
  }

  // Build heap and call custom_init (which registers foreign functions)
  Sbuild_heap(nullptr, custom_init);
  initialized = true;

  // Add Scheme library paths (source and binary)
  addLibraryPath(SCHEME_LIBRARIES_DIR, SCHEME_BINARY_DIR);
  addLibraryPath(RIME_DIR, RIME_DIR);

  if (logLevel <= SchemeLogLevel::Info) {
    llvm::errs() << "[info] ChezSchemeInterpreter: Initialization complete\n";
  }
}

void ChezSchemeInterpreter::shutdown() {
  if (!initialized) {
    return;
  }

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] ChezSchemeInterpreter: Shutting down Scheme runtime\n";
  }

  // Chez Scheme doesn't require explicit cleanup
  initialized = false;
}

void ChezSchemeInterpreter::addLibraryPath(const char* src_path, const char* bin_path) {
  if (!initialized) {
    llvm::errs() << "[error] ChezSchemeInterpreter: Cannot add library path - runtime not initialized\n";
    return;
  }

  // Build Scheme code to add paths as a single (src . bin) pair
  std::string code =
    "(library-directories "
    "  (cons (cons \"" + std::string(src_path) + "\" \"" + std::string(bin_path) + "\") "
    "  (library-directories)))";

  if (!eval(code.c_str())) {
    llvm::errs() << "[error] ChezSchemeInterpreter: Failed to add library paths: "
                 << src_path << ", " << bin_path << "\n";
    return;
  }

  if (logLevel <= SchemeLogLevel::Debug) {
    // Define a helper function to get library-directories as a string and call it
    std::string helper_code =
      "(define (__get-libdirs-string) "
      "  (call-with-port (open-output-string) "
      "    (lambda (p) (write (library-directories) p) (get-output-string p))))";
    eval(helper_code.c_str());

    std::string result = callFunction("__get-libdirs-string", {});
    llvm::errs() << "[debug] ChezSchemeInterpreter: library-directories = " << result << "\n";
  }
}

bool ChezSchemeInterpreter::load(const char* scriptPath) {
  if (!initialized) {
    llvm::errs() << "[error] ChezSchemeInterpreter: Cannot load script - runtime not initialized\n";
    return false;
  }

  // R5RS load: (load scriptPath)
  ptr load_sym = Stop_level_value(Sstring_to_symbol("load"));
  ptr path_str = Sstring(scriptPath);
  Scall1(load_sym, path_str);
  return true;
}

bool ChezSchemeInterpreter::eval(const char* code) {
  if (!initialized) {
    llvm::errs() << "[error] ChezSchemeInterpreter: Cannot evaluate code - runtime not initialized\n";
    return false;
  }

  // R5RS eval: (eval (read (open-string-input-port code)))
  ptr eval_sym = Stop_level_value(Sstring_to_symbol("eval"));
  ptr read_sym = Stop_level_value(Sstring_to_symbol("read"));
  ptr open_port_sym = Stop_level_value(Sstring_to_symbol("open-string-input-port"));

  ptr port = Scall1(open_port_sym, Sstring(code));
  ptr expr = Scall1(read_sym, port);
  Scall1(eval_sym, expr);

  return true;
}

// Create Scheme values from C++ primitives
ptr ChezSchemeInterpreter::makeString(const char* str) {
  return Sstring(str);
}

ptr ChezSchemeInterpreter::makeInteger(long value) {
  return Sinteger(value);
}

// Call a Scheme function with primitive arguments
std::string ChezSchemeInterpreter::callFunction(const char* functionName,
                                                const std::vector<ptr>& args) {
  if (!initialized)
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

// Call a Scheme function with a single MLIR operation argument
void ChezSchemeInterpreter::callPassFunction(const char* functionName, mlir::Operation* op) {
  if (!initialized)
    return;

  llvm::errs() << "[callPassFunction] Looking up function: " << functionName << "\n";

  ptr func = Stop_level_value(Sstring_to_symbol(functionName));
  if (func == Sfalse) {
    llvm::errs() << "Warning: Scheme function '" << functionName << "' not found\n";
    return;
  }

  llvm::errs() << "[callPassFunction] Function found, creating scheme operation ptr\n";
  ptr schemeOp = makeSchemeOperation(op);
  llvm::errs() << "[callPassFunction] Calling Scheme function with op=" << reinterpret_cast<uintptr_t>(op) << "\n";
  Scall1(func, schemeOp);
  llvm::errs() << "[callPassFunction] Scheme function returned\n";
}

} // namespace hipsr
} // namespace mlir
