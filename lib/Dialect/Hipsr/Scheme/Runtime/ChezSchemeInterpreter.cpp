/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Scheme/Runtime/ChezSchemeInterpreter.h"
#include "SchemeBindings.h"

#include "llvm/Support/raw_ostream.h"

// Include Chez Scheme C API header
extern "C" {
#include "boot/ta6le/scheme.h"
}

#include "ChezBootPetite.h"
#include "ChezBootScheme.h"

namespace {
const size_t petite_boot_size = sizeof(petite_boot_data) - 1;
const size_t scheme_boot_size = sizeof(scheme_boot_data) - 1;

// Cached Scheme symbols for script loading

// Custom init called by Sbuild_heap before loading boot files
static void custom_init() {
  // Register all MLIR foreign functions
  mlir::hipsr::registerMlirForeignFunctions();
}

} // anonymous namespace

namespace mlir {
namespace hipsr {

ChezSchemeInterpreter::ChezSchemeInterpreter(SchemeLogLevel logLevel)
    : logLevel(logLevel) {

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] ChezSchemeInterpreter: Initializing Chez Scheme runtime\n";
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
  initialized = true;  if (logLevel <= SchemeLogLevel::Info) {    llvm::errs() << "[info] ChezSchemeInterpreter: Initialization complete\n";  }

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] ChezSchemeInterpreter: Caching Scheme symbols\n";
    llvm::errs() << "[info] ChezSchemeInterpreter: Initialization complete\n";
  }
}

ChezSchemeInterpreter::~ChezSchemeInterpreter() {
  if (!initialized) {
    return;
  }

  if (logLevel <= SchemeLogLevel::Debug) {
    llvm::errs() << "[debug] ChezSchemeInterpreter: Shutting down Scheme runtime\n";
  }

  // Chez Scheme doesn't require explicit cleanup
  // The runtime will clean up on process exit

  initialized = false;
}

bool ChezSchemeInterpreter::loadScript(const char* scriptPath) {
  if (!initialized) {
    llvm::errs() << "[error] ChezSchemeInterpreter: Cannot load script - runtime not initialized\n";
    return false;
  }

  return loadSchemeScript(scriptPath);
}

bool ChezSchemeInterpreter::evaluateCode(const char* code) {
  if (!initialized) {
    llvm::errs() << "[error] ChezSchemeInterpreter: Cannot evaluate code - runtime not initialized\n";
    return false;
  }

  // Direct R5RS eval: (eval (read (open-string-input-port code)))
  ptr eval_sym = Stop_level_value(Sstring_to_symbol("eval"));
  ptr read_sym = Stop_level_value(Sstring_to_symbol("read"));
  ptr open_port_sym = Stop_level_value(Sstring_to_symbol("open-string-input-port"));
  
  ptr port = Scall1(open_port_sym, Sstring(code));
  ptr expr = Scall1(read_sym, port);
  Scall1(eval_sym, expr);
  
  return true;
}

} // namespace hipsr
} // namespace mlir
