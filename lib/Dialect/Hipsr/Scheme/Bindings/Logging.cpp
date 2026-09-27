/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Scheme/Bindings/SchemeMlirBindings.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "scheme-logging-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

extern "C" {

void mlir_log_trace(const char* msg) {
  if (mlir::hipsr::ChezSchemeInterpreter::getLogLevel() <= mlir::hipsr::SchemeLogLevel::Trace)
    llvm::errs() << "[trace] " << msg << "\n";
}

void mlir_log_debug(const char* msg) {
  if (mlir::hipsr::ChezSchemeInterpreter::getLogLevel() <= mlir::hipsr::SchemeLogLevel::Debug)
    llvm::errs() << "[debug] " << msg << "\n";
}

void mlir_log_info(const char* msg) {
  if (mlir::hipsr::ChezSchemeInterpreter::getLogLevel() <= mlir::hipsr::SchemeLogLevel::Info)
    llvm::errs() << "[info] " << msg << "\n";
}

void mlir_log_warning(const char* msg) {
  if (mlir::hipsr::ChezSchemeInterpreter::getLogLevel() <= mlir::hipsr::SchemeLogLevel::Warning)
    llvm::errs() << "[warning] " << msg << "\n";
}

void mlir_log_error(const char* msg) {
  if (mlir::hipsr::ChezSchemeInterpreter::getLogLevel() <= mlir::hipsr::SchemeLogLevel::Error)
    llvm::errs() << "[error] " << msg << "\n";
}

void mlir_log_fatal(const char* msg) {
  if (mlir::hipsr::ChezSchemeInterpreter::getLogLevel() <= mlir::hipsr::SchemeLogLevel::Fatal)
    llvm::errs() << "[fatal] " << msg << "\n";
}

//===----------------------------------------------------------------------===//
// Phase 1: Type System FFI
//===----------------------------------------------------------------------===//

// Set memory space on a RankedTensorType
// Returns: new Type* with memory space set

} // extern "C"

namespace mlir {
namespace hipsr {

void registerLoggingBindings() {
  Sregister_symbol("mlir_log_trace", (void*)::mlir_log_trace);
  Sregister_symbol("mlir_log_debug", (void*)::mlir_log_debug);
  Sregister_symbol("mlir_log_info", (void*)::mlir_log_info);
  Sregister_symbol("mlir_log_warning", (void*)::mlir_log_warning);
  Sregister_symbol("mlir_log_error", (void*)::mlir_log_error);
  Sregister_symbol("mlir_log_fatal", (void*)::mlir_log_fatal);
}

} // namespace hipsr
} // namespace mlir
