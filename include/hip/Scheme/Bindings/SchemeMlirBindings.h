/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_SCHEME_BINDINGS_SCHEMEMLIR_BINDINGS_H
#define HIP_SCHEME_BINDINGS_SCHEMEMLIR_BINDINGS_H

#include "hip/Scheme/Interpreter/ChezSchemeInterpreter.h"

// Logging functions (defined in Logging.cpp).
// Declared here so any Bindings/*.cpp file can call them without
// repeating extern "C" forward declarations inline.
extern "C" {
void mlir_log_trace(const char* msg);
void mlir_log_debug(const char* msg);
void mlir_log_info(const char* msg);
void mlir_log_warning(const char* msg);
void mlir_log_error(const char* msg);
void mlir_log_fatal(const char* msg);
} // extern "C"

namespace mlir {
class Operation;
class Value;
class Type;
class Attribute;

namespace hipsr {

// Register all MLIR foreign functions accessible from Scheme
void registerMlirForeignFunctions();

// Per-module registration functions — implemented in the corresponding .cpp files
void registerConversionBindings();
void registerCoreBindings();
void registerHipsrBindings();
void registerOnnxBindings();
void registerShapeBindings();
void registerTensorBindings();
void registerLoggingBindings();

// MLIR C++ to Scheme conversions - wrap MLIR objects as GC-safe Scheme integers
ptr makeSchemeOperation(mlir::Operation* op);
ptr makeSchemeValue(mlir::Value val);
ptr makeSchemeType(mlir::Type type);
ptr makeSchemeAttribute(mlir::Attribute attr);

} // namespace hipsr
} // namespace mlir

#endif
