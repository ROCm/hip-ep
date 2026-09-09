/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H

#include <string>
#include <vector>
#include <functional>

namespace mlir {
class Operation;
class Value;
class Type;
class Attribute;

namespace hipsr {

using SchemeValue = void*;

// Initialize Scheme runtime and register MLIR FFI bindings
bool initializeSchemeRuntime();

// Call a Scheme function with primitive arguments (legacy API)
std::string callSchemeFunction(const char* functionName,
                                const std::vector<SchemeValue>& args);

// Create Scheme values from C++ primitives
SchemeValue makeSchemeString(const char* str);
SchemeValue makeSchemeInteger(long value);

// MLIR C++ to Scheme conversions - wrap MLIR objects as foreign pointers
SchemeValue makeSchemeOperation(mlir::Operation* op);
SchemeValue makeSchemeValue(mlir::Value val);
SchemeValue makeSchemeType(mlir::Type type);
SchemeValue makeSchemeAttribute(mlir::Attribute attr);

// Register MLIR foreign functions accessible from Scheme
void registerMlirForeignFunctions();

// Load and evaluate a Scheme script file
bool loadSchemeScript(const char* scriptPath);

// Call a Scheme function with a single MLIR operation argument
// Used to invoke Scheme-defined pass entry points
void callSchemePassFunction(const char* functionName, mlir::Operation* op);

} // namespace hipsr
} // namespace mlir

#endif
