/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H

#include "hip/Dialect/Hipsr/Scheme/Runtime/ChezSchemeInterpreter.h"

namespace mlir {
class Operation;
class Value;
class Type;
class Attribute;

namespace hipsr {

// MLIR C++ to Scheme conversions - wrap MLIR objects as foreign pointers
SchemeValue makeSchemeOperation(mlir::Operation* op);
SchemeValue makeSchemeValue(mlir::Value val);
SchemeValue makeSchemeType(mlir::Type type);
SchemeValue makeSchemeAttribute(mlir::Attribute attr);

// Register all MLIR foreign functions accessible from Scheme
void registerMlirForeignFunctions();

} // namespace hipsr
} // namespace mlir

#endif
