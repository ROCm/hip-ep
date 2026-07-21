/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_DIALECT_H
#define HIPSR_DIALECT_H

#include "mlir/IR/Dialect.h"
// Needed by the generated attribute parser/printer.
#include "mlir/IR/OpImplementation.h"

// The generated dialect decls (below) reference these in the file-map cache
// declared via `extraClassDeclaration` in HipsrDialect.td.
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <mutex>

// The dialect holds a non-owning FileSystem* (constants sidecar sink). Forward
// declaration only -- avoids pulling morphizen-foundation into every dialect
// consumer.
namespace morphizen {
class FileSystem;
} // namespace morphizen

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h.inc"

// Enum header first: MemorySpaceAttr uses MemorySpaceKind.
#include "hip/Dialect/Hipsr/IR/HipsrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.h.inc"

// Type declarations (e.g. !hipsr.context).
#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h.inc"

// Predicates for the Hipsr_*MemRef type constraints (used by op verifiers once
// ops adopt them).
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"

#endif // HIPSR_DIALECT_H
