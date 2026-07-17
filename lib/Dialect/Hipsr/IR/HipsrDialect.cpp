/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "hip/Dialect/Hipsr/IR/HipsrCastOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrPlaceholderOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::hipsr;

#include "hip/Dialect/Hipsr/IR/HipsrDialect.cpp.inc"

// Enum code first: the attribute's parser/printer below calls these
// enum name<->value helpers.
#include "hip/Dialect/Hipsr/IR/HipsrEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrTypes.cpp.inc"

void HipsrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.cpp.inc"
      ,
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.cpp.inc"
      ,
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.cpp.inc"
      ,
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.cpp.inc"
      ,
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrCastOp.cpp.inc"
      ,
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.cpp.inc"
      ,
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrPlaceholderOp.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hip/Dialect/Hipsr/IR/HipsrTypes.cpp.inc"
      >();
}

llvm::MemoryBuffer *HipsrDialect::getOrLoadFileMap(llvm::StringRef path) {
  // The dialect is a shared singleton and MLIR runs passes multi-threaded;
  // guard the cache against concurrent lookups/inserts.
  std::lock_guard<std::mutex> lock(fileMapsMutex);

  auto it = fileMaps.find(path);
  if (it != fileMaps.end())
    return it->second.get();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOr =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false);
  if (!bufOr)
    return nullptr;

  llvm::MemoryBuffer *raw = bufOr->get();
  fileMaps[path] = std::move(*bufOr);
  return raw;
}
