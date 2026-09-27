/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scheme-bindings"

// Forward declarations of per-dialect registration functions
namespace mlir {
namespace hipsr {
extern void registerConversionBindings();
extern void registerCoreBindings();
extern void registerHipsrBindings();
extern void registerOnnxBindings();
extern void registerShapeBindings();
extern void registerTensorBindings();
extern void registerLoggingBindings();
} // namespace hipsr
} // namespace mlir

namespace mlir {
namespace hipsr {

// Box MLIR C++ objects as GC-safe Scheme integers (Sunsigned64).
// mlir::Value opaque pointer has tag bits (bit 0 = BlockArgument).
ptr makeSchemeOperation(mlir::Operation* op) {
  return Sunsigned64(reinterpret_cast<uint64_t>(op));
}
ptr makeSchemeValue(mlir::Value val) {
  return Sunsigned64(reinterpret_cast<uint64_t>(val.getAsOpaquePointer()));
}
ptr makeSchemeType(mlir::Type type) {
  return Sunsigned64(reinterpret_cast<uint64_t>(type.getAsOpaquePointer()));
}
ptr makeSchemeAttribute(mlir::Attribute attr) {
  return Sunsigned64(reinterpret_cast<uint64_t>(attr.getAsOpaquePointer()));
}

} // namespace hipsr
} // namespace mlir

namespace mlir {
namespace hipsr {

void registerMlirForeignFunctions() {
  registerConversionBindings();
  registerCoreBindings();
  registerHipsrBindings();
  registerOnnxBindings();
  registerShapeBindings();
  registerTensorBindings();
  registerLoggingBindings();
  LLVM_DEBUG(llvm::dbgs() << "All MLIR FFI functions registered\n");
}

} // namespace hipsr
} // namespace mlir
