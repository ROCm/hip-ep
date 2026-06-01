/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_COMPILER_INITALLPASSES_H
#define HIP_COMPILER_INITALLPASSES_H

#include "hip/Conversion/Passes.h"
#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"

namespace hip::compiler {

namespace detail {
/// Minimal ONNX dialect stub that claims the "onnx" namespace and permits
/// unknown operations.  This avoids depending on the full onnx-mlir library.
///
/// Beyond simply allowing unknown operations, the dialect also exposes a
/// process-global fallback registry for OpInterfaces. The override of
/// `getRegisteredInterfaceForOp` consults the static `s_fallbacks` map
/// keyed by interface `TypeID`. Callers (e.g. the OnnxResultTypeInference
/// rules library, see lib/Conversion/OnnxToHip/OnnxResultTypeInference.cpp)
/// install a singleton FallbackModel pointer at startup and never remove
/// it. Behaviour is unchanged when no fallback is installed for a given
/// interface (the default base-class no-op is restored). One fallback
/// slot per interface type; the same interface registered twice with
/// different fallbacks is undefined (in practice a single rules library
/// owns the slot for the whole process).
class OnnxStubDialect : public mlir::Dialect {
public:
  explicit OnnxStubDialect(mlir::MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx,
                mlir::TypeID::get<OnnxStubDialect>()) {
    allowUnknownOperations();
  }
  static constexpr llvm::StringLiteral getDialectNamespace() { return "onnx"; }

  // Dialect-level fallback dispatch for unregistered onnx.* ops.
  void *getRegisteredInterfaceForOp(mlir::TypeID interfaceID,
                                    mlir::OperationName) override {
    auto it = s_fallbacks.find(interfaceID);
    return it != s_fallbacks.end() ? it->second : nullptr;
  }

  // Install / remove a fallback for `interfaceID`. Caller owns the
  // lifetime of the pointed-to model (typically a function-local static
  // alive for the process lifetime). `unregister` exists for symmetry
  // and tests; production callers register once and never remove.
  static void registerInterfaceFallback(mlir::TypeID interfaceID,
                                        void *fallback) {
    s_fallbacks[interfaceID] = fallback;
  }
  static void unregisterInterfaceFallback(mlir::TypeID interfaceID) {
    s_fallbacks.erase(interfaceID);
  }

private:
  // Brace-init bypasses the `explicit DenseMap(unsigned)` constructor's
  // implicit-call ban that bites a default-init of an inline static.
  inline static llvm::DenseMap<mlir::TypeID, void *> s_fallbacks{};
};
} // namespace detail

/// Register all required dialects into a DialectRegistry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<detail::OnnxStubDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);
}

/// Load all required dialects into an MLIRContext.
inline void loadAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

/// Register all passes and pipelines.
inline void registerAllPasses() {
  mlir::registerCanonicalizerPass();
  mlir::hip::registerOptimizeMemRefsPass();
  mlir::hip::registerPoolAllocsPass();
  mlir::hip::registerSimplifyOnnxPass();
  registerConversionPasses();
}

} // namespace hip::compiler

#endif // HIP_COMPILER_INITALLPASSES_H
